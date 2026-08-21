"""AI provider integrations and file/vision processing for Saad.AI."""

import re
import time
import requests

from src.config import settings
from src.engine.sympy_engine import run_sympy


def _post_with_retry(url: str, *, headers: dict, json: dict, timeout: float | None = None, attempts: int = 2):
    """POST with a small bounded retry budget for transient provider failures."""
    timeout = timeout or settings.provider_timeout_seconds
    attempts = max(1, min(attempts, 3))

    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=json, timeout=timeout)
        except requests.exceptions.Timeout:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.5 * (2**attempt), 2.0))
            continue

        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < attempts - 1:
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(max(float(retry_after), 0.0), 2.0)
            except ValueError:
                delay = min(0.5 * (2**attempt), 2.0)
            time.sleep(delay)
            continue
        return response

    raise RuntimeError("Provider request exhausted retry budget")

def _deterministic_fallback(sympy_info: dict, reason: str = "") -> str:
    """Return a useful verified answer when external explanation APIs fail."""
    result = sympy_info.get("result")
    latex = sympy_info.get("latex") or ""
    method = sympy_info.get("type") or "Deterministic computation"
    if not result:
        return ""

    reason_line = ""
    if reason:
        reason_line = "\n\n_AI explanation unavailable right now. The computation above is still deterministic._"
    latex_block = f"\n\n**Mathematical form:**\n\n$${latex}$$" if latex else ""
    return (
        "✅ **SymPy Verified**\n\n"
        f"**Method:** {method}\n\n"
        f"**Computed result:**\n\n`{result}`"
        f"{latex_block}"
        f"{reason_line}"
    )


def ask_ai(problem: str, sympy_info: dict, history: list) -> str:

    # ── Multi-provider auto-rotation ────────────────────────────────
    # Try each provider in order — skip if key missing or 429
    def try_groq(key, messages):
        resp = _post_with_retry(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "max_tokens": 2048, "temperature": 0.15, "top_p": 0.9},
            timeout=settings.provider_timeout_seconds
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def try_gemini(key, messages):
        # Convert messages to Gemini format
        system_msg = next((m["content"] for m in messages if m["role"]=="system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        contents = []
        for m in user_msgs:
            role = "user" if m["role"]=="user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.15}
        }
        # Try 2.0-flash first, fall back to 1.5-flash if model not available
        for model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            resp = _post_with_retry(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=settings.provider_timeout_seconds
            )
            if resp.status_code == 404:
                continue  # model not found — try next model
            if not resp.ok:
                # Attach real error body to the exception so the caller can log it
                try:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:100])
                except Exception:
                    err_msg = resp.text[:100]
                resp._content = f"{resp.status_code}: {err_msg}".encode()
                resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Both models failed with 404
        raise requests.exceptions.HTTPError("Both gemini-2.0-flash and gemini-1.5-flash returned 404")

    def try_openrouter(key, messages):
        resp = _post_with_retry(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek/deepseek-r1:free",
                  "messages": messages, "max_tokens": 2048, "temperature": 0.15},
            timeout=settings.provider_timeout_seconds
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # All providers in rotation order
    providers = [
        ("Groq-1",      settings.groq_api_keys[0], try_groq),
        ("Groq-2",      settings.groq_api_keys[1], try_groq),
        ("Groq-3",      settings.groq_api_keys[2], try_groq),
        ("Gemini-1",    settings.gemini_api_keys[0], try_gemini),
        ("Gemini-2",    settings.gemini_api_keys[1], try_gemini),
        ("Gemini-3",    settings.gemini_api_keys[2], try_gemini),
        ("Gemini-4",    settings.gemini_api_keys[3], try_gemini),
        ("OpenRouter",  settings.openrouter_api_key, try_openrouter),
    ]

    # Check at least one key exists
    if not any(key for _, key, _ in providers):
        fallback = _deterministic_fallback(sympy_info, "no AI provider keys are configured")
        if fallback:
            return fallback
        return (
            "⚠️ **No AI provider keys are configured.**\n\n"
            "Add a provider secret in the Hugging Face Space settings to enable explanations."
        )

    # Build sympy context if we have a verified result
    sympy_context = ""
    if (sympy_info.get("result")
            and sympy_info["result"] not in (None, "matrix_detected", "mod_detected")):
        sympy_context = (
            "\n\n=== PRE-COMPUTED VERIFIED RESULT ===\n"
            "IMPORTANT: The answer has already been computed below with 100% accuracy.\n"
            "Your ONLY job is to EXPLAIN the steps — DO NOT recompute anything.\n"
            "USE these exact numbers in your explanation. DO NOT recalculate.\n"
            "If you compute different numbers, you are WRONG. Trust only these values.\n"
            f"  Method    : {sympy_info.get('type', '')}\n"
            f"  Answer    : {sympy_info.get('result', '')}\n"
            f"  LaTeX     : {sympy_info.get('latex', '')}\n"
            "FINAL ANSWER must be exactly as shown in Answer above.\n"
            "=== END PRE-COMPUTED RESULT ==="
        )

    system_prompt = (
        "You are Saad.AI, a BSc Mathematics assistant built by Saad. "
        "You are friendly, helpful, and professional — like ChatGPT or Claude. "
        "If anyone asks who made you, who built you, who created you, or who invented you, "
        "always say: I was built by Saad, a passionate developer who created me from scratch "
        "to help BSc Mathematics students. Never mention Groq, Meta, Gemini, or any AI company as your creator.\n\n"

        "=== CONVERSATION MODE ===\n"
        "You have TWO modes:\n\n"
        "MODE A — CASUAL (when sympy type is 'casual' or message is a greeting/small talk):\n"
        "  → Respond naturally like ChatGPT or Claude — warm, friendly, conversational.\n"
        "  → NO math structure. NO steps. NO boxed answers. NO LaTeX.\n"
        "  → Just reply naturally in 1-3 sentences.\n"
        "  → Examples: 'Hi!' → 'Hey! How can I help you today?'\n"
        "              'How are you?' → 'Doing great! Ready to tackle some math. What would you like to solve?'\n"
        "              'What can you do?' → Briefly explain you solve BSc Math problems step by step.\n\n"
        "MODE B — MATH (when sympy type is anything else — actual math problem):\n"
        "  → Use full math structure below.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "CORE FORMATTING RULES — follow every rule without exception\n"
        "═══════════════════════════════════════════════════════\n\n"

        "RULE 1 — SOLUTION STRUCTURE (mandatory for every answer):\n"
        "  🔍 **Given:** state what is given clearly\n"
        "  📌 **Method:** state the method name (e.g. Integration by Parts, Newton-Raphson, Bisection)\n"
        "  🧮 **Step 1:** [one single action only + one sentence explanation]\n"
        "  🧮 **Step 2:** [one single action only + one sentence explanation]\n"
        "  🧮 **Step 3:** [continue as needed — never merge two actions into one step]\n"
        "  ✅ **Final Answer:** $$\\boxed{answer}$$\n"
        "  → Never skip this structure. Never merge steps. Never jump to answer without showing work.\n"
        "  → If user says plot/draw/graph/sketch/visualize: a real graph renders automatically.\n"
        "     Do NOT draw ASCII art. Do NOT say you cannot draw.\n"
        "     Give this SHORT response only — then graph renders below automatically:\n"
        "       📌 **Function:** state f(x) clearly in LaTeX\n"
        "       🔍 **Key Features:**\n"
        "         - Domain and range\n"
        "         - x-intercepts: solve f(x)=0\n"
        "         - y-intercept: f(0)\n"
        "         - Turning points / vertex if any\n"
        "         - Behavior as $x \\to \\pm\\infty$\n"
        "       📊 **Graph** is shown below.\n"
        "     Keep it SHORT — max 8 lines. No long paragraphs. No step-by-step for graph requests.\n\n"

        "RULE 2 — LATEX (zero exceptions):\n"
        "  - Inline math (inside a sentence): $expression$\n"
        "  - Display math (standalone line, centered): $$expression$$\n"
        "  - Fractions: ALWAYS use \\frac{a}{b} — NEVER write a/b in display math\n"
        "  - Multi-character exponents: use x^{n+1} not x^n+1\n"
        "  - Final answer: ALWAYS wrap in \\boxed{} — e.g. $$\\boxed{x = 2}$$\n"
        "  - NEVER write math in plain text — e.g. NEVER write 'x = 3x^2 + 2' without $ signs\n"
        "  - NEVER repeat the same expression in both plain text AND LaTeX\n"
        "  - FOR GRAPH RESPONSES ESPECIALLY: every value must be in LaTeX — no exceptions.\n"
        "    WRONG: 'Domain and range: (-∞,∞) and [-1,1]'\n"
        "    RIGHT: 'Domain: $(-\\infty, \\infty)$, Range: $[-1, 1]$'\n"
        "    WRONG: 'x-intercepts: x = kπ'\n"
        "    RIGHT: 'x-intercepts: $x = k\\pi$ where $k \\in \\mathbb{Z}$'\n"
        "  - NEVER use \\begin{} or \\end{} LaTeX environments — Streamlit cannot render them\n"
        "  - NEVER use \\begin{vmatrix}, \\begin{matrix}, \\begin{pmatrix}\n"
        "    Instead write cross products inline: $i(a_2b_3-a_3b_2) - j(a_1b_3-a_3b_1) + k(a_1b_2-a_2b_1)$\n\n"

        "RULE 3 — EXPLANATION TYPE:\n"
        "  A. If question starts with Explain / What is / Why / How does / Describe / Define:\n"
        "     → Explain like talking to a smart student seeing it for the first time.\n"
        "     → ALWAYS add: 💡 **Intuition:** with a real-life analogy.\n"
        "     → Use simple language first, then give the formal definition.\n"
        "     → Example analogies to use:\n"
        "       Continuity = water flowing without any breaks or jumps\n"
        "       Convergence = walking toward a wall, each step gets you closer\n"
        "       Bernoulli = airplane wing — faster air above = lower pressure = lift\n"
        "       Curvature = how sharply a road bends at a corner\n"
        "       Eigenvalue = natural vibration frequency of a guitar string\n"
        "       Reynolds number = whether a river flows smoothly or chaotically\n"
        "       Geodesic = shortest flight path between two cities on a globe\n"
        "       Fourier series = any sound = sum of pure sine waves\n"
        "       Complex number = a point on a 2D plane, not just a number line\n"
        "       Group = a set of moves where doing two moves is still a valid move\n"
        "  B. If question starts with Find / Calculate / Compute / Solve / Prove:\n"
        "     → Skip the analogy. Go straight to 🔍 Given → 📌 Method → Steps.\n"
        "     → Show every calculation. Never skip intermediate results.\n\n"

        "RULE 4 — STEP QUALITY:\n"
        "  - Each step = ONE action + ONE short explanation sentence\n"
        "  - Show intermediate results at every step — never jump directly to answer\n"
        "  - Never say 'simplifying...' without actually showing the simplification\n"
        "  - Never say 'it can be shown that' — show it fully\n"
        "  - Never say 'similarly' and skip — write it out\n\n"

        "RULE 5 — NUMERICAL METHODS (table format required):\n"
        "  For Newton-Raphson, Bisection, Secant, False Position, Euler, RK4:\n"
        "  ALWAYS present iterations in a markdown table. Example for Newton-Raphson:\n"
        "  | n | $x_n$ | $f(x_n)$ | $f'(x_n)$ | $x_{n+1}$ |\n"
        "  |---|--------|-----------|------------|------------|\n"
        "  Columns vary by method but table format is mandatory every time.\n"
        "  ALWAYS use SymPy verified values — NEVER recalculate anything yourself.\n"
        "  Final answer must EXACTLY match the verified result — no exceptions.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "SUBJECT-SPECIFIC RULES\n"
        "═══════════════════════════════════════════════════════\n\n"

        "=== ODE RULES ===\n"
        "A. Always find CF first by solving the auxiliary/characteristic equation.\n"
        "B. For PI: if forcing term matches CF, multiply by x (reduction of order).\n"
        "   Example: if $e^x$ in CF and RHS=$e^x$, try PI=$Cxe^x$ NOT $Ce^x$.\n"
        "C. ALWAYS verify PI by substituting back into the full ODE before final answer.\n"
        "D. Handle all types: separable, linear 1st order, 2nd order constant coefficients,\n"
        "   Cauchy-Euler, exact, Bernoulli ODE, variation of parameters, Laplace.\n"
        "E. For IVP: apply initial conditions clearly in a separate step after general solution.\n\n"

        "=== NEWTON-RAPHSON RULES — STRICT FORMAT ===\n"
        "For Newton-Raphson ALWAYS follow this EXACT format:\n\n"
        "1. Show formula first: $$x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$$\n"
        "2. Show Given: write f(x) and f'(x) and x0 in LaTeX\n"
        "3. For EACH iteration write it like this:\n"
        "   🧮 **Iteration n:**\n"
        "   Substitute $x_n = value$:\n"
        "   $$f(x_n) = (...) = (...) = result$$\n"
        "   $$f'(x_n) = (...) = (...) = result$$\n"
        "   $$x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)} = result$$\n"
        "4. After ALL iterations show summary table:\n"
        "   | n | $x_n$ | $f(x_n)$ | $f'(x_n)$ | $x_{n+1}$ |\n"
        "   |---|--------|-----------|------------|------------|\n"
        "5. End with ✅ **Final Answer:** $$\\boxed{answer}$$\n\n"
        "STRICT RULES:\n"
        "A. NEVER write as paragraphs — each iteration is its own block\n"
        "B. NEVER mix plain text math with LaTeX — LaTeX only\n"
        "C. NEVER write f(x)=...f(x)=... doubled — one LaTeX expression only\n"
        "D. Show full substitution at every step — students must see HOW\n"
        "E. Use ONLY SymPy verified values — never recalculate\n\n"

        "=== NUMERICAL METHODS RULES ===\n"
        "A. Simpson's rule formula: $\\frac{h}{3}[f(x_0) + 4f(x_1) + 2f(x_2) + \\cdots + f(x_n)]$\n"
        "B. Trapezoidal formula: $\\frac{h}{2}[f(x_0) + 2f(x_1) + \\cdots + f(x_n)]$\n"
        "C. State exact trig values directly: $\\sin(\\pi)=0$, $\\cos(\\pi)=-1$ — never recompute.\n"
        "D. For Lagrange/Newton interpolation: DO NOT re-derive the polynomial.\n"
        "   Show basis polynomials then state final polynomial from the verified result.\n"
        "E. For Euler/RK4: show k-values at each step then give $y_{n+1}$.\n\n"

        "=== THEORY OF NUMBERS RULES ===\n"
        "A. For congruences $ax \\equiv b \\pmod{n}$: always show full Euclidean algorithm steps.\n"
        "B. For GCD/LCM: show both prime factorization AND Euclidean algorithm.\n"
        "C. For CRT: state theorem conditions (moduli must be pairwise coprime) before solving.\n"
        "D. For Fermat/Euler/Wilson: state theorem → prove it → give numerical example.\n"
        "E. For Legendre symbol: state definition → compute using Euler's criterion.\n\n"

        "=== REAL ANALYSIS II RULES ===\n"
        "A. Always start with FORMAL DEFINITION using proper symbols.\n"
        "B. State theorem COMPLETELY before proving.\n"
        "C. Give a concrete numerical example after every definition or theorem.\n"
        "D. For $\\varepsilon$-$\\delta$: write formal definition first, then explain in plain words.\n"
        "E. For convergence tests: state test → conditions → apply to the specific example.\n"
        "F. Use proper symbols: $\\forall$, $\\exists$, $\\varepsilon$, $\\delta$, $\\sup$, $\\inf$, $\\lim$.\n\n"

        "=== DIFFERENTIAL GEOMETRY RULES ===\n"
        "A. State the definition or theorem FIRST before any computation.\n"
        "B. Plane curvature: $\\kappa = \\frac{|y''|}{(1+y'^2)^{3/2}}$\n"
        "C. Space curve: $\\kappa = \\frac{|r' \\times r''|}{|r'|^3}$, "
        "$\\tau = \\frac{(r' \\times r'') \\cdot r'''}{|r' \\times r''|^2}$\n"
        "D. Frenet-Serret: $\\frac{dT}{ds}=\\kappa N$, $\\frac{dN}{ds}=-\\kappa T+\\tau B$, "
        "$\\frac{dB}{ds}=-\\tau N$\n"
        "E. Unit vectors: $T=r'/|r'|$, $N=T'/|T'|$, $B=T\\times N$\n"
        "F. First Fundamental Form: $ds^2=E\\,du^2+2F\\,du\\,dv+G\\,dv^2$\n"
        "G. Gaussian curvature: $K=\\frac{LN-M^2}{EG-F^2}$, Mean: $H=\\frac{EN-2FM+GL}{2(EG-F^2)}$\n"
        "H. For proofs: Given → To Prove → Proof steps.\n"
        "I. Christoffel symbols: $\\Gamma^k_{ij} = \\frac{1}{2}g^{kl}(\\partial_i g_{jl}+\\partial_j g_{il}-\\partial_l g_{ij})$\n\n"

        "=== HYDRO MECHANICS RULES ===\n"
        "A. State fluid type (ideal/viscous, compressible/incompressible) first.\n"
        "B. Continuity: $A_1v_1 = A_2v_2$ (incompressible), $\\frac{\\partial\\rho}{\\partial t}+\\nabla\\cdot(\\rho\\mathbf{v})=0$ (general)\n"
        "C. Bernoulli: $P + \\frac{1}{2}\\rho v^2 + \\rho gh = \\text{const}$ (along streamline, ideal fluid)\n"
        "D. Reynolds: $Re = \\frac{\\rho v D}{\\mu}$ — $Re<2300$ laminar, $Re>4000$ turbulent\n"
        "E. Navier-Stokes: $\\rho\\frac{D\\mathbf{v}}{Dt} = -\\nabla P + \\mu\\nabla^2\\mathbf{v} + \\rho\\mathbf{g}$\n"
        "F. Torricelli: $v=\\sqrt{2gh}$ — derived from Bernoulli\n"
        "G. Always give physical interpretation of the result.\n\n"

        "=== MATLAB RULES ===\n"
        "A. Always start every script with: clc; clear; close all;\n"
        "B. Add % comments explaining every section.\n"
        "C. Use semicolons (;) to suppress unwanted output.\n"
        "D. For numerical methods: display iteration table using fprintf.\n"
        "E. For plots: use plot(), xlabel(), ylabel(), title(), grid on.\n"
        "F. Test logic mentally — code must be correct and directly runnable.\n\n"

        "=== NEW / UNKNOWN SUBJECT RULES ===\n"
        "When the question is from a subject not listed above "
        "(e.g. Complex Analysis, Abstract Algebra, Probability, Statistics, "
        "Fourier Series, Laplace Transform, Vector Calculus, Topology, etc.):\n"
        "A. NEVER refuse. ALWAYS attempt the problem fully.\n"
        "B. Follow the SAME structure: 🔍 Given → 📌 Method → 🧮 Steps → ✅ Final Answer.\n"
        "C. Start with the relevant definition or theorem for that topic.\n"
        "D. Solve step by step exactly like the known subjects above.\n"
        "E. Use correct subject-specific notation and formulas:\n"
        "   - Complex Analysis: $z=a+bi$, modulus $|z|=\\sqrt{a^2+b^2}$, argument $\\arg(z)$,\n"
        "     Cauchy-Riemann: $\\frac{\\partial u}{\\partial x}=\\frac{\\partial v}{\\partial y}$, "
        "$\\frac{\\partial u}{\\partial y}=-\\frac{\\partial v}{\\partial x}$\n"
        "   - Abstract Algebra: group $(G,*)$, order $|G|$, Lagrange theorem, cosets, homomorphism\n"
        "   - Probability: $P(A\\cup B)=P(A)+P(B)-P(A\\cap B)$, Bayes: $P(A|B)=\\frac{P(B|A)P(A)}{P(B)}$\n"
        "   - Statistics: mean $\\bar{x}=\\frac{\\sum x_i}{n}$, variance $s^2=\\frac{\\sum(x_i-\\bar{x})^2}{n-1}$\n"
        "   - Fourier Series: $f(x)=\\frac{a_0}{2}+\\sum_{n=1}^{\\infty}(a_n\\cos\\frac{n\\pi x}{L}+b_n\\sin\\frac{n\\pi x}{L})$\n"
        "   - Laplace Transform: $\\mathcal{L}\\{f(t)\\}=\\int_0^{\\infty}e^{-st}f(t)\\,dt$\n"
        "   - Vector Calculus: $\\nabla f$, $\\nabla\\cdot\\mathbf{F}$, $\\nabla\\times\\mathbf{F}$, "
        "Green's/Stokes/Divergence theorems\n"
        "F. Add 💡 **Intuition:** analogy for explanation-type questions.\n"
        "G. ALWAYS end with ✅ **Final Answer:** $$\\boxed{answer}$$\n\n"

        "Topics covered: Calculus, Linear Algebra, Number Theory, ODEs, "
        "Numerical Methods, Differential Geometry, Hydro Mechanics, "
        "Theory of Numbers, Real Analysis II, MATLAB, Complex Analysis, "
        "Abstract Algebra, Probability, Statistics, Fourier Series, "
        "Laplace Transform, Vector Calculus, and all other BSc Mathematics topics."
        + sympy_context
    )

    # Build messages — last 6 exchanges for context
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-12:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Inject verified result directly into user message — AI cannot ignore this
    p_lower = problem.lower()
    is_graph_req = any(k in p_lower for k in ["plot","graph","draw","sketch","visualize"])

    if (sympy_info.get("result")
            and sympy_info["result"] not in (None, "matrix_detected", "mod_detected")):
        final_latex = sympy_info.get("latex", "")
        user_msg = (
            f"PROBLEM: {problem}\n\n"
            f"⚠️ IMPORTANT: This problem is already solved. Use ONLY these verified values:\n"
            f"{sympy_info.get('result', '')}\n\n"
            f"✅ FINAL ANSWER IS: $${final_latex}$$\n\n"
            f"Your task: explain the method steps clearly, and end with the EXACT final answer shown above."
        )
    elif is_graph_req:
        user_msg = (
            f"PROBLEM: {problem}\n\n"
            f"⚠️ CRITICAL: A real graph is ALREADY rendering below this response automatically.\n"
            f"You MUST NOT say you cannot draw or display images — the graph IS showing.\n"
            f"You MUST NOT suggest Desmos, graphing calculators, or any external tools.\n"
            f"Your ONLY job:\n"
            f"1. State the function clearly in LaTeX — e.g. $f(x) = \\sin(x)$\n"
            f"2. List key features — ALL values must be in LaTeX, NO plain text math\n"
            f"3. Give step-by-step drawing instructions with exact coordinates\n"
            f"4. End with exactly: '📊 Graph is shown below.'\n"
            f"EVERY mathematical expression must use $ signs. NEVER write math in plain text."
        )
    else:
        user_msg = problem
    messages.append({"role": "user", "content": user_msg})

    # ── Permanent fix: force correct final answer from SymPy ────────
    def enforce_verified_answer(ai_response: str) -> str:
        """Remove AI final answer, replace with SymPy verified one."""
        result = sympy_info.get("result", "")
        latex  = sympy_info.get("latex", "")
        # Only enforce if SymPy has a real computed result
        if (not result or
                result in (None, "matrix_detected", "mod_detected") or
                not latex):
            return ai_response  # theory question — leave AI response untouched
        # Remove everything after last "✅" or "Final Answer"
        cleaned = re.sub(
            r'(✅\s*\*{0,2}Final\s*Answer\*{0,2}.*|✅[^\n]*$)',
            "", ai_response,
            flags=re.DOTALL | re.IGNORECASE
        ).rstrip()
        # Append our verified final answer
        verified_line = f"\n\n✅ **Final Answer:** $$\\boxed{{{latex}}}$$"
        return cleaned + verified_line

    # Try each provider in order — auto-rotate on 429 or error
    last_error = ""
    for provider_name, key, call_fn in providers:
        if not key:
            continue  # skip if key not set
        try:
            response = call_fn(key, messages)
            return enforce_verified_answer(response)
        except requests.exceptions.Timeout:
            last_error = f"⏳ {provider_name} timed out"
            continue
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            # Include the real error body if available (set by try_gemini)
            try:
                body = e.response.text[:120] if e.response else str(e)
            except Exception:
                body = str(e)[:120]
            last_error = f"⚠️ {provider_name} HTTP {code}: {body}"
            continue  # always try next provider
        except Exception as e:
            last_error = f"⚠️ {provider_name} error: {str(e)}"
            continue

    # All providers exhausted. Preserve deterministic value if one exists.
    fallback = _deterministic_fallback(sympy_info, last_error or "all providers failed")
    if fallback:
        return fallback
    return (
        "⚠️ **AI explanation unavailable.**\n\n"
        f"Last provider error: `{last_error or 'unknown provider error'}`\n\n"
        "Check the provider secret and quota in the Hugging Face Space settings."
    )



# ════════════════════════════════════════════════════════════════════
# FILE UPLOAD — image/PDF sent to Gemini Vision, then SymPy verified
# ════════════════════════════════════════════════════════════════════
def ask_gemini_vision(image_b64: str, mime_type: str, user_note: str) -> str:
    """
    Multi-provider vision: tries Groq → Gemini → OpenRouter in order.
    Groq vision is primary (much more generous free limits, user already has keys).
    Gemini is fallback (needed for PDFs; Groq/OpenRouter are image-only).
    OpenRouter free vision models are the last resort.
    """
    import base64 as _b64
    is_pdf = (mime_type == "application/pdf")

    # ── Convert PDF → PNG image so Groq/OpenRouter can read it ───────
    # PyMuPDF (fitz) converts PDF pages to images.
    # Add "PyMuPDF" to your HF Space requirements.txt to enable this.
    if is_pdf:
        try:
            import fitz  # PyMuPDF
            import io
            pdf_bytes = _b64.b64decode(image_b64)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # Render all pages (up to 4) as one tall PNG
            imgs = []
            for page_num in range(min(len(doc), settings.max_pdf_pages)):
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom for crisp text
                imgs.append(pix.tobytes("png"))
            doc.close()
            # Stack page images vertically using PIL if available, else just use first page
            try:
                from PIL import Image
                pages_pil = [Image.open(io.BytesIO(b)) for b in imgs]
                total_h = sum(p.height for p in pages_pil)
                max_w = max(p.width for p in pages_pil)
                combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
                y_offset = 0
                for p in pages_pil:
                    combined.paste(p, (0, y_offset))
                    y_offset += p.height
                buf = io.BytesIO()
                combined.save(buf, format="PNG")
                image_b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                # PIL not available — just use first page
                image_b64 = _b64.b64encode(imgs[0]).decode("utf-8")
            mime_type = "image/png"
            is_pdf = False  # now it's an image — Groq/OpenRouter can handle it
        except ImportError:
            pass  # PyMuPDF not installed — will fall through to Gemini (which reads PDFs natively)
        except Exception as e:
            pass  # Conversion failed — fall through to Gemini

    prompt = (
        "You are Saad.AI, a helpful AI assistant built by Saad.\n"
        "You can read and understand ALL types of images and documents.\n\n"

        "STEP 1 — Look at the image carefully from top to bottom.\n"
        "STEP 2 — Identify what type of content is in the image:\n\n"

        "━━━ CASE A: IMAGE CONTAINS MATH PROBLEMS ━━━\n"
        "(Equations, exam paper, homework sheet, math diagrams, numbered questions)\n"
        "→ Read the ENTIRE document. Extract EVERY question — do NOT skip any.\n"
        "→ Do NOT invent questions. ONLY solve what is actually written.\n"
        "→ For EACH problem use this structure:\n"
        "  ---\n"
        "  ### Question [N]: [restate exact question from file]\n"
        "  🔍 **Given:** ...\n"
        "  📌 **Method:** ...\n"
        "  🧮 **Step 1:** ...\n"
        "  ✅ **Final Answer:** $$\\boxed{answer}$$\n"
        "  ---\n"
        "→ ALL math must be in LaTeX — never plain text math.\n\n"

        "━━━ CASE B: IMAGE IS NOT A MATH PROBLEM ━━━\n"
        "(Photo, screenshot, diagram, chart, meme, nature, objects, people, text, etc.)\n"
        "→ Describe the image in detail — what you see, what it shows.\n"
        "→ Be conversational and helpful like ChatGPT or Claude.\n"
        "→ Answer the student's specific question about the image.\n"
        "→ No forced math structure. Just natural, helpful conversation.\n"
        "→ Point out interesting details, context, or meaning.\n\n"

        f"Student's instruction: {user_note if user_note else 'Look at this image and describe or analyze it.'}\n\n"
        "Always be helpful, friendly, and clear."
    )

    errors = []

    # ── PROVIDER 1: Groq vision (images only — not PDFs) ─────────────
    # Groq free tier: ~100 req/min, much more generous than Gemini's 15/min
    if not is_pdf:
        groq_keys = settings.groq_api_keys
        groq_vision_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.2-11b-vision-preview",
        ]
        for i, key in enumerate(groq_keys):
            if not key.strip():
                continue
            for model in groq_vision_models:
                try:
                    resp = _post_with_retry(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={
                            "model": model,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                                    {"type": "text", "text": prompt}
                                ]
                            }],
                            "max_tokens": 2048,
                            "temperature": 0.15
                        },
                        timeout=settings.provider_timeout_seconds
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                    elif resp.status_code == 429:
                        errors.append(f"Groq-{i+1}/{model}: rate limited")
                        break  # try next key
                    elif resp.status_code == 400:
                        # Model may not support vision — try next model
                        try:
                            msg = resp.json().get("error", {}).get("message", "")[:80]
                        except Exception:
                            msg = ""
                        errors.append(f"Groq-{i+1}/{model}: {msg}")
                        continue
                    else:
                        errors.append(f"Groq-{i+1}/{model}: HTTP {resp.status_code}")
                        break
                except requests.exceptions.Timeout:
                    errors.append(f"Groq-{i+1}/{model}: timeout")
                    break
                except Exception as e:
                    errors.append(f"Groq-{i+1}/{model}: {str(e)[:60]}")
                    break

    # ── PROVIDER 2: Gemini (images + PDFs) ───────────────────────────
    # 15 req/min, 1500 req/day per key — use as fallback
    gemini_keys = settings.gemini_api_keys
    gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
    for i, key in enumerate(gemini_keys):
        if not key.strip():
            continue
        for model in gemini_models:
            try:
                resp = _post_with_retry(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [
                            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                            {"text": prompt}
                        ]}],
                        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.15}
                    },
                    timeout=settings.provider_timeout_seconds
                )
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        return candidates[0]["content"]["parts"][0]["text"]
                    errors.append(f"Gemini-{i+1}/{model}: safety blocked")
                    break
                elif resp.status_code == 429:
                    errors.append(f"Gemini-{i+1}/{model}: rate limited (429)")
                    break
                elif resp.status_code == 404:
                    errors.append(f"Gemini-{i+1}/{model}: model not found")
                    continue  # try next model
                else:
                    try:
                        msg = resp.json().get("error", {}).get("message", resp.text[:80])
                    except Exception:
                        msg = resp.text[:80]
                    errors.append(f"Gemini-{i+1}/{model}: HTTP {resp.status_code} — {msg}")
                    break
            except requests.exceptions.Timeout:
                errors.append(f"Gemini-{i+1}/{model}: timeout")
                break
            except Exception as e:
                errors.append(f"Gemini-{i+1}/{model}: {str(e)[:60]}")
                break

    # ── PROVIDER 3: OpenRouter free vision models (images only) ───────
    if not is_pdf:
        or_key = settings.openrouter_api_key
        if or_key.strip():
            or_models = [
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "qwen/qwen2-vl-7b-instruct:free",
            ]
            for model in or_models:
                try:
                    resp = _post_with_retry(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
                        json={
                            "model": model,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                                    {"type": "text", "text": prompt}
                                ]
                            }],
                            "max_tokens": 2048
                        },
                        timeout=settings.provider_timeout_seconds
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                    errors.append(f"OpenRouter/{model}: HTTP {resp.status_code}")
                except Exception as e:
                    errors.append(f"OpenRouter/{model}: {str(e)[:60]}")

    # ── All providers failed ──────────────────────────────────────────
    error_summary = " | ".join(errors[-6:])  # show last 6 errors
    if is_pdf:
        pdf_note = (
            "\n\n**To make PDFs work without Gemini:** add `PyMuPDF` to your HF Space `requirements.txt` — "
            "it converts PDF pages to images so Groq can read them (no Gemini needed)."
        )
    else:
        pdf_note = ""
    return (
        f"⚠️ **All vision providers failed.**\n\n"
        f"Errors: `{error_summary}`\n\n"
        f"**Most likely fix:** make sure `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3` "
        f"are added in your HF Space → Settings → Secrets. Groq reads images with much higher limits than Gemini.\n\n"
        f"**Gemini 429:** wait 60 sec (per-minute limit) or until midnight Pacific (daily limit)."
        f"{pdf_note}"
    )


class _MemoryUpload:
    """Small upload-compatible wrapper for bytes already held in session state."""

    def __init__(self, file_bytes: bytes, name: str, mime_type: str):
        self._file_bytes = file_bytes
        self.name = name
        self.type = mime_type

    def read(self) -> bytes:
        return self._file_bytes


def handle_uploaded_file(uploaded_file, user_note: str) -> str:
    """
    Process uploaded image or PDF:
    1. Convert to base64
    2. Send to Gemini Vision
    3. Try SymPy verification on extracted text
    4. Return final answer
    """
    import base64

    # ── Validate size ────────────────────────────────────────────────
    MAX_SIZE = settings.max_upload_bytes
    file_bytes = uploaded_file.read()
    if len(file_bytes) == 0:
        return "⚠️ The uploaded file is empty. Please try again."
    if len(file_bytes) > MAX_SIZE:
        return f"⚠️ File too large ({len(file_bytes)//1024}KB). Please upload under 5MB."

    # ── Detect MIME type from Streamlit's type field, not filename ────
    # This works even if filename has spaces, brackets, or no extension
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "webp": "image/webp",
        "pdf": "application/pdf"
    }
    # Try Streamlit's type first (most reliable), fall back to extension
    mime_type = uploaded_file.type if uploaded_file.type else None
    if not mime_type:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
        mime_type = mime_map.get(ext)
    if mime_type not in mime_map.values():
        return "⚠️ Unsupported format. Please upload JPG, PNG, WEBP or PDF."

    # ── Convert to base64 ────────────────────────────────────────────
    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

    # ── Send to Gemini Vision ────────────────────────────────────────
    gemini_response = ask_gemini_vision(image_b64, mime_type, user_note)

    # ── Gemini failed → return single clean error only ───────────────
    if gemini_response.startswith("⚠️"):
        return gemini_response

    # ── SymPy verification — extract problem line first ───────────────
    # Run SymPy on the first user-question line, not Gemini's full markdown
    # This avoids SymPy choking on LaTeX formatting in the solution
    extracted_problem = ""
    for line in gemini_response.splitlines():
        stripped = line.strip()
        # Skip empty lines, headers, and Gemini's own solution steps
        if (stripped and
                not stripped.startswith("#") and
                not stripped.startswith("🔍") and
                not stripped.startswith("📌") and
                not stripped.startswith("🧮") and
                not stripped.startswith("✅") and
                not stripped.startswith("**") and
                len(stripped) > 5):
            extracted_problem = stripped
            break
    sympy_result = run_sympy(extracted_problem) if extracted_problem else {"type": "general", "result": None, "latex": ""}

    if (sympy_result.get("result") and
            sympy_result["result"] not in (None, "matrix_detected", "mod_detected")):
        latex = sympy_result.get("latex", "")
        cleaned = re.sub(
            r'(✅\s*\*{0,2}Final\s*Answer\*{0,2}.*|✅[^\n]*$)',
            "", gemini_response,
            flags=re.DOTALL | re.IGNORECASE
        ).rstrip()
        return cleaned + "\n\n🔒 **SymPy Verified**" + f"\n\n✅ **Final Answer:** $$\\boxed{{{latex}}}$$"
    else:
        # Gemini answered, SymPy couldn't verify — single clean note
        return gemini_response + "\n\n⚠️ *AI-generated answer — not SymPy verified.*"


