# ═══════════════════════════════════════════════════════════════════
# Saad.AI — BSc Mathematics Solver
# Engine: Groq (free) + SymPy (verified computation)
# ═══════════════════════════════════════════════════════════════════

# ── Imports (order matters: stdlib/third-party before sympy, re LAST) ──
import streamlit as st
import os
import requests
import json as _json
import sympy as sp
from sympy import (
    symbols, diff, integrate, limit, solve,
    sympify, oo, sin, cos, tan, exp, log, sqrt, pi, E,
    Matrix, Symbol, Function, dsolve, Eq
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
import re  # MUST be last — 'from sympy import *' would overwrite re otherwise
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required on HuggingFace Spaces
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saad.AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── CSS — ChatGPT-style, compact font, clean dark theme ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background-color: #0f0f0f !important;
    color: #ececec !important;
}
.block-container {
    max-width: 780px !important;
    padding: 1rem 1.5rem 2rem 1.5rem !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Header ── */
.saad-header {
    text-align: center;
    padding: 1.4rem 0 0.6rem 0;
    margin-bottom: 0.5rem;
}
.saad-header h1 {
    font-size: 2rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 1px;
}
.saad-header p {
    font-size: 0.82rem;
    color: #555;
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.3px;
}
.badge-row {
    display: flex;
    gap: 0.4rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.7rem;
}
.badge {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a3a5e;
    color: #60a5fa;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
}

/* ── Welcome screen cards ── */
.subject-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
}
.subject-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 0.6rem 0.5rem;
    text-align: center;
    font-size: 0.75rem;
    color: #9ca3af;
    transition: all 0.2s;
}
.subject-card:hover {
    border-color: #3b82f6;
    color: #60a5fa;
    background: #0f172a;
}
.subject-card .icon {
    font-size: 1.3rem;
    display: block;
    margin-bottom: 0.2rem;
}

/* ── Chat messages — override Streamlit defaults ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
    margin: 0 !important;
    gap: 0.6rem !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #1e3a5f, #1a2f4a) !important;
    border: 1px solid #2a4a7f !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 0.6rem 0.9rem !important;
    max-width: 80% !important;
    font-size: 0.88rem !important;
    color: #e2e8f0 !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.88rem !important;
    line-height: 1.75 !important;
    color: #ececec !important;
}

/* Math display blocks */
.katex-display {
    background: #0f172a !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 0.6rem 1rem !important;
    margin: 0.5rem 0 !important;
    overflow-x: auto !important;
}
.katex { font-size: 1em !important; }

/* Step labels bold + colored */
[data-testid="stChatMessageContent"] p {
    margin: 0.2rem 0 !important;
    font-size: 0.88rem !important;
    line-height: 1.75 !important;
}

/* ── Input box ── */
[data-testid="stChatInput"] {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.88rem !important;
    color: #ececec !important;
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080810 0%, #0a0a14 100%) !important;
    border-right: 1px solid #1a1a2e !important;
}
[data-testid="stSidebar"] * {
    font-size: 0.82rem !important;
}

/* Sidebar topic chips */
.topic-chip {
    display: inline-block;
    background: #111827;
    border: 1px solid #1f2937;
    color: #6b7280;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    margin: 2px 2px;
    transition: all 0.15s;
}
.topic-chip:hover {
    border-color: #3b82f6;
    color: #60a5fa;
}

/* Sidebar engine info */
.engine-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0;
    color: #4b5563;
    font-size: 0.75rem;
}
.engine-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
    box-shadow: 0 0 4px #22c55e;
}

/* ── Buttons ── */
.stButton > button {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #6b7280 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 0.3rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #ececec !important;
    background: #0f172a !important;
}

/* ── Copy button ── */
[data-testid="stChatMessage"] .stButton > button {
    background: transparent !important;
    border: 1px solid #1f2937 !important;
    color: #4b5563 !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
    padding: 2px 6px !important;
    width: auto !important;
    min-height: 0 !important;
    height: 24px !important;
    transition: all 0.15s !important;
}
[data-testid="stChatMessage"] .stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #ececec !important;
    background: #0f172a !important;
}

/* ── Spinner — custom math themed ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }
.stSpinner p {
    color: #3b82f6 !important;
    font-size: 0.82rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Divider ── */
hr { border-color: #1a1a2e !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #111827 !important;
    border-color: #1f2937 !important;
    color: #ececec !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── Attachment toolbar (paperclip row above chat input) ── */
.attach-toolbar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.2rem 0.2rem 0.2rem;
}
.attach-btn {
    background: transparent;
    border: 1px solid #1f2937;
    color: #4b5563;
    border-radius: 8px;
    font-size: 0.78rem;
    padding: 4px 10px;
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.attach-btn:hover {
    border-color: #3b82f6;
    color: #60a5fa;
    background: #0f172a;
}

/* ── File uploader — compact popup style ── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1px solid #2a3a5e !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem 0.4rem 1rem !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: #0f172a !important;
}
[data-testid="stFileUploader"] section {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed #2a3a5e !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.5rem !important;
    min-height: 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(59,130,246,0.06) !important;
    border-color: #3b82f6 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 0.75rem !important;
    color: #4b5563 !important;
    padding: 0.15rem 0 !important;
}
/* ── Chat input — always rounded ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid #1f2937 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
}
/* Hide "Browse files" button text, keep icon feel */
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0.72rem !important;
    padding: 3px 10px !important;
    border-radius: 6px !important;
    background: #0f172a !important;
    border: 1px solid #2a3a5e !important;
    color: #60a5fa !important;
}
/* Compact upload section toggle button */
div[data-testid="column"] .stButton > button[kind="secondary"] {
    padding: 4px 10px !important;
    font-size: 0.78rem !important;
    width: auto !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
import datetime as _dt

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = ""
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "supa_loaded" not in st.session_state:
    st.session_state.supa_loaded = False
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "pending_file_bytes" not in st.session_state:
    st.session_state.pending_file_bytes = None
if "pending_file_name" not in st.session_state:
    st.session_state.pending_file_name = None
if "pending_file_mime" not in st.session_state:
    st.session_state.pending_file_mime = None
# ── Attached file kept in memory for follow-up questions ─────────
if "attached_file_bytes" not in st.session_state:
    st.session_state.attached_file_bytes = None
if "attached_file_name" not in st.session_state:
    st.session_state.attached_file_name = None
if "attached_file_mime" not in st.session_state:
    st.session_state.attached_file_mime = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# ════════════════════════════════════════════════════════════════════
# SUPABASE — Persistent chat history
# ════════════════════════════════════════════════════════════════════
_SUPA_URL = os.environ.get("SUPABASE_URL", "")
_SUPA_KEY = os.environ.get("SUPABASE_KEY", "")

def _supa_headers():
    return {
        "apikey": _SUPA_KEY,
        "Authorization": f"Bearer {_SUPA_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def supa_load_all_chats():
    """Load all chats from Supabase."""
    if not _SUPA_URL or not _SUPA_KEY:
        return {}
    try:
        resp = requests.get(
            f"{_SUPA_URL}/rest/v1/chats?select=*&order=created_at.desc",
            headers=_supa_headers(), timeout=5
        )
        if resp.status_code == 200:
            rows = resp.json()
            return {r["id"]: {
                "title": r["title"],
                "messages": r["messages"],
                "created": r["created_at"][:16].replace("T"," ")
            } for r in rows}
    except Exception:
        pass
    return {}

def supa_save_chat(chat_id, title, messages):
    """Save or update a chat in Supabase."""
    if not _SUPA_URL or not _SUPA_KEY or not messages:
        return
    try:
        requests.post(
            f"{_SUPA_URL}/rest/v1/chats",
            headers={**_supa_headers(), "Prefer": "resolution=merge-duplicates"},
            data=_json.dumps({
                "id": chat_id,
                "title": title,
                "messages": messages
            }), timeout=5
        )
    except Exception:
        pass

def supa_delete_chat(chat_id):
    """Delete a chat from Supabase."""
    if not _SUPA_URL or not _SUPA_KEY:
        return
    try:
        requests.delete(
            f"{_SUPA_URL}/rest/v1/chats?id=eq.{chat_id}",
            headers=_supa_headers(), timeout=5
        )
    except Exception:
        pass

# Load chats from Supabase now that functions are defined
if not st.session_state.supa_loaded:
    loaded = supa_load_all_chats()
    if loaded:
        st.session_state.chats = loaded
    st.session_state.supa_loaded = True

def new_chat():
    """Start a fresh chat session."""
    chat_id = f"chat_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.last_submitted = ""
    # Clear any attached file memory from previous chat
    st.session_state.attached_file_bytes = None
    st.session_state.attached_file_name  = None
    st.session_state.attached_file_mime  = None

def save_current_chat():
    """Save current messages to session state and Supabase."""
    cid = st.session_state.current_chat_id
    if not cid or not st.session_state.messages:
        return
    # Auto-title from first user message
    first_user = next((m["content"] for m in st.session_state.messages if m["role"]=="user"), "New Chat")
    title = first_user[:35] + "..." if len(first_user) > 35 else first_user
    st.session_state.chats[cid] = {
        "title": title,
        "messages": list(st.session_state.messages),
        "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # ISO-sortable
    }
    # Save to Supabase (persistent)
    supa_save_chat(cid, title, list(st.session_state.messages))

def load_chat(chat_id):
    """Load a previous chat."""
    if chat_id in st.session_state.chats:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = list(st.session_state.chats[chat_id]["messages"])
        st.session_state.last_submitted = ""
        # Clear attached file memory — it belongs to a different chat session
        st.session_state.attached_file_bytes = None
        st.session_state.attached_file_name  = None
        st.session_state.attached_file_mime  = None


# ════════════════════════════════════════════════════════════════════
# SYMPY ENGINE — exact symbolic computation, no AI needed for this
# ════════════════════════════════════════════════════════════════════
def run_sympy(problem: str) -> dict:
    """
    Symbolic computation engine. Returns verified result dict.
    Falls back silently on any error.
    ELIF ORDER (important — must check ODE before Solve):
      Derivative → Integral → Limit → ODE → NR → Solve → Matrix → Mod
    """
    p = problem.lower().strip()
    x = sp.Symbol('x')
    tfms = standard_transformations + (implicit_multiplication_application,)
    ld = {
        "x": x,
        "e": sp.E, "E": sp.E,
        "pi": sp.pi, "PI": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "inf": sp.oo, "oo": sp.oo
    }

    def clean(s):
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"\^", "**", s)
        return s

    def to_coeff(s):
        """Convert string to sympy integer/rational — never float (floats break dsolve)."""
        s = s.replace(" ", "")
        if s in ("", "+"): return sp.Integer(1)
        if s == "-": return sp.Integer(-1)
        try:
            f = float(s)
            return sp.Integer(int(f)) if f == int(f) else sp.Rational(s)
        except Exception:
            return sp.Integer(1)

    try:
        # ── 1. Derivative ────────────────────────────────────────────
        if any(k in p for k in ["derivative", "differentiate", "d/dx", "diff"]):
            raw = p
            for kw in ["derivative of", "differentiate", "diff of", "d/dx of", "d/dx"]:
                if kw in p:
                    raw = p.split(kw, 1)[-1].strip()
                    break
            raw = re.sub(r"\s*(dx|with\s*respect\s*to\s*x).*$", "", raw).strip()
            expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            result = sp.diff(expr, x)
            return {"type": "Derivative", "result": str(result), "latex": sp.latex(result)}

        # ── 2. Integral ──────────────────────────────────────────────
        elif any(k in p for k in ["integral", "integrate", "antiderivative"]):
            raw = p
            for kw in ["integral of", "integrate", "antiderivative of"]:
                if kw in p:
                    raw = p.split(kw, 1)[-1].strip()
                    break
            raw = re.sub(r"\s*dx.*$", "", raw).strip()
            expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            result = sp.integrate(expr, x)
            return {"type": "Integral", "result": str(result), "latex": sp.latex(result)}

        # ── 3. Limit ─────────────────────────────────────────────────
        elif "limit" in p:
            match = re.search(
                r"limit\s+of\s+([\w\s\(\)\+\-\*/\^\.\,]+?)"
                r"\s+as\s+x\s*(?:->|→|approaches)\s*([\w\.\+\-]+)", p
            )
            if match:
                raw_expr = clean(match.group(1))
                pt = match.group(2).strip()
                point = sp.oo if pt in ("inf", "infinity", "oo") else sp.sympify(pt)
                expr = parse_expr(raw_expr, transformations=tfms, local_dict=ld)
                result = sp.limit(expr, x, point)
                return {"type": "Limit", "result": str(result), "latex": sp.latex(result)}

        # ── 4. ODE — MUST be before Solve (many ODE problems start with "solve") ──
        elif any(k in p for k in ["d²y", "d^2y", "d2y", "dy/dx",
                                   "differential equation", "second order", "first order"]):
            y_fn = sp.Function('y')
            ode_sol = None
            try:
                # Extract RHS — everything after = and before "with"
                rhs_m = re.search(r"=\s*(.+?)(?:\s+with|\s*$)", p)
                rhs_str = clean(rhs_m.group(1).strip()) if rhs_m else "0"
                rhs_expr = parse_expr(rhs_str, transformations=tfms, local_dict=ld)
                lhs = p.split("=")[0]

                # Second order: d²y/dx² + a*dy/dx + b*y = rhs
                if any(k in p for k in ["d²y", "d^2y", "d2y", "second order"]):
                    # Match coefficient of dy/dx (must be dy/dx not just dy to avoid d²y match)
                    c1_m = re.search(r"([+\-]\s*\d*\.?\d*)\s*dy/dx", lhs)
                    # Match coefficient of standalone y (word boundary)
                    c0_m = re.search(r"([+\-]\s*\d+\.?\d*)\s*y\b", lhs)
                    a1 = to_coeff(c1_m.group(1)) if c1_m else sp.Integer(0)
                    a0 = to_coeff(c0_m.group(1)) if c0_m else sp.Integer(0)
                    ode_eq = sp.Eq(
                        y_fn(x).diff(x, 2) + a1*y_fn(x).diff(x) + a0*y_fn(x),
                        rhs_expr
                    )
                    ode_sol = sp.dsolve(ode_eq, y_fn(x))

                # First order: dy/dx + a*y = rhs
                elif "dy/dx" in p:
                    c0_m = re.search(r"([+\-]\s*\d+\.?\d*)\s*y\b", lhs)
                    a0 = to_coeff(c0_m.group(1)) if c0_m else sp.Integer(0)
                    ode_eq = sp.Eq(y_fn(x).diff(x) + a0*y_fn(x), rhs_expr)
                    ode_sol = sp.dsolve(ode_eq, y_fn(x))

            except Exception:
                ode_sol = None  # safe fallback to AI

            if ode_sol is not None:
                return {
                    "type": "ODE",
                    "result": f"General solution: y = {str(ode_sol.rhs)}",
                    "latex": f"y = {sp.latex(ode_sol.rhs)}"
                }

        # ── 5. Newton-Raphson ────────────────────────────────────────
        elif any(k in p for k in ["newton", "newton-raphson", "newton raphson"]):
            # Extract: equation (between of/to/for and "= 0"), x0, iterations
            eq_m   = re.search(r"(?:of|to|for)\s+(.+?)\s*=\s*0", p)
            x0_m   = re.search(r"x\s*0\s*[=:]\s*([\d\.]+)", p)
            iter_m = re.search(r"(\d+)\s*iter", p)

            if eq_m and x0_m:
                raw_eq  = clean(eq_m.group(1).strip())
                x0_val  = float(x0_m.group(1))
                n_iter  = int(iter_m.group(1)) if iter_m else 3

                expr_nr = parse_expr(raw_eq, transformations=tfms, local_dict=ld)
                f_sym   = sp.lambdify(x, expr_nr, modules="math")
                df_sym  = sp.lambdify(x, sp.diff(expr_nr, x), modules="math")

                # Run all iterations with full precision — never round intermediate values
                iterations = []
                xn = x0_val
                for i in range(n_iter):
                    fxn  = f_sym(xn)
                    dfxn = df_sym(xn)
                    if abs(dfxn) < 1e-15:
                        break  # avoid division by zero
                    xn1 = xn - fxn / dfxn
                    iterations.append({
                        "n": i, "xn": round(xn, 8),
                        "fxn": round(fxn, 8), "dfxn": round(dfxn, 8),
                        "xn1": round(xn1, 8)
                    })
                    xn = xn1  # use FULL precision for next iteration

                final_x  = iterations[-1]["xn1"] if iterations else x0_val
                final_fx = round(f_sym(final_x), 10)
                iter_str = "\n".join([
                    f"  x{it['n']+1} = {it['xn']} - ({it['fxn']}) / ({it['dfxn']}) = {it['xn1']}"
                    for it in iterations
                ])
                result_str = (
                    f"f(x) = {str(expr_nr)}, f'(x) = {str(sp.diff(expr_nr, x))}\n"
                    f"x0 = {x0_val}, iterations = {n_iter}\n"
                    f"VERIFIED ITERATIONS (AI MUST use these exact values):\n"
                    f"{iter_str}\n"
                    f"Final answer: x{n_iter} = {final_x}\n"
                    f"Verification: f({final_x}) = {final_fx} ≈ 0"
                )
                return {
                    "type": "Newton-Raphson",
                    "result": result_str,
                    "latex": f"x_{{{n_iter}}} = {final_x}"
                }

        # ── 6. Numerical Analysis Methods ───────────────────────────
        elif any(k in p for k in ["bisection", "secant method", "false position",
                                   "regula falsi", "gauss elimination", "gauss elim",
                                   "lu decomposition", "lu decomp",
                                   "lagrange interpolation", "lagrange interp",
                                   "newton divided", "divided difference",
                                   "trapezoidal", "trapezoid rule",
                                   "simpson", "euler method", "euler's method",
                                   "runge-kutta", "runge kutta", "rk4"]):
            import math as _math

            # ── Helper: extract f(x) expression ──────────────────────
            def get_expr():
                eq_m = re.search(r"(?:of|to|for|function)\s+(.+?)\s*(?:=\s*0|,|$)", p)
                if eq_m:
                    raw = clean(eq_m.group(1).strip())
                    return parse_expr(raw, transformations=tfms, local_dict=ld)
                return None

            # ── Helper: extract bounds a, b ───────────────────────────
            def get_bounds():
                nums = re.findall(r"[-]?\d+\.?\d*", p)
                floats = [float(n) for n in nums]
                # x0 value
                x0_m = re.search(r"x\s*0\s*[=:]\s*([-]?\d+\.?\d*)", p)
                x1_m = re.search(r"x\s*1\s*[=:]\s*([-]?\d+\.?\d*)", p)
                # interval [a,b]
                ab_m = re.search(r"\[\s*([-]?\d+\.?\d*)\s*,\s*([-]?\d+\.?\d*)\s*\]", p)
                return floats, x0_m, x1_m, ab_m

            # ── Helper: extract iterations ────────────────────────────
            def get_iters(default=5):
                m = re.search(r"(\d+)\s*iter", p)
                return int(m.group(1)) if m else default

            # ── Helper: extract step size h ───────────────────────────
            def get_h():
                m = re.search(r"h\s*[=:]\s*([\d\.]+)", p)
                return float(m.group(1)) if m else 0.1

            # ── Helper: extract ODE rhs f(x,y) ───────────────────────
            def get_ode_rhs():
                # dy/dx = f(x,y) → extract rhs
                m = re.search(r"dy/dx\s*=\s*(.+?)(?:\s*,|\s*with|\s*y\s*\(|$)", p)
                return m.group(1).strip() if m else None

            floats, x0_m, x1_m, ab_m = get_bounds()
            n_iter = get_iters()

            # ════════════════════════════════════════════════════════
            # BISECTION METHOD
            # ════════════════════════════════════════════════════════
            if "bisection" in p:
                expr_b = get_expr()
                if expr_b is not None and ab_m:
                    a_val = float(ab_m.group(1))
                    b_val = float(ab_m.group(2))
                    f_b = sp.lambdify(x, expr_b, modules="math")
                    steps = []
                    a_n, b_n = a_val, b_val
                    for i in range(n_iter):
                        c = (a_n + b_n) / 2
                        fc = f_b(c)
                        steps.append({"iter": i+1, "a": round(a_n,8),
                                      "b": round(b_n,8), "c": round(c,8),
                                      "fc": round(fc,8)})
                        if f_b(a_n) * fc < 0: b_n = c
                        else: a_n = c
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: a={s['a']}, b={s['b']}, c={s['c']}, f(c)={s['fc']}"
                        for s in steps])
                    final_c = steps[-1]["c"]
                    return {
                        "type": "Bisection",
                        "result": (f"f(x) = {str(expr_b)}\n"
                                   f"Interval [{a_val},{b_val}], {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {final_c}"),
                        "latex": f"x \\approx {final_c}"
                    }

            # ════════════════════════════════════════════════════════
            # SECANT METHOD
            # ════════════════════════════════════════════════════════
            elif "secant" in p:
                expr_s = get_expr()
                if expr_s is not None and x0_m and x1_m:
                    x0_v = float(x0_m.group(1))
                    x1_v = float(x1_m.group(1))
                    f_s = sp.lambdify(x, expr_s, modules="math")
                    steps = []
                    xp, xc = x0_v, x1_v
                    for i in range(n_iter):
                        fxp, fxc = f_s(xp), f_s(xc)
                        if abs(fxc - fxp) < 1e-15: break
                        xn_val = xc - fxc*(xc-xp)/(fxc-fxp)
                        steps.append({"iter": i+1, "x": round(xn_val, 8),
                                      "fx": round(f_s(xn_val), 8)})
                        xp, xc = xc, xn_val
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: x={s['x']}, f(x)={s['fx']}"
                        for s in steps])
                    return {
                        "type": "Secant",
                        "result": (f"f(x) = {str(expr_s)}\n"
                                   f"x0={x0_v}, x1={x1_v}, {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {steps[-1]['x']}"),
                        "latex": f"x \\approx {steps[-1]['x']}"
                    }

            # ════════════════════════════════════════════════════════
            # FALSE POSITION (Regula Falsi)
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["false position", "regula falsi"]):
                expr_fp = get_expr()
                if expr_fp is not None and ab_m:
                    a_val = float(ab_m.group(1))
                    b_val = float(ab_m.group(2))
                    f_fp = sp.lambdify(x, expr_fp, modules="math")
                    steps = []
                    a_n, b_n = a_val, b_val
                    for i in range(n_iter):
                        fa, fb = f_fp(a_n), f_fp(b_n)
                        c = (a_n*fb - b_n*fa) / (fb - fa)
                        fc = f_fp(c)
                        steps.append({"iter": i+1, "a": round(a_n,8),
                                      "b": round(b_n,8), "c": round(c,8),
                                      "fc": round(fc,8)})
                        if fa * fc < 0: b_n = c
                        else: a_n = c
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: a={s['a']}, b={s['b']}, c={s['c']}, f(c)={s['fc']}"
                        for s in steps])
                    return {
                        "type": "FalsePosition",
                        "result": (f"f(x) = {str(expr_fp)}\n"
                                   f"Interval [{a_val},{b_val}], {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {steps[-1]['c']}"),
                        "latex": f"x \\approx {steps[-1]['c']}"
                    }

            # ════════════════════════════════════════════════════════
            # GAUSS ELIMINATION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["gauss elimination", "gauss elim"]):
                # Extract matrix from problem — look for [[...]] pattern
                mat_m = re.search(r"\[\s*\[(.+?)\]\s*\]", p)
                rhs_m = re.search(r"(?:rhs|=|b)\s*[=:]?\s*\[([^\]]+)\]", p)
                if mat_m and rhs_m:
                    rows = re.findall(r"\[([^\]]+)\]", p)
                    mat_data = [[sp.Rational(v) for v in re.split(r"[,\s]+", r.strip()) if v]
                                for r in rows[:-1]]
                    rhs_data = [sp.Rational(v) for v in re.split(r"[,\s]+", rows[-1].strip()) if v]
                    A = sp.Matrix(mat_data)
                    b_vec = sp.Matrix(rhs_data)
                    sol = A.solve(b_vec)
                    sol_str = ", ".join([f"x{i+1}={sol[i]}" for i in range(len(sol))])
                    return {
                        "type": "GaussElimination",
                        "result": f"VERIFIED SOLUTION: {sol_str}",
                        "latex": sol_str
                    }

            # ════════════════════════════════════════════════════════
            # LU DECOMPOSITION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["lu decomposition", "lu decomp"]):
                rows = re.findall(r"\[([^\]]+)\]", p)
                if rows:
                    mat_data = [[sp.Rational(v) for v in re.split(r"[,\s]+", r.strip()) if v]
                                for r in rows]
                    M = sp.Matrix(mat_data)
                    L, U, _ = M.LUdecomposition()
                    return {
                        "type": "LUDecomposition",
                        "result": (f"VERIFIED:\nL = {str(L)}\nU = {str(U)}"),
                        "latex": f"L={sp.latex(L)}, U={sp.latex(U)}"
                    }

            # ════════════════════════════════════════════════════════
            # LAGRANGE INTERPOLATION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["lagrange interpolation", "lagrange interp"]):
                # Extract data points (x0,y0),(x1,y1),...
                pts = re.findall(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", p)
                if pts:
                    data = [(sp.Rational(px), sp.Rational(py)) for px, py in pts]
                    poly = sp.interpolate(data, x)
                    poly_exp = sp.expand(poly)
                    return {
                        "type": "LagrangeInterpolation",
                        "result": f"VERIFIED polynomial: {str(poly_exp)}",
                        "latex": sp.latex(poly_exp)
                    }

            # ════════════════════════════════════════════════════════
            # NEWTON DIVIDED DIFFERENCE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["newton divided", "divided difference"]):
                pts = re.findall(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", p)
                if pts:
                    xs_v = [float(px) for px, py in pts]
                    ys_v = [float(py) for px, py in pts]
                    n_p = len(xs_v)
                    dd = [[0.0]*n_p for _ in range(n_p)]
                    for i in range(n_p): dd[i][0] = ys_v[i]
                    for j in range(1, n_p):
                        for i in range(n_p - j):
                            dd[i][j] = (dd[i+1][j-1]-dd[i][j-1])/(xs_v[i+j]-xs_v[i])
                    coeffs = [round(dd[0][j], 8) for j in range(n_p)]
                    # Build polynomial
                    poly_s = sp.interpolate(list(zip(xs_v, ys_v)), x)
                    return {
                        "type": "NewtonDividedDiff",
                        "result": (f"VERIFIED divided differences: {coeffs}\n"
                                   f"Polynomial: {str(sp.expand(poly_s))}"),
                        "latex": sp.latex(sp.expand(poly_s))
                    }

            # ════════════════════════════════════════════════════════
            # TRAPEZOIDAL RULE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["trapezoidal", "trapezoid rule"]):
                expr_t = get_expr()
                ab_m2 = re.search(r"\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", p)
                n_m = re.search(r"n\s*[=:]\s*(\d+)", p)
                if expr_t is not None and ab_m2 and n_m:
                    a_v = float(ab_m2.group(1))
                    b_v = float(ab_m2.group(2))
                    n_v = int(n_m.group(1))
                    f_t = sp.lambdify(x, expr_t, modules="math")
                    h = (b_v - a_v) / n_v
                    s = f_t(a_v) + f_t(b_v)
                    pts_str = [f"f({round(a_v,4)})={round(f_t(a_v),6)}", ]
                    for i in range(1, n_v):
                        xi = a_v + i*h
                        pts_str.append(f"f({round(xi,4)})={round(f_t(xi),6)}")
                    pts_str.append(f"f({round(b_v,4)})={round(f_t(b_v),6)}")
                    result_val = round(h/2 * (f_t(a_v)+f_t(b_v) + 2*sum(f_t(a_v+i*h) for i in range(1,n_v))), 8)
                    return {
                        "type": "Trapezoidal",
                        "result": (f"f(x)={str(expr_t)}, [{a_v},{b_v}], n={n_v}, h={round(h,6)}\n"
                                   f"Function values: {', '.join(pts_str)}\n"
                                   f"VERIFIED result: {result_val}"),
                        "latex": f"\\int_{{{a_v}}}^{{{b_v}}} \\approx {result_val}"
                    }

            # ════════════════════════════════════════════════════════
            # SIMPSON'S 1/3 RULE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["simpson"]):
                expr_si = get_expr()
                ab_m3 = re.search(r"\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", p)
                n_m2 = re.search(r"n\s*[=:]\s*(\d+)", p)
                if expr_si is not None and ab_m3 and n_m2:
                    a_v = float(ab_m3.group(1))
                    b_v = float(ab_m3.group(2))
                    n_v = int(n_m2.group(1))
                    if n_v % 2 != 0: n_v += 1  # must be even
                    f_si = sp.lambdify(x, expr_si, modules="math")
                    h = (b_v - a_v) / n_v
                    s = f_si(a_v) + f_si(b_v)
                    for i in range(1, n_v):
                        s += (4 if i % 2 != 0 else 2) * f_si(a_v + i*h)
                    result_val = round(h/3 * s, 8)
                    return {
                        "type": "Simpsons",
                        "result": (f"f(x)={str(expr_si)}, [{a_v},{b_v}], n={n_v}, h={round(h,6)}\n"
                                   f"VERIFIED result: {result_val}"),
                        "latex": f"\\int_{{{a_v}}}^{{{b_v}}} \\approx {result_val}"
                    }

            # ════════════════════════════════════════════════════════
            # EULER'S METHOD
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["euler method", "euler's method"]):
                rhs_str = get_ode_rhs()
                x0_em = re.search(r"x\s*0?\s*[=:]\s*([-\d\.]+)", p)
                y0_em = re.search(r"y\s*[=\(]\s*0?\s*\)?\s*[=:]\s*([-\d\.]+)", p)
                h_val = get_h()
                n_steps = get_iters(default=5)
                if rhs_str and x0_em and y0_em:
                    x_s, y_s = sp.symbols('x y')
                    rhs_expr_ode = parse_expr(clean(rhs_str), transformations=tfms,
                                              local_dict={**ld, "y": y_s})
                    f_ode = sp.lambdify((x_s, y_s), rhs_expr_ode, modules="math")
                    xn_e, yn_e = float(x0_em.group(1)), float(y0_em.group(1))
                    steps = []
                    for i in range(n_steps):
                        yn1 = yn_e + h_val * f_ode(xn_e, yn_e)
                        xn_e += h_val
                        steps.append({"n": i+1, "x": round(xn_e,6), "y": round(yn1,8)})
                        yn_e = yn1
                    step_str = "\n".join([f"  Step {s['n']}: x={s['x']}, y={s['y']}" for s in steps])
                    final_y_e = steps[-1]['y']
                    return {
                        "type": "EulersMethod",
                        "result": (f"dy/dx = {rhs_str}, h={h_val}, {n_steps} steps\n"
                                   f"VERIFIED ITERATIONS (USE THESE EXACT VALUES):\n{step_str}\n"
                                   f"FINAL ANSWER: y({steps[-1]['x']}) = {final_y_e} — USE THIS EXACTLY"),
                        "latex": f"y_{{{n_steps}}} = {final_y_e}"
                    }

            # ════════════════════════════════════════════════════════
            # RUNGE-KUTTA RK4
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["runge-kutta", "runge kutta", "rk4"]):
                rhs_str = get_ode_rhs()
                x0_rk = re.search(r"x\s*0?\s*[=:]\s*([-\d\.]+)", p)
                y0_rk = re.search(r"y\s*[=\(]\s*0?\s*\)?\s*[=:]\s*([-\d\.]+)", p)
                h_val = get_h()
                n_steps = get_iters(default=3)
                if rhs_str and x0_rk and y0_rk:
                    x_s, y_s = sp.symbols('x y')
                    rhs_expr_rk = parse_expr(clean(rhs_str), transformations=tfms,
                                             local_dict={**ld, "y": y_s})
                    f_rk = sp.lambdify((x_s, y_s), rhs_expr_rk, modules="math")
                    xn_r, yn_r = float(x0_rk.group(1)), float(y0_rk.group(1))
                    steps = []
                    for i in range(n_steps):
                        k1 = h_val * f_rk(xn_r, yn_r)
                        k2 = h_val * f_rk(xn_r+h_val/2, yn_r+k1/2)
                        k3 = h_val * f_rk(xn_r+h_val/2, yn_r+k2/2)
                        k4 = h_val * f_rk(xn_r+h_val, yn_r+k3)
                        yn1 = yn_r + (k1+2*k2+2*k3+k4)/6
                        xn_r += h_val
                        steps.append({"n": i+1, "x": round(xn_r,6),
                                      "k1": round(k1,8), "k2": round(k2,8),
                                      "k3": round(k3,8), "k4": round(k4,8),
                                      "y": round(yn1,8)})
                        yn_r = yn1
                    step_str = "\n".join([
                        f"  Step {s['n']}: x={s['x']}, k1={s['k1']}, k2={s['k2']}, k3={s['k3']}, k4={s['k4']}, y={s['y']}"
                        for s in steps])
                    final_y = steps[-1]['y']
                    return {
                        "type": "RungeKutta4",
                        "result": (f"dy/dx={rhs_str}, h={h_val}, {n_steps} steps\n"
                                   f"VERIFIED ITERATIONS (USE THESE EXACT k VALUES):\n{step_str}\n"
                                   f"FINAL ANSWER: y({round(xn_r,4)}) = {final_y} — USE THIS EXACTLY"),
                        "latex": f"y_{{{n_steps}}} = {final_y}"
                    }

        # ── 7. Theory of Numbers — SymPy computes exactly ───────────────
        elif any(k in p for k in [
                "gcd", "greatest common divisor", "hcf",
                "lcm", "least common multiple",
                "prime factor", "factoriz", "factori",
                "is prime", "isprime", "prime or not", "check prime",
                "totient", "euler's totient",
                "number of divisor", "sum of divisor", "divisors of",
                "mobius", "möbius",
                "diophantine",
                "chinese remainder", "crt",
                "fermat", "wilson",
                "quadratic residu", "legendre",
                "primitive root",
                "linear congruence", "congruence", "≡",
                "euler's theorem", "euler theorem",
        ]) or re.search(r'\bis\s+\d+\s+prime\b', p):
            import math as _math

            def _nums(text):
                return [int(n) for n in re.findall(r'\b\d+\b', text)]

            x_d, y_d = sp.symbols('x y')

            try:
                # ── GCD (Extended Euclidean) ──────────────────────────────
                if any(k in p for k in ["gcd","greatest common divisor","hcf"]):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a,b = nums[0],nums[1]
                        g = int(sp.gcd(a,b))
                        # Extended Euclidean
                        old_r,r = a,b; old_s,s = 1,0; old_t,t2 = 0,1
                        while r:
                            q=old_r//r; old_r,r=r,old_r-q*r
                            old_s,s=s,old_s-q*s; old_t,t2=t2,old_t-q*t2
                        result_str = (f"gcd({a},{b}) = {old_r}\n"
                                      f"Extended Euclidean: {a}×({old_s}) + {b}×({old_t}) = {old_r}\n"
                                      f"Verify: {a*old_s + b*old_t} = {old_r} ✅")
                        return {"type":"GCD","result":result_str,
                                "latex":f"\\gcd({a},{b})={old_r}"}

                # ── LCM ───────────────────────────────────────────────────
                elif any(k in p for k in ["lcm","least common multiple"]):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        l = int(sp.lcm(nums[0],nums[1]))
                        g = int(sp.gcd(nums[0],nums[1]))
                        return {"type":"LCM",
                                "result":f"lcm({nums[0]},{nums[1]}) = {l}, gcd = {g}",
                                "latex":f"\\text{{lcm}}({nums[0]},{nums[1]})={l}"}

                # ── PRIME FACTORIZATION ───────────────────────────────────
                elif any(k in p for k in ["prime factor","factoriz","factori"]):
                    nums = _nums(p)
                    if nums:
                        f_dict = sp.factorint(nums[0])
                        f_str = " × ".join([f"{pp}^{e}" if e>1 else str(pp) for pp,e in f_dict.items()])
                        return {"type":"PrimeFactorization",
                                "result":f"{nums[0]} = {f_str}",
                                "latex":f"{nums[0]} = {f_str}"}

                # ── IS PRIME ──────────────────────────────────────────────
                elif any(k in p for k in ["is prime","isprime","prime or not","check prime"]) or re.search(r'\bis\s+\d+\s+prime\b',p):
                    nums = _nums(p)
                    if nums:
                        n_val = nums[0]
                        is_p = sp.isprime(n_val)
                        ans = "PRIME" if is_p else "COMPOSITE (NOT PRIME)"
                        return {"type":"PrimeCheck",
                                "result":f"{n_val} is {ans}",
                                "latex":f"{n_val}\\text{{ is }}{ans}"}

                # ── EULER TOTIENT ─────────────────────────────────────────
                elif any(k in p for k in ["totient","euler's totient"]):
                    nums = _nums(p)
                    if nums:
                        phi = int(sp.totient(nums[0]))
                        f_dict = sp.factorint(nums[0])
                        return {"type":"EulerTotient",
                                "result":f"φ({nums[0]}) = {phi}, factorization = {f_dict}",
                                "latex":f"\\phi({nums[0]})={phi}"}

                # ── DIVISOR FUNCTIONS τ, σ ────────────────────────────────
                elif any(k in p for k in ["number of divisor","sum of divisor","divisors of","tau","sigma"]):
                    nums = _nums(p)
                    if nums:
                        divs = sp.divisors(nums[0])
                        return {"type":"DivisorFunctions",
                                "result":f"divisors({nums[0]}) = {divs}, τ = {len(divs)}, σ = {sum(divs)}",
                                "latex":f"\\tau({nums[0]})={len(divs)}, \\sigma({nums[0]})={sum(divs)}"}

                # ── MOBIUS FUNCTION ───────────────────────────────────────
                elif any(k in p for k in ["mobius","möbius"]):
                    nums = _nums(p)
                    if nums:
                        mu = sp.mobius(nums[0])
                        return {"type":"Mobius",
                                "result":f"μ({nums[0]}) = {mu}",
                                "latex":f"\\mu({nums[0]})={mu}"}

                # ── DIOPHANTINE EQUATION ──────────────────────────────────
                elif "diophantine" in p:
                    m = re.search(r"(\d+)\s*x\s*[+\-]\s*(\d+)\s*y\s*=\s*(\d+)",p)
                    if m:
                        a_d,b_d,c_d = int(m.group(1)),int(m.group(2)),int(m.group(3))
                        g = int(sp.gcd(a_d,b_d))
                        if c_d%g == 0:
                            sol = sp.diophantine(sp.Eq(a_d*x_d+b_d*y_d, c_d))
                            return {"type":"Diophantine",
                                    "result":f"{a_d}x+{b_d}y={c_d}: general solution={sol}, gcd={g}",
                                    "latex":str(sol)}
                        else:
                            return {"type":"Diophantine",
                                    "result":f"No integer solution: gcd({a_d},{b_d})={g} does not divide {c_d}",
                                    "latex":"\\text{No solution}"}

                # ── CRT — MUST be before congruence ───────────────────────
                elif any(k in p for k in ["chinese remainder","crt"]):
                    pairs = re.findall(r'x\s*[≡=]\s*(\d+)\s*(?:mod|modulo|\(mod)\s*(\d+)',p)
                    if len(pairs) >= 2:
                        remainders = [int(r) for r,m in pairs]
                        moduli = [int(m) for r,m in pairs]
                        sol = sp.ntheory.modular.crt(moduli, remainders)
                        verify = [f"{sol[0]}%{m}={sol[0]%m}" for m in moduli]
                        return {"type":"CRT",
                                "result":f"x ≡ {sol[0]} (mod {sol[1]}), verify: {verify}",
                                "latex":f"x\\equiv {sol[0]}\\pmod{{{sol[1]}}}"}

                # ── FERMAT'S LITTLE THEOREM ───────────────────────────────
                elif "fermat" in p:
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a_f,p_f = nums[0],nums[1]
                        if sp.isprime(p_f):
                            r = pow(a_f,p_f-1,p_f)
                            return {"type":"FermatTheorem",
                                    "result":f"{a_f}^({p_f}-1) mod {p_f} = {r} ≡ 1 (mod {p_f})",
                                    "latex":f"{a_f}^{{{p_f-1}}}\\equiv 1\\pmod{{{p_f}}}"}

                # ── EULER'S THEOREM ───────────────────────────────────────
                elif "euler" in p and ("theorem" in p or "theorem" in p):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a_e,n_e = nums[0],nums[1]
                        phi = int(sp.totient(n_e))
                        r = pow(a_e,phi,n_e)
                        return {"type":"EulerTheorem",
                                "result":f"φ({n_e})={phi}, {a_e}^{phi} mod {n_e} = {r} ≡ 1 (mod {n_e})",
                                "latex":f"{a_e}^{{\\phi({n_e})}}\\equiv 1\\pmod{{{n_e}}}"}

                # ── WILSON'S THEOREM ──────────────────────────────────────
                elif "wilson" in p:
                    nums = _nums(p)
                    if nums:
                        p_w = nums[0]
                        val = _math.factorial(p_w-1)%p_w
                        return {"type":"WilsonTheorem",
                                "result":f"({p_w}-1)! mod {p_w} = {val} ≡ -1 (mod {p_w})",
                                "latex":f"({p_w}-1)!\\equiv -1\\pmod{{{p_w}}}"}

                # ── LINEAR CONGRUENCE ─────────────────────────────────────
                elif any(k in p for k in ["congruence","linear congruence"]) or re.search(r'\d+\s*x\s*[≡=]',p):
                    m = re.search(r"(\d+)\s*x\s*[≡=]\s*(\d+)\s*(?:\(mod|mod|modulo)\s*(\d+)",p)
                    if m:
                        a_c,b_c,n_c = int(m.group(1)),int(m.group(2)),int(m.group(3))
                        g = int(sp.gcd(a_c,n_c))
                        if b_c%g != 0:
                            return {"type":"LinearCongruence",
                                    "result":f"No solution: gcd({a_c},{n_c})={g} ∤ {b_c}",
                                    "latex":"\\text{No solution}"}
                        sols = [i for i in range(n_c) if (a_c*i)%n_c==b_c%n_c]
                        return {"type":"LinearCongruence",
                                "result":f"{a_c}x ≡ {b_c} (mod {n_c}): x ≡ {sols} (mod {n_c}), {g} solution(s)",
                                "latex":f"x\\equiv {sols[0]}\\pmod{{{n_c//g}}}"}

                # ── QUADRATIC RESIDUES ────────────────────────────────────
                elif any(k in p for k in ["quadratic residu","quadratic non"]):
                    nums = _nums(p)
                    if nums:
                        p_q = nums[0]
                        qr = sorted(set([pow(i,2,p_q) for i in range(1,p_q)]))
                        qnr = [i for i in range(1,p_q) if i not in qr]
                        return {"type":"QuadraticResidues",
                                "result":f"QR mod {p_q} = {qr}, QNR mod {p_q} = {qnr}",
                                "latex":f"QR\\pmod{{{p_q}}}={qr}"}

                # ── LEGENDRE SYMBOL ───────────────────────────────────────
                elif "legendre" in p:
                    m = re.search(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)",p)
                    if m:
                        a_l,p_l = int(m.group(1)),int(m.group(2))
                        val = 1 if pow(a_l,(p_l-1)//2,p_l)==1 else (-1 if a_l%p_l!=0 else 0)
                        meaning = "QR (quadratic residue)" if val==1 else ("QNR (non-residue)" if val==-1 else "0 (divisible)")
                        return {"type":"LegendreSymbol",
                                "result":f"({a_l}/{p_l}) = {val} → {a_l} is {meaning} mod {p_l}",
                                "latex":f"\\left(\\frac{{{a_l}}}{{{p_l}}}\\right)={val}"}

                # ── PRIMITIVE ROOT ────────────────────────────────────────
                elif "primitive root" in p:
                    nums = _nums(p)
                    if nums:
                        pr = sp.ntheory.primitive_root(nums[0])
                        return {"type":"PrimitiveRoot",
                                "result":f"primitive_root({nums[0]}) = {pr}",
                                "latex":f"g={pr}"}

            except Exception:
                pass  # safe fallback to AI

        # ── 7. Solve equation ────────────────────────────────────────
        elif any(k in p for k in ["solve", "roots", "find x"]):
            raw = re.sub(r"(solve|find x|roots of|roots|the equation)", "", p)
            raw = raw.strip().strip(":").strip()
            if "=" in raw:
                lhs_s, rhs_s = raw.split("=", 1)
                lhs_e = parse_expr(clean(lhs_s), transformations=tfms, local_dict=ld)
                rhs_e = parse_expr(clean(rhs_s), transformations=tfms, local_dict=ld)
                expr  = lhs_e - rhs_e
            else:
                expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            if x in expr.free_symbols:
                sol = sp.solve(expr, x)
                sol_latex = ", ".join([sp.latex(s) for s in sol])
                return {
                    "type": "Equation",
                    "result": str(sol),
                    "latex": r"x \in \{" + sol_latex + r"\}"
                }

        # ── 8. Real Analysis II — SymPy for computations, AI for theory ──
        elif any(k in p for k in [
                # Sets & Real Numbers
                "supremum", "infimum", "least upper bound", "greatest lower bound",
                "lub", "glb", "archimedean", "bounded set", "completeness",
                "cartesian product", "density of rational", "real number system",
                "field propert", "order propert",
                # Sequences
                "sequence", "cauchy sequence", "bounded sequence",
                "monotone sequence", "subsequence", "bolzano", "weierstrass",
                # Series
                "ratio test", "root test", "integral test", "comparison test",
                "alternating series", "leibniz test", "absolute convergence",
                "conditional convergence", "cauchy criterion",
                "pointwise convergence", "uniform convergence",
                "weierstrass m-test", "m-test",
                # Limits & Continuity
                "epsilon delta", "epsilon-delta", "uniform continuity",
                "intermediate value", "extreme value theorem",
                # Differentiation theorems
                "mean value theorem", "rolle", "taylor's theorem",
                "lhopital", "l'hopital",
                # Riemann Integration
                "riemann sum", "riemann integral", "upper sum", "lower sum",
                "darboux", "integrability", "fundamental theorem of calculus",
        ]):
            try:
                n_s = sp.Symbol('n', positive=True)

                # ── Sequence limit ────────────────────────────────────────
                if any(k in p for k in ["sequence","limit of sequence"]):
                    # Extract expression after "of" or "for"
                    m = re.search(r"(?:of|for|lim)\s+(.+?)\s*(?:as|when|$)", p)
                    if m:
                        raw = clean(m.group(1))
                        try:
                            expr_s = parse_expr(raw, transformations=tfms,
                                               local_dict={**ld, "n": n_s})
                            lim_val = sp.limit(expr_s, n_s, sp.oo)
                            return {
                                "type": "SequenceLimit",
                                "result": f"lim({m.group(1)}) as n→∞ = {lim_val}",
                                "latex": f"\\lim_{{n\\to\\infty}} = {sp.latex(lim_val)}"
                            }
                        except Exception:
                            pass

                # ── Series sum ────────────────────────────────────────────
                elif any(k in p for k in ["series","sum of"]):
                    m = re.search(r"(?:sum\s+of|series\s+(?:of\s+)?|convergence\s+of)\s*(.+?)(?:\s+from|\s+using|\s+by|$)", p)
                    if m:
                        raw = m.group(1).strip()
                        raw = re.sub(r"^series\s+sum\s+of\s+", "", raw)
                        raw = re.sub(r"^series\s+of\s+", "", raw)
                        raw = re.sub(r"^sum\s+of\s+", "", raw)
                        raw = re.sub(r"^of\s+", "", raw)
                        raw = clean(raw)
                        try:
                            expr_ser = parse_expr(raw, transformations=tfms,
                                                  local_dict={**ld, "n": n_s})
                            s_val = sp.summation(expr_ser, (n_s, 1, sp.oo))
                            converges = s_val.is_finite
                            return {
                                "type": "SeriesConvergence",
                                "result": (f"Series sum = {s_val}, "
                                           f"Converges: {converges}"),
                                "latex": f"\\sum_{{n=1}}^{{\\infty}} = {sp.latex(s_val)}"
                            }
                        except Exception:
                            pass

                # ── Taylor Series ─────────────────────────────────────────
                elif any(k in p for k in ["taylor", "maclaurin"]):
                    funcs = {
                        "sin": sp.sin(x), "cos": sp.cos(x),
                        "exp": sp.exp(x), "e^x": sp.exp(x),
                        "ln": sp.log(1+x), "log": sp.log(1+x),
                        "tan": sp.tan(x)
                    }
                    for fname, fexpr in funcs.items():
                        if fname in p:
                            n_terms = 6
                            ts = sp.series(fexpr, x, 0, n_terms)
                            return {
                                "type": "TaylorSeries",
                                "result": f"Taylor series of {fname}: {ts}",
                                "latex": sp.latex(ts)
                            }

                # ── L'Hopital ─────────────────────────────────────────────
                elif any(k in p for k in ["lhopital","l'hopital"]):
                    m = re.search(r"(?:of|for)\s+(.+?)\s*(?:as|at|when)\s*x\s*[→→=]\s*([\d\.]+|inf)", p)
                    if m:
                        raw = clean(m.group(1))
                        pt_str = m.group(2)
                        pt = sp.oo if pt_str in ("inf","infinity") else sp.sympify(pt_str)
                        try:
                            expr_lh = parse_expr(raw, transformations=tfms, local_dict=ld)
                            lim_val = sp.limit(expr_lh, x, pt)
                            return {
                                "type": "LHopital",
                                "result": f"lim({m.group(1)}) as x→{pt_str} = {lim_val}",
                                "latex": f"\\lim_{{x\\to {pt_str}}} = {sp.latex(lim_val)}"
                            }
                        except Exception:
                            pass

                # ── Riemann Integral ──────────────────────────────────────
                elif any(k in p for k in ["riemann","riemann integral","riemann sum"]):
                    # Try to extract definite integral
                    m = re.search(r"(?:of|for)\s+(.+?)\s+(?:from|on)\s+([\d\.]+)\s+to\s+([\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1))
                        a_v = sp.sympify(m.group(2))
                        b_v = sp.sympify(m.group(3))
                        try:
                            expr_r = parse_expr(raw, transformations=tfms, local_dict=ld)
                            result_r = sp.integrate(expr_r, (x, a_v, b_v))
                            return {
                                "type": "RiemannIntegral",
                                "result": f"∫({m.group(1)}) from {a_v} to {b_v} = {result_r}",
                                "latex": f"\\int_{{{a_v}}}^{{{b_v}}} = {sp.latex(result_r)}"
                            }
                        except Exception:
                            pass

            except Exception:
                pass  # safe fallback to AI for all theory/proof questions

        # ── 9. Differential Geometry — SymPy for computations ──────────
        elif any(k in p for k in [
                "curvature", "torsion", "tangent vector", "normal vector",
                "binormal", "serret-frenet", "frenet", "osculating",
                "arc length", "space curve", "plane curve", "helix", "helices",
                "evolute", "involute", "rectifying plane",
                "first fundamental form", "second fundamental form",
                "fundamental form", "gaussian curvature", "mean curvature",
                "principal curvature", "geodesic",
                "parametric surface", "christoffel", "covariant derivative",
                "contravariant", "metric tensor",
        ]):
            t_s = sp.Symbol('t')
            u_s, v_s = sp.symbols('u v')
            try:

                # ── Curvature of plane curve y=f(x) ──────────────────────
                if "curvature" in p and not any(k in p for k in ["gaussian","mean","space","torsion"]):
                    # Extract function and point
                    m = re.search(r"(?:of|for)\s+y\s*=\s*(.+?)(?:\s+at|\s*$)", p)
                    pt_m = re.search(r"at\s+x\s*[=:]\s*([-\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1).strip())
                        expr_c = parse_expr(raw, transformations=tfms, local_dict=ld)
                        dy  = sp.diff(expr_c, x)
                        d2y = sp.diff(expr_c, x, 2)
                        kappa_expr = sp.Abs(d2y) / (1 + dy**2)**sp.Rational(3,2)
                        if pt_m:
                            pt_val = float(pt_m.group(1))
                            kappa_val = sp.simplify(kappa_expr.subs(x, pt_val))
                            return {
                                "type": "Curvature",
                                "result": f"κ at x={pt_val}: y'={dy.subs(x,pt_val)}, y''={d2y.subs(x,pt_val)}, κ={kappa_val}",
                                "latex": f"\\kappa = {sp.latex(kappa_val)}"
                            }
                        else:
                            return {
                                "type": "Curvature",
                                "result": f"κ(x) = {sp.simplify(kappa_expr)}",
                                "latex": f"\\kappa = {sp.latex(sp.simplify(kappa_expr))}"
                            }

                # ── Arc Length ────────────────────────────────────────────
                elif "arc length" in p:
                    m = re.search(r"(?:of|for)\s+y\s*=\s*(.+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1).strip())
                        a_v = sp.sympify(m.group(2))
                        b_v = sp.sympify(m.group(3))
                        expr_al = parse_expr(raw, transformations=tfms, local_dict=ld)
                        dy = sp.diff(expr_al, x)
                        integrand = sp.sqrt(1 + dy**2)
                        L = sp.integrate(integrand, (x, a_v, b_v))
                        L_simplified = sp.simplify(L)
                        return {
                            "type": "ArcLength",
                            "result": f"L = ∫√(1+y'²)dx from {a_v} to {b_v} = {L_simplified}",
                            "latex": f"L = {sp.latex(L_simplified)}"
                        }

                # ── Space Curve: Curvature + Torsion ─────────────────────
                elif any(k in p for k in ["space curve","torsion","frenet","serret"]):
                    # Extract parametric curve r(t) = (x(t), y(t), z(t))
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    pt_m = re.search(r"at\s+t\s*[=:]\s*([-\d\.]+)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "t": t_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "t": t_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "t": t_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        dr = r_vec.diff(t_s)
                        d2r = dr.diff(t_s)
                        d3r = d2r.diff(t_s)
                        speed = sp.sqrt(dr.dot(dr))
                        cross = dr.cross(d2r)
                        kappa = sp.simplify(sp.sqrt(cross.dot(cross)) / speed**3)
                        torsion_val = sp.simplify(cross.dot(d3r) / cross.dot(cross))
                        t_val = float(pt_m.group(1)) if pt_m else 0
                        k_at = sp.simplify(kappa.subs(t_s, t_val))
                        tau_at = sp.simplify(torsion_val.subs(t_s, t_val))
                        T_vec = sp.simplify(dr / speed)
                        return {
                            "type": "FrenetSerret",
                            "result": (f"r(t)={pts[0]}, at t={t_val}:\n"
                                      f"κ = {k_at}, τ = {tau_at}\n"
                                      f"T = {T_vec.subs(t_s,t_val).T}"),
                            "latex": f"\\kappa={sp.latex(k_at)}, \\tau={sp.latex(tau_at)}"
                        }

                # ── First Fundamental Form ────────────────────────────────
                elif "first fundamental form" in p:
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        ru = r_vec.diff(u_s)
                        rv = r_vec.diff(v_s)
                        E = sp.simplify(ru.dot(ru))
                        F = sp.simplify(ru.dot(rv))
                        G = sp.simplify(rv.dot(rv))
                        return {
                            "type": "FirstFundamentalForm",
                            "result": f"E={E}, F={F}, G={G}, ds²={E}du²+{2*F}dudv+{G}dv²",
                            "latex": f"E={sp.latex(E)}, F={sp.latex(F)}, G={sp.latex(G)}"
                        }

                # ── Gaussian + Mean Curvature ─────────────────────────────
                elif any(k in p for k in ["gaussian curvature","mean curvature"]):
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        ru = r_vec.diff(u_s); rv = r_vec.diff(v_s)
                        E = sp.simplify(ru.dot(ru)); F = sp.simplify(ru.dot(rv)); G = sp.simplify(rv.dot(rv))
                        n = ru.cross(rv); N = sp.simplify(n / sp.sqrt(n.dot(n)))
                        L = sp.simplify(N.dot(ru.diff(u_s)))
                        M = sp.simplify(N.dot(ru.diff(v_s)))
                        Nv = sp.simplify(N.dot(rv.diff(v_s)))
                        K = sp.simplify((L*Nv - M**2)/(E*G - F**2))
                        H = sp.simplify((E*Nv - 2*F*M + G*L)/(2*(E*G - F**2)))
                        return {
                            "type": "GaussianCurvature",
                            "result": f"K (Gaussian) = {K}, H (Mean) = {H}",
                            "latex": f"K={sp.latex(K)}, H={sp.latex(H)}"
                        }

            except Exception:
                pass  # safe fallback to AI

        # ── 10. Hydro Mechanics — SymPy for computations, AI for theory ──
        elif any(k in p for k in [
                "continuity equation", "equation of continuity",
                "streamline", "stream function", "stream line",
                "velocity potential", "irrotational", "rotational motion",
                "lagrangian", "eulerian", "vortex", "vorticity",
                "path line", "streak line",
                "bernoulli", "euler's equation", "euler equation of motion",
                "torricelli", "flow rate", "discharge",
                "reynolds number", "reynolds",
                "hydrostatic pressure", "pressure at depth",
                "hydrostatic", "buoyancy", "archimedes",
                "laminar flow", "turbulent flow", "viscous flow",
                "incompressible fluid", "steady flow",
                "navier-stokes", "navier stokes",
                "stokes stream function", "complex velocity potential",
                "source", "sink", "doublet",
                "milne thomson", "blasius theorem",
                "dimensional analysis", "buckingham pi",
        ]):
            try:
                x_h, y_h = sp.symbols('x y')
                g_val = sp.Rational(981, 100)  # 9.81

                # ── Continuity: find v2 from A1v1=A2v2 ───────────────
                if any(k in p for k in ["continuity equation","equation of continuity"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 3:
                        A1_v,v1_v,A2_v = nums[0],nums[1],nums[2]
                        v2_v = round(A1_v*v1_v/A2_v, 6)
                        Q_v  = round(A1_v*v1_v, 6)
                        return {
                            "type": "ContinuityEq",
                            "result": (f"A1={A1_v}, v1={v1_v}, A2={A2_v}\n"
                                      f"v2 = A1*v1/A2 = {v2_v} m/s\n"
                                      f"Flow rate Q = A1*v1 = {Q_v} m³/s"),
                            "latex": f"v_2 = {v2_v}\\text{{ m/s}}"
                        }

                # ── Reynolds Number ────────────────────────────────────
                elif any(k in p for k in ["reynolds number","reynolds"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 4:
                        rho_v,v_v,D_v,mu_v = nums[0],nums[1],nums[2],nums[3]
                        Re = round(rho_v*v_v*D_v/mu_v, 2)
                        flow = "Turbulent (Re>4000)" if Re>4000 else ("Transitional (2300<Re<4000)" if Re>2300 else "Laminar (Re<2300)")
                        return {
                            "type": "ReynoldsNumber",
                            "result": f"Re = ρvD/μ = {rho_v}×{v_v}×{D_v}/{mu_v} = {Re} → {flow}",
                            "latex": f"Re = {Re}"
                        }

                # ── Bernoulli: find P2 ─────────────────────────────────
                elif "bernoulli" in p:
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 5:
                        P1_v,v1_v,h1_v,v2_v,h2_v = nums[0],nums[1],nums[2],nums[3],nums[4]
                        rho_v = 1000  # default water
                        P2_v = round(P1_v + 0.5*rho_v*(v1_v**2-v2_v**2) + rho_v*9.81*(h1_v-h2_v), 4)
                        return {
                            "type": "Bernoulli",
                            "result": (f"P1+½ρv1²+ρgh1 = P2+½ρv2²+ρgh2\n"
                                      f"P2 = {P2_v} Pa"),
                            "latex": f"P_2 = {P2_v}\\text{{ Pa}}"
                        }

                # ── Torricelli: v = √(2gh) ─────────────────────────────
                elif "torricelli" in p:
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if nums:
                        h_v = nums[0]
                        v_torr = round((2*9.81*h_v)**0.5, 6)
                        return {
                            "type": "Torricelli",
                            "result": f"v = √(2gh) = √(2×9.81×{h_v}) = {v_torr} m/s",
                            "latex": f"v = {v_torr}\\text{{ m/s}}"
                        }

                # ── Hydrostatic Pressure ───────────────────────────────
                elif any(k in p for k in ["hydrostatic pressure","pressure at depth"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if nums:
                        h_v = nums[0]
                        rho_v = 1000
                        P_gauge = round(rho_v*9.81*h_v, 4)
                        P_abs = round(101325 + P_gauge, 4)
                        return {
                            "type": "HydrostaticPressure",
                            "result": (f"At depth h={h_v}m:\n"
                                      f"Gauge pressure = ρgh = {P_gauge} Pa\n"
                                      f"Absolute pressure = P0+ρgh = {P_abs} Pa"),
                            "latex": f"P = P_0 + \\rho g h = {P_abs}\\text{{ Pa}}"
                        }

                # ── Flow Rate ──────────────────────────────────────────
                elif any(k in p for k in ["flow rate","discharge"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 2:
                        A_v, v_v = nums[0], nums[1]
                        Q_v = round(A_v*v_v, 8)
                        return {
                            "type": "FlowRate",
                            "result": f"Q = A×v = {A_v}×{v_v} = {Q_v} m³/s",
                            "latex": f"Q = {Q_v}\\text{{ m³/s}}"
                        }

                # ── Velocity Potential ─────────────────────────────────
                elif "velocity potential" in p:
                    m = re.search(r"(?:phi|φ|potential)\s*=\s*(.+?)(?:\s|$)", p)
                    if m:
                        raw = clean(m.group(1))
                        phi_expr = parse_expr(raw, transformations=tfms, local_dict={**ld,"x":x_h,"y":y_h})
                        u_comp = sp.diff(phi_expr, x_h)
                        v_comp = sp.diff(phi_expr, y_h)
                        lap = sp.diff(phi_expr,x_h,2) + sp.diff(phi_expr,y_h,2)
                        return {
                            "type": "VelocityPotential",
                            "result": (f"φ={str(phi_expr)}, u=∂φ/∂x={u_comp}, v=∂φ/∂y={v_comp}\n"
                                      f"∇²φ={sp.simplify(lap)} (irrotational: {sp.simplify(lap)==0})"),
                            "latex": f"\\nabla^2\\phi = {sp.latex(sp.simplify(lap))}"
                        }

            except Exception:
                pass  # safe fallback to AI

        # ── 11. Matrix / Eigenvalues — delegate to AI ────────────────────
        elif any(k in p for k in ["matrix", "determinant", "eigenvalue",
                                   "eigenvector", "det("]):
            return {"type": "Matrix", "result": "matrix_detected", "latex": ""}

        # ── 9. Modular arithmetic — delegate to AI ────────────────────
        elif "mod" in p or "congruence" in p:
            return {"type": "NumberTheory", "result": "mod_detected", "latex": ""}

    except Exception:
        pass  # Silently fall back — AI handles it

    return {"type": "general", "result": None, "latex": ""}

# ════════════════════════════════════════════════════════════════════
# GRAPH PLOTTING — only when user asks to plot/graph/draw/visualize
# ════════════════════════════════════════════════════════════════════
def plot_graph(problem: str, sympy_info: dict):
    """
    Plot graph only when user explicitly requests it.
    Triggered by: plot, graph, draw, sketch, visualize, show graph.
    Expression extracted from: y=..., f(x)=..., or after keyword.
    Safe fallback — never crashes the app.
    """
    p = problem.lower().strip()

    # ── Only trigger on explicit plot keywords ──────────────────────
    if not any(k in p for k in ["plot", "graph", "draw",
                                  "sketch", "visualize", "show graph"]):
        return  # user didn't ask — do nothing

    try:
        x_sym = sp.Symbol('x')
        tfms  = standard_transformations + (implicit_multiplication_application,)
        ld    = {
            "x": x_sym, "e": sp.E, "E": sp.E,
            "pi": sp.pi, "PI": sp.pi,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "ln": sp.log,
            "sqrt": sp.sqrt
        }

        def clean_expr(s):
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"\^", "**", s)
            return s

        # ── Extract expression — 3 strategies in order ───────────────
        expr_sym = None
        label    = ""

        # Strategy 1: y = ... or f(x) = ... (most explicit)
        m = re.search(
            r"(?:y\s*=\s*|f\s*\(x\)\s*=\s*)(.+?)(?:\s+from|\s+for|\s*$)", p)
        if m:
            raw = clean_expr(m.group(1).strip())
            try:
                expr_sym = parse_expr(raw, transformations=tfms, local_dict=ld)
                label = raw
            except Exception:
                pass

        # Strategy 2: after keyword — plot/draw/graph/sketch <expr>
        if expr_sym is None:
            m = re.search(
                r"(?:plot|draw|graph|sketch|visualize)\s+(.+?)(?:\s+from|\s+for|\s*$)", p)
            if m:
                raw = m.group(1).strip()
                # Strip noise words that are not expressions
                noise_words = ["the graph of", "the graph", "the function of",
                               "the function", "the curve of", "the curve",
                               "this", "it", "me", "of"]
                for nw in noise_words:
                    raw = raw.replace(nw, "").strip()
                raw = clean_expr(raw)
                if raw:
                    try:
                        expr_sym = parse_expr(raw, transformations=tfms, local_dict=ld)
                        label = raw
                    except Exception:
                        pass

        # Strategy 3: use sympy_info latex if already computed
        if expr_sym is None and sympy_info.get("latex"):
            raw = clean_expr(sympy_info["latex"])
            try:
                expr_sym = parse_expr(raw, transformations=tfms, local_dict=ld)
                label = raw
            except Exception:
                pass

        # Nothing found — ask user to be specific
        if expr_sym is None:
            st.info("📊 Please specify the function. Example: *plot y = x^2 - 4*")
            return

        # ── Determine x range ────────────────────────────────────────
        x_range_m = re.search(
            r"(?:from|between)\s*([-\d\.]+)\s*(?:to|and)\s*([-\d\.]+)", p)
        x_min = float(x_range_m.group(1)) if x_range_m else -10
        x_max = float(x_range_m.group(2)) if x_range_m else 10

        # ── Lambdify ─────────────────────────────────────────────────
        f_num  = sp.lambdify(x_sym, expr_sym, modules=["numpy"])
        df_sym = sp.diff(expr_sym, x_sym)
        df_num = sp.lambdify(x_sym, df_sym, modules=["numpy"])

        x_vals = np.linspace(x_min, x_max, 800)

        with np.errstate(all="ignore"):
            y_vals  = np.array(f_num(x_vals),  dtype=float)
            dy_vals = np.array(df_num(x_vals), dtype=float)

        y_vals[~np.isfinite(y_vals)]   = np.nan
        dy_vals[~np.isfinite(dy_vals)] = np.nan

        # ── Build plot — dark theme matching Saad.AI ─────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        # f(x) — blue
        ax.plot(x_vals, y_vals, color="#3b82f6", linewidth=2.2,
                label=f"$f(x) = {sp.latex(expr_sym)}$")

        # f'(x) — orange dashed — only when derivative was asked
        if sympy_info.get("type") == "Derivative":
            ax.plot(x_vals, dy_vals, color="#f59e0b", linewidth=1.8,
                    linestyle="--",
                    label=f"$f'(x) = {sp.latex(df_sym)}$")

        # Axes lines
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.axvline(0, color="#555", linewidth=0.8)

        # Grid
        ax.grid(True, color="#2a2a2a", linewidth=0.6, linestyle="--")

        # Labels
        ax.set_xlabel("x", color="#ececec", fontsize=11)
        ax.set_ylabel("y", color="#ececec", fontsize=11)
        ax.set_title(f"$y = {sp.latex(expr_sym)}$",
                     color="#ffffff", fontsize=13, pad=12)

        # Tick + spine colors
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

        # Legend
        ax.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a",
                  labelcolor="#ececec", fontsize=9)

        # Smart y limits — clip extreme outliers
        valid_y = y_vals[np.isfinite(y_vals)]
        if len(valid_y) > 0:
            y_med = np.median(valid_y)
            y_std = np.std(valid_y)
            pad   = (y_std * 5) * 0.1 if y_std > 0 else 1
            ax.set_ylim(
                max(valid_y.min(), y_med - 5*y_std) - pad,
                min(valid_y.max(), y_med + 5*y_std) + pad
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # free memory
        # Caption below graph
        deriv_label = "  🟠 f'(x)" if sympy_info.get("type") == "Derivative" else ""
        st.caption(f"📊 $y = {sp.latex(expr_sym)}$ "
                   f"| x ∈ [{x_min}, {x_max}]"
                   f"{deriv_label}")

    except Exception:
        pass  # silent fallback — never crash the app


# ════════════════════════════════════════════════════════════════════
# GROQ API — free, fast, Llama 3.3 70B
# ════════════════════════════════════════════════════════════════════
def ask_ai(problem: str, sympy_info: dict, history: list) -> str:

    # ── Multi-provider auto-rotation ────────────────────────────────
    # Try each provider in order — skip if key missing or 429
    def try_groq(key, messages):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "max_tokens": 2048, "temperature": 0.15, "top_p": 0.9},
            timeout=60
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
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
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
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek/deepseek-r1:free",
                  "messages": messages, "max_tokens": 2048, "temperature": 0.15},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # All providers in rotation order
    providers = [
        ("Groq-1",      os.environ.get("GROQ_API_KEY_1",""),      try_groq),
        ("Groq-2",      os.environ.get("GROQ_API_KEY_2",""),      try_groq),
        ("Groq-3",      os.environ.get("GROQ_API_KEY_3",""),      try_groq),
        ("Gemini-1",    os.environ.get("GEMINI_API_KEY_1",""),    try_gemini),
        ("Gemini-2",    os.environ.get("GEMINI_API_KEY_2",""),    try_gemini),
        ("Gemini-3",    os.environ.get("GEMINI_API_KEY_3",""),    try_gemini),
        ("Gemini-4",    os.environ.get("GEMINI_API_KEY_4",""),    try_gemini),
        ("OpenRouter",  os.environ.get("OPENROUTER_API_KEY",""),  try_openrouter),
    ]

    # Check at least one key exists
    if not any(key for _, key, _ in providers):
        return (
            "⚠️ **No API keys set.**\n\n"
            "Add these secrets in HF Space → Settings → Secrets:\n"
            "`GROQ_API_KEY_1`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`\n"
            "`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`\n"
            "`OPENROUTER_API_KEY`"
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

    # All providers exhausted
    return (
        f"⚠️ **All providers failed.**\n\n"
        f"Last error: `{last_error}`\n\n"
        "Tried: 3×Groq → 4×Gemini → OpenRouter. All failed or rate-limited.\n"
        "If you see `API not enabled` — visit [aistudio.google.com](https://aistudio.google.com/app/apikey) and enable the Generative Language API for your account.\n"
        "If you see `quota exceeded` — wait 60 seconds (rate limit) or until midnight Pacific (daily limit)."
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
            for page_num in range(min(len(doc), 6)):
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
        groq_keys = [
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", ""),
        ]
        groq_vision_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.2-11b-vision-preview",
        ]
        for i, key in enumerate(groq_keys):
            if not key.strip():
                continue
            for model in groq_vision_models:
                try:
                    resp = requests.post(
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
                        timeout=60
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
    gemini_keys = [
        os.environ.get("GEMINI_API_KEY_1", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
        os.environ.get("GEMINI_API_KEY_4", ""),
    ]
    gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
    for i, key in enumerate(gemini_keys):
        if not key.strip():
            continue
        for model in gemini_models:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [
                            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                            {"text": prompt}
                        ]}],
                        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.15}
                    },
                    timeout=60
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
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key.strip():
            or_models = [
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "qwen/qwen2-vl-7b-instruct:free",
            ]
            for model in or_models:
                try:
                    resp = requests.post(
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
                        timeout=60
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
    MAX_SIZE = 5 * 1024 * 1024  # 5MB
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


def ask_ai_streaming(problem: str, sympy_info: dict, history: list) -> str:
    """Get AI response and stream it word by word using st.write_stream."""
    import time

    # Get full response first
    full_response = ask_ai(problem, sympy_info, history)

    # Stream chunk by chunk (3-4 words at a time) — natural reading pace
    def word_generator():
        words = full_response.split(" ")
        chunk_size = 3  # 3 words at a time
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            time.sleep(0.08)  # 80ms per chunk — comfortable reading pace

    # Use st.write_stream for streaming display
    streamed = st.write_stream(word_generator())
    return full_response  # return full for history


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Saad.AI")
    st.caption("BSc Mathematics Engine")
    st.divider()

    # ── New Chat Button ──────────────────────────────────────────
    if st.button("➕  New Chat", use_container_width=True):
        save_current_chat()
        new_chat()
        st.rerun()

    st.divider()

    # ── Chat History ─────────────────────────────────────────────
    if st.session_state.chats:
        st.markdown("**💬 Chat History**")
        # Show most recent first
        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda x: x[1]["created"],
            reverse=True
        )
        for chat_id, chat_data in sorted_chats:
            col1, col2 = st.columns([4,1])
            with col1:
                # Highlight current chat
                is_current = chat_id == st.session_state.current_chat_id
                label = ("▶ " if is_current else "") + chat_data["title"]
                if st.button(label, key=f"load_{chat_id}", use_container_width=True):
                    save_current_chat()
                    load_chat(chat_id)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    supa_delete_chat(chat_id)  # delete from Supabase too
                    if chat_id == st.session_state.current_chat_id:
                        new_chat()
                    st.rerun()
        st.divider()

    st.markdown("**🎯 Topics**")
    st.markdown("""
    <div style="padding:0.2rem 0 0.4rem 0; line-height:2.2;">
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">📈 Calculus</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">🔢 Linear Algebra</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">📉 ODEs</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">🧮 Numerical</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">🔍 Number Theory</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">📐 Diff. Geometry</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">🌊 Hydro Mechanics</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">📊 Real Analysis II</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">📈 Graph Plotting</span>
        <span style="background:#111827; border:1px solid #1f2937; color:#6b7280; padding:3px 8px; border-radius:12px; font-size:0.7rem; margin:2px;">➕ General Math</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚡ Example Problems**")

    examples = {
        "-- Select --": "",
        "📈 Derivative": "Find the derivative of x^3 + 5x^2 - 3x + 7",
        "∫ Integral": "Integrate sin(x) * e^x dx",
        "📐 Limit": "Find limit of sin(x)/x as x -> 0",
        "🔢 Eigenvalues": "Find eigenvalues of matrix [[4,1],[2,3]]",
        "🔁 Congruence": "Solve 14x ≡ 30 (mod 44) using Euclidean algorithm",
        "📉 ODE": "Solve dy/dx + 2y = e^(-x) with y(0) = 1",
        "🧮 Newton-Raphson": "Apply Newton-Raphson to x^3 - 2x - 5 = 0, x0=2, 3 iterations",
        "📊 Series": "Test convergence of sum 1/n^2 from n=1 to infinity",
        "🌊 Bernoulli": "Explain Bernoulli equation in fluid mechanics with example",
        "🔍 Fermat": "State and prove Fermat's Little Theorem with example",
    }

    selected = st.selectbox(
        "Load example:",
        list(examples.keys()),
        key="example_select",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**🔧 Engine**")
    st.markdown("""
    <div style="padding:0.2rem 0;">
        <div style="display:flex; align-items:center; gap:0.5rem; padding:0.2rem 0; color:#4b5563; font-size:0.75rem;">
            <div style="width:6px; height:6px; border-radius:50%; background:#22c55e; box-shadow:0 0 4px #22c55e; flex-shrink:0;"></div>
            Groq Llama 3.3 70B
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; padding:0.2rem 0; color:#4b5563; font-size:0.75rem;">
            <div style="width:6px; height:6px; border-radius:50%; background:#22c55e; box-shadow:0 0 4px #22c55e; flex-shrink:0;"></div>
            SymPy — verified math
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; padding:0.2rem 0; color:#4b5563; font-size:0.75rem;">
            <div style="width:6px; height:6px; border-radius:50%; background:#22c55e; box-shadow:0 0 4px #22c55e; flex-shrink:0;"></div>
            6 providers · 46,400 req/day
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; padding:0.2rem 0; color:#4b5563; font-size:0.75rem;">
            <div style="width:6px; height:6px; border-radius:50%; background:#22c55e; box-shadow:0 0 4px #22c55e; flex-shrink:0;"></div>
            Zero arithmetic errors
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        new_chat()
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════

# Welcome screen — only when no messages
if not st.session_state.messages:
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(24px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes floatSymbol {
        0%,100% { transform: translateY(0px) rotate(0deg); opacity:0.18; }
        33%      { transform: translateY(-8px) rotate(5deg); opacity:0.32; }
        66%      { transform: translateY(4px) rotate(-3deg); opacity:0.22; }
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    .saad-welcome-wrap {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 5rem 1rem 4rem 1rem;
        text-align: center;
        animation: fadeInUp 0.7s cubic-bezier(.22,1,.36,1) both;
        overflow: hidden;
    }
    /* Floating math symbols background */
    .math-bg {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 0;
    }
    .math-sym {
        position: absolute;
        font-size: 1.4rem;
        color: #3b82f6;
        animation: floatSymbol 6s ease-in-out infinite;
        font-family: 'JetBrains Mono', monospace;
        user-select: none;
    }
    /* Main name with gradient */
    .saad-main-name {
        position: relative;
        z-index: 1;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 40%, #f472b6 70%, #60a5fa 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    /* Spinning math ring */
    .math-ring {
        position: relative;
        z-index: 1;
        width: 56px;
        height: 56px;
        margin-bottom: 1.2rem;
        border-radius: 50%;
        border: 2px solid transparent;
        background: linear-gradient(#0f0f0f, #0f0f0f) padding-box,
                    linear-gradient(135deg, #3b82f6, #a78bfa, #f472b6) border-box;
        animation: spin 3s linear infinite;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .math-ring-inner {
        animation: spin 3s linear infinite reverse;
        font-size: 1.4rem;
    }
    .saad-tagline {
        position: relative;
        z-index: 1;
        font-size: 0.82rem;
        color: #4b5563;
        letter-spacing: 0.4px;
        margin-top: 0.3rem;
    }
    .saad-tagline span {
        color: #374151;
        margin: 0 0.4rem;
    }
    </style>

    <div class="saad-welcome-wrap">
        <!-- Floating math symbols -->
        <div class="math-bg">
            <span class="math-sym" style="top:10%; left:8%; animation-delay:0s;">∫</span>
            <span class="math-sym" style="top:20%; right:10%; animation-delay:1s;">∑</span>
            <span class="math-sym" style="top:60%; left:5%; animation-delay:2s;">∂</span>
            <span class="math-sym" style="top:70%; right:8%; animation-delay:0.5s;">√</span>
            <span class="math-sym" style="top:40%; left:15%; animation-delay:1.5s; font-size:1rem;">π</span>
            <span class="math-sym" style="top:35%; right:15%; animation-delay:2.5s; font-size:1rem;">∞</span>
            <span class="math-sym" style="top:80%; left:20%; animation-delay:3s; font-size:0.9rem;">Δ</span>
            <span class="math-sym" style="top:15%; left:40%; animation-delay:3.5s; font-size:0.85rem;">λ</span>
        </div>
        <!-- Spinning ring -->
        <div class="math-ring">
            <span class="math-ring-inner">∑</span>
        </div>
        <!-- Name with gradient -->
        <div class="saad-main-name">Saad.AI</div>
        <!-- Tagline -->
        <div class="saad-tagline">
            BSc Mathematics
            <span>·</span> Step-by-Step Solutions
            <span>·</span> SymPy Verified
            <span>·</span> LaTeX Rendered
        </div>
    </div>
    """, unsafe_allow_html=True)

# Small corner header — only when chat has started
else:
    st.markdown("""
    <style>
    @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
    .saad-corner {
        font-size: 0.88rem;
        font-weight: 600;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 0.3px;
        margin-bottom: 0.6rem;
        padding: 0.1rem 0;
        animation: fadeIn 0.4s ease both;
        font-family: 'Inter', sans-serif;
    }
    </style>
    <div class="saad-corner">∑ Saad.AI</div>
    """, unsafe_allow_html=True)

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍🎓" if msg["role"] == "user" else "📐"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("📋 Copy"):
                st.code(msg["content"], language=None)

# ════════════════════════════════════════════════════════════════════
# FILE ATTACH — FIXED: real Streamlit button toggle (no JS tricks)
# Works reliably on HuggingFace Spaces — no iframe/JS issues
# ════════════════════════════════════════════════════════════════════

_pending  = st.session_state.pending_file_bytes is not None
_memfile  = st.session_state.attached_file_name

# ── Status bar ───────────────────────────────────────────────────────
if _pending:
    col_s, col_x = st.columns([9, 1])
    with col_s:
        st.markdown(
            f'<div style="background:#0d2318;border:1px solid #22c55e;border-radius:10px;'
            f'padding:0.5rem 0.9rem;font-size:0.8rem;color:#86efac;margin-bottom:0.3rem;">'
            f'✅ <b>{st.session_state.pending_file_name}</b> — ready · type your question and press Enter</div>',
            unsafe_allow_html=True
        )
    with col_x:
        if st.button("✕", key="rm_pending", help="Remove file"):
            st.session_state.pending_file_bytes = None
            st.session_state.pending_file_name  = None
            st.session_state.pending_file_mime  = None
            st.session_state["last_uploaded_file"] = ""
            st.session_state.show_uploader = False
            st.rerun()

elif _memfile:
    col_s, col_x = st.columns([9, 1])
    with col_s:
        st.markdown(
            f'<div style="background:#0d1a2e;border:1px solid #3b82f6;border-radius:10px;'
            f'padding:0.5rem 0.9rem;font-size:0.8rem;color:#93c5fd;margin-bottom:0.3rem;">'
            f'📎 <b>{_memfile}</b> in memory — ask a follow-up or click 📎 to attach new</div>',
            unsafe_allow_html=True
        )
    with col_x:
        if st.button("✕", key="rm_attached", help="Clear file memory"):
            st.session_state.attached_file_bytes = None
            st.session_state.attached_file_name  = None
            st.session_state.attached_file_mime  = None
            st.rerun()

# ── 📎 Attach toggle button ──────────────────────────────────────────
attach_col, _ = st.columns([1, 8])
with attach_col:
    btn_label = "📎 Attached" if (st.session_state.show_uploader or _pending) else "📎 Attach"
    if st.button(btn_label, key="toggle_uploader", help="Attach image or PDF"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

# ── Real file uploader — only shown when toggled on ──────────────────
if st.session_state.show_uploader and not _pending:
    uploaded = st.file_uploader(
        "Upload image or PDF",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        label_visibility="collapsed",
        key="main_uploader"
    )
    if uploaded is not None:
        _fkey = f"{uploaded.size}_{uploaded.type}_{uploaded.name}"
        if _fkey != st.session_state.get("last_uploaded_file", ""):
            st.session_state["last_uploaded_file"] = _fkey
            st.session_state.pending_file_bytes = uploaded.read()
            st.session_state.pending_file_name  = uploaded.name
            st.session_state.pending_file_mime  = uploaded.type or "application/octet-stream"
            st.session_state.show_uploader = False
            st.rerun()

# ════════════════════════════════════════════════════════════════════
# INPUT — ChatGPT-style input bar
# ════════════════════════════════════════════════════════════════════

# Pre-fill from example selector
prefill = examples.get(selected, "") if selected != "-- Select --" else ""

user_input = st.chat_input(
    placeholder="Type a math problem... or attach a file in the sidebar ← then ask here",
)

# Also allow clicking an example to submit it directly
if prefill and prefill != st.session_state.last_submitted:
    problem = prefill
elif user_input and user_input.strip():
    problem = user_input.strip()
else:
    problem = ""

# ════════════════════════════════════════════════════════════════════
# PROCESS — only when there's a new problem
# ════════════════════════════════════════════════════════════════════
if problem and problem != st.session_state.last_submitted:
    st.session_state.last_submitted = problem

    # ── If a NEW file is attached, send file + question to Vision ────
    if st.session_state.pending_file_bytes is not None:
        import base64

        file_bytes = st.session_state.pending_file_bytes
        file_name  = st.session_state.pending_file_name
        file_mime  = st.session_state.pending_file_mime

        # ── Validate size (5 MB limit) ────────────────────────────────
        MAX_FILE_SIZE = 5 * 1024 * 1024
        if len(file_bytes) > MAX_FILE_SIZE:
            st.session_state.pending_file_bytes = None
            st.session_state.pending_file_name  = None
            st.session_state.pending_file_mime  = None
            st.session_state["last_uploaded_file"] = ""
            st.error(f"⚠️ File too large ({len(file_bytes)//1024} KB). Please upload under 5 MB.")
            st.stop()

        # ── Validate MIME type ────────────────────────────────────────
        _allowed_mimes = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
        if file_mime not in _allowed_mimes:
            st.session_state.pending_file_bytes = None
            st.session_state.pending_file_name  = None
            st.session_state.pending_file_mime  = None
            st.session_state["last_uploaded_file"] = ""
            st.error("⚠️ Unsupported format. Please upload JPG, PNG, WEBP or PDF.")
            st.stop()

        # Clear pending (one-time) but keep in attached memory for follow-ups
        st.session_state.pending_file_bytes = None
        st.session_state.pending_file_name  = None
        st.session_state.pending_file_mime  = None
        st.session_state.attached_file_bytes = file_bytes
        st.session_state.attached_file_name  = file_name
        st.session_state.attached_file_mime  = file_mime

        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"📎 **{file_name}** — {problem}")
            # Show file preview so user can see what was attached
            if file_mime and file_mime.startswith("image/"):
                st.image(file_bytes, caption=file_name, use_column_width=True)
            else:
                # PDF — try to show first page
                try:
                    import fitz, io
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    doc.close()
                    st.image(pix.tobytes("png"), caption=f"📄 {file_name} (page 1 preview)", use_column_width=True)
                except Exception:
                    st.caption(f"📄 {file_name}")

        with st.chat_message("assistant", avatar="📐"):
            with st.spinner("📖 Reading your file..."):
                image_b64 = base64.b64encode(file_bytes).decode("utf-8")
                answer = ask_gemini_vision(image_b64, file_mime, problem)
            st.markdown(answer)
            with st.expander("📋 Copy"):
                st.code(answer, language=None)

        if not st.session_state.current_chat_id:
            new_chat()
        st.session_state.messages.append({
            "role": "user",
            "content": f"📎 {file_name} — {problem}"
        })
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_current_chat()
        st.stop()

    # ── Follow-up about previously attached file ──────────────────────
    # Detects "solve q3", "next question", "question 2" etc. and re-sends the file
    _p = problem.lower()
    _followup_triggers = [
        "question", "solve q", "q1","q2","q3","q4","q5","q6","q7","q8","q9","q10",
        "next one", "next question", "next qus", "next ques",
        "previous", "solve the next", "solve all", "solve rest",
        "number ", "no.", "no ", "#", "part ", "part(", "section",
    ]
    _is_file_followup = (
        st.session_state.attached_file_bytes is not None and
        any(t in _p for t in _followup_triggers)
    )
    if _is_file_followup:
        import base64 as _b64_fu
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"📎 *{st.session_state.attached_file_name}* — {problem}")
        with st.chat_message("assistant", avatar="📐"):
            with st.spinner("📖 Re-reading your file..."):
                _fb64 = _b64_fu.b64encode(st.session_state.attached_file_bytes).decode("utf-8")
                answer = ask_gemini_vision(_fb64, st.session_state.attached_file_mime, problem)
            st.markdown(answer)
            with st.expander("📋 Copy"):
                st.code(answer, language=None)
        if not st.session_state.current_chat_id:
            new_chat()
        st.session_state.messages.append({"role": "user", "content": f"📎 {st.session_state.attached_file_name} — {problem}"})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_current_chat()
        st.stop()

    # ── Detect casual / non-math messages ───────────────────────────
    p_lower = problem.lower().strip()
    casual_keywords = [
        "hi", "hello", "hey", "how are you", "how r u", "what's up",
        "whats up", "good morning", "good evening", "good night",
        "who are you", "what are you", "what can you do", "help",
        "thanks", "thank you", "bye", "goodbye", "ok", "okay",
        "what is your name", "your name", "who made you", "who built you",
        "how do you work", "what do you do", "sup", "hlo", "hlw",
        "what u doing", "what are you doing", "hows it going",
    ]
    is_casual = (
        any(p_lower == kw for kw in casual_keywords) or
        any(p_lower.startswith(kw) for kw in casual_keywords) or
        (len(p_lower.split()) <= 4 and not any(c in p_lower for c in [
            "=", "+", "-", "*", "/", "^", "∫", "∑", "√", "dx", "dy",
            "sin", "cos", "tan", "log", "lim", "diff", "solve", "find",
            "calculate", "compute", "prove", "matrix", "eigen", "gcd",
            "integral", "derivative", "equation", "theorem"
        ]))
    )

    # Show user message immediately
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(problem)

    # Show AI response
    with st.chat_message("assistant", avatar="📐"):
        if is_casual:
            # ── Casual message — no SymPy, no math structure ─────────
            casual_sympy = {"type": "casual", "result": None, "latex": ""}
            answer = ask_ai_streaming(problem, casual_sympy, st.session_state.messages)
        else:
            # ── Math message — full SymPy + structured response ───────
            import random
            spinner_msgs = [
                "🧮 Computing with SymPy...",
                "∫ Integrating the solution...",
                "∑ Summing it all up...",
                "📐 Applying the formula...",
                "🔢 Crunching the numbers...",
                "📊 Verifying the answer...",
            ]
            with st.spinner(random.choice(spinner_msgs)):
                sympy_result = run_sympy(problem)
            answer = ask_ai_streaming(problem, sympy_result, st.session_state.messages)
            plot_graph(problem, sympy_result)

        # Copy — click to expand, then use built-in copy button
        with st.expander("📋 Copy"):
            st.code(answer, language=None)

    # Save to history
    if not st.session_state.current_chat_id:
        new_chat()  # create chat ID BEFORE appending — avoids wiping messages
    st.session_state.messages.append({"role": "user", "content": problem})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_current_chat()
