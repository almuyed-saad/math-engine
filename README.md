# Saad.AI — B.Sc. Mathematics Engine

Saad.AI is an academic mathematics assistant for university students. It combines a deterministic **SymPy computation engine** with configurable AI providers for explanations, proofs, theory questions, graph descriptions, and image/PDF-based problem solving. See [PORTFOLIO.md](PORTFOLIO.md) for the project story and demo flow.

> **Important:** SymPy verification applies only when a request matches one of the implemented deterministic adapters. General proofs, theory questions, unsupported matrix formats, and unsupported subjects are treated as AI-generated unless a deterministic adapter returns a verified result.

## Current capabilities

| Area | Examples | Verification mode |
|---|---|---|
| Calculus | Derivatives, integrals, limits | SymPy when parsed successfully |
| Equations | Polynomial equations and roots | SymPy when parsed successfully |
| Differential equations | Selected first- and second-order ODE forms | SymPy for supported forms |
| Numerical methods | Newton–Raphson, bisection, secant, Simpson, trapezoidal, Euler, RK4 | Deterministic numeric adapter |
| Number theory | GCD, LCM, factorization, totient, congruences, CRT, selected theorems, modulo | SymPy / deterministic adapter |
| Linear algebra | Determinants, eigenvalues/eigenvectors, inverses, ranks, transposes | Deterministic SymPy adapter |
| Real analysis | Selected sequence, series, Taylor, and integral computations | SymPy for supported computations; AI for theory/proofs |
| Differential geometry | Curvature, arc length, Frenet–Serret, fundamental forms | SymPy for supported parametric forms |
| Hydro mechanics | Continuity, Bernoulli, Reynolds, flow rate, pressure, Torricelli | Deterministic formula adapter for supported prompts |
| Graphing | Explicit requests to plot or graph a function | Matplotlib rendering |
| Attachments | JPG, PNG, WEBP, and PDF questions | Vision provider; deterministic verification when extractable |

The engine is intentionally not presented as a universal proof checker. For questions that cannot be deterministically parsed, the application sends the prompt to the configured AI provider and labels the result as AI-generated where appropriate.

## Architecture

The current stabilization refactor keeps Streamlit as the user interface while separating the main responsibilities:

```text
math-engine/
├── app.py                         # Streamlit UI, session flow, and graph rendering
├── src/
│   ├── engine/
│   │   └── sympy_engine.py        # Deterministic symbolic and numeric adapters
│   └── services/
│       └── ai.py                  # Provider rotation, vision, uploads, verification
├── tests/
│   └── test_engine.py             # Deterministic engine regression tests
├── requirements.txt               # Runtime dependencies actually used by the app
└── README.md
```

The next recommended phase is to split the remaining UI, persistence, and plotting concerns into their own modules and add provider mocks and upload fixtures.

## Run locally

```bash
git clone https://github.com/almuyed-saad/math-engine.git
cd math-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in a browser.

Run the deterministic regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Configuration

AI providers are optional for deterministic SymPy requests but required for explanations and unsupported subjects. Configure provider credentials through environment variables or Hugging Face Space secrets:

```text
GROQ_API_KEY_1
GROQ_API_KEY_2
GROQ_API_KEY_3
GEMINI_API_KEY_1
GEMINI_API_KEY_2
GEMINI_API_KEY_3
GEMINI_API_KEY_4
OPENROUTER_API_KEY
```

Chat history is intentionally **session-local** in the portfolio edition. This keeps the application easy to understand and deploy while still allowing users to create, switch, and delete conversations during a demo session. A database is not required.

## Example prompts

```text
Find the derivative of x^3 + 5x^2 - 3x + 7
Integrate sin(x) * e^x dx
Find limit of sin(x)/x as x -> 0
Apply Newton-Raphson to x^3 - 2x - 5 = 0, x0=2, 3 iterations
Apply bisection of x^3 - x on [0, 2], 4 iterations
Find gcd of 84 and 30
Solve 14x ≡ 30 (mod 44) using Euclidean algorithm
Plot y = x^2 - 4 from -3 to 3
```

## Deployment notes

The active source uses hosted API providers rather than loading a local Hugging Face model at runtime. The dependency list therefore excludes the previously declared `transformers` and `torch` packages, which were not used by the current application and added unnecessary deployment weight.

Runtime configuration is centralized in `src/config.py`. Copy `.env.example` to `.env` for local development, or add the same variables as Hugging Face Space secrets. Provider timeouts, upload size, and PDF page limits are validated and clamped at startup so malformed deployment values cannot create unbounded resource usage.

For a portfolio deployment, add the provider secrets to Hugging Face Spaces or another Streamlit host, then launch the app with `streamlit run app.py`. No database or authentication setup is required. Every push and pull request runs the deterministic test suite and Python compilation checks through [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Credits

The project was created by Saad for B.Sc. Mathematics students at Shahjalal University of Science and Technology. It uses [Streamlit](https://streamlit.io), [SymPy](https://www.sympy.org), [Matplotlib](https://matplotlib.org), and hosted AI provider APIs.
