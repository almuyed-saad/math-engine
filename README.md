# Saad.AI — B.Sc. Mathematics Engine

[![CI](https://github.com/almuyed-saad/math-engine/actions/workflows/ci.yml/badge.svg?branch=refactor%2Fstabilize-saadai)](https://github.com/almuyed-saad/math-engine/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?logo=huggingface)](https://saad-sust-saad-ai.hf.space)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Saad.AI** is a portfolio-focused academic mathematics assistant for B.Sc. mathematics students. It combines a deterministic [SymPy](https://www.sympy.org/) computation engine with hosted AI providers that explain results in a clear, step-by-step format.

> **Live application:** [saad-sust-saad-ai.hf.space](https://saad-sust-saad-ai.hf.space)
>
> **Repository branch:** [`refactor/stabilize-saadai`](https://github.com/almuyed-saad/math-engine/tree/refactor/stabilize-saadai)

The project is designed around a simple principle: **use deterministic mathematics whenever a problem can be parsed reliably, and use AI for explanation, theory, proofs, and unsupported requests.** This makes the application more transparent than an AI-only chatbot while keeping the experience conversational and accessible.

## Why this project is technically interesting

Saad.AI demonstrates how a mathematical assistant can combine structured computation with natural-language interaction without treating every answer as equally reliable. For supported problem types, the application produces a result through SymPy or a deterministic numerical adapter and labels the response as verified. For theory questions, proofs, broad explanations, and requests outside the deterministic adapters, the application delegates explanation to a configured provider and clearly identifies when deterministic verification is unavailable.

The portfolio edition intentionally avoids unnecessary enterprise infrastructure. Chat history is session-local, there is no database or authentication layer, and deployment requires only a Streamlit-compatible host plus provider secrets. This keeps the project easy to inspect, run, and demonstrate while preserving a clean separation between the user interface, mathematics engine, configuration, and AI services.

## Core capabilities

| Area | Supported examples | Result handling |
|---|---|---|
| **Calculus** | Derivatives, integrals, limits, substitutions, and selected differential-calculus prompts | SymPy verification when the expression is parsed successfully |
| **Equations** | Polynomial equations, roots, and selected symbolic equations | Deterministic SymPy result for supported forms |
| **Differential equations** | Selected first- and second-order ordinary differential equations | SymPy for supported equation structures; AI explanation for broader theory |
| **Numerical methods** | Newton–Raphson, bisection, secant, Simpson, trapezoidal, Euler, and RK4 methods | Deterministic numerical adapters with input validation |
| **Number theory** | GCD, LCM, factorization, Euler’s totient, congruences, CRT, modulo arithmetic, and selected theorems | SymPy or deterministic arithmetic adapters |
| **Linear algebra** | Determinants, eigenvalues, eigenvectors, inverses, ranks, and transposes | Deterministic matrix adapters |
| **Real analysis** | Selected sequences, series, Taylor expansions, and integral computations | SymPy for supported computations; AI for theory and proof-oriented questions |
| **Differential geometry** | Curvature, arc length, Frenet–Serret quantities, and fundamental forms | Deterministic SymPy calculations for supported parametric forms |
| **Hydro mechanics** | Continuity, Bernoulli, Reynolds number, flow rate, pressure, and Torricelli-style problems | Deterministic formula adapters for supported prompts |
| **Graphing** | Requests to plot explicit functions or compare curves | Matplotlib rendering inside the Streamlit interface |
| **Attachments** | JPG, PNG, WEBP, and PDF-based questions | Vision provider workflow with file validation and previews |

The engine is intentionally not presented as a universal theorem prover or proof verifier. A deterministic result is shown only when the request matches an implemented adapter and the computation succeeds.

## User experience

The application uses a chat-first interface inspired by modern conversational assistants while retaining mathematics-specific feedback. The empty state presents the product identity and the primary input rather than forcing users through a collection of demo cards. The sidebar provides topic guidance, optional example prompts, session-local chat management, attachment controls, and the current provider-availability status.

Every response can include:

- A natural-language explanation from the configured AI provider.
- A deterministic SymPy or numerical result when a supported adapter succeeds.
- A visible verification caption distinguishing AI explanation from deterministic computation.
- A collapsible solution-tools section for downloading or copying the Markdown response.
- A graph when the prompt requests a supported visualization.

## Architecture

The repository separates the main responsibilities of the original monolithic application into focused modules:

```text
math-engine/
├── app.py                         # Streamlit UI, session flow, chat rendering, and graphing
├── src/
│   ├── config.py                  # Typed settings and environment-variable loading
│   ├── engine/
│   │   └── sympy_engine.py        # Deterministic symbolic and numerical adapters
│   └── services/
│       └── ai.py                  # Provider rotation, prompts, verification, and uploads
├── tests/
│   ├── test_ai_service.py         # Provider retry and answer-normalization tests
│   ├── test_config.py             # Settings and configuration safety tests
│   └── test_engine.py             # Mathematics-engine regression tests
├── .github/
│   └── workflows/
│       └── ci.yml                 # Python compilation and regression checks
├── .env.example                   # Local configuration template
├── requirements.txt               # Runtime dependencies
├── PORTFOLIO.md                   # Project story and portfolio context
└── README.md
```

### Request flow

```text
User prompt
    │
    ▼
Streamlit chat interface
    │
    ├── Attachment validation and preview, when applicable
    │
    ├── SymPy / numerical adapter
    │       │
    │       ├── Supported and successful → verified result
    │       └── Unsupported or unavailable → explanation path
    │
    └── AI provider rotation
            │
            ├── Groq
            ├── Gemini fallback
            └── OpenRouter fallback
                    │
                    ▼
            Explanation + verification label
```

The deterministic engine is the source of mathematical verification for supported requests. The AI service is responsible for explanation, provider rotation, response cleanup, attachment workflows, and safe fallback behavior. Provider failures are not exposed as raw technical tracebacks to end users.

## AI provider strategy

The application supports multiple hosted providers so a temporary provider timeout, model retirement, or quota issue does not unnecessarily prevent the rest of the application from working.

| Priority | Provider path | Purpose |
|---|---|---|
| 1 | Groq | Primary text explanation provider; the deployed configuration uses `openai/gpt-oss-20b` |
| 2 | Gemini | Fallback text and vision workflow where configured |
| 3 | OpenRouter | Last-resort text provider where configured |
| 4 | Deterministic fallback | Preserves a verified result when an AI explanation cannot be obtained |

AI providers are not required for supported deterministic calculations, but they are required for conversational explanations, broad theory questions, proofs, and some attachment workflows.

## Local development

### Requirements

The project supports Python 3.10 and Python 3.11. A virtual environment is recommended for local development.

```bash
git clone https://github.com/almuyed-saad/math-engine.git
cd math-engine
git checkout refactor/stabilize-saadai

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the application

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in a browser.

### Run the tests

The repository uses Python’s built-in `unittest` discovery so the regression suite can run without an additional test framework:

```bash
python -m unittest discover -s tests -v
```

To compile the main modules directly:

```bash
python -m py_compile \
  app.py \
  src/config.py \
  src/engine/sympy_engine.py \
  src/services/ai.py
```

## Configuration

Provider credentials should be supplied through environment variables during local development or through private secrets in Hugging Face Spaces. **Never commit API keys to the repository or place them in public documentation.**

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

Optional runtime controls are also supported:

```text
PROVIDER_TIMEOUT_SECONDS
MAX_UPLOAD_BYTES
MAX_PDF_PAGES
```

The settings loader normalizes and clamps these values at startup. This prevents malformed deployment settings from creating unbounded timeouts, upload sizes, or PDF-processing limits.

For local development, copy the template and fill in only the secrets required for the provider path you want to test:

```bash
cp .env.example .env
```

The application intentionally keeps chat history in Streamlit session state. A database is not required for the current portfolio scope, and no user authentication is implemented.

## Deployment on Hugging Face Spaces

The production demonstration is deployed as a Streamlit Space at [saad-sust/SAAD_AI](https://huggingface.co/spaces/saad-sust/SAAD_AI). The Space contains a flat deployment bundle because Hugging Face launches the root `app.py` directly:

```text
app.py
ai.py
config.py
sympy_engine.py
requirements.txt
README.md
```

To reproduce the deployment on another Streamlit-compatible host:

1. Upload the application files and install the dependencies from `requirements.txt`.
2. Store provider credentials as private environment variables or platform secrets.
3. Launch the root application with `streamlit run app.py`.
4. Confirm that the deterministic test prompt returns both a correct result and the expected verification caption.

The active runtime uses hosted API providers and does not load a local Hugging Face transformer model. The dependency list therefore remains focused on Streamlit, requests, SymPy, NumPy, Matplotlib, Pillow, and PyMuPDF rather than including unused `transformers` or `torch` packages.

## Continuous integration

GitHub Actions runs on pushes to `main`, the stabilization branch pattern, and pull requests. The workflow tests Python 3.10 and 3.11, installs the runtime dependencies, compiles the core modules, and runs the complete regression suite.

Workflow definition: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## Limitations and design decisions

Saad.AI is a polished portfolio project rather than a multi-user enterprise platform. The following constraints are intentional:

- Chat history is session-local and is lost when the Streamlit session ends.
- There is no database, authentication layer, or user account system.
- Deterministic verification applies only to supported adapters and successfully parsed inputs.
- AI-generated proofs and explanations should be reviewed by a mathematics instructor or student rather than treated as formal certification.
- Provider availability depends on valid secrets, provider quotas, model availability, and network access.
- Attachment solving depends on the configured vision-capable provider and the platform’s upload limits.

These boundaries keep the deployment simple, make the architecture easy to explain, and leave clear opportunities for future extensions without implying guarantees the current system does not provide.

## Possible future extensions

Future work could add broader parser coverage, more formal proof tooling, richer graphing controls, stronger attachment fixtures, provider mocks, and optional persistent accounts. Those features are deliberately outside the current portfolio scope; the present version prioritizes correctness of supported calculations, transparent verification labels, maintainable structure, and a straightforward permanent deployment.

## Credits

Saad.AI was created by **Saad** for B.Sc. mathematics learning and portfolio demonstration. The project builds on the following open-source and hosted technologies:

- [Streamlit](https://streamlit.io/) for the interactive application interface.
- [SymPy](https://www.sympy.org/) for symbolic mathematics and deterministic computation.
- [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) for numerical work and graphing.
- [Pillow](https://python-pillow.org/) and [PyMuPDF](https://pymupdf.readthedocs.io/) for attachment handling.
- Hosted AI provider APIs for explanations and vision-assisted workflows.

## License

No license file is currently included in the repository. If this project is intended for public reuse, add an explicit license before accepting external contributions or redistributing the code.
