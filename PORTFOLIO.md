# Saad.AI — Portfolio Project Brief

## One-line description

Saad.AI is a Streamlit mathematics assistant that combines a deterministic SymPy engine with AI-generated explanations, graphing, and image/PDF problem input for undergraduate mathematics students.

## The problem

Mathematics learners often receive either a calculator-style answer with no explanation or a conversational explanation that may not be mathematically reliable. Saad.AI addresses that gap by attempting exact symbolic or numerical computation first, then using an AI provider to explain the result in a structured, student-friendly way.

## What the project demonstrates

| Capability | Evidence in the project |
|---|---|
| Applied mathematics | Derivatives, integrals, limits, equations, numerical methods, ODEs, number theory, matrices, graphing, and selected applied mathematics topics. |
| AI integration | Provider rotation, structured prompts, image/PDF interpretation, and explanation generation. |
| Reliability engineering | Bounded provider retries, timeout handling, graceful fallback, and explicit verification labels. |
| Software architecture | Deterministic engine and AI/file services extracted from the Streamlit presentation layer. |
| Testing | Regression tests for configuration, provider retry behavior, parser edge cases, matrices, congruences, modulo, calculus, and numerical methods. |
| Product design | Responsive dark UI, subject examples, chat history, upload workflow, verification states, and downloadable Markdown solutions. |
| Deployment | Simple environment-based configuration and CI checks for Python 3.10 and 3.11. |

## Architecture

The application uses a deliberately understandable architecture for a portfolio project:

```text
User prompt / image / PDF
          |
          v
Streamlit interface (app.py)
          |
          +--> SymPy deterministic engine (src/engine/sympy_engine.py)
          |          |
          |          +--> verified computation when a supported adapter matches
          |
          +--> AI and file services (src/services/ai.py)
                     |
                     +--> explanation, vision extraction, provider fallback

Configuration: src/config.py
Tests: tests/
CI: .github/workflows/ci.yml
```

Chat history is session-local by design. This keeps the demo easy to run and avoids adding authentication or database infrastructure that is not necessary to demonstrate the project’s core engineering value.

## Recommended demo flow

Start with a calculus example such as “Find the derivative of `x^3 + 5x^2 - 3x + 7`.” Show the visible SymPy verification label and the generated explanation. Then try the matrix example “Find eigenvalues of matrix `[[4,1],[2,3]]`” to demonstrate deterministic linear algebra. Follow with a numerical method such as bisection or Newton–Raphson, and finish with an image/PDF upload if provider secrets are configured.

For a portfolio interview, emphasize the distinction between **computation** and **explanation**. SymPy provides the exact calculation for supported prompts, while the AI layer turns that computation into an understandable learning response. The interface tells the user when deterministic verification is available rather than presenting every answer with the same confidence.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The application can run with only deterministic SymPy functionality. AI explanations and vision features require at least one configured provider key.

## Portfolio talking points

The strongest story for this project is not that it contains a large number of mathematical keywords. It is that the project evolved from a monolithic prototype into a more maintainable application while preserving the valuable domain logic. The refactor introduced module boundaries, regression tests, bounded external calls, clear verification semantics, and a simpler deployment model.

A concise interview summary is:

> “I built a mathematics assistant that separates exact computation from AI explanation. I refactored the original monolith into a testable SymPy engine and service layer, added deterministic coverage for common mathematics operations, hardened provider failures, and redesigned the UI to communicate when an answer is verified.”
