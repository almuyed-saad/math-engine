# 🧮 SAAD AI — B.Sc. Mathematics Engine
### AI-Powered Academic Math Solver | SUST Department of Mathematics

---

## 🔗 [▶ LIVE DEMO — Click Here to Use](https://huggingface.co/spaces/saad-sust/SAAD_AI)
> **https://huggingface.co/spaces/saad-sust/SAAD_AI**

---

## What is SAAD AI?

An AI-powered mathematics assistant designed for university-level problem solving. Built by a B.Sc. Mathematics student at SUST, Bangladesh — from scratch, using real machine learning tools.

It does not just give answers. It explains every step, cites the theorems used, and formats solutions in clean academic style.

---

## What It Can Solve

| Topic | Examples |
|---|---|
| Algebra | Quadratic, cubic, polynomial equations |
| Calculus | Derivatives, integrals, chain rule, limits |
| Real Analysis | Epsilon-delta proofs, convergence tests |
| Group Theory | Normal subgroups, cosets, index proofs |
| Number Theory | Divisibility, Fermat's theorem, congruences |
| Complex Analysis | Cauchy-Riemann equations, Residue theorem |
| Word Problems | Applied math with full reasoning |

---

## Tech Stack

| Component | Tool |
|---|---|
| Language Model | SmolLM2-1.7B-Instruct |
| UI | Streamlit |
| ML Framework | HuggingFace Transformers |
| Deployment | HuggingFace Spaces (Free) |
| Fine-tuning | Unsloth + LoRA on Kaggle GPU |
| Language | Python 3.10+ |

---

## Run Locally

```bash
git clone https://github.com/almuyed-saad/math-engine.git
cd SAAD-AI-Math
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
SAAD-AI-Math/
├── app.py
├── requirements.txt
└── README.md
```

---

## Example Problems

```
solve x^2 + 5x + 6 = 0
derivative of (x^2 + 1)^3
integrate x^2 + 3x + 2 dx
prove that sqrt(2) is irrational
prove n^3 - n is divisible by 6
solve 3x = 7 (mod 11)
prove a subgroup of index 2 is normal
epsilon-delta proof for limit of 2x as x approaches 3
```

---

## How It Works

1. You enter a math problem in the chat
2. A strict academic system prompt structures the request
3. SmolLM2-1.7B generates a step-by-step solution
4. Response is rendered with LaTeX in the browser

Every solution follows this structure:

- **Definitions and Prerequisites**
- **Step-by-Step Proof or Solution**
- **Conclusion**

---

## About the Developer

**Saad**

B.Sc. Mathematics — Shahjalal University of Science and Technology (SUST), Sylhet, Bangladesh

Built this project independently as a first AI project — from fine-tuning LLMs on Kaggle to deploying a full web application with a public URL.

---

## Acknowledgements

- [HuggingFace](https://huggingface.co) — Model hosting and free deployment
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning
- [Streamlit](https://streamlit.io) — UI framework
- SUST Department of Mathematics

---

⭐ Star this repo if it helped you!
