# ═══════════════════════════════════════════════════════════════════
# Saad.AI — BSc Mathematics Solver
# Engine: Groq (free) + SymPy (verified computation)
# ═══════════════════════════════════════════════════════════════════

# ── Runtime imports ───────────────────────────────────────────────────
import streamlit as st
import os
import uuid as _uuid
from html import escape as _escape_html
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

from src.config import settings
from src.engine.sympy_engine import run_sympy
from src.services.ai import ask_ai, ask_gemini_vision, handle_uploaded_file

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saad.AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — ChatGPT-style, compact font, clean dark theme ─────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
    font-size: 14px !important;
    background-color: #0f0f0f !important;
    color: #ececec !important;
}
.block-container {
    max-width: 980px !important;
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
    font-size: 0.72rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
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
    font-size: 0.8rem;
    color: #cbd5e1;
    transition: all 0.2s;
}
.subject-card:hover {
    border-color: #3b82f6;
    color: #60a5fa;
    background: #0f172a;
}
    @media (max-width: 640px) {
        .subject-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .saad-main-name { font-size: clamp(2.2rem, 12vw, 3rem); }
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
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
    color: #94a3b8;
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
    color: #94a3b8;
    font-size: 0.8rem;
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
    color: #94a3b8;
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

# Chat history is intentionally session-local for the portfolio edition.
# This keeps the demo simple, avoids database configuration, and makes the
# project straightforward to deploy on Streamlit or Hugging Face Spaces.

def new_chat():
    """Start a fresh chat session."""
    chat_id = f"chat_{_uuid.uuid4().hex}"
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.last_submitted = ""
    # Clear any attached file memory from previous chat
    st.session_state.attached_file_bytes = None
    st.session_state.attached_file_name  = None
    st.session_state.attached_file_mime  = None

def save_current_chat():
    """Save current messages to the current Streamlit session."""
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
# AI / VISION SERVICES
# ════════════════════════════════════════════════════════════════════

def ask_ai_streaming(problem: str, sympy_info: dict, history: list) -> str:
    """Fetch the full provider response and render it without artificial delay."""
    full_response = ask_ai(problem, sympy_info, history)

    # Provider calls are currently non-streaming. Render the completed response
    # once instead of replaying it with a synthetic sleep between word chunks.
    st.write_stream(iter((full_response,)))
    return full_response


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Saad.AI")
    st.caption("B.Sc. Mathematics Engine")
    st.markdown("Deterministic calculations with AI-powered explanations.")
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
        "Try an example problem",
        list(examples.keys()),
        key="example_select",
        help="Choose a prompt to place in the chat input.",
    )

    st.divider()
    st.markdown("**🔧 How Saad.AI works**")
    st.markdown(
        "1. **SymPy** handles supported calculations exactly.\n"
        "2. **AI providers** explain the result step by step.\n"
        "3. The response is labeled when deterministic verification is available."
    )

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
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
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
        color: #94a3b8;
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

    st.markdown("### Explore a worked example")
    st.caption("Start with a guided problem, then try your own question. Supported calculations are verified by SymPy before the AI explanation is shown.")
    demo_cols = st.columns(3)
    demo_prompts = [
        ("📈 Calculus", "Find the derivative of x^3 + 5x^2 - 3x + 7"),
        ("🔢 Linear algebra", "Find eigenvalues of matrix [[4,1],[2,3]]"),
        ("🧮 Numerical", "Apply Newton-Raphson to x^3 - 2x - 5 = 0, x0=2, 3 iterations"),
    ]
    for demo_col, (label, prompt) in zip(demo_cols, demo_prompts):
        with demo_col:
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                st.session_state.demo_problem = prompt
                st.rerun()

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
            if msg.get("verified"):
                st.caption("✓ SymPy verified computation")
            else:
                st.caption("AI-generated explanation — deterministic verification was unavailable for this request.")
            with st.expander("Solution tools"):
                st.download_button(
                    "Download Markdown",
                    data=msg["content"],
                    file_name=f"saad-ai-solution-{i + 1}.md",
                    mime="text/markdown",
                    key=f"download_solution_{i}",
                )
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
            f'✅ <b>{_escape_html(st.session_state.pending_file_name or "")}</b> — ready · type your question and press Enter</div>',
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
            f'📎 <b>{_escape_html(_memfile or "")}</b> in memory — ask a follow-up or click 📎 to attach new</div>',
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
        "Upload an image or PDF",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        help="Maximum size is controlled by MAX_UPLOAD_BYTES.",
        key="main_uploader",
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
demo_problem = st.session_state.pop("demo_problem", "")
if demo_problem and demo_problem != st.session_state.last_submitted:
    problem = demo_problem
elif prefill and prefill != st.session_state.last_submitted:
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
        MAX_FILE_SIZE = settings.max_upload_bytes
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
                answer = handle_uploaded_file(
                    _MemoryUpload(file_bytes, file_name, file_mime),
                    problem,
                )
            st.markdown(answer)
            with st.expander("📋 Copy"):
                st.code(answer, language=None)

        if not st.session_state.current_chat_id:
            new_chat()
        st.session_state.messages.append({
            "role": "user",
            "content": f"📎 {file_name} — {problem}"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "verified": "SymPy Verified" in answer,
        })
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "verified": "SymPy Verified" in answer,
        })
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

        if is_casual:
            st.caption("AI response")
        elif sympy_result.get("result") and sympy_result.get("result") not in ("matrix_detected", "mod_detected"):
            st.caption("✓ SymPy verified computation")
        else:
            st.caption("AI-generated explanation — deterministic verification was unavailable for this request.")

        with st.expander("Solution tools"):
            st.download_button(
                "Download Markdown",
                data=answer,
                file_name="saad-ai-solution.md",
                mime="text/markdown",
                key="download_current_solution",
            )
            st.code(answer, language=None)

    # Save to history
    if not st.session_state.current_chat_id:
        new_chat()  # create chat ID BEFORE appending — avoids wiping messages
    st.session_state.messages.append({"role": "user", "content": problem})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "verified": bool(
            not is_casual
            and sympy_result.get("result")
            and sympy_result.get("result") not in ("matrix_detected", "mod_detected")
        ) if not is_casual else False,
    })
    save_current_chat()
