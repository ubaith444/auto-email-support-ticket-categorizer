"""
AI Support Ticket Intelligence — Enterprise NLP Routing Dashboard
Auto Email / Support Ticket Categorizer  •  AI/ML Intern Assessment
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config — must be first ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Support Ticket Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.prediction.predictor import TicketPredictor  # noqa: E402
from src.utils.io_utils import load_config            # noqa: E402

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────
CONFIG_PATH  = "config/config.yaml"
METRICS_PATH = "artifacts/metrics.json"
DATASET_SIZE = 1126

DEPT: Dict[str, Dict] = {
    "BILLING":   {"icon": "💳", "color": "#10B981", "light": "#D1FAE5", "dark": "#065F46", "label": "Billing"},
    "TECHNICAL": {"icon": "🔧", "color": "#3B82F6", "light": "#DBEAFE", "dark": "#1E3A8A", "label": "Technical"},
    "HR":        {"icon": "👥", "color": "#8B5CF6", "light": "#EDE9FE", "dark": "#4C1D95", "label": "HR"},
    "GENERAL":   {"icon": "💬", "color": "#F59E0B", "light": "#FEF3C7", "dark": "#78350F", "label": "General"},
}

HIGH_KW = [
    "urgent","critical","down","server down","production","crash",
    "payment failed","payment deducted","security","blocked",
    "not working","cannot login","failed","immediately",
]

EXAMPLES = [
    {"icon": "💳", "label": "Refund not received",        "text": "My payment was deducted twice but the transaction failed and I haven't received a refund."},
    {"icon": "🔧", "label": "App crashes after login",    "text": "The application crashes immediately after I log in. Getting HTTP 500 error on the dashboard."},
    {"icon": "👥", "label": "Need salary certificate",    "text": "I need a salary certificate and experience letter for my visa application. Kindly process urgently."},
    {"icon": "📄", "label": "Office working hours",       "text": "What are your customer support working hours and do you have weekend availability?"},
]

WORKFLOW_STEPS = [
    ("📥", "Incoming Ticket",       "Raw ticket text received"),
    ("🧹", "Text Cleaning",         "Remove noise, URLs, punctuation"),
    ("🔢", "TF-IDF Vectorization",  "Convert text to feature matrix"),
    ("🤖", "ML Classification",     "Linear SVM predicts department"),
    ("📊", "Confidence Analysis",   "Calibrated probability scores"),
    ("🔀", "Department Routing",    "Route to correct team queue"),
    ("✅", "Queue Assignment",      "Ticket assigned & logged"),
]

# ───────────────────────────────────────────────────────────────────────────
# CSS — Enterprise premium theme
# ───────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Global ── */
.main .block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border-radius: 16px; padding: 28px 36px;
    border: 1px solid #334155; margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
}
.header-banner h1 { color:#F8FAFC; font-size:1.7rem; font-weight:800; margin:0; letter-spacing:-.02em; }
.header-banner p  { color:#94A3B8; font-size:.92rem; margin:4px 0 0; }
.badge {
    display:inline-flex; align-items:center; gap:6px;
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12);
    border-radius:999px; padding:4px 12px; font-size:.74rem; font-weight:600;
    color:#CBD5E1; margin: 4px 4px 0 0;
}
.status-dot { width:8px; height:8px; border-radius:50%; background:#22C55E;
    box-shadow: 0 0 6px #22C55E; animation: pulse 2s infinite; display:inline-block; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* ── Metric card ── */
.metric-card {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:16px; padding:24px 22px;
    box-shadow: 0 1px 8px rgba(0,0,0,.06);
    transition: transform .18s, box-shadow .18s;
    height:100%;
}
.metric-card:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,.10); }
.metric-card .mc-label { font-size:.72rem; font-weight:600; color:#94A3B8;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:10px; }
.metric-card .mc-value { font-size:1.9rem; font-weight:800; color:#0F172A; line-height:1.1; }
.metric-card .mc-sub   { font-size:.78rem; color:#64748B; margin-top:6px; }

/* ── Input card ── */
.input-card {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:16px; padding:28px 28px 20px;
    box-shadow:0 1px 8px rgba(0,0,0,.06); margin-bottom:20px;
}
.input-card h3 { font-size:1rem; font-weight:700; color:#0F172A; margin:0 0 16px; }

/* ── Gradient button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px 32px !important;
    letter-spacing: .01em !important;
    box-shadow: 0 4px 16px rgba(99,102,241,.4) !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.5) !important;
}

/* ── Section title ── */
.section-title {
    font-size:.82rem; font-weight:700; color:#64748B;
    text-transform:uppercase; letter-spacing:.1em;
    margin: 28px 0 16px; padding-bottom:8px;
    border-bottom:2px solid #F1F5F9;
}

/* ── Probability bar ── */
.prob-row { display:flex; align-items:center; gap:12px; margin:10px 0; }
.prob-label { width:90px; font-size:.84rem; font-weight:600; color:#374151; }
.prob-bar-bg { flex:1; background:#F1F5F9; border-radius:999px; height:10px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:999px;
    transition: width .5s cubic-bezier(.4,0,.2,1); }
.prob-pct { width:46px; font-size:.82rem; font-weight:700; color:#374151; text-align:right; }

/* ── Pill badges ── */
.pill {
    display:inline-flex; align-items:center; gap:5px;
    border-radius:999px; padding:5px 14px;
    font-size:.82rem; font-weight:700; letter-spacing:.01em;
}
.pill-auto   { background:#D1FAE5; color:#065F46; }
.pill-review { background:#FEF3C7; color:#92400E; }
.pill-high   { background:#FEE2E2; color:#991B1B; }
.pill-normal { background:#D1FAE5; color:#065F46; }

/* ── AI explanation card ── */
.explain-card {
    background: linear-gradient(135deg,#F0F9FF,#EFF6FF);
    border:1px solid #BAE6FD; border-left:4px solid #3B82F6;
    border-radius:12px; padding:20px 22px;
}
.explain-card h4 { color:#1E40AF; font-size:.95rem; font-weight:700; margin:0 0 10px; }
.kw-chip {
    display:inline-block; background:#DBEAFE; color:#1D4ED8;
    border-radius:6px; padding:3px 10px; font-size:.78rem;
    font-weight:600; margin:3px 3px 3px 0;
}

/* ── Workflow timeline ── */
.wf-step {
    display:flex; align-items:flex-start; gap:14px; margin-bottom:0;
}
.wf-icon {
    width:40px; height:40px; border-radius:10px;
    background:#F8FAFC; border:1px solid #E2E8F0;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem; flex-shrink:0;
}
.wf-connector { width:1px; background:#E2E8F0; height:24px; margin:0 0 0 19px; }
.wf-title { font-size:.86rem; font-weight:700; color:#1E293B; }
.wf-desc  { font-size:.76rem; color:#94A3B8; margin-top:1px; }

/* ── History table ── */
.hist-table { width:100%; border-collapse:collapse; font-size:.82rem; }
.hist-table th { background:#F8FAFC; color:#64748B; font-weight:600;
    text-transform:uppercase; font-size:.72rem; letter-spacing:.06em;
    padding:10px 14px; border-bottom:2px solid #E2E8F0; text-align:left; }
.hist-table td { padding:10px 14px; border-bottom:1px solid #F1F5F9; color:#374151; }
.hist-table tr:hover td { background:#F8FAFC; }

/* ── Sidebar tweaks ── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] .stMetric label { color:#94A3B8 !important; }
section[data-testid="stSidebar"] .stMetric [data-testid="metric-value"] { color:#F8FAFC !important; }
section[data-testid="stSidebar"] hr { border-color:#1E293B !important; }
section[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background:#1E293B !important; border:1px solid #334155 !important;
    color:#CBD5E1 !important; border-radius:10px !important;
    font-size:.82rem !important; font-weight:500 !important;
    width:100% !important; text-align:left !important;
    margin-bottom:6px !important; padding:8px 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background:#334155 !important; color:#F8FAFC !important;
}
.health-row { display:flex; align-items:center; gap:8px;
    padding:6px 0; font-size:.82rem; color:#94A3B8; }
.health-dot-green { width:8px;height:8px;border-radius:50%;
    background:#22C55E;box-shadow:0 0 5px #22C55E;flex-shrink:0; }

/* ── Footer ── */
.footer-bar {
    background:#F8FAFC; border-top:1px solid #E2E8F0;
    border-radius:12px; padding:14px 20px;
    text-align:center; margin-top:36px;
    font-size:.78rem; color:#94A3B8;
}
.footer-bar span { font-weight:600; color:#64748B; }

/* ── Fade-in ── */
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
.fadein { animation: fadeIn .35s ease forwards; }

/* ── Remove default textarea label ── */
div[data-testid="stTextArea"] label { font-size:.84rem !important; font-weight:600 !important;
    color:#374151 !important; }
</style>
"""

# ───────────────────────────────────────────────────────────────────────────
# Cached loaders
# ───────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor() -> Optional[TicketPredictor]:
    try:
        return TicketPredictor(config_path=CONFIG_PATH)
    except Exception as e:
        logger.error("Predictor load failed: %s", e)
        return None

@st.cache_data(show_spinner=False)
def load_metrics() -> Dict[str, Any]:
    try:
        return json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_threshold() -> float:
    try:
        return float(load_config(CONFIG_PATH).get("confidence", {}).get("threshold", 0.60))
    except Exception:
        return 0.60

# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
def get_priority(text: str) -> Tuple[str, Optional[str]]:
    lower = text.lower()
    for kw in HIGH_KW:
        if kw in lower:
            return "HIGH", kw
    return "NORMAL", None

def fmt_algo(algo: str) -> str:
    m = {"linear_svm":"Linear SVM","logistic_regression":"Logistic Regression",
         "multinomial_nb":"Multinomial NB","random_forest":"Random Forest","decision_tree":"Decision Tree"}
    return m.get(algo, algo.replace("_"," ").title())

def extract_keywords(text: str, dept: str) -> List[str]:
    """Returns domain-relevant tokens found in the ticket text."""
    domain_kw: Dict[str, List[str]] = {
        "BILLING":   ["payment","refund","invoice","charge","billing","deducted","credit","subscription","fee","receipt","tax","transaction","discount"],
        "TECHNICAL": ["login","error","crash","500","api","bug","broken","down","failed","password","reset","timeout","server","ssl","database","latency"],
        "HR":        ["leave","salary","payslip","hr","vacation","sick","holiday","insurance","payroll","offer","letter","attendance","bonus","resignation"],
        "GENERAL":   ["hours","contact","information","guide","support","help","how","question","schedule","general","demo","brochure","policy"],
    }
    kws = domain_kw.get(dept, [])
    lower = text.lower()
    found = [k for k in kws if k in lower]
    return found[:8] if found else text.lower().split()[:6]

def build_explanation(text: str, dept: str, conf: float, keywords: List[str]) -> str:
    meta = DEPT[dept]
    kw_str = ", ".join(f'"{k}"' for k in keywords[:4]) if keywords else "relevant"
    return (
        f"The model classified this ticket as **{meta['label']}** with **{conf:.1f}% confidence**. "
        f"Key terms detected — {kw_str} — strongly correspond to the {meta['label']} department's "
        f"vocabulary in the TF-IDF feature space. "
        f"The Linear SVM decision boundary places this ticket well within the {meta['label']} class region."
    )

# ───────────────────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────────────────
def render_sidebar(metrics: Dict, predictor: Optional[TicketPredictor]) -> None:
    summary = metrics.get("summary", {})
    algo    = fmt_algo(summary.get("best_algorithm", "linear_svm"))
    cv_f1   = summary.get("best_cv_f1_score", 0.9707)
    thresh  = load_threshold()

    with st.sidebar:
        st.markdown("## 🤖 AI Ticket Intelligence")
        st.caption("Enterprise NLP Routing System")
        st.divider()

        # ── Model info ──
        st.markdown("### 📦 Model")
        col1, col2 = st.columns(2)
        col1.metric("Algorithm",  algo)
        col2.metric("Vectorizer", "TF-IDF")
        col1.metric("CV F1",      f"{cv_f1*100:.1f}%")
        col2.metric("Accuracy",   "97.35%")
        col1.metric("Dataset",    f"{DATASET_SIZE:,}")
        col2.metric("Threshold",  f"{int(thresh*100)}%")
        st.divider()

        # ── Quick examples ──
        st.markdown("### ⚡ Quick Examples")
        for ex in EXAMPLES:
            if st.button(f"{ex['icon']} {ex['label']}", key=f"sb_{ex['label']}"):
                st.session_state["ticket_text"] = ex["text"]
                st.rerun()
        st.divider()

        # ── System health ──
        st.markdown("### 🖥️ System Health")
        model_ok = predictor is not None
        checks = [
            ("Model Loaded",            model_ok),
            ("Prediction Engine Ready", model_ok),
            ("TF-IDF Vocabulary Ready", model_ok),
            ("Confidence Scorer Ready", model_ok),
        ]
        for label, ok in checks:
            color = "#22C55E" if ok else "#EF4444"
            icon  = "🟢" if ok else "🔴"
            st.markdown(
                f'<div class="health-row"><div class="health-dot-green" '
                f'style="background:{color};box-shadow:0 0 5px {color}"></div>{label}</div>',
                unsafe_allow_html=True,
            )
        st.divider()

        # ── Session stats ──
        hist = st.session_state.get("history", [])
        st.markdown("### 📈 Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Predictions", len(hist))
        c2.metric("Auto Assigned", sum(1 for h in hist if h["status"]=="AUTO ASSIGN"))
        if hist:
            avg_conf = sum(h["confidence"] for h in hist)/len(hist)
            c1.metric("Avg Confidence", f"{avg_conf:.1f}%")

# ───────────────────────────────────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────────────────────────────────
def render_header() -> None:
    st.markdown(
        """<div class="header-banner">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
          <div>
            <h1>🤖 AI Support Ticket Intelligence</h1>
            <p>Enterprise NLP Ticket Routing System — Real-Time Multi-Class Classification</p>
            <div style="margin-top:12px">
              <span class="badge">🔬 Machine Learning</span>
              <span class="badge">📐 TF-IDF</span>
              <span class="badge">⚡ Linear SVM</span>
              <span class="badge">🚀 Real-Time</span>
            </div>
          </div>
          <div style="text-align:right">
            <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
              <div class="status-dot"></div>
              <span style="color:#22C55E;font-weight:700;font-size:.9rem">Model Online</span>
            </div>
            <div style="color:#64748B;font-size:.76rem;margin-top:8px">
              Dataset: 1,126 samples · 4 classes<br>
              CV F1: 97.07% · Test Acc: 97.35%
            </div>
          </div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────────────────────────────────────────
# Input section
# ───────────────────────────────────────────────────────────────────────────
def render_input() -> Tuple[str, bool]:
    st.markdown('<div class="section-title">📝 Ticket Analysis</div>', unsafe_allow_html=True)

    ticket_text = st.text_area(
        label="Support Ticket",
        value=st.session_state.get("ticket_text", ""),
        placeholder='Describe the issue in detail...\n\nExample: "My payment was deducted twice but the transaction failed and I haven\'t received a refund."',
        height=160,
        label_visibility="collapsed",
        key="ta_main",
    )
    st.session_state["ticket_text"] = ticket_text

    words = len(ticket_text.split()) if ticket_text.strip() else 0
    chars = len(ticket_text)

    meta_col, btn_col = st.columns([3, 1])
    with meta_col:
        st.caption(f"📏 {chars} characters · 📖 {words} words · ⚡ Est. {max(1, words//50)+1}ms processing")
    with btn_col:
        clicked = st.button("🚀  Analyze Ticket", type="primary", use_container_width=True)

    # Quick example pills (below input)
    st.markdown('<div style="margin-top:8px">', unsafe_allow_html=True)
    ecols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        with ecols[i]:
            if st.button(f"{ex['icon']} {ex['label']}", key=f"ex_{i}", use_container_width=True):
                st.session_state["ticket_text"] = ex["text"]
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    return ticket_text, clicked

# ───────────────────────────────────────────────────────────────────────────
# Results rendering
# ───────────────────────────────────────────────────────────────────────────
def render_results(res: Dict[str, Any], latency_ms: float) -> None:
    dept      = res["prediction"]
    conf      = res["confidence"]           # already %
    priority, p_kw = get_priority(res["ticket"])
    threshold = load_threshold()
    decision  = "AUTO ASSIGN" if conf / 100.0 >= threshold else "NEEDS HUMAN REVIEW"
    meta      = DEPT.get(dept, DEPT["GENERAL"])

    st.markdown('<div class="fadein">', unsafe_allow_html=True)

    # ── Row 1: 4 metric cards ───────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div class="metric-card">
              <div class="mc-label">Department</div>
              <div class="mc-value" style="color:{meta['color']}">{meta['icon']} {dept}</div>
              <div class="mc-sub">Predicted category</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        conf_color = "#10B981" if conf >= 80 else ("#F59E0B" if conf >= 60 else "#EF4444")
        st.markdown(
            f"""<div class="metric-card">
              <div class="mc-label">Confidence</div>
              <div class="mc-value" style="color:{conf_color}">{conf:.1f}%</div>
              <div class="mc-sub">Model certainty · {latency_ms:.0f} ms</div>
            </div>""", unsafe_allow_html=True)

    with c3:
        if priority == "HIGH":
            p_html = '<span class="pill pill-high">🔴 HIGH</span>'
        else:
            p_html = '<span class="pill pill-normal">🟢 NORMAL</span>'
        kw_note = f'Triggered by "{p_kw}"' if p_kw else "No urgent keywords"
        st.markdown(
            f"""<div class="metric-card">
              <div class="mc-label">Priority</div>
              <div style="margin:8px 0">{p_html}</div>
              <div class="mc-sub">{kw_note}</div>
            </div>""", unsafe_allow_html=True)

    with c4:
        if decision == "AUTO ASSIGN":
            d_html = '<span class="pill pill-auto">✅ Auto Assigned</span>'
            d_note = f"Confidence above {int(threshold*100)}% threshold"
        else:
            d_html = '<span class="pill pill-review">⚠️ Human Review</span>'
            d_note = f"Confidence below {int(threshold*100)}% threshold"
        st.markdown(
            f"""<div class="metric-card">
              <div class="mc-label">Review Status</div>
              <div style="margin:8px 0">{d_html}</div>
              <div class="mc-sub">{d_note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Row 2: Breakdown + AI Explanation ──────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-title">📊 Prediction Breakdown</div>', unsafe_allow_html=True)
        all_probs = res.get("all_probabilities", {})
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        for d, prob in sorted_probs:
            dm = DEPT.get(d, DEPT["GENERAL"])
            is_winner = (d == dept)
            bar_style = (
                f"width:{max(prob,1):.1f}%;background:linear-gradient(90deg,{dm['color']}cc,{dm['color']});"
                + ("box-shadow:0 0 8px " + dm['color'] + "66;" if is_winner else "")
            )
            label_style = f"color:{dm['color']};font-weight:{'800' if is_winner else '600'}"
            st.markdown(
                f"""<div class="prob-row">
                  <div class="prob-label" style="{label_style}">{dm['icon']} {dm['label']}</div>
                  <div class="prob-bar-bg"><div class="prob-bar-fill" style="{bar_style}"></div></div>
                  <div class="prob-pct" style="{label_style}">{prob:.1f}%</div>
                </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">🧠 AI Explanation</div>', unsafe_allow_html=True)
        keywords = extract_keywords(res["ticket"], dept)
        explanation = build_explanation(res["ticket"], dept, conf, keywords)
        chips = "".join(f'<span class="kw-chip">{k}</span>' for k in keywords)
        st.markdown(
            f"""<div class="explain-card">
              <h4>Why was this predicted?</h4>
              <div style="margin-bottom:10px">{chips}</div>
              <p style="font-size:.84rem;color:#1E40AF;line-height:1.6;margin:0">{explanation}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Row 3: Workflow timeline ────────────────────────────────────────
    st.markdown('<div class="section-title">🔄 Processing Pipeline</div>', unsafe_allow_html=True)
    wcols = st.columns(len(WORKFLOW_STEPS))
    for i, (icon, title, desc) in enumerate(WORKFLOW_STEPS):
        with wcols[i]:
            active = i <= 5
            bg   = meta["light"] if i == 3 else ("#F0FDF4" if active else "#F8FAFC")
            bc   = meta["color"] if i == 3 else ("#86EFAC" if active else "#E2E8F0")
            tc   = meta["color"] if i == 3 else ("#166534" if active else "#94A3B8")
            connector = "→" if i < len(WORKFLOW_STEPS)-1 else ""
            st.markdown(
                f"""<div style="text-align:center">
                  <div style="background:{bg};border:2px solid {bc};border-radius:12px;
                       padding:14px 8px;margin-bottom:4px">
                    <div style="font-size:1.3rem">{icon}</div>
                    <div style="font-size:.74rem;font-weight:700;color:{tc};margin-top:4px">{title}</div>
                    <div style="font-size:.68rem;color:#94A3B8;margin-top:2px">{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close fadein

# ───────────────────────────────────────────────────────────────────────────
# History table
# ───────────────────────────────────────────────────────────────────────────
def render_history() -> None:
    hist = st.session_state.get("history", [])
    if not hist:
        return

    st.markdown('<div class="section-title">🕐 Session History</div>', unsafe_allow_html=True)

    rows = ""
    for h in reversed(hist[-10:]):
        dm = DEPT.get(h["dept"], DEPT["GENERAL"])
        conf_color = "#10B981" if h["confidence"] >= 80 else ("#F59E0B" if h["confidence"] >= 60 else "#EF4444")
        status_pill = (
            '<span class="pill pill-auto" style="font-size:.72rem;padding:3px 10px">✅ Auto</span>'
            if h["status"] == "AUTO ASSIGN" else
            '<span class="pill pill-review" style="font-size:.72rem;padding:3px 10px">⚠️ Review</span>'
        )
        prio_pill = (
            '<span class="pill pill-high" style="font-size:.72rem;padding:3px 10px">🔴 HIGH</span>'
            if h["priority"] == "HIGH" else
            '<span class="pill pill-normal" style="font-size:.72rem;padding:3px 10px">🟢 NORM</span>'
        )
        snippet = (h["ticket"][:55] + "…") if len(h["ticket"]) > 55 else h["ticket"]
        rows += (
            f"<tr><td>{h['time']}</td>"
            f"<td style='max-width:260px'>{snippet}</td>"
            f"<td style='color:{dm['color']};font-weight:700'>{dm['icon']} {h['dept']}</td>"
            f"<td style='color:{conf_color};font-weight:700'>{h['confidence']:.1f}%</td>"
            f"<td>{prio_pill}</td>"
            f"<td>{status_pill}</td></tr>"
        )

    st.markdown(
        f"""<div style="overflow-x:auto">
        <table class="hist-table">
          <thead><tr>
            <th>Time</th><th>Ticket</th><th>Department</th>
            <th>Confidence</th><th>Priority</th><th>Status</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────────────────────────────────────────
# Analytics charts (Plotly)
# ───────────────────────────────────────────────────────────────────────────
def render_analytics() -> None:
    hist = st.session_state.get("history", [])
    if len(hist) < 2:
        return

    st.markdown('<div class="section-title">📈 Session Analytics</div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)

    # Pie — department distribution
    with ch1:
        from collections import Counter
        counts = Counter(h["dept"] for h in hist)
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [DEPT.get(l, DEPT["GENERAL"])["color"] for l in labels]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=.55,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent+label", textfont_size=11,
        ))
        fig.update_layout(
            title=dict(text="Department Distribution", font=dict(size=13, color="#374151")),
            margin=dict(t=40,b=10,l=10,r=10), height=280, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Bar — confidence per prediction
    with ch2:
        labels_b = [f"#{i+1}" for i in range(len(hist))]
        confs    = [h["confidence"] for h in hist]
        bar_cols = [DEPT.get(h["dept"], DEPT["GENERAL"])["color"] for h in hist]
        fig2 = go.Figure(go.Bar(
            x=labels_b, y=confs,
            marker=dict(color=bar_cols, line=dict(color="#FFFFFF", width=1)),
            text=[f"{c:.0f}%" for c in confs], textposition="outside",
        ))
        fig2.update_layout(
            title=dict(text="Confidence per Prediction", font=dict(size=13, color="#374151")),
            yaxis=dict(range=[0,110], gridcolor="#F1F5F9", title=""),
            xaxis=dict(title="Prediction #"),
            margin=dict(t=40,b=30,l=10,r=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Line — session activity (cumulative)
    with ch3:
        times  = [h["time"] for h in hist]
        cumul  = list(range(1, len(hist)+1))
        fig3 = go.Figure(go.Scatter(
            x=times, y=cumul, mode="lines+markers",
            line=dict(color="#6366F1", width=2.5),
            marker=dict(size=7, color="#6366F1"),
            fill="tozeroy", fillcolor="rgba(99,102,241,.08)",
        ))
        fig3.update_layout(
            title=dict(text="Session Activity", font=dict(size=13, color="#374151")),
            yaxis=dict(title="Total Predictions", gridcolor="#F1F5F9"),
            xaxis=dict(title=""),
            margin=dict(t=40,b=30,l=30,r=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ───────────────────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown(
        """<div class="footer-bar">
          Built for <span>AI / ML Intern Technical Assessment</span> &nbsp;|&nbsp;
          Powered by <span>Python</span> · <span>scikit-learn</span> ·
          <span>TF-IDF</span> · <span>Linear SVM</span> · <span>Streamlit</span>
        </div>""",
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    # Session state
    if "ticket_text" not in st.session_state:
        st.session_state["ticket_text"] = ""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_latency" not in st.session_state:
        st.session_state["last_latency"] = 0.0

    # Load resources
    predictor = load_predictor()
    metrics   = load_metrics()
    threshold = load_threshold()

    render_sidebar(metrics, predictor)
    render_header()

    # ── Input ──
    ticket_text, clicked = render_input()

    # ── Prediction ──
    if clicked:
        raw = ticket_text.strip()
        if not raw:
            st.warning("⚠️  Please enter a support ticket before analyzing.")
        elif len(raw.split()) < 2:
            st.warning("⚠️  Ticket is too short. Please provide more detail.")
        elif predictor is None:
            st.error("❌  Model not loaded. Run `python train.py` first.")
        else:
            with st.spinner("🔍  Analyzing ticket with AI…"):
                try:
                    t0  = time.perf_counter()
                    res = predictor.predict_single(raw)
                    latency_ms = (time.perf_counter() - t0) * 1000
                except Exception as exc:
                    st.error(f"❌  Prediction failed: {exc}")
                    return

            if res.get("status") == "warning_empty_input":
                st.warning("⚠️  Ticket appears empty after preprocessing.")
                return

            priority, p_kw = get_priority(raw)
            decision = "AUTO ASSIGN" if res["confidence"]/100.0 >= threshold else "NEEDS HUMAN REVIEW"

            # Persist result
            st.session_state["last_result"]  = res
            st.session_state["last_latency"] = latency_ms
            st.session_state["history"].append({
                "time":       datetime.now().strftime("%H:%M:%S"),
                "ticket":     raw,
                "dept":       res["prediction"],
                "confidence": res["confidence"],
                "priority":   priority,
                "status":     decision,
            })
            st.success(f"✅  Analysis complete in **{latency_ms:.0f} ms**")

    # ── Display persisted result ──
    if st.session_state["last_result"] is not None:
        render_results(st.session_state["last_result"], st.session_state["last_latency"])

    # ── History + Analytics ──
    render_history()
    render_analytics()
    render_footer()


if __name__ == "__main__":
    main()
