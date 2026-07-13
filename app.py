"""
NLP-Based Candidate Job Match System
--------------------------------------
A Streamlit application that compares a candidate's resume against a job
description using TF-IDF similarity and skill matching, then produces an
overall match score and recommendation.

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from utils import (
    extract_text_from_pdf,
    clean_text,
    final_score,
    recommendation,
)

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Candidate Job Match System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Global Styling
# =========================================================
st.markdown(
    """
    <style>
        /* ---------- General ---------- */
        .main {
            background-color: #0e1117;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---------- Hero header ---------- */
        .hero {
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .hero p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.02rem;
            max-width: 780px;
            margin-bottom: 0;
        }

        /* ---------- Upload cards ---------- */
        .upload-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.2rem 1.4rem 0.6rem 1.4rem;
            height: 100%;
        }
        .upload-card h4 {
            margin-bottom: 0.6rem;
        }

        /* ---------- Metric cards ---------- */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1.2rem;
        }

        /* ---------- Section titles ---------- */
        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin: 0.2rem 0 0.8rem 0;
        }

        /* ---------- Skill chips ---------- */
        .chip {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            margin: 0.25rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .chip-match {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }
        .chip-missing {
            background: rgba(248, 113, 113, 0.12);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.35);
        }

        /* ---------- Buttons ---------- */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border: none;
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
            transition: transform 0.05s ease-in-out, box-shadow 0.15s ease-in-out;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
            box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45);
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:active {
            transform: translateY(1px);
        }
        div.stButton > button[kind="secondary"] {
            border: 1px solid rgba(124, 58, 237, 0.5);
            color: #c4b5fd;
            font-weight: 600;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: #7c3aed;
            color: #ffffff;
            background: rgba(124, 58, 237, 0.12);
        }

        /* ---------- Footer note ---------- */
        .footnote {
            color: rgba(255,255,255,0.45);
            font-size: 0.82rem;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================
SCORE_ICONS = {
    "Strong Match": "🟢",
    "Good Match": "🔵",
    "Moderate Match": "🟡",
    "Low Match": "🔴",
}


def score_indicator(value: float) -> str:
    """Return a colored dot + label based on the same thresholds as
    utils.recommendation() (>=80 Strong, >=60 Good, >=40 Moderate, else Low)."""
    label = recommendation(value)
    icon = SCORE_ICONS.get(label, "⚪")
    return f"{icon} {label}"


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ How it works")
    st.markdown(
        """
        1. Upload a **Job Description** (PDF)
        2. Upload a **Candidate Resume** (PDF)
        3. Click **Generate Match Report**

        The system extracts and cleans text from both documents,
        computes **TF-IDF cosine similarity**, and cross-references
        required skills to produce a weighted match score.
        """
    )
    st.divider()
    st.markdown("**Scoring formula**")
    st.code("Final Score = 30% TF-IDF + 70% Skill Match", language="text")
    st.divider()
    st.caption("Built with Streamlit • Powered by NLP")

# =========================================================
# Hero Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🤖 NLP-Based Candidate Job Match System</h1>
        <p>
            Automatically compare a candidate's resume with a job description using
            Natural Language Processing. This tool extracts text from PDFs, performs
            TF-IDF vectorization, calculates cosine similarity, identifies matched
            and missing skills, and generates an overall hiring recommendation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session State Setup
# =========================================================
if "results" not in st.session_state:
    st.session_state.results = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================================================
# Upload Section
# =========================================================
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 📄 Job Description")
    jd_file = st.file_uploader(
        "Upload the job description PDF",
        type=["pdf"],
        key=f"jd_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 📑 Candidate Resume")
    resume_file = st.file_uploader(
        "Upload the candidate resume PDF",
        type=["pdf"],
        key=f"resume_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
generate = st.button("🚀 Generate Match Report", use_container_width=True, type="primary")

st.divider()

# =========================================================
# Processing
# =========================================================
if generate:
    if jd_file is None or resume_file is None:
        st.warning("⚠️ Please upload both the Job Description and Resume PDFs before continuing.")
        st.stop()

    try:
        with st.spinner("🔍 Analyzing documents... this will only take a moment"):
            # Use a temp directory so files never collide or leak between runs.
            with tempfile.TemporaryDirectory() as tmp_dir:
                jd_path = Path(tmp_dir) / "jd.pdf"
                resume_path = Path(tmp_dir) / "resume.pdf"

                jd_path.write_bytes(jd_file.getvalue())
                resume_path.write_bytes(resume_file.getvalue())

                jd_text = extract_text_from_pdf(str(jd_path))
                resume_text = extract_text_from_pdf(str(resume_path))

            if not jd_text.strip() or not resume_text.strip():
                st.error(
                    "❌ Couldn't extract readable text from one of the PDFs. "
                    "Please make sure the files are not scanned images and try again."
                )
                st.stop()

            jd_clean = clean_text(jd_text)
            resume_clean = clean_text(resume_text)

            score, tfidf_score, skill_score, matched, missing = final_score(
                jd_clean,
                resume_clean,
            )

        st.session_state.results = {
            "score": score,
            "tfidf_score": tfidf_score,
            "skill_score": skill_score,
            "matched": matched,
            "missing": missing,
        }

    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Something went wrong while processing the files: {exc}")
        st.session_state.results = None
        st.stop()

# =========================================================
# Results
# =========================================================
if st.session_state.results:
    r = st.session_state.results
    score = r["score"]
    tfidf_score = r["tfidf_score"]
    skill_score = r["skill_score"]
    matched = r["matched"]
    missing = r["missing"]
    total_skills = len(matched) + len(missing)

    st.success("Analysis completed successfully ✅")
    st.divider()

    # ---------------------------------------------------
    # Overall Score
    # ---------------------------------------------------
    st.markdown('<p class="section-title">🎯 Overall Match Score</p>', unsafe_allow_html=True)

    score_col, gauge_col = st.columns([1, 2])
    with score_col:
        st.metric("Candidate Match", f"{score}%")
        st.markdown(f"**{score_indicator(score)}**")
    with gauge_col:
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.caption("Final Score = 30% TF-IDF Similarity + 70% Skill Match")

    st.divider()

    # ---------------------------------------------------
    # Score Breakdown
    # ---------------------------------------------------
    st.markdown('<p class="section-title">📊 Score Breakdown</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("TF-IDF Similarity", f"{tfidf_score}%")
        st.caption(score_indicator(tfidf_score))

    with c2:
        st.metric("Skill Match", f"{skill_score}%")
        st.caption(score_indicator(skill_score))

    with c3:
        skills_pct = (len(matched) / total_skills * 100) if total_skills else 0
        st.metric("Matched Skills", f"{len(matched)}/{total_skills}" if total_skills else "0/0")
        st.caption(score_indicator(skills_pct))

    st.divider()

    # ---------------------------------------------------
    # Recommendation
    # ---------------------------------------------------
    st.markdown('<p class="section-title">📢 Recommendation</p>', unsafe_allow_html=True)
    rec = recommendation(score)
    icon = SCORE_ICONS.get(rec, "⚪")
    rec_style = {
        "Strong Match": st.success,
        "Good Match": st.info,
        "Moderate Match": st.warning,
    }
    rec_style.get(rec, st.error)(f"**{icon} {rec}**")

    st.divider()

    # ---------------------------------------------------
    # Skills Detail
    # ---------------------------------------------------
    st.markdown('<p class="section-title">🧩 Skill Analysis</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("**✅ Matched Skills**")
            if matched:
                st.markdown(
                    "".join(f'<span class="chip chip-match">✔ {s}</span>' for s in matched),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No matched skills found.")

    with col2:
        with st.container(border=True):
            st.markdown("**❌ Missing Skills**")
            if missing:
                st.markdown(
                    "".join(f'<span class="chip chip-missing">✘ {s}</span>' for s in missing),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No missing skills — great coverage!")

    st.divider()

    # ---------------------------------------------------
    # Reset
    # ---------------------------------------------------
    if st.button("🔄 Start a New Comparison", type="secondary"):
        st.session_state.results = None
        st.session_state.uploader_key += 1  # forces file_uploader widgets to reset
        st.rerun()

st.markdown(
    '<p class="footnote">NLP Candidate Job Match System • Built with Streamlit</p>',
    unsafe_allow_html=True,
)