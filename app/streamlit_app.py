"""
Legal Judgment Prediction & Explanation — Streamlit Application

Main entry point for the web interface.
Run: streamlit run app/streamlit_app.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.utils.logger import get_logger

logger = get_logger("streamlit_app")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Judgment Prediction & Explanation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 2.5em;
        color: #0d47a1;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    div.stButton > button:first-child {
        background-color: #0d47a1;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #1565c0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚖️ Navigation")

page = st.sidebar.radio(
    "Choose a feature:",
    ["🏠 Home", "📄 Predict Judgment", "📊 Model Comparison", "💬 Legal Chatbot"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by** [Srujan Kothuri](https://github.com/srujankothuri)\n\n"
    "Published at Springer ICDSA 2025"
)

# ── Page Router ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from app.pages.home import render
    render()
elif page == "📄 Predict Judgment":
    from app.pages.predict import render
    render()
elif page == "📊 Model Comparison":
    from app.pages.compare_models import render
    render()
elif page == "💬 Legal Chatbot":
    from app.pages.chatbot import render
    render()