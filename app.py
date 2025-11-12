import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from io import StringIO

# ---------- STYLING & PAGE CONFIG ----------
st.set_page_config(page_title="TruthLens: The NLP Intelligence Comparator", page_icon="🧠", layout="wide")

# Inject custom CSS for modern look
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #1e1e2f, #2b2b40);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Section titles */
h2, h3 {
    color: #f8f9fa;
    text-shadow: 0 0 10px rgba(255,255,255,0.2);
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}

/* Buttons */
div.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}

/* Humor box */
.humor-box {
    background: rgba(0,0,0,0.4);
    border-radius: 15px;
    padding: 15px;
    font-style: italic;
    border-left: 5px solid #00c6ff;
}

/* Chart borders */
.plot-container {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


# ---------- SPLASH SCREEN ----------
with st.spinner("⚙️ Initializing AI humor protocols..."):
    time.sleep(1.5)

st.markdown("""
<div style='text-align:center; margin-bottom: 30px;'>
    <h1>🧠 TruthLens: The NLP Intelligence Comparator</h1>
    <p style='color:#cfcfcf;'>Where machine learning models compete... and the truth shines through.</p>
</div>
""", unsafe_allow_html=True)


# ---------- LAYOUT ----------
left_col, center_col, right_col = st.columns([1, 2, 2])


# ---------- LEFT COLUMN: INPUT & CONFIG ----------
with left_col:
    st.markdown("### 📅 Data Sourcing")

    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    phase = st.selectbox("🧠 Choose NLP Phase", [
        "Lexic
