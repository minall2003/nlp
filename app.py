import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from io import StringIO

# ---------- STYLING & PAGE CONFIG ----------
st.set_page_config(page_title="AI vs. Fact: NLP Comparator", page_icon="🤖", layout="wide")

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
    <h1>🤖 AI vs. Fact: NLP Comparator</h1>
    <p style='color:#cfcfcf;'>Where machine learning models compete... and sometimes get roasted.</p>
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
        "Lexical & Morphological", 
        "Syntactic", 
        "Semantic", 
        "Discourse", 
        "Pragmatic"
    ])

    st.markdown("---")

    st.markdown("### ⚙️ Analysis Settings")
    st.slider("Train/Test Split Ratio", 0.1, 0.9, 0.8)
    st.selectbox("Evaluation Metric", ["Accuracy", "F1-Score", "Precision", "Recall"])

    if st.button("🚀 Run Comparison"):
        st.toast("Running models... please wait ⏳")
        time.sleep(2)


# ---------- CENTER COLUMN: RESULTS ----------
with center_col:
    st.markdown("### 📊 Model Benchmarking Results")

    # Example data (you will replace with real model outputs)
    models = ["Naive Bayes", "Decision Tree", "Logistic Regression", "SVM"]
    metrics = {
        "Accuracy": [0.86, 0.78, 0.89, 0.91],
        "F1-Score": [0.84, 0.75, 0.88, 0.90],
        "Precision": [0.85, 0.77, 0.88, 0.92],
        "Recall": [0.83, 0.74, 0.87, 0.88],
        "Training Time (s)": [0.2, 0.5, 0.7, 1.3],
        "Inference Latency (ms)": [2, 5, 3, 8]
    }
    df_metrics = pd.DataFrame(metrics, index=models)

    st.dataframe(df_metrics.style.highlight_max(color="#0072ff", axis=0))

    st.markdown("### 📈 Performance Visualization")

    metric_choice = st.selectbox("Select Metric to Visualize", list(metrics.keys())[:-2])
    plt.figure(figsize=(7, 4))
    plt.bar(models, df_metrics[metric_choice], color="#00c6ff", alpha=0.8)
    plt.title(f"{metric_choice} Comparison", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel(metric_choice)
    plt.grid(alpha=0.2)
    st.pyplot(plt)


# ---------- RIGHT COLUMN: HUMOROUS CRITIQUE ----------
with right_col:
    st.markdown("### 😂 AI Roast Zone")

    best_model = "SVM"
    best_phase = phase
    roasts = [
        f"{best_model} walked into the {best_phase} phase and said, 'Is this all you got, human?' 😎",
        f"In the {best_phase} phase, {best_model} just flexed its margins and left everyone speechless. 💪",
        f"{best_model} performed so well, the other models applied for early retirement. 🏆",
        f"Even ChatGPT blushed at {best_model}'s performance in the {best_phase} phase. 💬🔥"
    ]
    st.markdown(f"<div class='humor-box'>{random.choice(roasts)}</div>", unsafe_allow_html=True)

    # Scatter plot for trade-off
    st.markdown("### ⚖️ Speed vs. Quality Trade-Off")

    plt.figure(figsize=(6, 4))
    plt.scatter(df_metrics["Training Time (s)"], df_metrics["F1-Score"], color="#00c6ff", s=100)
    for i, model in enumerate(models):
        plt.text(df_metrics["Training Time (s)"][i] + 0.02,
                 df_metrics["F1-Score"][i],
                 model, fontsize=9)
    plt.xlabel("Training Time (s)")
    plt.ylabel("F1-Score")
    plt.grid(alpha=0.3)
    st.pyplot(plt)


# ---------- FOOTER ----------
st.markdown("""
---
<div style='text-align:center; color: #aaaaaa; font-size: 13px; margin-top: 15px;'>
    Built with ❤️ using Streamlit | Designed by <b>AI vs. Fact</b> Team © 2025
</div>
""", unsafe_allow_html=True)
