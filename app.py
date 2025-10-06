# ==========================================================
# 🧠 NLP Phase vs ML Model Comparator (High Accuracy Version)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ML models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Helper functions
# ==========================================

def clean_text(text):
    """Preprocess text to remove noise"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)              # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# Streamlit App
# ==========================================

def main():
    st.title("🧠 NLP Phase vs ML Model Comparator (Accuracy Boosted 🚀)")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("👀 **Data Preview**")
        st.write(df.head())
        st.write(f"Shape: {df.shape}")

        text_col = st.selectbox("📜 Select Text Column", df.columns)
        label_col = st.selectbox("🏷️ Select Label Column", df.columns)

        if st.button("🚀 Train Model"):
            X = df[text_col].astype(str).apply(clean_text)
            y = df[label_col]

            # --- Safe Train/Test Split ---
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
            except ValueError:
                st.warning("⚠️ Adjusting labels with few samples...")
                y_series = pd.Series(y)
                y_counts = y_series.value_counts()
                valid_classes = y_counts[y_counts > 1].index
                mask = y_series.isin(valid_classes)
                X = X[mask]
                y = y_series[mask].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

            # --- TF-IDF Vectorization (optimized) ---
            st.info("🔠 Converting text to vectors using TF-IDF...")
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2)
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # --- Apply SMOTE ---
            try:
                st.info("⚖️ Applying SMOTE to balance classes...")
                smote = SMOTE(random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)
                st.success(f"✅ SMOTE applied! Training samples: {X_train_bal.shape[0]}")
            except Exception as e:
                st.warning(f"⚠️ SMOTE skipped: {e}")
                X_train_bal, y_train_bal = X_train_vec, y_train

            # --- Choose Model ---
            model_choice = st.selectbox(
                "🤖 Choose ML Algorithm",
                ["Naive Bayes", "Logistic Regression", "SVM", "Decision Tree", "Random Forest"]
            )

            if model_choice == "Naive Bayes":
                model = MultinomialNB()
            elif model_choice == "Logistic Regression":
                st.info("🧩 Scaling features for Logistic Regression...")
                scaler = StandardScaler(with_mean=False)
                X_train_bal = scaler.fit_transform(X_train_bal)
                X_test_vec = scaler.transform(X_test_vec)
                model = LogisticRegression(C=3, solver='lbfgs', max_iter=2000)
            elif model_choice == "SVM":
                st.info("🧩 Scaling features for SVM...")
                scaler = StandardScaler(with_mean=False)
                X_train_bal = scaler.fit_transform(X_train_bal)
                X_test_vec = scaler.transform(X_test_vec)
                model = SVC(kernel='linear', C=2)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=None)
            else:
                model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)

            # --- Train & Evaluate ---
            st.info("🏋️ Training the model, please wait...")
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Accuracy: {acc * 100:.2f}%")
            st.text("📊 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.balloons()

if __name__ == "__main__":
    main()
