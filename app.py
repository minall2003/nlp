# ==========================================================
# 🧠 NLP Phase vs ML Model Comparator
# Enhanced with SMOTE + safe stratified splitting
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE  # 🧩 New Import

# Example ML algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Streamlit app
# ==========================================

def main():
    st.title("🧠 NLP Phase vs ML Model Comparator")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("👀 **Data Preview**")
        st.write(df.head())
        st.write(f"Shape: {df.shape}")

        # Select text and label columns
        text_col = st.selectbox("📜 Select Text Column", df.columns)
        label_col = st.selectbox("🏷️ Select Label Column", df.columns)

        if st.button("🚀 Train Model"):
            X = df[text_col]
            y = df[label_col]

            # --- Safe Train/Test Split ---
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
            except ValueError as e:
                if "least populated class" in str(e):
                    st.warning("⚠️ Some labels have only one sample — removing them and retrying split.")
                    y_series = pd.Series(y)
                    y_counts = y_series.value_counts()
                    valid_classes = y_counts[y_counts > 1].index
                    mask = y_series.isin(valid_classes)
                    X = X[mask]
                    y = y_series[mask].values
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )
                else:
                    st.warning("⚠️ Stratified split failed — falling back to random split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

            # --- Vectorization (TF-IDF Phase Example) ---
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # --- Apply SMOTE to balance classes ---
            try:
                st.info("🧩 Applying SMOTE to balance training data...")
                smote = SMOTE(random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)
                st.success(f"✅ SMOTE applied! Training samples: {X_train_bal.shape[0]}")
            except Exception as e:
                st.warning(f"⚠️ SMOTE skipped due to: {e}")
                X_train_bal, y_train_bal = X_train_vec, y_train

            # --- Choose ML Model ---
            model_choice = st.selectbox(
                "🤖 Choose ML Algorithm",
                ["Naive Bayes", "Logistic Regression", "SVM", "Decision Tree", "Random Forest"]
            )

            if model_choice == "Naive Bayes":
                model = MultinomialNB()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "SVM":
                model = SVC(kernel='linear')
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = RandomForestClassifier()

            # --- Train and Evaluate ---
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Accuracy: {acc*100:.2f}%")
            st.text("📊 Classification Report:")
            st.text(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
