import os
import streamlit as st
import pandas as pd
import numpy as np
import re, string
import warnings
warnings.filterwarnings("ignore")

# ✅ Auto-install required packages (only if missing)
os.system('pip install -q nltk imbalanced-learn scikit-learn pandas numpy')

# --- NLTK setup (fast + safe)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Download stopwords only if not already present
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="⚡ Fast High-Accuracy NLP Comparator", layout="wide")
    st.title("🧠 High-Accuracy NLP Model Comparator — Optimized & Fast 🚀")

    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a dataset first.")
        return

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
    st.dataframe(df.head())

    text_col = st.selectbox("📜 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    if st.button("🚀 Train Models"):
        st.info("🧹 Cleaning text data...")
        df[text_col] = df[text_col].astype(str).apply(clean_text)
        df = df[df[text_col].str.strip() != ""]

        X = df[text_col]
        y = df[label_col].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # ✅ Remove classes with only one sample
        y_series = pd.Series(y)
        valid_labels = y_series.value_counts()[y_series.value_counts() > 1].index
        mask = y_series.isin(valid_labels)
        X, y = X[mask], y_series[mask]

        # ✅ Smaller dataset sampling for speed (optional)
        if len(X) > 10000:
            X = X.sample(10000, random_state=42)
            y = y.loc[X.index]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        st.info("🔠 Vectorizing text with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english"
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        st.info("⚖️ Applying SMOTE for balancing...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        # --- Optimized Models (Fast & Accurate)
        models = {
            "Naive Bayes": MultinomialNB(alpha=0.05),
            "Logistic Regression": LogisticRegression(max_iter=1000, C=3, class_weight="balanced"),
            "SVM": LinearSVC(C=2, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=50,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}
        progress = st.progress(0)
        total = len(models)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        st.info("🏋️ Training all models (optimized speed mode)...")
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_vec)
            acc = round(accuracy_score(y_test, y_pred) * 100, 2)
            cv_acc = round(np.mean(cross_val_score(model, X_train_res, y_train_res, cv=skf)) * 100, 2)
            results[name] = (acc, cv_acc)
            progress.progress((i + 1) / total)

        leaderboard = pd.DataFrame({
            "Model": list(results.keys()),
            "Test Accuracy (%)": [v[0] for v in results.values()],
            "CV Accuracy (%)": [v[1] for v in results.values()]
        }).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)

        st.subheader("🏆 Model Accuracy Leaderboard")
        st.dataframe(leaderboard)
        st.bar_chart(leaderboard.set_index("Model")[["Test Accuracy (%)", "CV Accuracy (%)"]])

        # --- Fast Stacking Ensemble (Final Boost)
        st.info("🧩 Combining top models into a stacking ensemble...")
        top_models = [
            ('lr', models["Logistic Regression"]),
            ('svm', models["SVM"]),
            ('rf', models["Random Forest"])
        ]

        stack = StackingClassifier(
            estimators=top_models,
            final_estimator=LogisticRegression(max_iter=2000),
            n_jobs=-1
        )
        stack.fit(X_train_res, y_train_res)
        y_pred_stack = stack.predict(X_test_vec)
        stack_acc = round(accuracy_score(y_test, y_pred_stack) * 100, 2)

        st.success(f"🎯 Stacking Ensemble Accuracy: {stack_acc}%")
        st.text(classification_report(y_test, y_pred_stack, digits=3, zero_division=0))

        if stack_acc > leaderboard.iloc[0]["Test Accuracy (%)"]:
            st.balloons()
            st.success("🚀 Ensemble achieved highest accuracy!")

if __name__ == "__main__":
    main()
