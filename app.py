# ============================================================
# 🚀 High-Accuracy NLP Model Comparator (70–80% Target Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ==============================================
# TEXT CLEANING + FEATURE FUNCTIONS
# ==============================================
def clean_text(text):
    """Clean text: lowercase, remove punctuation, URLs, digits."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_extra_features(texts):
    """Extract numeric features: sentiment + length stats."""
    feats = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        feats.append([
            s["neg"], s["neu"], s["pos"], s["compound"],
            len(str(t).split()), len(str(t))
        ])
    return np.array(feats)


# ==============================================
# APP START
# ==============================================
def main():
    st.set_page_config(page_title="NLP Accuracy Booster", layout="wide")
    st.title("🧠 NLP Model Comparator (70–80% Accuracy Version)")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV dataset first.")
        return

    df = pd.read_csv(uploaded_file)
    st.write("👀 Preview of Data:")
    st.dataframe(df.head())

    text_col = st.selectbox("📜 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    if st.button("🚀 Train All Models"):
        X = df[text_col].astype(str).apply(clean_text)
        y = df[label_col].astype(str)

        if len(np.unique(y)) < 2:
            st.error("❌ Need at least two unique labels.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # TF-IDF features
        st.info("🔠 Generating text embeddings...")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2)
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Extra sentiment/length features
        st.info("🧩 Adding sentiment & length features...")
        train_extra = get_extra_features(X_train)
        test_extra = get_extra_features(X_test)

        # Combine features
        X_train_combined = np.hstack((X_train_tfidf.toarray(), train_extra))
        X_test_combined = np.hstack((X_test_tfidf.toarray(), test_extra))

        # Optional SMOTE
        try:
            st.info("⚖️ Applying SMOTE balancing...")
            sm = SMOTE(random_state=42)
            X_train_combined, y_train = sm.fit_resample(X_train_combined, y_train)
        except Exception as e:
            st.warning(f"⚠️ SMOTE skipped: {e}")

        # Scale features for linear models
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = scaler.transform(X_test_combined)

        # ===============================
        # MODELS
        # ===============================
        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(C=4, max_iter=3000, solver='lbfgs'),
            "SVM": LinearSVC(C=2, max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(max_depth=40, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=50, random_state=42)
        }

        # Ensemble Voting Classifier (Best Combo)
        ensemble = VotingClassifier(
            estimators=[
                ('lr', models["Logistic Regression"]),
                ('rf', models["Random Forest"]),
                ('svm', models["SVM"])
            ],
            voting='hard'
        )
        models["Ensemble (LR + SVM + RF)"] = ensemble

        # ===============================
        # TRAIN & EVALUATE
        # ===============================
        results = {}
        progress = st.progress(0)
        total = len(models)
        for i, (name, model) in enumerate(models.items(), 1):
            with st.spinner(f"Training {name}..."):
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds)
                results[name] = acc
                progress.progress(int(i / total * 100))

        # ===============================
        # RESULTS
        # ===============================
        st.success("✅ Training Completed!")

        result_df = pd.DataFrame(
            {"Model": list(results.keys()), "Accuracy": [f"{a*100:.2f}%" for a in results.values()]}
        ).sort_values("Accuracy", ascending=False)

        st.subheader("🏆 Model Accuracy Comparison")
        st.dataframe(result_df, use_container_width=True)

        chart_data = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        chart_data["Accuracy"] *= 100
        st.bar_chart(chart_data.set_index("Model"))

        best_model = max(results, key=results.get)
        st.success(f"🎯 Best Model: **{best_model}** with accuracy **{results[best_model]*100:.2f}%**")

        # Show detailed report for best model
        st.subheader("📄 Classification Report for Best Model")
        best = models[best_model]
        y_pred_best = best.predict(X_test_scaled)
        st.text(classification_report(y_test, y_pred_best))

        st.balloons()


if __name__ == "__main__":
    main()
