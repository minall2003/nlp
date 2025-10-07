import streamlit as st
import pandas as pd
import numpy as np
import re, string, emoji, spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

nlp = spacy.load("en_core_web_sm")

# ------------------ Text Cleaning ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = emoji.replace_emoji(text, replace="")          # remove emojis
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized.strip()

# ------------------ Streamlit App ------------------
def main():
    st.set_page_config(page_title="🚀 NLP Model Accuracy Comparator (Pro)", layout="wide")
    st.title("🧠 NLP Model Accuracy Comparator — Optimized Version")

    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV dataset first.")
        return

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded! Shape: {df.shape}")
    st.dataframe(df.head())

    text_col = st.selectbox("📜 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    if st.button("🚀 Train & Compare Models (Optimized)"):
        st.info("🧹 Cleaning and preprocessing text data...")
        df[text_col] = df[text_col].astype(str).apply(clean_text)
        df = df[df[text_col].str.strip() != ""]

        X = df[text_col]
        y = df[label_col].astype(str)

        le = LabelEncoder()
        y = le.fit_transform(y)

        y_series = pd.Series(y)
        valid_labels = y_series.value_counts()[y_series.value_counts() > 1].index
        mask = y_series.isin(valid_labels)
        X = X[mask]
        y = y_series[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        st.info("🔠 Applying TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,3),
            sublinear_tf=True,
            stop_words="english"
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Feature selection
        st.info("📊 Selecting top features (Chi2)...")
        selector = SelectKBest(chi2, k=min(8000, X_train_vec.shape[1]))
        X_train_vec = selector.fit_transform(X_train_vec, y_train)
        X_test_vec = selector.transform(X_test_vec)

        # Balance classes
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_vec, y_train)

        st.info("🏋️ Training models, please wait...")
        progress = st.progress(0)

        models = {
            "Naive Bayes": MultinomialNB(alpha=0.05),
            "Logistic Regression": LogisticRegression(max_iter=2000, C=3, class_weight="balanced"),
            "SVM": LinearSVC(C=1.5, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=50, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=50, class_weight="balanced", random_state=42
            )
        }

        results = {}
        total = len(models)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            progress.progress((i + 1) / total)

        leaderboard = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": [round(v * 100, 2) for v in results.values()]
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        st.subheader("🏆 Model Accuracy Leaderboard")
        st.dataframe(leaderboard)
        st.bar_chart(leaderboard.set_index("Model")["Accuracy (%)"])

        best_model_name = leaderboard.iloc[0]["Model"]
        best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test_vec)

        labels_in_test = np.unique(y_test)
        target_names = [le.classes_[i] for i in labels_in_test]
        st.success(f"🎯 Best Model: {best_model_name} with Accuracy {leaderboard.iloc[0]['Accuracy (%)']}%")

        st.subheader(f"📄 Classification Report for {best_model_name}")
        st.text(classification_report(y_test, y_pred_best, labels=labels_in_test, target_names=target_names))

        st.balloons()

if __name__ == "__main__":
    main()
