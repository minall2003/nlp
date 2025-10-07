import streamlit as st
import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
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

# optional modules
try:
    import emoji
except ImportError:
    emoji = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    if emoji:
        text = emoji.replace_emoji(text, replace="")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    if nlp:
        doc = nlp(text)
        text = " ".join([t.lemma_ for t in doc if not t.is_stop])
    return text


# -------------------- Main Streamlit App --------------------
def main():
    st.set_page_config(page_title="🔥 High Accuracy NLP Comparator", layout="wide")
    st.title("🧠 NLP Model Accuracy Comparator — Enhanced with SMOTE + Random Forest Boost")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a dataset first.")
        return

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
    st.dataframe(df.head())

    text_col = st.selectbox("📜 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    if st.button("🚀 Train & Compare Models"):
        st.info("🧹 Cleaning text data — this may take a few seconds...")
        df[text_col] = df[text_col].astype(str).apply(clean_text)
        df = df[df[text_col].str.strip() != ""]

        X = df[text_col]
        y = df[label_col].astype(str)

        le = LabelEncoder()
        y = le.fit_transform(y)

        # Keep only labels with more than one sample
        y_series = pd.Series(y)
        valid_labels = y_series.value_counts()[y_series.value_counts() > 1].index
        mask = y_series.isin(valid_labels)
        X, y = X[mask], y_series[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        st.info("🔠 Applying TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english"
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        st.info("📊 Feature Selection (Chi2)...")
        selector = SelectKBest(chi2, k=min(10000, X_train_vec.shape[1]))
        X_train_vec = selector.fit_transform(X_train_vec, y_train)
        X_test_vec = selector.transform(X_test_vec)

        # ✅ Apply SMOTE for balanced training
        st.info("⚖️ Balancing dataset using SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

        # -------------------- Models --------------------
        st.info("🏋️ Training all models, please wait...")
        progress = st.progress(0)
        models = {
            "Naive Bayes": MultinomialNB(alpha=0.05),
            "Logistic Regression": LogisticRegression(max_iter=3000, C=3, class_weight="balanced"),
            "SVM": LinearSVC(C=2, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=60, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=500,
                max_depth=70,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}
        total = len(models)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_vec)
            acc = round(accuracy_score(y_test, y_pred) * 100, 2)
            results[name] = acc
            progress.progress((i + 1) / total)

        leaderboard = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": list(results.values())
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        st.subheader("🏆 Model Accuracy Leaderboard")
        st.dataframe(leaderboard)
        st.bar_chart(leaderboard.set_index("Model")["Accuracy (%)"])

        # -------------------- Best Model Report --------------------
        best_model_name = leaderboard.iloc[0]["Model"]
        best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test_vec)

        labels_in_test = np.unique(y_test)
        target_names = [le.classes_[i] for i in labels_in_test]
        st.success(f"🎯 Best Model: {best_model_name} with {leaderboard.iloc[0]['Accuracy (%)']}% accuracy")

        st.subheader(f"📄 Classification Report for {best_model_name}")
        st.text(classification_report(
            y_test, y_pred_best, labels=labels_in_test,
            target_names=target_names, digits=3, zero_division=0
        ))

        # 🌳 If Random Forest is best — show feature importance
        if best_model_name == "Random Forest":
            st.subheader("🌲 Top Feature Importance (Random Forest)")
            importances = best_model.feature_importances_
            feature_names = np.array(vectorizer.get_feature_names_out())[selector.get_support()]
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(20)
            st.dataframe(importance_df)

        st.balloons()


if __name__ == "__main__":
    main()
