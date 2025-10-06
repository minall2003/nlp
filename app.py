import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def main():
    st.set_page_config(page_title="NLP Model Accuracy Comparator", layout="wide")
    st.title("🧠 NLP Model Accuracy Comparator (All Models)")

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV dataset first.")
        return

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded! Shape: {df.shape}")
    st.dataframe(df.head())

    text_col = st.selectbox("📜 Text Column", df.columns)
    label_col = st.selectbox("🏷️ Label Column", df.columns)

    if st.button("🚀 Train & Compare All Models"):
        # Preprocessing
        X = df[text_col].astype(str).str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.strip()
        y = df[label_col].astype(str)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Remove rare labels for stratify
        y_series = pd.Series(y)
        valid_labels = y_series.value_counts()[y_series.value_counts() > 1].index
        mask = y_series.isin(valid_labels)
        X = X[mask]
        y = y_series[mask]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # TF-IDF Vectorization (bigrams)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # SMOTE for class balancing
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_vec, y_train)

        # Models to compare
        models = {
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "Logistic Regression": LogisticRegression(max_iter=1000, C=2, class_weight='balanced'),
            "SVM": SVC(kernel='linear', C=2, class_weight='balanced'),
            "Decision Tree": DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=30, class_weight='balanced', random_state=42)
        }

        # Training & evaluating all models
        st.info("🏋️ Training models, please wait...")
        results = {}
        for name, model in models.items():
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

        # Display leaderboard
        st.subheader("🏆 Model Accuracy Leaderboard")
        leaderboard = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": [round(v*100,2) for v in results.values()]
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
        st.dataframe(leaderboard)

        st.bar_chart(leaderboard.set_index("Model")["Accuracy (%)"])

        # Show classification report for best model
        best_model_name = leaderboard.iloc[0]["Model"]
        st.success(f"🎯 Best Model: {best_model_name} with Accuracy {leaderboard.iloc[0]['Accuracy (%)']}%")
        best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test_vec)
        st.subheader(f"📄 Classification Report for {best_model_name}")
        st.text(classification_report(y_test, y_pred_best, target_names=le.classes_))

        st.balloons()

if __name__ == "__main__":
    main()
