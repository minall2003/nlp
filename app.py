import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# NLTK downloads (safe)
# --------------------------
def ensure_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "wordnet",
        "omw-1.4",
        "stopwords",
    ]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}") if "punkt" in r else nltk.data.find(f"corpora/{r}")
        except LookupError:
            nltk.download(r, quiet=True)

ensure_nltk_resources()

# --------------------------
# Text Preprocessing
# --------------------------
def lexical_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# --------------------------
# Model Evaluation Function
# --------------------------
def evaluate_models(X_features, y):
    if y.value_counts().min() < 2:
        st.warning("⚠️ One or more target classes have less than 2 samples — skipping stratified split.")
        stratify_opt = None
    else:
        stratify_opt = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=stratify_opt
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': acc,
        'report': cls_report,
        'confusion_matrix': conf_mat,
    }

# --------------------------
# Streamlit UI Configuration
# --------------------------
st.set_page_config(page_title="⚡ AI vs. Fact: NLP Comparator", layout="wide")

# --------------------------
# 🌈 Animated Gradient Title
# --------------------------
st.markdown("""
<style>
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

h1.gradient-text {
  font-size: 58px;
  font-weight: 900;
  letter-spacing: 1px;
  background: linear-gradient(270deg, #00f5ff, #0072ff, #ff00c8);
  background-size: 600% 600%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradientShift 6s ease infinite;
  text-shadow: 0px 0px 20px rgba(255,255,255,0.25);
}

p.subtitle {
  color: #e5e5e5;
  font-size: 18px;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-shadow: 0 0 8px rgba(0,255,255,0.4);
}
</style>

<div style='text-align:center; margin-bottom: 40px;'>
    <h1 class='gradient-text'>⚡ AI vs. Fact ⚖️</h1>
    <p class='subtitle'>Where Algorithms Battle Truth... and Only the Wittiest Survive 🤖🔥</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# File Upload Section
# --------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataset Preview", df.head())

    text_col = st.selectbox("📝 Select Text Column", df.columns)
    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Train Model"):
        with st.spinner("Preprocessing text and training model..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            # Text preprocessing
            X_processed = X.apply(lexical_preprocess)

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=5000)
            X_features = vectorizer.fit_transform(X_processed)

            # Model evaluation
            results = evaluate_models(X_features, y)

        st.success("✅ Model trained successfully!")
        st.write(f"**Accuracy:** {results['accuracy']:.2f}")

        st.write("### 📊 Classification Report")
        st.dataframe(pd.DataFrame(results['report']).transpose())

        st.write("### 🧩 Confusion Matrix")
        st.write(results['confusion_matrix'])
else:
    st.info("📤 Please upload a CSV file to begin.")
