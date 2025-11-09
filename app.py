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
# 📦 Ensure NLTK dependencies are available
# --------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# --------------------------
# ⚙️ Text Preprocessing
# --------------------------
def lexical_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)

    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# --------------------------
# 🧠 Model Evaluation Function
# --------------------------
def evaluate_models(X_features, y):
    # Handle small-class issue safely
    if y.value_counts().min() < 2:
        st.warning("⚠️ One or more target classes have less than 2 samples — skipping stratified split for safety.")
        stratify_opt = None
    else:
        stratify_opt = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=stratify_opt
    )

    # Logistic Regression with balanced class weights (instead of SMOTE)
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
# 🌐 Streamlit UI
# --------------------------
st.set_page_config(page_title="🧠 NLP Classifier", layout="wide")

st.title("🧠 NLP Text Classification App")
st.write("Upload your dataset and automatically train a Logistic Regression model with TF-IDF features.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Dataset", df.head())

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
