# streamlit_nlp_classifier.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --------------------------
# ✅ FIXED NLTK resource handling (works in all environments)
# --------------------------
def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

ensure_nltk_resources()

# --------------------------
# Text Cleaning Utilities
# --------------------------
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
}

def expand_contractions(text):
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return text

_LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    """Clean and tokenize text safely."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        tokens = word_tokenize(text)
    except LookupError:
        ensure_nltk_resources()
        tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --------------------------
# Model Training Function
# --------------------------
@st.cache_data
def build_and_tune_model(X, y, model_name='logreg', cv=4, n_jobs=-1):
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, max_df=0.95)
    svd = TruncatedSVD(n_components=200, random_state=42)
    
    if model_name == 'logreg':
        clf = LogisticRegression(max_iter=5000, solver='saga', random_state=42, class_weight='balanced')
        param_grid = {'clf__C': [0.1, 1, 5]}
    elif model_name == 'svc':
        clf = LinearSVC(max_iter=5000, dual=False, class_weight='balanced', random_state=42)
        param_grid = {'clf__C': [0.1, 1, 5]}
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        param_grid = {'clf__max_depth': [None, 30], 'clf__n_estimators': [200, 300]}

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svd', svd),
        ('clf', clf)
    ])

    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs, refit=True, verbose=0)
    gs.fit(X, y)
    return gs

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🧠 NLP Classifier (Fixed)", layout="wide")
st.title("🧠 NLP Text Classification — Fixed Version")
st.write("Upload CSV → Clean text → Train & Tune Logistic/SVM/RandomForest model automatically.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("📤 Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### 🔍 Dataset Preview", df.head())

text_col = st.selectbox("📝 Select Text Column", df.columns, index=0)
target_col = st.selectbox("🎯 Select Target Column", df.columns, index=1)

if df[target_col].nunique() < 2:
    st.error("❌ Target column must contain at least 2 unique labels.")
    st.stop()

le = LabelEncoder()
y_series = le.fit_transform(df[target_col].astype(str))
label_mapping = dict(enumerate(le.classes_))
st.write("Label mapping:", label_mapping)

if st.button("🚀 Train Model"):
    with st.spinner("Cleaning text..."):
        X_raw = df[text_col].astype(str).fillna("")
        X_clean = X_raw.apply(clean_text)

    st.write("Sample cleaned text:")
    st.dataframe(pd.DataFrame({"original": X_raw.head(5), "cleaned": X_clean.head(5)}))

    model_choice = st.selectbox("Choose model", ["logreg", "svc", "rf"], index=0)
    n_samples = len(X_clean)
    cv_folds = 4 if n_samples >= 200 else 3

    with st.spinner("Training and tuning model... please wait ⏳"):
        try:
            gs = build_and_tune_model(X_clean, y_series, model_name=model_choice, cv=cv_folds)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success("✅ Model training complete!")
    st.write("Best Parameters:", gs.best_params_)
    best_pipeline = gs.best_estimator_

    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.4f}")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    st.write("### 🧩 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.download_button(
        label="📥 Download Trained Model (Pickle)",
        data=pd.to_pickle(best_pipeline),
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )
