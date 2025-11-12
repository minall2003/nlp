# streamlit_nlp_classifier.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

# --------------------------
# NLTK downloads (safe, idempotent)
# --------------------------
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --------------------------
# Lightweight preprocessing utilities
# --------------------------
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
}

def expand_contractions(text: str) -> str:
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return text

_LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    """Robust text cleaning: lower, remove urls, emails, punctuation, digits, expand contractions, tokenize & lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # Keep basic sentence content, remove special characters and digits
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # tokenize + lemmatize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 1]  # drop single letters
    tokens = [ _LEMMATIZER.lemmatize(t) for t in tokens ]
    return " ".join(tokens)

# --------------------------
# Model training / evaluation helper
# --------------------------
@st.cache_data
def build_and_tune_model(X, y, model_name='logreg', cv=4, n_jobs=-1):
    """
    Build a scikit-learn pipeline and run GridSearchCV.
    model_name: 'logreg', 'svc', or 'rf'
    Returns fitted GridSearchCV object (best_estimator_ is ready).
    """
    # Vectorizer + optional dimensionality reduction + classifier
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=3, max_df=0.95)
    svd = TruncatedSVD(n_components=200, random_state=42)  # helps if many features / sparse
    if model_name == 'logreg':
        clf = LogisticRegression(max_iter=5000, solver='saga', random_state=42, class_weight='balanced')
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 5],
            'pipeline__svd__n_components': [100, 200],  # tune SVD size
        }
    elif model_name == 'svc':
        clf = LinearSVC(max_iter=5000, dual=False, class_weight='balanced', random_state=42)
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 5],
            'pipeline__svd__n_components': [100, 200],
        }
    elif model_name == 'rf':
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        param_grid = {
            'clf__n_estimators': [200, 300],
            'clf__max_depth': [None, 30],
            'pipeline__svd__n_components': [100, 200],
        }
    else:
        raise ValueError("Unknown model_name")

    # pipeline: tfidf -> svd -> clf
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svd', svd),
        ('clf', clf)
    ])

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=0,
        refit=True
    )

    gs.fit(X, y)
    return gs

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🧠 NLP Classifier (Improved)", layout="wide")
st.title("🧠 NLP Text Classification — Improved")
st.write("Uploads → cleaning → TF-IDF (uni+bi-grams) → optional SVD → model search (LogReg/SVM/RandomForest).")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("📤 Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### 🔍 Dataset preview", df.head())

text_col = st.selectbox("📝 Select text column", df.columns, index=0)
target_col = st.selectbox("🎯 Select target column", df.columns, index=1)

# Quick checks / encoding
if df[target_col].nunique() < 2:
    st.error("Target column must contain at least 2 classes.")
    st.stop()

# Optional label encoding (for string labels)
le = LabelEncoder()
y_series = le.fit_transform(df[target_col].astype(str))
label_mapping = dict(enumerate(le.classes_))
st.write("Label mapping:", label_mapping)

# Preprocessing progress
if st.button("🚀 Preprocess & Train (takes a little time)"):
    with st.spinner("Cleaning text..."):
        X_raw = df[text_col].astype(str).fillna("")
        # apply cleaning (this can be cached if needed)
        X_clean = X_raw.apply(clean_text)

    st.write("Sample cleaned text:")
    st.dataframe(pd.DataFrame({"original": X_raw.head(5), "cleaned": X_clean.head(5)}))

    # choose model family
    model_choice = st.selectbox("Model family to tune", ["logreg", "svc", "rf"], index=0)

    # If dataset is small, reduce cv folds
    n_samples = len(X_clean)
    cv_folds = 4 if n_samples >= 200 else 3

    with st.spinner("Running GridSearchCV... This can take a few minutes depending on dataset size."):
        try:
            gs = build_and_tune_model(X_clean, y_series, model_name=model_choice, cv=cv_folds, n_jobs= -1)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success("✅ Grid search complete")
    st.write("Best params:", gs.best_params_)
    best_pipeline = gs.best_estimator_

    # Evaluate with a fresh stratified split (holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_series, test_size=0.2, random_state=42, stratify=y_series if len(set(y_series))>1 else None
    )
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Holdout Accuracy:** {acc:.4f}")

    st.write("### 📊 Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    st.write("### 🧩 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("### Notes & next steps to push accuracy higher")
    st.markdown("""
    - **Quality & quantity of data**: More labeled examples and less label noise often yield the biggest gains.
    - **Class balance**: If classes are imbalanced, consider targeted up/down sampling or use of specialized losses.
    - **Feature engineering**: Domain-specific features (e.g., presence of certain tokens, metadata fields) often help more than raw text alone.
    - **Try larger models**: Transformer-based models (BERT, RoBERTa) fine-tuned on your dataset often outperform classical pipelines for many NLP tasks.
    - **Clean labels**: Manually inspect misclassified samples — label errors are common and correcting them yields big improvements.
    - **Ensembling**: Combine multiple models (e.g., logistic + SVM + tree) for more robust predictions.
    """)

    # offer download of the fitted model if desired
    st.download_button(
        label="📥 Download best sklearn pipeline (pickle)",
        data=pd.to_pickle(best_pipeline),
        file_name="best_pipeline.pkl",
        mime="application/octet-stream"
    )

else:
    st.info("Press the 'Preprocess & Train' button to start training.")
