"""
streamlit_nlp_models_app.py

Updated Streamlit app (safe & enhanced):
- Implements five ML classifiers (NB, Decision Tree, SVM, Logistic, KNN)
- Implements five NLP phases (Lexical, Semantic, Synaptic, Pragmatic, Discloser Integration)
- Auto-handles stratify errors (rare labels)
- Uses regex tokenization (no punkt errors)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------ NLTK downloads (safe) ------------------
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    pass
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception:
    pass

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ Safe utilities ------------------
def safe_pos_tag(tokens):
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            return nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            return nltk.pos_tag(tokens)

def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def regex_tokenize(text: str):
    text = clean_text(text)
    if not text:
        return []
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def simple_tokenize(text: str):
    return regex_tokenize(text)

def lemma_tokenize(text: str):
    text = clean_text(text)
    if not text:
        return []
    tokens = text.split()
    pos_tags = safe_pos_tag(tokens)
    lemmas = []
    for token, pos in pos_tags:
        wn_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token, pos=wn_pos)
        if lemma not in STOPWORDS and len(lemma) > 1:
            lemmas.append(lemma)
    return lemmas

def pos_tokenize(text: str):
    text = clean_text(text)
    if not text:
        return []
    tokens = text.split()
    pos_tags = safe_pos_tag(tokens)
    tokens_with_pos = [f"{w}_{p}" for w, p in pos_tags if w not in STOPWORDS and len(w) > 1]
    return tokens_with_pos

def lemma_synonym_tokenize(text: str):
    tokens = lemma_tokenize(text)
    expanded = []
    for token in tokens:
        expanded.append(token)
        try:
            syns = wordnet.synsets(token)
            if syns:
                for lemma in syns[0].lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name != token:
                        expanded.append(name)
                        break
        except Exception:
            pass
    return expanded

# ------------------ sklearn Transformers ------------------
class SentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for doc in X:
            s = self.sid.polarity_scores(str(doc))
            compound_scaled = (s['compound'] + 1.0) / 2.0
            words = re.findall(r"[a-z]+", str(doc).lower())
            doc_len = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if doc_len > 0 else 0.0
            feats.append([s['neg'], s['neu'], s['pos'], compound_scaled, doc_len, avg_word_len])
        return sparse.csr_matrix(np.array(feats))

# ------------------ classifier factory ------------------
def get_classifier(name: str):
    name = name.lower()
    if 'naive' in name:
        return MultinomialNB()
    elif 'decision' in name or 'tree' in name:
        return DecisionTreeClassifier(random_state=42)
    elif 'svc' in name or 'svm' in name or 'support' in name:
        return LinearSVC(max_iter=10000, random_state=42)
    elif 'logistic' in name:
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    elif 'knn' in name or 'nearest' in name:
        return KNeighborsClassifier()
    else:
        raise ValueError(f"Unknown classifier: {name}")

# ------------------ pipelines per NLP phase ------------------
def build_pipeline(phase: str, classifier_name: str):
    clf = get_classifier(classifier_name)
    if phase == 'Lexical':
        vect = CountVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False)
        return Pipeline([('vect', vect), ('clf', clf)])
    elif phase == 'Semantic':
        vect = TfidfVectorizer(tokenizer=lemma_synonym_tokenize, token_pattern=None, lowercase=False)
        return Pipeline([('vect', vect), ('clf', clf)])
    elif phase == 'Synaptic':
        vect = TfidfVectorizer(tokenizer=pos_tokenize, token_pattern=None, lowercase=False, ngram_range=(1,2), max_features=50000)
        return Pipeline([('vect', vect), ('clf', clf)])
    elif phase == 'Pragmatic':
        union = FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False, max_features=50000)),
            ('sent', SentimentTransformer())
        ])
        return Pipeline([('union', union), ('clf', clf)])
    elif phase == 'Discloser Integration':
        union = FeatureUnion([
            ('lex', TfidfVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False, max_features=30000)),
            ('sem', TfidfVectorizer(tokenizer=lemma_synonym_tokenize, token_pattern=None, lowercase=False, max_features=30000)),
            ('syn', TfidfVectorizer(tokenizer=pos_tokenize, token_pattern=None, lowercase=False, ngram_range=(1,2), max_features=30000)),
            ('sent', SentimentTransformer())
        ])
        return Pipeline([('union', union), ('clf', clf)])
    else:
        raise ValueError(f"Unknown NLP phase: {phase}")

# ------------------ evaluation ------------------
def evaluate_phase(phase, classifier_name, X_train, X_test, y_train, y_test):
    pipe = build_pipeline(phase, classifier_name)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    return {'pipeline': pipe, 'accuracy': acc, 'report': report, 'confusion_matrix': cm, 'preds': preds}

# ------------------ Streamlit UI ------------------
def main():
    st.set_page_config(page_title='🧠 NLP Phase vs ML Model Comparator', page_icon="🤖", layout='wide')
    st.title('🧠 NLP Phase vs ML Model Comparator')
    st.markdown("""
    Upload your CSV, pick the text & label columns, choose one ML algorithm,
    and the app will train it using **five different NLP phases**.  
    Compare accuracy and see detailed reports! 🚀
    """)

    with st.sidebar:
        st.header('⚙️ Configuration')
        uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
        sample_n = st.number_input('Max rows to use (0 = all)', min_value=0, value=2000, step=500)
        classifier_name = st.selectbox('Choose ML algorithm', [
            'Naive Bayes Classification',
            'Decision Tree Classification',
            'Support Vector Machine',
            'Logistic Regression',
            'K - Nearest Neighbour'
        ])
        run_button = st.button('🚀 Run Experiment')

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
            return
    else:
        try:
            df = pd.read_csv('/mnt/data/politifact_full.csv')
            st.info("📂 Loaded default dataset from /mnt/data/politifact_full.csv")
        except:
            st.warning("⚠️ Please upload a CSV file (or place politifact_full.csv at /mnt/data).")

    if df is not None:
        st.subheader('👀 Data Preview')
        st.write('Shape: ', df.shape)
        st.dataframe(df.head())

        cols = df.columns.tolist()
        text_col = st.selectbox('📜 Text column', cols, index=0)
        label_col = st.selectbox('🏷️ Label column', cols, index=min(1, len(cols)-1))

        if run_button:
            data = df[[text_col, label_col]].dropna()
            if sample_n and sample_n > 0:
                data = data.sample(min(sample_n, len(data)), random_state=42)
            data = data.reset_index(drop=True)

            X = data[text_col].astype(str)
            y_raw = data[label_col]

            le = LabelEncoder()
            try:
                y = le.fit_transform(y_raw.astype(str))
            except Exception:
                y = y_raw

            if len(np.unique(y)) < 2:
                st.error('❌ Need at least two classes in the target label.')
                return

            # --- Safe train/test split ---
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
                    st.warning("⚠️ Stratified split failed — using random split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

            st.info(f"✅ Using {len(X)} samples across {len(np.unique(y))} unique labels after cleaning.")

            phases = ['Lexical', 'Semantic', 'Synaptic', 'Pragmatic', 'Discloser Integration']
            results = {}

            progress = st.progress(0)
            total = len(phases)
            for i, phase in enumerate(phases, start=1):
                with st.spinner(f'Training {phase} with {classifier_name}...'):
                    res = evaluate_phase(phase, classifier_name, X_train, X_test, y_train, y_test)
                    results[phase] = res
                progress.progress(int(i / total * 100))

            tab1, tab2, tab3 = st.tabs(["📊 Accuracy", "📑 Reports", "🔢 Confusion Matrices"])

            with tab1:
                accs = {p: results[p]['accuracy'] for p in phases}
                acc_df = pd.DataFrame.from_dict(accs, orient='index', columns=['accuracy']).sort_values('accuracy', ascending=False)
                st.subheader("📊 Accuracy per Phase")
                st.table(acc_df)
                st.bar_chart(acc_df['accuracy'])

            with tab2:
                st.subheader("📑 Classification Reports")
                for phase in phases:
                    st.markdown(f"### 🔹 {phase}")
                    report_df = pd.DataFrame(results[phase]['report']).transpose()
                    st.dataframe(report_df)

            with tab3:
                st.subheader("🔢 Confusion Matrices")
                for phase in phases:
                    st.markdown(f"### 🔹 {phase}")
                    st.write(results[phase]['confusion_matrix'])

if __name__ == '__main__':
    main()
