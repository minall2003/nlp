# ============================================
# 🌈 Streamlit NLP Phase-wise Model Comparison (Optimized)
# ============================================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ============================
# NLTK Setup
# ============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================
# Cleaning & Feature Functions
# ============================
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def lexical_preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

def syntactic_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return " ".join([tag for _, tag in pos_tags])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    sents = sent_tokenize(text)
    return f"{len(sents)} {' '.join([s.split()[0] for s in sents if s])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Model Evaluation
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel='linear', probability=True)
    }

    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

    # Handle imbalance if any
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except:
        pass

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase-wise Analysis", layout="wide")
st.title("🧠 NLP Phase-wise Model Comparison (Optimized)")
st.markdown("<p style='color:gray;'>Upload a dataset, choose an NLP phase, and compare multiple ML models with improved preprocessing.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.write("---")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Select Text Column", df.columns)
    with col2:
        target_col = st.selectbox("Select Target Column", df.columns)

    phase = st.selectbox(
        "🔎 Choose NLP Phase",
        ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
    )

    if st.button("🚀 Run Model Comparison"):
        X = df[text_col].astype(str)
        y = df[target_col]

        # Feature extraction
        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_preprocess)
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X_features = vectorizer.fit_transform(X_processed)

        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            X_features = TfidfVectorizer(max_features=4000).fit_transform(X_processed)

        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                      columns=["polarity", "subjectivity"])
            scaler = StandardScaler()
            X_features = scaler.fit_transform(X_features)

        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            X_features = TfidfVectorizer(max_features=4000).fit_transform(X_processed)

        else:  # Pragmatic
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                      columns=pragmatic_words)
            scaler = StandardScaler()
            X_features = scaler.fit_transform(X_features)

        # Evaluate models
        results = evaluate_models(X_features, y)
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df[results_df["Accuracy"].apply(lambda x: isinstance(x, (int,float)))]
        results_df = results_df.sort_values(by="Accuracy", ascending=False)

        st.subheader("🏆 Model Accuracy")
        st.table(results_df)

        # Plot
        plt.figure(figsize=(7, 4))
        plt.bar(results_df["Model"], results_df["Accuracy"], color="#FF9800", alpha=0.8)
        plt.ylabel("Accuracy (%)")
        plt.title(f"Performance on {phase}")
        for i, v in enumerate(results_df["Accuracy"]):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        st.pyplot(plt)
else:
    st.info("⬅️ Please upload a CSV file to start.")
