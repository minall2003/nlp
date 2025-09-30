import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# keep your original imports + functions here (not repeated for brevity)
from streamlit_nlp_models_app import evaluate_phase  # assuming core logic in same file

warnings.filterwarnings("ignore")

# ------------------ Streamlit UI ------------------
def main():
    st.set_page_config(page_title='🧠 NLP Phases vs ML Models', layout='wide')

    st.title('🧠 NLP Phase vs ML Model Comparator')
    st.markdown(
        """
        This app lets you **compare multiple NLP preprocessing phases** (Lexical, Semantic, Synaptic, Pragmatic, Disclosure Integration) 
        across **different ML classifiers**. 🚀
        
        Upload your dataset, pick a text + label column, choose a model, and visualize results!
        """
    )

    with st.sidebar:
        st.header('⚙️ Configuration')
        with st.expander("📂 Dataset Options", expanded=True):
            uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
            sample_n = st.number_input('Max rows to use (0 = all)', min_value=0, value=2000, step=500)

        with st.expander("🤖 Model Options", expanded=True):
            classifier_name = st.selectbox('Choose ML algorithm', [
                'Naive Bayes Classification',
                'Decision Tree Classification',
                'Support Vector Machine',
                'Logistic Regression',
                'K - Nearest Neighbour'
            ])

        run_button = st.button('🚀 Run Experiment')

    # load dataframe
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
            return
    else:
        default_path = '/mnt/data/politifact_full.csv'
        try:
            df = pd.read_csv(default_path)
            st.info(f"📂 Loaded default dataset from {default_path}")
        except Exception:
            st.warning('⚠️ Please upload a CSV file (or place politifact_full.csv at /mnt/data).')

    if df is not None:
        st.subheader('👀 Data Preview')
        st.write('Shape: ', df.shape)
        st.dataframe(df.head())

        cols = df.columns.tolist()
        text_col = st.selectbox('📜 Select Text Column (input)', cols, index=0)
        label_col = st.selectbox('🏷️ Select Label Column (target)', cols, index=min(1, len(cols)-1))

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

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            phases = ['Lexical', 'Semantic', 'Synaptic', 'Pragmatic', 'Discloser Integration']
            results = {}

            progress = st.progress(0)
            for i, phase in enumerate(phases, start=1):
                with st.spinner(f'Training {phase} pipeline with {classifier_name}...'):
                    try:
                        res = evaluate_phase(phase, classifier_name, X_train, X_test, y_train, y_test)
                        results[phase] = res
                    except Exception as e:
                        st.error(f'Error while processing phase {phase}: {e}')
                        results[phase] = {'accuracy': 0.0, 'error': str(e)}
                progress.progress(int(i/len(phases)*100))

            # --- Results Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Accuracy", "📑 Reports", "🔢 Confusion Matrices"])

            with tab1:
                st.subheader('📊 Accuracy Comparison')
                accs = {p: float(results[p]['accuracy']) if 'accuracy' in results[p] else 0.0 for p in phases}
                acc_df = pd.DataFrame.from_dict(accs, orient='index', columns=['accuracy']).sort_values('accuracy', ascending=False)
                col1, col2 = st.columns([1,2])
                with col1:
                    st.table(acc_df)
                with col2:
                    st.bar_chart(acc_df['accuracy'])

            with tab2:
                st.subheader('📑 Classification Reports')
                for phase in phases:
                    st.markdown(f"### 🔹 {phase}")
                    item = results[phase]
                    if 'error' in item:
                        st.error(f"Phase failed: {item['error']}")
                        continue
                    report_df = pd.DataFrame(item['report']).transpose()
                    st.dataframe(report_df)

            with tab3:
                st.subheader('🔢 Confusion Matrices')
                for phase in phases:
                    st.markdown(f"### 🔹 {phase}")
                    item = results[phase]
                    if 'error' in item:
                        st.error(f"Phase failed: {item['error']}")
                        continue
                    st.write(item['confusion_matrix'])

    st.markdown('---')
    st.markdown('✅ **Notes & Tips**')
    st.markdown(
        """
        - 🚫 Avoids `nltk.word_tokenize`, so no `punkt_tab` errors.
        - Use fewer rows (e.g., 1000–2000) for quicker tests.
        - Try different ML models in sidebar for performance comparison.
        - If you’d like caching or per-class metrics visualization → let me know! 🎯
        """
    )

if __name__ == '__main__':
    main()
