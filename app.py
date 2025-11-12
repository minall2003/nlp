import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import requests
from bs4 import BeautifulSoup  # <-- NEW IMPORT for PolitiFact scraping
from io import StringIO


# ---------- HELPER FUNCTION: FETCH GOOGLE FACT CHECK DATA ----------
def fetch_google_factcheck(query, api_key):
    """
    Fetches fact-checking results for a given query using Google Fact Check Tools API.
    Returns a list of claims, verdicts, publishers, and URLs.
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": api_key, "pageSize": 3}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("claims", []):
            claim = item.get("text", "")
            review = item.get("claimReview", [{}])[0]
            results.append({
                "Claim": claim,
                "Verdict": review.get("textualRating", "Unknown"),
                "Publisher": review.get("publisher", {}).get("name", "N/A"),
                "URL": review.get("url", "")
            })

        return results

    except Exception as e:
        return [{"error": str(e)}]


# ---------- HELPER FUNCTION: FETCH POLITIFACT DATA ----------
def fetch_politifact_data(pages=2):
    """
    Scrapes PolitiFact fact-check statements and verdicts.
    Automatically handles missing verdicts and HTML changes.
    """
    base_url = "https://www.politifact.com/factchecks/?page="
    all_claims = []

    for page in range(1, pages + 1):
        url = base_url + str(page)
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, "html.parser")

        statements = soup.find_all("li", class_="o-listicle__item")
        for s in statements:
            try:
                statement = s.find("div", class_="m-statement__quote").get_text(strip=True)
            except:
                statement = None

            # 🧩 FIXED VERDICT EXTRACTION
            try:
                meter_div = s.find("div", class_="m-statement__meter")
                verdict = meter_div.find("div", class_="c-meter__rating").get_text(strip=True)
            except:
                verdict = "Not Rated"

            try:
                speaker = s.find("a", class_="m-statement__name").get_text(strip=True)
            except:
                speaker = None

            try:
                date = s.find("footer", class_="m-statement__footer").get_text(strip=True)
            except:
                date = None

            if statement:
                all_claims.append({
                    "statement": statement,
                    "verdict": verdict,
                    "speaker": speaker,
                    "date": date
                })

        time.sleep(1)  # prevent rate limiting

    df = pd.DataFrame(all_claims)
    return df


# ---------- PAGE CONFIG & STYLE ----------
st.set_page_config(page_title="Battle of the Bots: The NLP Showdown", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e2f, #2b2b40);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}
h2, h3 {
    color: #f8f9fa;
    text-shadow: 0 0 10px rgba(255,255,255,0.2);
}
.metric-box {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}
div.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}
.humor-box {
    background: rgba(0,0,0,0.4);
    border-radius: 15px;
    padding: 15px;
    font-style: italic;
    border-left: 5px solid #00c6ff;
}
.plot-container {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


# ---------- SPLASH ----------
with st.spinner("⚙️ Initializing AI humor protocols..."):
    time.sleep(1.5)

st.markdown("""
<div style='text-align:center; margin-bottom: 30px;'>
    <h1>⚔️ Battle of the Bots: The NLP Showdown</h1>
    <p style='color:#cfcfcf;'>Where machine learning models duel for dominance... and get roasted for fun!</p>
</div>
""", unsafe_allow_html=True)


# ---------- POLITIFACT SECTION ----------
st.markdown("### 🗞️ PolitiFact Data (Live Scrape Test)")
if st.button("Fetch PolitiFact Data"):
    with st.spinner("Fetching latest fact-checks from PolitiFact..."):
        df_politifact = fetch_politifact_data(2)
        st.success("✅ Fetched successfully!")
        st.dataframe(df_politifact.head(10))

# ---------- REST OF YOUR EXISTING CODE CONTINUES ----------
# (left_col, center_col, right_col sections remain unchanged)

# ---------- FACT CHECK TEST ----------
st.markdown("### 🔍 Google Fact Check Quick Test")

query = st.text_input("Enter a claim or topic to verify (e.g., 'COVID vaccine safe')")
api_key = st.text_input("Enter your Google Fact Check API Key", type="password")

if st.button("Fetch Fact Check Data"):
    if query and api_key:
        with st.spinner("Fetching fact-check results..."):
            results = fetch_google_factcheck(query, api_key)
            st.success("✅ Results fetched successfully!")
            st.dataframe(pd.DataFrame(results))
    else:
        st.warning("Please enter both query and API key.")

# ---------- FOOTER ----------
st.markdown("""
---
<div style='text-align:center; color: #aaaaaa; font-size: 13px; margin-top: 15px;'>
    Built with ❤️ using Streamlit | Designed by <b>Battle of the Bots</b> Team © 2025
</div>
""", unsafe_allow_html=True)
