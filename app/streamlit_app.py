import streamlit as st
import pickle
import nltk
import re
import os
import requests

from nltk.corpus import stopwords

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

API_KEY = "YOUR_API_KEY_HERE"  # 🔥 ADD YOUR KEY

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# -------------------------------
# NLTK
# -------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# NEWS API CHECK
# -------------------------------
def check_news_api(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "ok" and data["totalResults"] > 0:
        return True, data["articles"][:3]  # return top 3 articles
    return False, []

# -------------------------------
# UI
# -------------------------------
st.title("📰 Fake News Detector (Hybrid AI + API)")
st.write("Combines Machine Learning + Real-time News Verification")

user_input = st.text_area("Enter News Text")

# -------------------------------
# ANALYZE
# -------------------------------
if st.button("Analyze"):
    if user_input.strip() != "":
        
        # ML Prediction
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        ml_result = "REAL" if prediction == 1 else "FAKE"

        # API Check
        found, articles = check_news_api(user_input)

        # -------------------------------
        # FINAL DECISION
        # -------------------------------
        st.subheader("🧠 ML Prediction")
        if prediction == 1:
            st.success("Model: REAL")
        else:
            st.error("Model: FAKE")

        st.subheader("🌐 News API Verification")

        if found:
            st.success("News found in real sources ✅")
            for article in articles:
                st.write(f"🔗 {article['title']}")
        else:
            st.warning("No matching news found ⚠️")

        # -------------------------------
        # FINAL RESULT
        # -------------------------------
        st.subheader("🎯 Final Verdict")

        if prediction == 1 and found:
            st.success("✅ Highly Likely REAL NEWS")
        elif prediction == 0 and not found:
            st.error("❌ Highly Likely FAKE NEWS")
        else:
            st.warning("⚠️ Uncertain (Mixed Signals)")

    else:
        st.warning("Enter some text")