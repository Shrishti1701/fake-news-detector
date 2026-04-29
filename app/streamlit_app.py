import streamlit as st
import pickle
import nltk
import re
import os

from nltk.corpus import stopwords

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

# -------------------------------
# Load Model (CORRECT PATH)
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
# Clean Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# UI
# -------------------------------
st.title("📰 Fake News Detector")
st.write("Detect whether news is Fake or Real using Machine Learning")

user_input = st.text_area("Enter News Text")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        # ✅ CORRECT LABEL MAPPING
        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
    else:
        st.warning("Enter some text")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Built with NLP + TF-IDF + Logistic Regression")