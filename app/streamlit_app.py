import streamlit as st
import pickle
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

# -------------------------------
# Load Model (IMPORTANT)
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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
st.write("Detect whether news is Fake or Real using ML")

user_input = st.text_area("Enter News Text")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
    else:
        st.warning("Enter some text")

# -------------------------------
# Dummy Visualization (optional)
# -------------------------------
st.subheader("Sample Distribution")

labels = ["Fake", "Real"]
values = [50, 50]

fig, ax = plt.subplots()
ax.bar(labels, values)
st.pyplot(fig)