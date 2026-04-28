import streamlit as st
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    page_icon="📰"
)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    h1, h2 {
        color: #00BFFF;
    }
    .stButton>button {
        background-color: #00BFFF;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Setup
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/cleaned_news.csv")
df = df.dropna(subset=['clean_text'])

# -------------------------------
# Model Training
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About")
st.sidebar.info("""
This app detects whether a news article is FAKE or REAL using NLP and Machine Learning.
""")

st.sidebar.subheader("Example Input")
st.sidebar.write("Breaking: Government announces new policy to boost economy.")

# -------------------------------
# Main UI
# -------------------------------
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is **Fake or Real** using AI")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Enter News Text")
    user_input = st.text_area("")

with col2:
    st.subheader("📌 Prediction")

    if st.button("Analyze News"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.success("✅ This News is REAL")
            else:
                st.error("❌ This News is FAKE")
        else:
            st.warning("Please enter some text")

# -------------------------------
# Data Visualization
# -------------------------------
st.subheader("📊 Dataset Distribution")

label_counts = df['label'].value_counts()

fig, ax = plt.subplots()
label_counts.plot(kind='bar', ax=ax)
ax.set_xticklabels(["Fake", "Real"])
ax.set_title("Fake vs Real News Distribution")

st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 Built using NLP + Machine Learning | Streamlit App")