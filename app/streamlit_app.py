import streamlit as st
import pandas as pd
import nltk
import re
import os
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
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #00BFFF;
}
.subtitle {
    font-size: 18px;
    color: #A9A9A9;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# NLTK Setup
# -------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -------------------------------
# Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Options")

uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

# -------------------------------
# Load Dataset (Smart)
# -------------------------------
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded ✅")
    elif os.path.exists("data/sample_news.csv"):
        df = pd.read_csv("data/sample_news.csv")
    else:
        raise Exception

except:
    st.sidebar.warning("Using demo dataset ⚠️")
    df = pd.DataFrame({
        "clean_text": [
            "government announces new policy",
            "economy shows growth and stability",
            "shocking conspiracy spreads online",
            "fake rumor spreads rapidly"
        ],
        "label": [1, 1, 0, 0]
    })

# -------------------------------
# Prepare Data
# -------------------------------
df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)

# -------------------------------
# Train Model
# -------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# UI Header
# -------------------------------
st.markdown('<p class="big-title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether a news article is Fake or Real using Machine Learning</p>', unsafe_allow_html=True)

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter News Text")
    user_input = st.text_area("Paste news content here...", height=200)

with col2:
    st.subheader("📊 Model Info")
    st.metric("Dataset Size", len(df))
    st.metric("Features", X.shape[1])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Analyze News"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.markdown('<div class="result-box" style="background-color:#1e7e34;color:white;">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background-color:#c82333;color:white;">❌ FAKE NEWS</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📈 Dataset Insights")

label_counts = df['label'].value_counts()

fig, ax = plt.subplots()
label_counts.plot(kind='bar', ax=ax)
ax.set_xticklabels(["Fake", "Real"])
ax.set_title("Fake vs Real Distribution")

st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 Built with NLP, TF-IDF & Logistic Regression | Streamlit App")