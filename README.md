# 📰 Fake News Detector (ML + Real-Time API Verification)

An intelligent web application that detects whether a news article is **Real or Fake** using Machine Learning and enhances reliability with **real-time news verification via API**.

---

## 🚀 Live Demo

🔗 https://fake-news-detector-fefxyybivnmzeudmr28vvk.streamlit.app/

---

## 📌 Features

* 🔍 Detects Fake vs Real News using Machine Learning
* 🌐 Verifies news using real-time News API
* 🧠 Hybrid system (ML + API validation)
* 📊 Dataset insights visualization
* 📁 Upload custom dataset support
* ⚡ Fast & interactive Streamlit UI

---

## 🧠 How It Works

### 1. Text Processing

* Cleans input text (removes punctuation, stopwords, etc.)

### 2. Feature Extraction

* Uses **TF-IDF Vectorization** to convert text into numerical features

### 3. Machine Learning Model

* **Logistic Regression** model trained on labeled dataset

### 4. Real-Time Verification

* Uses News API to check if similar news exists online

### 5. Final Verdict

| ML Prediction | API Result | Final Output      |
| ------------- | ---------- | ----------------- |
| Real          | Found      | ✅ Highly Reliable |
| Fake          | Not Found  | ❌ Likely Fake     |
| Mixed         | Partial    | ⚠️ Uncertain      |

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NLTK
* Matplotlib
* Requests (API integration)

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app/
│   └── streamlit_app.py     # Main Streamlit app
│
├── data/                    # Dataset (ignored in GitHub)
│
├── src/
│   ├── model.py             # Model training script
│   └── preprocessing.py     # Text cleaning logic
│
├── .streamlit/
│   └── secrets.toml         # API key (not uploaded)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔐 API Setup (IMPORTANT)

This project uses News API for real-time verification.

### Step 1: Get API Key

👉 https://newsapi.org/

### Step 2: Create file

```
.streamlit/secrets.toml
```

### Step 3: Add this

```toml
NEWS_API_KEY = "your_api_key_here"
```

### ⚠️ Do NOT upload this file to GitHub

---

## ⚙️ Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run app

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Sample Output

* ✅ REAL NEWS
* ❌ FAKE NEWS
* 🌐 API verification links
* 📈 Dataset visualization

---

## 💡 Key Learnings

* Importance of data cleaning in NLP
* Feature extraction using TF-IDF
* Model training & debugging
* Real-world API integration
* Building end-to-end ML applications

---

## 🚧 Limitations

* Model depends on dataset quality
* API may not cover all news sources
* Cannot guarantee 100% factual accuracy

---

## 🚀 Future Improvements

* Add BERT / Transformer models
* Improve accuracy with larger datasets
* Add explainability (why prediction?)
* Multi-language support
* Better UI/UX

---

## 👩‍💻 Author

**Shrishti Banshiar**
📧 [shrishtibanshiar105@gmail.com](mailto:shrishtibanshiar105@gmail.com)

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
