import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/cleaned_news.csv")

# -------------------------------
# Fix Missing Values (IMPORTANT)
# -------------------------------
df = df.dropna(subset=['clean_text'])

# convert to string (extra safety)
df['clean_text'] = df['clean_text'].astype(str)

# -------------------------------
# Features & Target
# -------------------------------
X = df['clean_text']
y = df['label']

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Completed!")
print(f"\nAccuracy: {accuracy:.2f}")

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))