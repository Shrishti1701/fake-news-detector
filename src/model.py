import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/cleaned_news.csv")

# -------------------------------
# Clean Data
# -------------------------------
df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)

# -------------------------------
# FIX LABELS (VERY IMPORTANT)
# -------------------------------
# Convert text labels to numeric safely
df['label'] = df['label'].map({
    'Fake': 0,
    'Real': 1,
    'FAKE': 0,
    'TRUE': 1,
    0: 0,
    1: 1
})

# Drop invalid labels
df = df.dropna(subset=['label'])

print("✅ Label Distribution:\n", df['label'].value_counts())

# -------------------------------
# Features & Target
# -------------------------------
X = df['clean_text']
y = df['label']

# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Model (BALANCED)
# -------------------------------
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Training Completed!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer Saved!")