"""
categorize.py
Compares a keyword-based baseline against ML models (Logistic Regression,
Random Forest) for predicting transaction category from merchant text.
Saves the best-performing model for use in the dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# ----------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------
df = pd.read_csv('data/synthetic/transactions.csv')
df['text'] = df['merchant'].astype(str) + " " + df['description'].astype(str)

# ----------------------------------------------------------------
# 2. Baseline: keyword matching
# ----------------------------------------------------------------
KEYWORD_MAP = {
    "walmart": "Groceries", "trader joe": "Groceries", "whole foods": "Groceries",
    "kroger": "Groceries", "costco": "Groceries",
    "apartments": "Rent", "properties": "Rent",
    "chipotle": "Dining", "starbucks": "Dining", "diner": "Dining", "pizza": "Dining", "sushi": "Dining",
    "amc": "Entertainment", "steam": "Entertainment", "spotify": "Entertainment", "netflix": "Entertainment",
    "shell": "Transport", "uber": "Transport", "lyft": "Transport", "metro transit": "Transport",
    "power & light": "Utilities", "comcast": "Utilities", "water authority": "Utilities",
    "amazon": "Shopping", "target": "Shopping", "best buy": "Shopping", "h&m": "Shopping",
    "cvs": "Healthcare", "walgreens": "Healthcare", "medical": "Healthcare",
    "adobe": "Subscriptions", "icloud": "Subscriptions", "gym": "Subscriptions",
    "payroll": "Income",
}

def keyword_predict(text):
    text_lower = text.lower()
    for keyword, category in KEYWORD_MAP.items():
        if keyword in text_lower:
            return category
    return "Unknown"

df['baseline_pred'] = df['text'].apply(keyword_predict)
baseline_acc = accuracy_score(df['category'], df['baseline_pred'])
print(f"Baseline keyword accuracy: {baseline_acc:.2%}")

# ----------------------------------------------------------------
# 3. ML models: TF-IDF + Logistic Regression / Random Forest
# ----------------------------------------------------------------
X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))),
    ('clf', LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression accuracy: {lr_acc:.2%}")

rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest accuracy: {rf_acc:.2%}")

print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, lr_preds, zero_division=0))

# ----------------------------------------------------------------
# 4. Confusion matrix plot (Logistic Regression — our best model)
# ----------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, lr_preds, labels=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix — Logistic Regression Categorizer')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
print("\nSaved confusion matrix plot to reports/confusion_matrix.png")

# ----------------------------------------------------------------
# 5. Save the best model (Logistic Regression won)
# ----------------------------------------------------------------
joblib.dump(lr_pipeline, 'models/categorizer.joblib')
print("Saved trained model to models/categorizer.joblib")

# ----------------------------------------------------------------
# 6. Save a summary of results to a text file (for your README later)
# ----------------------------------------------------------------
with open('reports/categorization_results.txt', 'w') as f:
    f.write(f"Baseline keyword accuracy: {baseline_acc:.2%}\n")
    f.write(f"Logistic Regression accuracy: {lr_acc:.2%}\n")
    f.write(f"Random Forest accuracy: {rf_acc:.2%}\n\n")
    f.write("Classification report (Logistic Regression):\n")
    f.write(classification_report(y_test, lr_preds, zero_division=0))

print("Saved results summary to reports/categorization_results.txt")