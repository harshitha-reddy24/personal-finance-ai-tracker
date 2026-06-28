"""
anomaly.py
Detects unusual transactions using Isolation Forest, an unsupervised
anomaly detection algorithm. Evaluates against the known injected
anomalies from generate_data.py to measure precision/recall.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import os

df = pd.read_csv('data/synthetic/transactions.csv')
df['date'] = pd.to_datetime(df['date'])

spending = df[df['category'] != 'Income'].copy()

# Feature engineering: how unusual is this amount RELATIVE TO ITS CATEGORY?
# Raw amount alone is useless here — $1400 is normal for Rent but would be
# a huge anomaly for Dining. We compute a z-score within each category instead.
category_stats = spending.groupby('category')['amount'].agg(['mean', 'std']).reset_index()
category_stats.columns = ['category', 'cat_mean', 'cat_std']
spending = spending.merge(category_stats, on='category', how='left')
spending['amount_zscore'] = (spending['amount'] - spending['cat_mean']) / spending['cat_std']

# Isolation Forest: an unsupervised model that learns what "normal" looks
# like and isolates points that are easy to separate from the rest (outliers).
# contamination = expected proportion of anomalies in the data.
model = IsolationForest(contamination=0.01, random_state=42)
spending['anomaly_score'] = model.fit_predict(spending[['amount_zscore']])
spending['predicted_anomaly'] = (spending['anomaly_score'] == -1).astype(int)

print("=== Evaluation against known injected anomalies ===")
true_anomalies = spending[spending['is_anomaly'] == 1]
predicted_anomalies = spending[spending['predicted_anomaly'] == 1]

print(f"\nTrue anomalies (injected): {len(true_anomalies)}")
print(true_anomalies[['date', 'merchant', 'amount', 'category', 'amount_zscore']])

print(f"\nPredicted anomalies (Isolation Forest): {len(predicted_anomalies)}")
print(predicted_anomalies[['date', 'merchant', 'amount', 'category', 'amount_zscore']])

caught = spending[(spending['is_anomaly'] == 1) & (spending['predicted_anomaly'] == 1)]
print(f"\nTrue anomalies caught: {len(caught)} / {len(true_anomalies)}")

precision = precision_score(spending['is_anomaly'], spending['predicted_anomaly'], zero_division=0)
recall = recall_score(spending['is_anomaly'], spending['predicted_anomaly'], zero_division=0)
f1 = f1_score(spending['is_anomaly'], spending['predicted_anomaly'], zero_division=0)
print(f"\nPrecision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 score: {f1:.2%}")

# ----------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------
os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(10, 5))
normal = spending[spending['predicted_anomaly'] == 0]
flagged = spending[spending['predicted_anomaly'] == 1]
true_anom = spending[spending['is_anomaly'] == 1]

plt.scatter(normal['date'], normal['amount_zscore'], alpha=0.4, s=20, label='Normal', color='steelblue')
plt.scatter(flagged['date'], flagged['amount_zscore'], s=60, label='Flagged by model', color='orange', marker='x')
plt.scatter(true_anom['date'], true_anom['amount_zscore'], s=120, facecolors='none',
            edgecolors='red', linewidths=2, label='True injected anomaly')
plt.axhline(0, color='gray', linewidth=0.5)
plt.xlabel('Date')
plt.ylabel('Amount z-score (relative to category)')
plt.title('Anomaly Detection: Category-Relative Spending Outliers')
plt.legend()
plt.tight_layout()
plt.savefig('reports/anomaly_detection.png', dpi=150)
print("\nSaved anomaly plot to reports/anomaly_detection.png")

with open('reports/anomaly_results.txt', 'w') as f:
    f.write(f"True anomalies (injected): {len(true_anomalies)}\n")
    f.write(f"Predicted anomalies (Isolation Forest): {len(predicted_anomalies)}\n")
    f.write(f"True anomalies caught: {len(caught)} / {len(true_anomalies)}\n\n")
    f.write(f"Precision: {precision:.2%}\n")
    f.write(f"Recall: {recall:.2%}\n")
    f.write(f"F1 score: {f1:.2%}\n\n")
    f.write("Note: false positives include legitimate holiday-season shopping spikes\n")
    f.write("(Nov/Dec), since the model flags statistical outliers without accounting\n")
    f.write("for seasonality. A production system would need seasonal-aware baselines.\n")
print("Saved results summary to reports/anomaly_results.txt")