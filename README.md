# Personal Finance AI Tracker

An end-to-end personal finance dashboard that automatically categorizes transactions, forecasts future spending, flags unusual transactions, and answers natural-language questions about your finances — all powered by machine learning, with a local LLM for the Q&A layer.

**[Live demo / screenshots below]**

## Why this project

Manually categorizing transactions and tracking spending patterns is tedious. This project explores how far you can get with classic ML and a few well-chosen models, comparing naive baselines against trained models at every step rather than assuming "more AI = better."

## Features

- **Transaction categorization** — TF-IDF + Logistic Regression model classifies transactions (Groceries, Dining, Rent, etc.) from messy merchant text, achieving **99.4% accuracy vs. 62–66% for a keyword-matching baseline**.
- **Spending forecasting** — Prophet-based time series model predicts next month's spending with confidence intervals, built on 18 months of transaction history.
- **Anomaly detection** — Isolation Forest flags unusual transactions using category-relative z-scores (not raw amount, which fails when comparing ₹18,000 rent to ₹300 dining). Achieves 67–100% recall on injected anomalies, with documented false positives from legitimate seasonal spikes.
- **Natural language Q&A** — Ask questions like *"How much did I spend on Dining last month?"* and get an answer grounded in your actual transaction data, powered by a local LLM (Ollama/Llama 3.2) — no API costs, fully private, runs offline.
- **Interactive dashboard** — Built with Streamlit: spending breakdowns, monthly trends, forecast charts, and a flagged anomalies table.

## Tech stack

Python · pandas · scikit-learn · Prophet · SQLite · Streamlit · Ollama (Llama 3.2)

## Results

| Model | Accuracy |
|---|---|
| Keyword-matching baseline | ~62-66% |
| TF-IDF + Logistic Regression | **99.4%** |
| TF-IDF + Random Forest | 96-98% |

**Forecast**: Stable monthly spending prediction (~₹43,000/month in synthetic data) with a realistic confidence interval, after disabling yearly seasonality — 18 months of history wasn't enough to reliably learn a yearly cycle, and enabling it caused the model to overfit noise into unstable, even negative, predictions. (See `reports/` for plots.)

**Anomaly detection**: Engineered a category-relative z-score feature instead of using raw transaction amounts, since spending scale varies enormously by category. The model's false positives (e.g., flagging legitimate holiday shopping spikes) highlight a real limitation: outlier detection without seasonal awareness will confuse "unusual" with "fraudulent" — a genuine tradeoff worth understanding before deploying something like this.

## Project structure