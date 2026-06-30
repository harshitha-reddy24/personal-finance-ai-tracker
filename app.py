"""
app.py
Personal Finance AI Tracker — Streamlit dashboard.
Brings together categorization, forecasting, and anomaly detection
into a single interactive view.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Personal Finance AI Tracker", layout="wide")

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------

@st.cache_resource
def load_categorizer():
    if os.path.exists("models/categorizer.joblib"):
        return joblib.load("models/categorizer.joblib")
    return None


def categorize_transactions(df, model):
    """Predict category for any rows missing one, using the trained model."""
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = None
    text = df["merchant"].astype(str) + " " + df.get("description", "").astype(str)
    missing = df["category"].isna()
    if missing.any() and model is not None:
        df.loc[missing, "category"] = model.predict(text[missing])
    return df


def detect_anomalies(df):
    """Flag category-relative outliers using Isolation Forest."""
    from sklearn.ensemble import IsolationForest

    df = df.copy()
    spending = df[df["category"] != "Income"].copy()

    stats = spending.groupby("category")["amount"].agg(["mean", "std"]).reset_index()
    stats.columns = ["category", "cat_mean", "cat_std"]
    spending = spending.merge(stats, on="category", how="left")
    spending["cat_std"] = spending["cat_std"].replace(0, 1)  # avoid divide-by-zero
    spending["amount_zscore"] = (spending["amount"] - spending["cat_mean"]) / spending["cat_std"]

    model = IsolationForest(contamination=0.01, random_state=42)
    spending["pred"] = model.fit_predict(spending[["amount_zscore"]])
    spending["flagged_anomaly"] = (spending["pred"] == -1).astype(int)

    df = df.merge(
        spending[["date", "merchant", "amount", "flagged_anomaly", "amount_zscore"]],
        on=["date", "merchant", "amount"], how="left"
    )
    df["flagged_anomaly"] = df["flagged_anomaly"].fillna(0).astype(int)
    return df


@st.cache_data
def forecast_spending(monthly_df):
    """Run Prophet forecast on monthly totals. Returns forecast dataframe or None."""
    if len(monthly_df) < 4:
        return None
    from prophet import Prophet

    m = monthly_df.rename(columns={"month": "ds", "amount": "y"}).copy()
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        changepoint_prior_scale=0.05, growth="flat"
    )
    model.fit(m)
    future = model.make_future_dataframe(periods=3, freq="MS")
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    return forecast


# ----------------------------------------------------------------
# Sidebar: data source
# ----------------------------------------------------------------
st.sidebar.title("💰 Finance Tracker")
data_source = st.sidebar.radio("Data source", ["Use sample data", "Upload my own CSV"])

if data_source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload transactions CSV", type="csv")
    st.sidebar.caption("Expected columns: date, merchant, amount (description and category optional)")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV to get started, or switch to sample data in the sidebar.")
        st.stop()
else:
    df = pd.read_csv("data/synthetic/transactions.csv")

df["date"] = pd.to_datetime(df["date"])

# ----------------------------------------------------------------
# Categorize if needed
# ----------------------------------------------------------------
categorizer = load_categorizer()
df = categorize_transactions(df, categorizer)

# ----------------------------------------------------------------
# Header + headline metrics
# ----------------------------------------------------------------
st.title("Personal Finance AI Tracker")
st.caption("Transaction categorization, spending forecasts, and anomaly detection — powered by ML.")

spending_df = df[df["category"] != "Income"].copy()
total_spent = spending_df["amount"].sum()
top_category = spending_df.groupby("category")["amount"].sum().idxmax()
n_transactions = len(df)
n_months = spending_df["date"].dt.to_period("M").nunique()
avg_monthly_spend = total_spent / n_months if n_months > 0 else 0
date_min = df["date"].min().strftime("%b %Y")
date_max = df["date"].max().strftime("%b %Y")

st.caption(f"📅 Showing data from **{date_min}** to **{date_max}** ({n_months} months)")

df = detect_anomalies(df)
n_anomalies = int(df["flagged_anomaly"].sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Spending (all time)", f"₹{total_spent:,.0f}")
col2.metric("Avg. Monthly Spending", f"₹{avg_monthly_spend:,.0f}")
col3.metric("Top Category", top_category)
col4.metric("Transactions", f"{n_transactions:,}")
col5.metric("Flagged Anomalies", n_anomalies)

st.divider()

# ----------------------------------------------------------------
# Spending by category
# ----------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Total Spending by Category (all-time)")
    cat_totals = spending_df.groupby("category")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(cat_totals)

with right:
    st.subheader("Monthly Spending Trend")
    monthly = spending_df.groupby(spending_df["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly.columns = ["month", "amount"]
    st.line_chart(monthly.set_index("month")["amount"])

st.divider()

# ----------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------
st.subheader("Spending Forecast (Next 3 Months)")
forecast = forecast_spending(monthly)

if forecast is not None:
    chart_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
    st.line_chart(chart_data)
    next_month_pred = forecast.iloc[-3]
    st.markdown(
        f"Predicted next month's spending: **₹{next_month_pred['yhat']:,.0f}** "
        f"(range: ₹{next_month_pred['yhat_lower']:,.0f} – ₹{next_month_pred['yhat_upper']:,.0f})"
    )
else:
    st.info("Need at least 4 months of data to generate a forecast.")

st.divider()

# ----------------------------------------------------------------
# Anomalies table
# ----------------------------------------------------------------
st.subheader("Flagged Anomalies")
anomalies = df[df["flagged_anomaly"] == 1].sort_values("date", ascending=False)
if len(anomalies) > 0:
    st.dataframe(
        anomalies[["date", "merchant", "amount", "category"]],
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Flagged using Isolation Forest on category-relative spending z-scores. "
        "Some flags may reflect legitimate seasonal spikes (e.g. holiday shopping) "
        "rather than true anomalies — the model doesn't yet account for seasonality."
    )
else:
    st.write("No anomalies flagged in this dataset.")

st.divider()

# ----------------------------------------------------------------
# Natural language Q&A (local LLM via Ollama)
# ----------------------------------------------------------------
st.subheader("💬 Ask Your Finances")
st.caption(
    "Ask a question in plain English, e.g. \"How much did I spend on Dining last month?\" "
    "Powered by a local LLM (Ollama) — runs entirely on this machine, free, no API key."
)

user_question = st.text_input("Your question", placeholder="How much did I spend on Groceries this year?")

if st.button("Ask") and user_question.strip():
    try:
        import sys
        sys.path.insert(0, "src")
        from llm_query import answer_question
        with st.spinner("Thinking... (local model, may take a few seconds)"):
            answer = answer_question("finance.db", user_question)
        st.markdown(f"**Answer:** {answer}")
    except FileNotFoundError:
        st.error("finance.db not found. Run `python src/db.py` first to set up the database.")
    except Exception as e:
        st.error(
            f"Couldn't reach the local LLM. Make sure Ollama is running "
            f"(open the Ollama app, or run `ollama serve`) and that you've pulled "
            f"a model with `ollama pull llama3.2`.\n\nError: {e}"
        )

st.divider()
st.caption("Built with Streamlit · scikit-learn · Prophet · Ollama (local LLM) · synthetic data generator")