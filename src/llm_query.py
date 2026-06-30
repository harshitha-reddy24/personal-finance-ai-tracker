"""
llm_query.py
Natural-language Q&A over your transaction data using a LOCAL LLM (Ollama).
Free, runs entirely on your own machine — no API key, no cost, no internet
needed once the model is downloaded.
"""

import pandas as pd
import sqlite3
from datetime import timedelta
import ollama

CATEGORIES = ["Groceries", "Rent", "Dining", "Entertainment", "Transport",
              "Utilities", "Shopping", "Healthcare", "Subscriptions", "Income"]


def extract_filters(question: str, latest_date: pd.Timestamp):
    """
    Lightweight intent extraction: look for a known category name and
    simple time-period phrases in the question. This isn't NLP magic —
    it's a pragmatic filter so we don't have to send the entire transaction
    history to the LLM for every question.
    """
    q = question.lower()
    category = None
    for cat in CATEGORIES:
        if cat.lower() in q:
            category = cat
            break

    start_date = None
    end_date = latest_date

    if "last month" in q:
        first_of_this_month = latest_date.replace(day=1)
        end_date = first_of_this_month - timedelta(days=1)
        start_date = end_date.replace(day=1)
    elif "this month" in q:
        start_date = latest_date.replace(day=1)
    elif "last 3 months" in q or "past 3 months" in q:
        start_date = latest_date - pd.DateOffset(months=3)
    elif "this year" in q:
        start_date = latest_date.replace(month=1, day=1)
    elif "last year" in q:
        start_date = latest_date.replace(year=latest_date.year - 1, month=1, day=1)
        end_date = latest_date.replace(year=latest_date.year - 1, month=12, day=31)

    return category, start_date, end_date


def get_relevant_data(conn, question: str):
    df = pd.read_sql("SELECT * FROM transactions", conn)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()

    category, start_date, end_date = extract_filters(question, latest_date)

    filtered = df.copy()
    if category:
        filtered = filtered[filtered["category"] == category]
    if start_date is not None:
        filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]

    return filtered, category, start_date, end_date


def build_context_summary(filtered: pd.DataFrame) -> str:
    """Summarize the filtered data into compact text instead of dumping raw rows."""
    if len(filtered) == 0:
        return "No matching transactions found."

    spending = filtered[filtered["category"] != "Income"]
    income = filtered[filtered["category"] == "Income"]

    n = len(filtered)
    date_range = f"{filtered['date'].min().date()} to {filtered['date'].max().date()}"

    lines = [
        f"Number of transactions: {n}",
        f"Date range: {date_range}",
        f"Total spending (excluding income): Rs. {spending['amount'].sum():,.2f}",
        f"Total income: Rs. {income['amount'].sum():,.2f}",
        "Spending breakdown by category:",
    ]
    by_category = spending.groupby("category")["amount"].sum().sort_values(ascending=False)
    for cat, amt in by_category.items():
        lines.append(f"  - {cat}: Rs. {amt:,.2f}")

    top_merchants = spending.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(5)
    lines.append("Top merchants by spend (excluding income):")
    for merch, amt in top_merchants.items():
        lines.append(f"  - {merch}: Rs. {amt:,.2f}")

    return "\n".join(lines)


def ask_local_llm(question: str, context_summary: str, category, start_date, end_date,
                   model: str = "llama3.2") -> str:
    """
    Send the filtered data summary + question to a LOCAL LLM via Ollama.
    """
    filter_note = ""
    if category:
        filter_note += f" (filtered to category: {category})"
    if start_date is not None:
        filter_note += f" (date range: {start_date.date()} to {end_date.date()})"

    prompt = f"""You are a helpful personal finance assistant. Answer the user's
question using ONLY the transaction data summary below. Be concise (2-4 sentences).
Use Rs. for all amounts. If the data doesn't fully answer the question, say so.

Question: {question}

Relevant transaction data{filter_note}:
{context_summary}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def answer_question(db_path: str, question: str, model: str = "llama3.2") -> str:
    """Convenience wrapper: filter data, build summary, ask the LLM, return the answer."""
    conn = sqlite3.connect(db_path)
    filtered, category, start, end = get_relevant_data(conn, question)
    summary = build_context_summary(filtered)
    conn.close()
    return ask_local_llm(question, summary, category, start, end, model=model)


if __name__ == "__main__":
    test_questions = [
        "How much did I spend on Dining last month?",
        "What's my total Groceries spending this year?",
    ]
    for q in test_questions:
        print(f"\n=== Question: {q} ===")
        print(answer_question("finance.db", q))