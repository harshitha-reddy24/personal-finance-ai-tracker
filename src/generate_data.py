"""
generate_data.py
Generates realistic synthetic credit card / bank transaction data
for the Personal Finance AI Tracker project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Make results reproducible — same "random" data every time we run this
np.random.seed(42)
random.seed(42)

# ----------------------------------------------------------------
# 1. Define categories, their merchants, and typical spending amounts
# ----------------------------------------------------------------
# mean = average transaction amount for this category
# std = how much it typically varies
# per_month = roughly how many transactions per month in this category

CATEGORIES = {
    "Groceries": {
        "merchants": ["BigBasket", "DMart", "Reliance Fresh", "More Supermarket", "Nature's Basket"],
        "mean": 800, "std": 350, "per_month": 8
    },
    "Rent": {
        "merchants": ["Lakeview Residency", "Shanti Properties"],
        "mean": 18000, "std": 500, "per_month": 1
    },
    "Dining": {
        "merchants": ["Swiggy", "Zomato", "Barbeque Nation", "Domino's", "Cafe Coffee Day"],
        "mean": 350, "std": 200, "per_month": 10
    },
    "Entertainment": {
        "merchants": ["PVR Cinemas", "BookMyShow", "Spotify", "Netflix"],
        "mean": 300, "std": 180, "per_month": 4
    },
    "Transport": {
        "merchants": ["Indian Oil", "Uber", "Ola", "IRCTC"],
        "mean": 400, "std": 250, "per_month": 8
    },
    "Utilities": {
        "merchants": ["BSES Electricity", "Airtel Broadband", "Municipal Water Board"],
        "mean": 1500, "std": 400, "per_month": 3
    },
    "Shopping": {
        "merchants": ["Amazon", "Flipkart", "Myntra", "Lifestyle"],
        "mean": 900, "std": 600, "per_month": 5
    },
    "Healthcare": {
        "merchants": ["Apollo Pharmacy", "MedPlus", "Max Healthcare"],
        "mean": 600, "std": 400, "per_month": 2
    },
    "Subscriptions": {
        "merchants": ["Adobe", "iCloud Storage", "Cult.fit Membership"],
        "mean": 250, "std": 100, "per_month": 3
    },
    "Income": {
        "merchants": ["Employer Payroll"],
        "mean": 35000, "std": 1500, "per_month": 2
    },
}

# Months where shopping spending spikes (Nov=11, Dec=12 — holiday season)
SEASONAL_BOOST_MONTHS = {11: 1.6, 12: 2.0}

# ----------------------------------------------------------------
# 2. Function to add realistic "messiness" to merchant names
# ----------------------------------------------------------------
PROCESSOR_PREFIXES = ["SQ *", "TST* ", "PAYPAL *", "POS DEBIT ", "CKCD ", ""]
CITY_SUFFIXES = ["MUMBAI", "BANGLR", "DELHI NCR", "PUNE MH", "0210", "029384"]

def messify_merchant(merchant: str) -> str:
    """
    Real bank statements show merchant names inconsistently — often
    truncated, prefixed by a payment processor, or reduced to a code
    that doesn't obviously match the merchant name at all.
    """
    clean = merchant.upper().replace("'", "").replace(" ", "")
    abbrev = "".join(w[:3] for w in merchant.upper().split())[:8]

    variants = [
        merchant.upper(),
        clean,
        f"{merchant.upper()} #{random.randint(100, 9999)}",
        f"{merchant.upper()[:6]} SC",
        f"{random.choice(PROCESSOR_PREFIXES)}{clean[:10]}",
        f"{abbrev}{random.randint(100,999)}",
        f"{clean[:8]} {random.choice(CITY_SUFFIXES)}",
    ]
    return random.choice(variants)


# ----------------------------------------------------------------
# 3. Generate transactions for one category across the date range
# ----------------------------------------------------------------
def generate_category_transactions(category, info, start_date, end_date):
    transactions = []
    current = start_date

    # Fixed recurring bills (rent, income) happen EVERY month —
    # real recurring bills don't randomly skip months the way discretionary
    # spending does, so we treat them deterministically instead of with Poisson sampling.
    FIXED_MONTHLY = {"Rent", "Income"}

    while current <= end_date:
        if category in FIXED_MONTHLY:
            n_transactions = info["per_month"]
        else:
            n_transactions = max(0, int(np.random.poisson(info["per_month"])))

        for i in range(n_transactions):
            if category in FIXED_MONTHLY:
                day_offset = int(i * (27 / max(n_transactions, 1)))
            else:
                day_offset = random.randint(0, 27)

            txn_date = current + timedelta(days=day_offset)
            if txn_date > end_date:
                continue

            # Apply seasonal boost if this category is Shopping and month is Nov/Dec
            boost = 1.0
            if category == "Shopping":
                boost = SEASONAL_BOOST_MONTHS.get(txn_date.month, 1.0)

            amount = max(1.0, np.random.normal(info["mean"], info["std"]) * boost)
            amount = round(amount, 2)

            merchant_clean = random.choice(info["merchants"])
            merchant_messy = messify_merchant(merchant_clean)

            transactions.append({
                "date": txn_date.strftime("%Y-%m-%d"),
                "merchant": merchant_messy,
                "description": f"{merchant_messy} PURCHASE",
                "amount": amount,
                "category": category
            })

        # Move to the next month
        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    return transactions


# ----------------------------------------------------------------
# 4. Inject anomalies — unusually large transactions (rare, ~0.5%)
# ----------------------------------------------------------------
def inject_anomalies(df, fraction=0.005):
    n_anomalies = max(1, int(len(df) * fraction))
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    df["is_anomaly"] = 0
    for idx in anomaly_indices:
        # Make the amount 5-10x larger than normal for that row
        multiplier = np.random.uniform(5, 10)
        df.loc[idx, "amount"] = round(df.loc[idx, "amount"] * multiplier, 2)
        df.loc[idx, "is_anomaly"] = 1

    return df


# ----------------------------------------------------------------
# 5. Main script logic
# ----------------------------------------------------------------
def main():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 6, 30)

    all_transactions = []
    for category, info in CATEGORIES.items():
        txns = generate_category_transactions(category, info, start_date, end_date)
        all_transactions.extend(txns)

    df = pd.DataFrame(all_transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Only inject anomalies into spending categories, not Income
    spending_df = df[df["category"] != "Income"].copy()
    income_df = df[df["category"] == "Income"].copy()
    income_df["is_anomaly"] = 0

    spending_df = inject_anomalies(spending_df, fraction=0.005)

    final_df = pd.concat([spending_df, income_df]).sort_values("date").reset_index(drop=True)

    output_path = "data/synthetic/transactions.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Generated {len(final_df)} transactions.")
    print(f"Anomalies injected: {final_df['is_anomaly'].sum()}")
    print(f"Saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(final_df.head())


if __name__ == "__main__":
    main()