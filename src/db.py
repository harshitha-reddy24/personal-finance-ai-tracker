import sqlite3
import pandas as pd


def init_db(db_path="finance.db"):
    """Create the database and transactions table if they don't exist yet."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            merchant TEXT,
            description TEXT,
            amount REAL,
            category TEXT,
            is_anomaly INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def load_csv_into_db(conn, csv_path="data/synthetic/transactions.csv", replace=True):
    """Load a transactions CSV into the database."""
    df = pd.read_csv(csv_path)
    table_exists_mode = "replace" if replace else "append"
    df.to_sql("transactions", conn, if_exists=table_exists_mode, index=False)
    return len(df)


def fetch_all(conn):
    """Return all transactions as a DataFrame."""
    return pd.read_sql("SELECT * FROM transactions", conn)


def fetch_by_category(conn, category):
    """Return transactions for a specific category."""
    query = "SELECT * FROM transactions WHERE category = ?"
    return pd.read_sql(query, conn, params=(category,))


def fetch_anomalies(conn):
    """Return only the transactions flagged as anomalies."""
    return pd.read_sql("SELECT * FROM transactions WHERE is_anomaly = 1", conn)


if __name__ == "__main__":
    conn = init_db("finance.db")
    n_loaded = load_csv_into_db(conn)
    print(f"Loaded {n_loaded} transactions into finance.db")

    all_txns = fetch_all(conn)
    print(f"\nTotal rows in database: {len(all_txns)}")

    anomalies = fetch_anomalies(conn)
    print(f"\nAnomalies found in database: {len(anomalies)}")
    print(anomalies[["date", "merchant", "amount", "category"]])

    groceries = fetch_by_category(conn, "Groceries")
    print(f"\nGroceries transactions: {len(groceries)}")

    conn.close()