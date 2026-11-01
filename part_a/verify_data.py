
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
db_path = ROOT / "data" / "instamart.db"

with sqlite3.connect(db_path) as con:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        con
    )
    print("Database tables:")
    print(tables.to_string(index=False))

    result = pd.read_sql_query("""
        SELECT
            COUNT(*) AS completed_orders,
            ROUND(SUM(order_amount), 2) AS total_revenue
        FROM orders
        WHERE status = 'completed'
    """, con)

print("\nCompleted order summary:")
print(result.to_string(index=False))

# Check invalid order amounts in the raw dataset
raw_orders = pd.read_csv(
    ROOT / "data" / "raw" / "orders.csv",
    keep_default_na=False
)

raw_orders["amount_numeric"] = pd.to_numeric(
    raw_orders["order_amount"],
    errors="coerce"
)

print("\nRAW ORDER AMOUNT CHECK")
print("Invalid or non-numeric amounts:",
      raw_orders["amount_numeric"].isna().sum())

print("Negative amounts:",
      (raw_orders["amount_numeric"] < 0).sum())

print("Zero amounts:",
      (raw_orders["amount_numeric"] == 0).sum())

print("\nRaw order statuses:")
print(raw_orders["status"].value_counts())

print("\nNegative amount examples:")
print(
    raw_orders.loc[
        raw_orders["amount_numeric"] < 0,
        ["order_id", "status", "order_amount"]
    ].head(10).to_string(index=False)
)

# Inspect invalid order amount values
invalid_orders = raw_orders[
    raw_orders["amount_numeric"].isna()
]

print("\nINVALID ORDER AMOUNT VALUES:")
print(
    invalid_orders["order_amount"]
    .value_counts(dropna=False)
    .head(20)
    .to_string()
)

print("\nEXAMPLES OF INVALID ORDERS:")
print(
    invalid_orders[
        ["order_id", "product_id", "quantity",
         "discount_pct", "order_amount", "status"]
    ].head(10).to_string(index=False)
)
