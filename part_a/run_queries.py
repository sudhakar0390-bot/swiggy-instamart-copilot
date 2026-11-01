
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
db_path = ROOT / "data" / "instamart.db"
sql_path = ROOT / "sql" / "queries.sql"

# Read all three SQL queries
sql_text = sql_path.read_text(encoding="utf-8")
queries = sql_text.split(";")

with sqlite3.connect(db_path) as con:
    for number, query in enumerate(queries, start=1):
        if not query.strip():
            continue

        print(f"\nQUERY {number}")
        result = pd.read_sql_query(query, con)
        print(result.to_string(index=False))

# Cross-check SQL revenue with pandas

orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")

completed_orders = orders[
    orders["status"] == "completed"
]

pandas_revenue = round(
    completed_orders["order_amount"].sum(), 2
)

with sqlite3.connect(db_path) as con:
    sql_revenue = pd.read_sql_query("""
        SELECT ROUND(SUM(order_amount), 2) AS revenue
        FROM orders
        WHERE status = 'completed'
    """, con)["revenue"].iloc[0]

print("\nREVENUE CROSS-CHECK")
print("Pandas revenue:", pandas_revenue)
print("SQL revenue:", sql_revenue)

assert abs(pandas_revenue - sql_revenue) < 0.01

print("PASS: SQL and pandas revenue match!")
