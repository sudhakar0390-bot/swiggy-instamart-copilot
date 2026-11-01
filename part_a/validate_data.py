
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Check 1: Customer count
assert len(customers) == 1200
print("PASS 1: Customer count")

# Check 2: Product count
assert len(products) == 40
print("PASS 2: Product count")

# Check 3: Cleaned order count
assert len(orders) == 12309
print("PASS 3: Cleaned order count")

# Check 4: Completed order count
completed = orders[orders["status"] == "completed"]
assert len(completed) == 11110
print("PASS 4: Completed orders")

# Check 5: Cancelled order count
assert (orders["status"] == "cancelled").sum() == 1199
print("PASS 5: Cancelled orders")

# Check 6: No pending orders
assert not (orders["status"] == "pending").any()
print("PASS 6: No pending orders")

# Check 7: Unique order IDs
assert orders["order_id"].is_unique
print("PASS 7: Unique order IDs")

# Check 8: Revenue
assert abs(completed["order_amount"].sum() - 2998668.80) < 0.01
print("PASS 8: Revenue matches")

# Check 9: No missing order amounts
assert orders["order_amount"].notna().all()
print("PASS 9: No missing order amounts")

# Check 10: Database tables
with sqlite3.connect(DB) as con:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        con
    )["name"].tolist()

assert {"customers", "products", "orders"}.issubset(set(tables))
print("PASS 10: Database tables exist")

print("\nALL CHECKPOINTS PASSED")
