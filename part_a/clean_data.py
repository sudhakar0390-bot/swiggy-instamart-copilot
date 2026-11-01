from pathlib import Path
import sqlite3
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load raw datasets
# --------------------------------------------------
customers = pd.read_csv(RAW_DIR / "customers.csv", keep_default_na=False)
products = pd.read_csv(RAW_DIR / "products.csv", keep_default_na=False)
orders = pd.read_csv(RAW_DIR / "orders.csv", keep_default_na=False)

print("Raw data loaded successfully.")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

# --------------------------------------------------
# 2. Clean customer data
# --------------------------------------------------
customers["city"] = (
    customers["city"]
    .astype(str)
    .str.strip()
    .str.title()
)

customers["gender"] = (
    customers["gender"]
    .astype(str)
    .str.strip()
    .str.title()
)

customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Treat ages outside the assignment's valid range as missing.
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = pd.NA

# Standardize blank gender values.
customers.loc[
    customers["gender"].isin(["", "Nan", "None"]),
    "gender"
] = pd.NA

# --------------------------------------------------
# 3. Clean product data
# --------------------------------------------------
products["product_name"] = products["product_name"].astype(str).str.strip()
products["category"] = products["category"].astype(str).str.strip()
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# --------------------------------------------------
# 4. Clean order data
# --------------------------------------------------
orders = orders.drop_duplicates().copy()

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Repair missing order amounts using product prices

# Convert quantity and discount into numeric values
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)

# Create a lookup table containing product prices
price_lookup = products.set_index("product_id")["price_inr"]

# Find the price for each order's product
orders["unit_price"] = orders["product_id"].map(price_lookup)

# Calculate the amount using price, quantity and discount
calculated_amount = (
    orders["unit_price"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
)

# Replace only missing amounts
missing_amounts = orders["order_amount"].isna()

orders.loc[missing_amounts, "order_amount"] = (
    calculated_amount.loc[missing_amounts]
)

# Remove the temporary lookup column
orders = orders.drop(columns=["unit_price"])

print("Missing amounts repaired:", missing_amounts.sum())

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Keep orders, but mark invalid amounts as missing.
orders.loc[
    orders["order_amount"] < 0,
    "order_amount"
] = float("nan")

# Keep orders, but mark negative delivery times as missing.
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Keep only the two final order statuses.
orders["status"] = (
    orders["status"].astype(str).str.strip().str.lower()
)

orders = orders[
    orders["status"].isin(["completed", "cancelled"])
].copy()

# --------------------------------------------------
# 5. Save cleaned CSV files
# --------------------------------------------------
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

print("\nCleaned data saved.")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts())

# --------------------------------------------------
# 6. Create SQLite database
# --------------------------------------------------
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as con:
    customers.to_sql("customers", con, if_exists="replace", index=False)
    products.to_sql("products", con, if_exists="replace", index=False)
    orders.to_sql("orders", con, if_exists="replace", index=False)

print(f"\nSQLite database created at: {DB_PATH}")