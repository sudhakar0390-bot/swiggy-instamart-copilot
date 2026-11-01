
from pathlib import Path
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
customers = pd.read_csv(
    CLEAN_DIR / "customers.csv"
)

orders = pd.read_csv(
    CLEAN_DIR / "orders.csv"
)

# Convert order timestamps into dates
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Use completed orders only
completed = orders[
    orders["status"] == "completed"
].copy()

# Reference date: one day after the latest order
reference_date = completed["order_ts"].max() + pd.Timedelta(days=1)

# Calculate RFM for each customer
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency in days
rfm["recency"] = (
    reference_date - rfm["last_order"]
).dt.days

# Include customers with no completed orders
rfm = customers[["customer_id"]].merge(
    rfm,
    on="customer_id",
    how="left"
)

# Customers with no orders have zero frequency and spending
rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
rfm["monetary"] = rfm["monetary"].fillna(0)

# Give customers with no orders a recency larger than
# any customer who has ordered
max_recency = rfm["recency"].max()
rfm["recency"] = rfm["recency"].fillna(max_recency + 1)

# Keep the important columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display results
print("Total customers:", len(rfm))

print("\nFIRST 10 RFM RECORDS")
print(rfm.head(10).to_string(index=False))

print("\nRFM SUMMARY")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# Save the RFM table
rfm.to_csv(
    OUTPUT_DIR / "rfm_customers.csv",
    index=False
)

print("\nRFM data saved to outputs/rfm_customers.csv")
