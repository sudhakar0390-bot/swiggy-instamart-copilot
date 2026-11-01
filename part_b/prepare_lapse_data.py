
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned datasets
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Convert timestamps
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Use completed orders only
orders = orders[orders["status"] == "completed"].copy()

# Prediction cutoff
cutoff = pd.Timestamp("2025-09-01")

# Historical orders for model features
history = orders[orders["order_ts"] < cutoff].copy()

# Future orders for target labels
future = orders[
    (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < pd.Timestamp("2025-10-01"))
].copy()

# Include customers with historical completed orders
cohort = customers[
    customers["customer_id"].isin(history["customer_id"])
].copy()

# Calculate historical RFM
features = history.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_quantity=("quantity", "mean")
).reset_index()

features["recency"] = (
    cutoff - features["last_order"]
).dt.days

# Merge with customer information
dataset = cohort.merge(
    features,
    on="customer_id",
    how="inner"
)

# Lapsed = no completed order in September
active_future_customers = set(future["customer_id"])

dataset["lapsed"] = (
    ~dataset["customer_id"].isin(active_future_customers)
).astype(int)

# Drop last_order, which is already represented by recency
dataset = dataset.drop(columns=["last_order"])

# Save dataset
output_path = OUTPUT_DIR / "lapse_dataset.csv"
dataset.to_csv(output_path, index=False)

print("LAPSE PREDICTION DATASET")
print("------------------------------")
print("Cutoff date:", cutoff.date())
print("Customers:", len(dataset))

print("\nTARGET DISTRIBUTION")
print(dataset["lapsed"].value_counts().sort_index())

print("\nFIRST FIVE RECORDS")
print(dataset.head().to_string(index=False))

print("\nSaved:", output_path)
