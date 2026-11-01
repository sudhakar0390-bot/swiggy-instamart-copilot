
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

orders = pd.read_csv(
    ROOT / "data" / "clean" / "orders.csv"
)

# Use completed orders for business analysis
completed = orders[
    orders["status"] == "completed"
].copy()

print("Completed orders:", len(completed))

# ----------------------------------------
# 1. Outlier detection using IQR
# ----------------------------------------

amounts = completed["order_amount"].dropna()

Q1 = amounts.quantile(0.25)
Q3 = amounts.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = amounts[
    (amounts < lower_bound) |
    (amounts > upper_bound)
]

print("\nOUTLIER ANALYSIS")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower boundary:", round(lower_bound, 2))
print("Upper boundary:", round(upper_bound, 2))
print("Number of outliers:", len(outliers))

# ----------------------------------------
# 2. Correlation analysis
# ----------------------------------------

columns = [
    "quantity",
    "discount_pct",
    "order_amount",
    "delivery_minutes"
]

correlation = completed[columns].corr()

print("\nCORRELATION MATRIX")
print(correlation.round(3).to_string())

# ----------------------------------------
# 3. Missing values
# ----------------------------------------

print("\nMISSING VALUES")
print(completed[columns].isna().sum())
