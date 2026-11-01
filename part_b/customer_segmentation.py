
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Load RFM data
rfm = pd.read_csv(
    OUTPUT_DIR / "rfm_customers.csv"
)

# Features used for clustering
features = ["recency", "frequency", "monetary"]

X = rfm[features].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different numbers of clusters
results = []

print("K-MEANS CLUSTERING RESULTS")
print("-" * 40)

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "silhouette_score": score
    })

    print(f"K={k}, Silhouette Score={score:.4f}")

# Choose the best K from the assignment's allowed choices
eligible_results = [
    result for result in results
    if result["k"] in [3, 4, 5]
]

best_k = max(
    eligible_results,
    key=lambda result: result["silhouette_score"]
)["k"]

print("\nSelected K:", best_k)

# Train the final clustering model
final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["cluster"] = final_model.fit_predict(X_scaled)

# Display cluster profiles
profiles = rfm.groupby("cluster").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\nCUSTOMER CLUSTER PROFILES")
print(profiles.to_string())
# Add business segment names
segment_names = {
    0: "Premium",
    1: "Loyal",
    2: "Regular",
    3: "At-Risk"
}

rfm["segment"] = rfm["cluster"].map(segment_names)

# Save the updated file
rfm.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nSaved customer segments with names.")

print("\nSaved: outputs/customer_segments.csv")

# Assign business-friendly names based on cluster profiles
segment_names = {
    0: "Premium",
    1: "Loyal",
    2: "Regular",
    3: "At-Risk"
}

rfm["segment"] = rfm["cluster"].map(segment_names)

print("\nCUSTOMER SEGMENT COUNTS")
print(rfm["segment"].value_counts())
