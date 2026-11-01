
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Load customer segments
data = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

features = ["recency", "frequency", "monetary"]

# Standardize RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Reduce to two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA coordinates
data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]

print("PCA ANALYSIS")
print("------------------------------")

print("PC1 explained variance:",
      round(pca.explained_variance_ratio_[0], 4))

print("PC2 explained variance:",
      round(pca.explained_variance_ratio_[1], 4))

print("Total explained variance:",
      round(sum(pca.explained_variance_ratio_), 4))

print("\nPCA SUMMARY BY SEGMENT")

summary = data.groupby("segment")[
    ["PC1", "PC2"]
].mean().round(3)

print(summary.to_string())

# Save results
data.to_csv(
    OUTPUT_DIR / "customer_segments_pca.csv",
    index=False
)

print("\nSaved: outputs/customer_segments_pca.csv")
