
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the RFM table
rfm = pd.read_csv(OUTPUT_DIR / "rfm_table.csv")

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardise the RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the selected K-means model
K = 4
model = KMeans(n_clusters=K, n_init=10, random_state=42)
rfm["cluster"] = model.fit_predict(X_scaled)

# Profile each cluster using the original RFM values
profiles = (
    rfm.groupby("cluster")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

# Name clusters based on their observed profiles
def name_segment(row):
    if row["mean_monetary"] == profiles["mean_monetary"].max():
        return "Premium"
    elif row["mean_recency"] <= profiles["mean_recency"].median() and row["mean_frequency"] >= profiles["mean_frequency"].median():
        return "Loyal Customers"
    elif row["mean_recency"] > profiles["mean_recency"].median():
        return "At-Risk Customers"
    else:
        return "Growing Customers"

profiles["segment_name"] = profiles.apply(name_segment, axis=1)
profiles["recommended_action"] = profiles["segment_name"].map({
    "Premium": "Offer VIP rewards and personalised benefits",
    "Loyal Customers": "Reward repeat purchases with loyalty offers",
    "At-Risk Customers": "Send a win-back campaign",
    "Growing Customers": "Encourage a second or repeat purchase",
})

# Mark the highest-monetary cluster as Premium
premium_cluster = profiles.loc[
    profiles["mean_monetary"].idxmax(), "cluster"
]
rfm["segment_name"] = rfm["cluster"].map(
    profiles.set_index("cluster")["segment_name"]
)
rfm["Premium"] = rfm["cluster"].eq(premium_cluster)

# Save the required customer segment file
rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

print("=" * 55)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 55)
print(profiles.round(2).to_string(index=False))

print("\nPremium cluster:", premium_cluster)
print("Premium customers:", int(rfm["Premium"].sum()))
print("\nSaved: outputs/customer_segments.csv")

# PCA variance explained for the standardised RFM features
pca = PCA(n_components=3)
pca.fit(X_scaled)

print("\nPCA explained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {ratio:.4f} ({ratio:.1%})")
