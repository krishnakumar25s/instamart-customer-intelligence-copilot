
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Define file locations
OUTPUT_DIR = Path("outputs")
rfm = pd.read_csv(OUTPUT_DIR / "rfm_table.csv")

# 2. Select the three RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 3. Standardise the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Test K values from 2 through 8
results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    results.append({
        "K": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels)
    })

# 5. Display and save the results
results_df = pd.DataFrame(results)

print("=" * 55)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 55)
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "kmeans_evaluation.csv",
    index=False
)

print("\nSaved: outputs/kmeans_evaluation.csv")
