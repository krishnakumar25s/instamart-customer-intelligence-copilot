
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. File locations and prediction cutoff
# --------------------------------------------------
DATA_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

# --------------------------------------------------
# 2. Load the cleaned datasets
# --------------------------------------------------
customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["status"] = orders["status"].str.lower().str.strip()

# --------------------------------------------------
# 3. Build the prediction cohort
# Customers must have a completed order before cutoff
# --------------------------------------------------
completed_before = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] < CUTOFF)
].copy()

cohort_ids = completed_before["customer_id"].drop_duplicates()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

print("Customers in prediction cohort:", len(cohort))

# --------------------------------------------------
# 4. Calculate features using only pre-cutoff orders
# --------------------------------------------------
history = orders[orders["order_ts"] < CUTOFF].copy()

final_statuses = history[
    history["status"].isin(["completed", "cancelled"])
]

completed_history = history[
    history["status"] == "completed"
].copy()

# Completed-order behaviour
features = (
    completed_history.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)

features["recency"] = (
    CUTOFF - features["last_order_ts"]
).dt.days

# Cancellation rate: cancelled / all orders with a final status
status_counts = (
    final_statuses.assign(
        is_cancelled=(
            final_statuses["status"] == "cancelled"
        ).astype(int)
    )
    .groupby("customer_id")
    .agg(
        cancelled_orders=("is_cancelled", "sum"),
        final_orders=("order_id", "nunique"),
    )
    .reset_index()
)

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["final_orders"]
)

# Most frequent product category from completed history
category_history = completed_history.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

favourite_category = (
    category_history.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="category_orders")
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True],
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Combine the historical features
features = features.merge(
    status_counts[
        ["customer_id", "cancellation_rate"]
    ],
    on="customer_id",
    how="left",
)

features = features.merge(
    favourite_category,
    on="customer_id",
    how="left",
)

# Add customer details
features = features.merge(
    cohort[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ],
    on="customer_id",
    how="left",
    validate="one_to_one",
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

# --------------------------------------------------
# 5. Create labels using September orders ONLY
# September data is not included in model features
# --------------------------------------------------
september_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
]

september_active_ids = set(
    september_completed["customer_id"].unique()
)

features["lapsed"] = (
    ~features["customer_id"].isin(september_active_ids)
).astype(int)

# --------------------------------------------------
# 6. Handle missing feature values
# --------------------------------------------------
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

for column in numeric_features:
    features[column] = pd.to_numeric(
        features[column], errors="coerce"
    )
    features[column] = features[column].fillna(
        features[column].median()
    )

for column in ["city", "gender", "favourite_category"]:
    features[column] = (
        features[column]
        .fillna("unknown")
        .astype(str)
    )

# Remove helper columns; do not use customer IDs as features
features = features.drop(
    columns=["last_order_ts", "signup_date"],
    errors="ignore",
)

# Save the cohort features and target for inspection
features.to_csv(
    OUTPUT_DIR / "lapse_features.csv", index=False
)

# --------------------------------------------------
# 7. Validate the assignment checkpoints
# --------------------------------------------------
print("\n--- LAPSE COHORT CHECKS ---")
print("Cohort size:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))
print(
    f"Lapsed percentage: {features['lapsed'].mean():.2%}"
)

print("\nLabel distribution:")
print(features["lapsed"].value_counts().sort_index())

print("\nMissing feature values:")
print(
    features.drop(
        columns=["customer_id", "lapsed"]
    ).isna().sum().sum()
)

print("\nSaved: outputs/lapse_features.csv")
print("FEATURE ENGINEERING COMPLETE")
