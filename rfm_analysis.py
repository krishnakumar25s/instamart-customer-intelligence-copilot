
from pathlib import Path
import pandas as pd

# 1. Define the input and output locations
DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Set the fixed snapshot date required by the assignment
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# 3. Load the cleaned customer and order data
customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

# 4. Convert order timestamps into datetime values
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# 5. Keep completed orders only
completed_orders = orders[
    orders["status"].str.lower().eq("completed")
].copy()

# 6. Calculate Recency, Frequency, and Monetary values
rfm = (
    completed_orders.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

# Recency is the whole number of days since the last completed order
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_ts"]
).dt.days

# Keep the requested column order
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# 7. Save the RFM table
rfm.to_csv(OUTPUT_DIR / "rfm_table.csv", index=False)

# 8. Print the reference checks
print("=" * 55)
print("INSTAMART RFM ANALYSIS")
print("=" * 55)

print("\nRFM table preview:")
print(rfm.head(10).to_string(index=False))

print("\n--- RFM CHECKPOINTS ---")
print("Customers with completed orders:", len(rfm))
print("Median recency:", rfm["recency"].median())
print("Minimum recency:", rfm["recency"].min())
print("Maximum recency:", rfm["recency"].max())
print(f"Mean frequency: {rfm['frequency'].mean():.2f}")
print("Maximum frequency:", rfm["frequency"].max())
print(f"Mean monetary: INR {rfm['monetary'].mean():,.2f}")
print(f"Maximum monetary: INR {rfm['monetary'].max():,.2f}")

print("\nSaved: outputs/rfm_table.csv")
