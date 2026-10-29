
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/instamart.db")

with sqlite3.connect(DB_PATH) as connection:
    customers = pd.read_sql_query("SELECT * FROM customers", connection)
    products = pd.read_sql_query("SELECT * FROM products", connection)
    orders = pd.read_sql_query("SELECT * FROM orders", connection)

# Convert columns back to numeric types for reliable validation
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Checkpoint 1: Customer count
assert len(customers) == 1200, "Customer count mismatch"

# Checkpoint 2: Product count
assert len(products) == 40, "Product count mismatch"

# Checkpoint 3: Clean order count
assert len(orders) == 12309, "Order count mismatch"

# Checkpoint 4: Every order ID is unique
assert orders["order_id"].nunique() == 12309, "Duplicate order IDs remain"

# Checkpoint 5: No pending orders
assert not orders["status"].str.lower().eq("pending").any(), "Pending orders remain"

# Checkpoint 6: Completed and cancelled order counts
assert (orders["status"] == "completed").sum() == 11110, "Completed count mismatch"
assert (orders["status"] == "cancelled").sum() == 1199, "Cancelled count mismatch"

# Checkpoint 7: No missing order amounts
assert orders["order_amount"].notna().all(), "Missing order amounts remain"

# Checkpoint 8: Delivery-time missing values
completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0, (
    "Completed orders have missing delivery times"
)
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199, (
    "Cancelled delivery-time checkpoint mismatch"
)

# Checkpoint 9: Completed revenue
completed_revenue = orders.loc[completed, "order_amount"].sum()
assert abs(completed_revenue - 2998668.80) < 0.01, (
    f"Completed revenue mismatch: {completed_revenue:.2f}"
)

# Checkpoint 10: Customer ages are complete and within the expected range
assert customers["age"].notna().all(), "Missing customer ages remain"
assert customers["age"].between(18, 60).all(), "Customer age range mismatch"

# Checkpoint 11: Six standardised cities
expected_cities = {
    "Bengaluru", "Chennai", "Delhi",
    "Hyderabad", "Mumbai", "Pune"
}
assert set(customers["city"].unique()) == expected_cities, "City standardisation mismatch"

# Checkpoint 12: Gender counts
gender_counts = customers["gender"].value_counts().to_dict()
assert gender_counts == {"female": 610, "male": 560, "unknown": 30}, (
    f"Gender counts mismatch: {gender_counts}"
)

print("ALL CHECKPOINTS PASSED")
