
import pandas as pd
from pathlib import Path
import sqlite3

# Project folders
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load the raw files
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# --------------------------------------------------
# 1. CLEAN CUSTOMERS
# --------------------------------------------------

# Standardise city names: remove extra spaces and use title case
customers["city"] = customers["city"].str.strip().str.title()

# Standardise gender spellings
def clean_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"

    value = str(value).strip().lower()

    if value in ["m", "male", "man"]:
        return "male"
    elif value in ["f", "female", "woman"]:
        return "female"
    else:
        return "unknown"


customers["gender"] = customers["gender"].apply(clean_gender)

# Convert invalid ages into missing values
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    (customers["age"] < 10) | (customers["age"] > 100),
    "age"
] = pd.NA

# Fill missing ages using the median age of each city
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# Convert signup_date into datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

print("CUSTOMER CLEANING COMPLETE")
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print(
    "Age range:",
    customers["age"].min(),
    "to",
    customers["age"].max()
)

# --------------------------------------------------
# 2. CLEAN ORDERS
# --------------------------------------------------

# Convert order timestamps into datetime values
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

# Rule 1: Remove exact duplicate rows
rows_before_duplicates = len(orders)
orders = orders.drop_duplicates().copy()
duplicates_removed = rows_before_duplicates - len(orders)

print("\nExact duplicate rows removed:", duplicates_removed)

# Rule 2: Keep the latest record for each order_id
rows_before_latest = len(orders)

orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
).copy()

duplicate_order_records_removed = rows_before_latest - len(orders)

print(
    "Older records for repeated order IDs removed:",
    duplicate_order_records_removed
)

# Rule 3: Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Fill missing amounts using product price, quantity, and discount
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

missing_amount_before = orders["order_amount"].isna().sum()

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

print("Missing amounts filled:", missing_amount_before)

# Rule 4: Negative delivery times are invalid
negative_delivery_count = (orders["delivery_minutes"] < 0).sum()

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

print("Negative delivery times set to missing:", negative_delivery_count)

# Rule 5: Fill missing delivery times for completed orders
# using the median of completed orders in the same product category.
completed_mask = orders["status"].str.strip().str.lower().eq("completed")

completed_category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(completed_category_median)
)

# Cancelled orders intentionally retain missing delivery times.
# They were not delivered, so a delivery time must not be invented.

# Remove helper columns added by the merge
orders = orders.drop(columns=["price_inr", "category"])

# Ensure consistent status spelling
orders["status"] = orders["status"].str.strip().str.lower()

# --------------------------------------------------
# 3. SAVE CLEAN CSV FILES
# --------------------------------------------------

# Save signup_date and order_ts in reproducible text formats
customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# --------------------------------------------------
# 4. VALIDATE CLEANING CHECKPOINTS
# --------------------------------------------------

print("\nORDER CLEANING COMPLETE")
print("Clean order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times among completed orders:",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Missing delivery times among cancelled orders:",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)

completed_revenue = orders.loc[
    orders["status"] == "completed",
    "order_amount"
].sum()

print(f"Completed revenue: INR {completed_revenue:,.2f}")

print("\nClean CSV files saved in data/clean/")

# --------------------------------------------------
# 5. SAVE CLEAN DATA TO SQLITE DATABASE
# --------------------------------------------------

DB_PATH = Path("data/instamart.db")

# Connect to the database (creates it if it doesn't exist)
with sqlite3.connect(DB_PATH) as connection:

    # Save each clean DataFrame as a SQLite table
    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

print("\nSQLite database created:", DB_PATH)

# Verify that all three tables exist
with sqlite3.connect(DB_PATH) as connection:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        connection
    )

    print("\nTables in database:")
    print(tables)

print("\nDATA CLEANING AND DATABASE STORAGE COMPLETE")
