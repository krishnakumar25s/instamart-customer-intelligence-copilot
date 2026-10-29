
import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the database location
DB_PATH = Path("data/instamart.db")

# 2. Load cleaned tables from SQLite
with sqlite3.connect(DB_PATH) as connection:
    customers = pd.read_sql_query(
        "SELECT * FROM customers", connection
    )
    products = pd.read_sql_query(
        "SELECT * FROM products", connection
    )
    orders = pd.read_sql_query(
        "SELECT * FROM orders", connection
    )

# 3. Convert dates and numeric columns to correct data types
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

print("=" * 55)
print("INSTAMART EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 4. Basic dataset overview
print("\n--- DATASET SIZES ---")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

# 5. Order status distribution
print("\n--- ORDER STATUS DISTRIBUTION ---")
print(orders["status"].value_counts(dropna=False))

# 6. Revenue and spending analysis
completed = orders[orders["status"] == "completed"].copy()

print("\n--- COMPLETED ORDER ANALYSIS ---")
print("Completed orders:", len(completed))
print(f"Total revenue: INR {completed['order_amount'].sum():,.2f}")
print(f"Average order value: INR {completed['order_amount'].mean():,.2f}")
print(f"Median order value: INR {completed['order_amount'].median():,.2f}")

# 7. Revenue by city
customer_city = customers[["customer_id", "city"]]

city_revenue = completed.merge(
    customer_city, on="customer_id", how="left"
).groupby("city", as_index=False)["order_amount"].sum()

city_revenue = city_revenue.sort_values(
    "order_amount", ascending=False
)

print("\n--- REVENUE BY CITY ---")
print(city_revenue.to_string(index=False))

# 8. Revenue by product category
order_product = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

category_revenue = order_product.groupby(
    "category", as_index=False
)["order_amount"].sum().sort_values(
    "order_amount", ascending=False
)

print("\n--- REVENUE BY CATEGORY ---")
print(category_revenue.to_string(index=False))

# 9. Daily revenue trend
daily_revenue = (
    completed.assign(order_day=completed["order_ts"].dt.date)
    .groupby("order_day", as_index=False)["order_amount"]
    .sum()
    .sort_values("order_day")
)

print("\n--- DAILY REVENUE SUMMARY ---")
print("Days with completed orders:", len(daily_revenue))
print(daily_revenue.head(5).to_string(index=False))
print(daily_revenue.tail(5).to_string(index=False))

# 10. Save charts and analysis outputs
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

city_revenue.to_csv(OUTPUT_DIR / "revenue_by_city.csv", index=False)
category_revenue.to_csv(
    OUTPUT_DIR / "revenue_by_category.csv", index=False
)
daily_revenue.to_csv(
    OUTPUT_DIR / "daily_revenue.csv", index=False
)

plt.figure(figsize=(9, 5))
plt.bar(city_revenue["city"], city_revenue["order_amount"])
plt.title("Completed Revenue by City")
plt.xlabel("City")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "revenue_by_city.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(
    daily_revenue["order_day"].astype(str),
    daily_revenue["order_amount"]
)
plt.title("Daily Completed Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_revenue.png", dpi=150)
plt.close()

print("\n--- OUTPUT FILES SAVED ---")
print("outputs/revenue_by_city.csv")
print("outputs/revenue_by_category.csv")
print("outputs/daily_revenue.csv")
print("outputs/revenue_by_city.png")
print("outputs/daily_revenue.png")

print("\nEDA COMPLETE!")
