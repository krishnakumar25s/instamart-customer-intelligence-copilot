
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def audit_table(name, keep_default_na=True):
    path = RAW_DIR / name
    df = pd.read_csv(path, keep_default_na=keep_default_na)

    print(f"\n{'=' * 60}")
    print(f"FILE: {name}")
    print(f"Shape (rows, columns): {df.shape}")

    print("\nData types:")
    print(df.dtypes.to_string())

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })

    print("\nMissing values by column:")
    print(missing.to_string())

    return df


customers = audit_table("customers.csv")
products = audit_table("products.csv")
orders = audit_table("orders.csv")

# The assignment requires preserving empty ticket text as empty strings.
tickets = audit_table("tickets.csv", keep_default_na=False)

print(f"\n{'=' * 60}")
print("ORDER QUALITY CHECKS")

print("Raw order rows:", len(orders))
print("Exact duplicate rows:", int(orders.duplicated().sum()))
print("Pending rows:", int(orders["status"].eq("pending").sum()))
print("Distinct order IDs:", orders["order_id"].nunique())

amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
print("Non-numeric order amounts:", int(amount_numeric.isna().sum()))

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
print("Negative delivery times:", int(delivery_numeric.lt(0).sum()))
print(
    "Blank delivery times:",
    int(orders["delivery_minutes"].isna().sum()),
)

print(f"\n{'=' * 60}")
print("CUSTOMER QUALITY CHECKS")

age = pd.to_numeric(customers["age"], errors="coerce")
city = customers["city"].astype(str)
gender = customers["gender"].astype(str)

print("Blank ages:", int(customers["age"].isna().sum()))
print("Ages outside 10-100:", int((age.lt(10) | age.gt(100)).sum()))
print("Distinct raw city spellings:", city.nunique())
print(
    "Distinct non-blank raw gender spellings:",
    gender[gender.str.strip().ne("")].nunique(),
)
print(
    "Blank genders:",
    int((gender.str.strip().eq("") | customers["gender"].isna()).sum()),
)

print("\nAudit complete. Raw files were not modified.")
