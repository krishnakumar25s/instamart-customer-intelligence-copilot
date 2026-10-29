
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

# Read all seven SQL queries from the file.
sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split the file into individual queries.
queries = [
    statement.strip()
    for statement in sql_text.split(";")
    if statement.strip()
]

print("Number of SQL queries found:", len(queries))

if len(queries) != 7:
    raise ValueError("Expected exactly seven SQL queries in sql/queries.sql")

# Execute each query against the SQLite database.
with sqlite3.connect(DB_PATH) as connection:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)

        print("\n" + "=" * 60)
        print(f"QUERY {number}")
        print("=" * 60)
        print("Rows returned:", len(result))
        print(result.head(10).to_string(index=False))

        # Print useful summary checks.
        if number == 2:
            print("\nTotal revenue across all city/category groups:")
            print(f"INR {result['revenue'].sum():,.2f}")

            largest = result.loc[result["revenue"].idxmax()]
            print(
                "Largest revenue group:",
                largest["city"],
                "/",
                largest["category"],
                f"— INR {largest['revenue']:,.2f}",
            )

        elif number == 5 and not result.empty:
            print("\nAverage completed spend per customer:")
            print(f"INR {result['average_spend'].iloc[0]:,.2f}")

        elif number == 7 and not result.empty:
            print("\nFinal cumulative revenue:")
            print(f"INR {result['cumulative_revenue'].iloc[-1]:,.2f}")

print("\nAll seven SQL queries executed.")
