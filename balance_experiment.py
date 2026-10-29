
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Load the prepared lapse dataset
data = pd.read_csv("outputs/lapse_features.csv")

numeric_features = [
    "recency", "frequency", "monetary",
    "avg_discount_pct", "cancellation_rate",
    "avg_delivery_minutes", "age", "tenure_days",
]
categorical_features = ["city", "gender", "favourite_category"]

X = data[numeric_features + categorical_features].copy()
y = data["lapsed"].astype(int)

# 2. Recreate the same reproducible train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Duplicate lapsed training rows until both classes are equal
train = X_train.copy()
train["lapsed"] = y_train.to_numpy()

majority_count = int(y_train.value_counts().max())
minority_rows = train[train["lapsed"] == 1]

extra_rows = minority_rows.sample(
    n=majority_count - len(minority_rows),
    replace=True,
    random_state=42,
)

balanced_train = pd.concat([train, extra_rows], ignore_index=True)

X_balanced = balanced_train.drop(columns="lapsed")
y_balanced = balanced_train["lapsed"].astype(int)

print("Original training class counts:")
print(y_train.value_counts().sort_index())
print("\nBalanced training class counts:")
print(y_balanced.value_counts().sort_index())

# 4. Build preprocessing inside a pipeline
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    (
        "categorical",
        OneHotEncoder(drop="first", handle_unknown="ignore"),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=100, max_iter=1000)),
])

# Fit only on the oversampled training set
model.fit(X_balanced, y_balanced)

# 5. Evaluate on the original, unchanged test set
predictions = model.predict(X_test)

print("\n--- BALANCED LOGISTIC REGRESSION ---")
print("Precision:", round(
    precision_score(y_test, predictions, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, predictions, zero_division=0), 3
))
print("F1:", round(
    f1_score(y_test, predictions, zero_division=0), 3
))

print("\nBALANCE EXPERIMENT COMPLETE")
