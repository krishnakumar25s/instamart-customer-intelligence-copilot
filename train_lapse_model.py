
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# 1. Load the lapse dataset
DATA_PATH = Path("outputs/lapse_features.csv")
data = pd.read_csv(DATA_PATH)

# 2. Define features and target
target = "lapsed"

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

categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

X = data[numeric_features + categorical_features].copy()
y = data[target].astype(int)

# Ensure there are no missing values in model features
assert not X.isna().any().any(), "Missing feature values detected!"

# 3. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training lapsed customers:", int(y_train.sum()))
print("Test lapsed customers:", int(y_test.sum()))

# 4. Define preprocessing
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Baseline: always predict not lapsed
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n--- BASELINE ---")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
print(
    "Lapsed-class F1:",
    round(f1_score(y_test, baseline_predictions, zero_division=0), 3),
)

# 6. Tune Logistic Regression
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

logistic_search.fit(X_train, y_train)

print("\n--- LOGISTIC REGRESSION ---")
print("Best parameters:", logistic_search.best_params_)
print("Best CV F1:", round(logistic_search.best_score_, 3))

# 7. Tune Random Forest
forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    ),
])

forest_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}

forest_search = GridSearchCV(
    forest_pipeline,
    forest_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

forest_search.fit(X_train, y_train)

print("\n--- RANDOM FOREST ---")
print("Best parameters:", forest_search.best_params_)
print("Best CV F1:", round(forest_search.best_score_, 3))

# 8. Evaluate only after tuning is complete
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n--- {name}: TEST RESULTS ---")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Lapsed-class precision:", round(
        precision_score(y_test, predictions, zero_division=0), 3
    ))
    print("Lapsed-class recall:", round(
        recall_score(y_test, predictions, zero_division=0), 3
    ))
    print("Lapsed-class F1:", round(
        f1_score(y_test, predictions, zero_division=0), 3
    ))

evaluate_model("Logistic Regression", logistic_search.best_estimator_, X_test, y_test)
evaluate_model("Random Forest", forest_search.best_estimator_, X_test, y_test)

print("\nMODEL TUNING COMPLETE")
