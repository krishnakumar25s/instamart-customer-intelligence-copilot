
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the engineered lapse dataset
data = pd.read_csv(OUTPUT_DIR / "lapse_features.csv")

numeric_features = [
    "recency", "frequency", "monetary",
    "avg_discount_pct", "cancellation_rate",
    "avg_delivery_minutes", "age", "tenure_days",
]
categorical_features = [
    "city", "gender", "favourite_category"
]
feature_columns = numeric_features + categorical_features

X = data[feature_columns].copy()
y = data["lapsed"].astype(int)
customer_ids = data["customer_id"].copy()

assert not X.isna().any().any(), "Missing model features found"

# 2. Recreate the same split as the model evaluation
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, customer_ids,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# 3. Define preprocessing: fitted only on training data
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    (
        "categorical",
        OneHotEncoder(drop="first", handle_unknown="ignore"),
        categorical_features,
    ),
])

# 4. Fit the selected Random Forest configuration
best_forest = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
        ),
    ),
])

best_forest.fit(X_train, y_train)

# 5. Evaluate the selected model on the untouched test set
test_predictions = best_forest.predict(X_test)
test_probabilities = best_forest.predict_proba(X_test)[:, 1]

print("--- SELECTED RANDOM FOREST ---")
print("Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 3
))
print("F1:", round(
    f1_score(y_test, test_predictions, zero_division=0), 3
))

# 6. Save and reload the model
model_path = OUTPUT_DIR / "lapse_model.joblib"
joblib.dump(best_forest, model_path)

reloaded_model = joblib.load(model_path)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(test_predictions, reloaded_predictions), (
    "Reloaded model predictions do not match"
)

print("\nModel saved:", model_path)
print("Reloaded predictions match exactly: YES")

# 7. Show the five most important Random Forest features
fitted_preprocessor = best_forest.named_steps["preprocessor"]
fitted_forest = best_forest.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": fitted_forest.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n--- TOP FIVE RANDOM FOREST FEATURES ---")
print(importance_df.head(5).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
)

# 8. Fit the tuned Logistic Regression for coefficient interpretation
logistic_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features,
        ),
    ])),
    ("model", LogisticRegression(C=100, max_iter=1000)),
])

logistic_model.fit(X_train, y_train)

logistic_names = (
    logistic_model.named_steps["preprocessor"].get_feature_names_out()
)
logistic_coefficients = logistic_model.named_steps["model"].coef_[0]

coefficients_df = pd.DataFrame({
    "feature": logistic_names,
    "coefficient": logistic_coefficients,
}).sort_values("coefficient", ascending=False)

print("\n--- THREE LOGISTIC REGRESSION COEFFICIENTS ---")
print("Largest positive coefficients:")
print(coefficients_df.head(3).to_string(index=False))
print("\nLargest negative coefficients:")
print(coefficients_df.tail(3).sort_values("coefficient").to_string(index=False))

# 9. Generate the Premium-segment retention list
segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")

# Support both the assignment's requested names and the earlier script's names
if "is_premium" in segments.columns:
    premium_ids = set(
        segments.loc[segments["is_premium"].astype(bool), "customer_id"]
    )
elif "Premium" in segments.columns:
    premium_ids = set(
        segments.loc[segments["Premium"].astype(bool), "customer_id"]
    )
else:
    raise ValueError("No Premium flag found in customer_segments.csv")

retention = pd.DataFrame({
    "customer_id": ids_test.to_numpy(),
    "lapse_probability": test_probabilities,
    "actually_lapsed": y_test.to_numpy(),
})

retention = retention[
    retention["customer_id"].isin(premium_ids)
].sort_values("lapse_probability", ascending=False).head(10)

retention["lapse_probability"] = retention["lapse_probability"].round(4)
retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

print("\n--- PREMIUM RETENTION LIST ---")
print(retention.to_string(index=False))
print("Premium test customers listed:", len(retention))
print("Actually lapsed in this list:", int(retention["actually_lapsed"].sum()))

print("\nSaved: outputs/retention_list.csv")
print("\nPART B MODEL PERSISTENCE COMPLETE")
