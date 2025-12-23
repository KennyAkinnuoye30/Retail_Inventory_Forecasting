# logistic_stockout_model.py
# Predict stockout risk with Logistic Regression

# --- 0) Imports
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)
import joblib

# --- 1) Load cleaned dataset
CSV_PATH = Path("/Users/kenny/Projects/Retail_Inventory_Forecasting_Project/data/processed/clean.csv")
df = pd.read_csv(CSV_PATH)
print(f"Data loaded from {CSV_PATH} -> shape: {df.shape}")

print("Unique Product IDs:", df["Product ID"].nunique())
print("Unique Store IDs:", df["Store ID"].nunique())

# (optional) normalize column labels just in case
df.columns = df.columns.str.strip()

# --- 2) Target engineering (Stockout_Flag)
# 1 if demand exceeds inventory (shortage), else 0
df["Inventory Level"] = pd.to_numeric(df["Inventory Level"], errors="coerce")
df["Demand Forecast"] = pd.to_numeric(df["Demand Forecast"], errors="coerce")
df["Stockout_Flag"] = (df["Inventory Level"] < df["Demand Forecast"]).astype(int)

# --- Create derived ratio to avoid leakage ---
df["Inventory_Demand_Ratio"] = df["Inventory Level"] / (df["Demand Forecast"] + 1)

# Drop the original leak columns
df = df.drop(columns=["Inventory Level", "Demand Forecast"])

# Drop rows missing critical predictors (adjust as needed)
must_have = ["Inventory_Demand_Ratio", "Price", "Competitor Pricing"]
present = [c for c in must_have if c in df.columns]
df = df.dropna(subset=present)


# --- 3) Features / target
target = "Stockout_Flag"
drop_cols = [c for c in [target, "Date"] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[target]

# --- 4) Feature lists
numeric_features = [
    "Inventory_Demand_Ratio",
    "Units Sold", "Units Ordered",
    "Price", "Discount", "Competitor Pricing",
    "year", "month", "dayofweek", "is_weekend"
]

categorical_features = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Holiday/Promotion", "Seasonality"
]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

# --- 5) Train/test split (stratify preserves class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --- 6) Preprocess + model
numeric_transformer = Pipeline(steps=[
    # with_mean=False to support sparse concat after OHE
    ("scaler", StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# Tip: class_weight="balanced" helps if positives are rare
log_reg = LogisticRegression(
    max_iter=2000,
    solver="liblinear",       # use solver="saga" + penalty="l1" for sparse/feature selection
    class_weight="balanced",
    penalty="l2"
)

clf = Pipeline(steps=[("prep", preprocess), ("model", log_reg)])

# --- 7) Fit
clf.fit(X_train, y_train)

# --- 8) Evaluate (default 0.5 threshold)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== Evaluation @ threshold=0.50 ===")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("PR AUC (Average Precision):", round(average_precision_score(y_test, y_proba), 3))

# --- 9) Optional: threshold tuning example (e.g., more recall)
threshold = 0.35
y_pred_tuned = (y_proba >= threshold).astype(int)
print(f"\n=== Evaluation @ threshold={threshold:.2f} ===")
print(classification_report(y_test, y_pred_tuned, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))

# --- 10) Inspect drivers (odds ratios)
try:
    ohe = clf.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
except Exception:
    cat_names = []

num_names = numeric_features
all_feature_names = num_names + cat_names

coefs = clf.named_steps["model"].coef_.ravel()
# If feature name/coef mismatch due to empty categories, align length
if len(coefs) != len(all_feature_names):
    # fallback: skip names to avoid error
    print("\n[Note] Feature name/coef length mismatch; printing top coefficients only.")
    top_idx = np.argsort(np.abs(coefs))[::-1][:25]
    print(pd.DataFrame({
        "coef": coefs[top_idx],
        "odds_ratio": np.exp(coefs[top_idx])
    }).to_string(index=False))
else:
    coef_df = pd.DataFrame({
        "feature": all_feature_names,
        "coef": coefs,
        "odds_ratio": np.exp(coefs)  # exp(beta): multiplicative change in odds per +1 unit
    }).sort_values("coef", key=np.abs, ascending=False)
    print("\nTop positive/negative drivers (by |coef|):")
    print(coef_df.head(25).to_string(index=False))

# --- 11) Save model for later scoring
MODEL_PATH = CSV_PATH.parent / "log_reg_stockout_pipeline.joblib"
joblib.dump(clf, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# --- 12) How to score NEW data (example)
# Use the same columns your model expects. Easiest demo: take one row from X_test.
sample = X_test.iloc[[0]].copy()
proba = clf.predict_proba(sample)[:, 1][0]
pred = clf.predict(sample)[0]
print("\n=== Example prediction on one held-out row ===")
print("Predicted stockout probability:", round(proba, 3))
print("Predicted risk (0/1):", int(pred))


#Print stockout class distribution
print(df['Stockout_Flag'].value_counts(normalize=True))
df['Stockout_Flag'].value_counts().plot(kind='bar', title='Class Distribution')




# If you want to score a handcrafted new row, build a dict with your columns, e.g.:
# new_row = {
#     "Store ID": "A1",
#     "Product ID": "P200",
#     "Category": "Beverages",
#     "Region": "West",
#     "Inventory Level": 90,
#     "Units Sold": 100,
#     "Units Ordered": 120,
#     "Demand Forecast": 150,
#     "Price": 2.99,
#     "Discount": 10,
#     "Weather Condition": "Sunny",
#     "Holiday/Promotion": 0,
#     "Competitor Pricing": 3.20,
#     "Seasonality": "Summer",
#     "year": 2024,
#     "month": 7,
#     "dayofweek": 2,
#     "is_weekend": 0
# }
# new_df = pd.DataFrame([new_row])
# print("New row stockout prob:", clf.predict_proba(new_df)[:, 1][0])
# print("New row predicted flag:", int(clf.predict(new_df)[0]))
