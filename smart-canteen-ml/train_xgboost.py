
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "canteen_xgboost_model.pkl"

csv_files = list(DATA_DIR.glob("*.csv"))
if len(csv_files) != 1:
    raise ValueError(f"Expected one CSV in {DATA_DIR}; found {len(csv_files)}")

df = pd.read_csv(csv_files[0])
df["date"] = pd.to_datetime(df["date"], errors="raise")
df["orders_served"] = pd.to_numeric(df["orders_served"], errors="raise")

for col in ["class_break", "campus_event"]:
    df[col] = (
        df[col].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )
    if df[col].isna().any():
        raise ValueError(f"Unexpected values in {col}")

# Aggregate counter records into one row per date/time slot.
data = (
    df.groupby(["date", "time_slot"], as_index=False)
    .agg(
        orders_served=("orders_served", "sum"),
        class_break=("class_break", "first"),
        campus_event=("campus_event", "first"),
    )
    .sort_values(["date", "time_slot"])
    .reset_index(drop=True)
)

start_time = data["time_slot"].str.split("-").str[0]
parsed_time = pd.to_datetime(start_time, format="%H:%M", errors="coerce")
if parsed_time.isna().any():
    raise ValueError("Could not parse one or more time slots.")

data["hour"] = parsed_time.dt.hour
data["minute"] = parsed_time.dt.minute
data["day_of_week"] = data["date"].dt.dayofweek
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

FEATURES = [
    "hour", "minute", "day_of_week", "is_weekend",
    "class_break", "campus_event"
]
CLASSES = ["Low", "Medium", "High"]

# Chronological split by unique date: 70% / 15% / 15%.
dates = np.array(sorted(data["date"].unique()))
train_end = int(len(dates) * 0.70)
val_end = int(len(dates) * 0.85)

train_dates = dates[:train_end]
val_dates = dates[train_end:val_end]
test_dates = dates[val_end:]

train = data[data["date"].isin(train_dates)].copy()
val = data[data["date"].isin(val_dates)].copy()
test = data[data["date"].isin(test_dates)].copy()

# Demand thresholds are derived from training data only.
low_threshold = float(train["orders_served"].quantile(0.33))
high_threshold = float(train["orders_served"].quantile(0.67))

def classify_demand(orders):
    if orders < low_threshold:
        return "Low"
    if orders <= high_threshold:
        return "Medium"
    return "High"

for part in [train, val, test]:
    part["demand_class"] = part["orders_served"].apply(classify_demand)

class_to_number = {"Low": 0, "Medium": 1, "High": 2}
number_to_class = {v: k for k, v in class_to_number.items()}

X_train = train[FEATURES]
y_train_encoded = train["demand_class"].map(class_to_number)

X_val = val[FEATURES]
y_val = val["demand_class"]
y_val_encoded = y_val.map(class_to_number)

X_test = test[FEATURES]
y_test = test["demand_class"]

print("Rows:", len(train), "train |", len(val), "validation |", len(test), "test")
print("Demand thresholds:", low_threshold, high_threshold)

# Tune only against validation data.
parameter_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_weight": [1, 3],
}

best_model = None
best_score = -1
best_params = None

for params in ParameterGrid(parameter_grid):
    candidate = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        **params,
    )

    candidate.fit(X_train, y_train_encoded)
    pred = candidate.predict(X_val)

    score = f1_score(
        y_val_encoded, pred, average="macro", zero_division=0
    )
    print(f"{params} | validation macro F1: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = candidate
        best_params = params

print("\nBest parameters:", best_params)
print(f"Best validation macro F1: {best_score:.4f}")

# Final evaluation on the untouched test period.
test_encoded = y_test.map(class_to_number)
test_pred_encoded = best_model.predict(X_test).astype(int)
test_pred = pd.Series(test_pred_encoded).map(number_to_class)

print("\nFINAL TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Macro precision:", round(
    precision_score(y_test, test_pred, labels=CLASSES,
                    average="macro", zero_division=0), 4
))
print("Macro recall:", round(
    recall_score(y_test, test_pred, labels=CLASSES,
                 average="macro", zero_division=0), 4
))
print("Macro F1:", round(
    f1_score(y_test, test_pred, labels=CLASSES,
             average="macro", zero_division=0), 4
))

print("\nClassification report:")
print(classification_report(
    y_test, test_pred, labels=CLASSES, zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred, labels=CLASSES))

# Save the tuned model in the same bundle format.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "low_threshold": low_threshold,
    "high_threshold": high_threshold,
    "classes": CLASSES,
    "class_to_number": class_to_number,
    "target": "orders_served",
}, MODEL_PATH)

print(f"\nSaved tuned XGBoost model to: {MODEL_PATH}")
