
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

rf_path = MODEL_DIR / "canteen_demand_model.pkl"
xgb_path = MODEL_DIR / "canteen_xgboost_model.pkl"

# --------------------------------------------------
# 1. Load both saved models
# --------------------------------------------------

rf_bundle = joblib.load(rf_path)
xgb_bundle = joblib.load(xgb_path)

rf_model = rf_bundle["model"]
xgb_model = xgb_bundle["model"]

features = rf_bundle["features"]
classes = ["Low", "Medium", "High"]

# Verify that both models use the same features and thresholds.
assert features == xgb_bundle["features"], "Feature mismatch!"
assert np.isclose(
    rf_bundle["low_threshold"],
    xgb_bundle["low_threshold"],
), "Low-demand thresholds differ!"

assert np.isclose(
    rf_bundle["high_threshold"],
    xgb_bundle["high_threshold"],
), "High-demand thresholds differ!"

low_threshold = rf_bundle["low_threshold"]
high_threshold = rf_bundle["high_threshold"]

# --------------------------------------------------
# 2. Load and preprocess the same dataset
# --------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError(
        f"Expected one CSV in {DATA_DIR}, found {len(csv_files)}."
    )

df = pd.read_csv(csv_files[0])
df["date"] = pd.to_datetime(df["date"], errors="raise")

for col in ["class_break", "campus_event"]:
    df[col] = (
        df[col].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )

    if df[col].isna().any():
        raise ValueError(f"Unexpected values in {col}")

df["orders_served"] = pd.to_numeric(
    df["orders_served"], errors="raise"
)

slot_data = (
    df.groupby(["date", "time_slot"], as_index=False)
    .agg(
        orders_served=("orders_served", "sum"),
        class_break=("class_break", "first"),
        campus_event=("campus_event", "first"),
    )
    .sort_values(["date", "time_slot"])
    .reset_index(drop=True)
)

start_time = slot_data["time_slot"].str.split("-").str[0]
parsed_time = pd.to_datetime(
    start_time, format="%H:%M", errors="coerce"
)

if parsed_time.isna().any():
    raise ValueError("Could not parse some time slots.")

slot_data["hour"] = parsed_time.dt.hour
slot_data["minute"] = parsed_time.dt.minute
slot_data["day_of_week"] = slot_data["date"].dt.dayofweek
slot_data["is_weekend"] = (
    slot_data["day_of_week"] >= 5
).astype(int)

# --------------------------------------------------
# 3. Re-create the same chronological split
# --------------------------------------------------

unique_dates = np.array(
    sorted(slot_data["date"].unique())
)

train_end = int(len(unique_dates) * 0.70)
val_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]

train = slot_data[slot_data["date"].isin(train_dates)].copy()
val = slot_data[slot_data["date"].isin(val_dates)].copy()
test = slot_data[slot_data["date"].isin(test_dates)].copy()

def classify_demand(orders):
    if orders < low_threshold:
        return "Low"
    elif orders <= high_threshold:
        return "Medium"
    return "High"

for part in [train, val, test]:
    part["demand_class"] = part["orders_served"].apply(
        classify_demand
    )

X_val = val[features]
y_val = val["demand_class"]

X_test = test[features]
y_test = test["demand_class"]

print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# --------------------------------------------------
# 4. Evaluate both models
# --------------------------------------------------

def evaluate(name, model, X, y):
    start = time.perf_counter()
    raw_predictions = model.predict(X)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # XGBoost was trained using Low=0, Medium=1, High=2.
    if name == "XGBoost":
        predictions = pd.Series(raw_predictions).map({
            0: "Low",
            1: "Medium",
            2: "High",
        })
    else:
        predictions = pd.Series(raw_predictions)

    predictions = predictions.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Macro Precision": precision_score(
            y, predictions, labels=classes,
            average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y, predictions, labels=classes,
            average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y, predictions, labels=classes,
            average="macro", zero_division=0
        ),
        "Prediction time (ms)": elapsed_ms,
        "predictions": predictions,
        "y_true": y,
    }


for split_name, X, y in [
    ("VALIDATION", X_val, y_val),
    ("TEST", X_test, y_test),
]:
    rf_result = evaluate("Random Forest", rf_model, X, y)
    xgb_result = evaluate("XGBoost", xgb_model, X, y)

    print(f"\n{'=' * 65}")
    print(f"{split_name} COMPARISON")
    print("=" * 65)

    comparison = pd.DataFrame([
        {k: v for k, v in result.items()
         if k not in ["predictions", "y_true"]}
        for result in [rf_result, xgb_result]
    ])

    print(comparison.round(4).to_string(index=False))

    for result in [rf_result, xgb_result]:
        print(f"\n{result['Model']} classification report:")
        print(classification_report(
            result["y_true"],
            result["predictions"],
            labels=classes,
            zero_division=0,
        ))

        print("Confusion matrix (actual rows, predicted columns):")
        print(confusion_matrix(
            result["y_true"],
            result["predictions"],
            labels=classes,
        ))
