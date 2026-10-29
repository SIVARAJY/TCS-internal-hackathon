
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = next((BASE_DIR / "data").glob("*.csv"), None)

if DATA_PATH is None:
    raise FileNotFoundError("No CSV file found inside the data folder.")
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "canteen_demand_model.pkl"

FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "class_break",
    "campus_event",
]

CLASSES = ["Low", "Medium", "High"]


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------
print("Script location:", Path(__file__).resolve())
print("Expected CSV path:", DATA_PATH)
print("CSV exists:", DATA_PATH.exists())
print("Files in data folder:", list((BASE_DIR / "data").glob("*")))
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place your CSV inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "date",
    "time_slot",
    "counter_id",
    "orders_served",
    "queue_length",
    "avg_wait_min",
    "staff_on_duty",
    "class_break",
    "campus_event",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. Clean and aggregate counter-level records
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="raise")

for column in ["orders_served", "queue_length",
               "avg_wait_min", "staff_on_duty"]:
    df[column] = pd.to_numeric(df[column], errors="raise")

for column in ["class_break", "campus_event"]:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if df[column].isna().any():
        raise ValueError(
            f"Unexpected values found in {column}. "
            "Expected only Yes/No."
        )

# One row per date and 15-minute slot.
# This assumes each counter's queue length is separate.
slot_data = (
    df.groupby(["date", "time_slot"], as_index=False)
    .agg(
        orders_served=("orders_served", "sum"),
        queue_length=("queue_length", "sum"),
        avg_wait_min=("avg_wait_min", "mean"),
        staff_on_duty=("staff_on_duty", "sum"),
        class_break=("class_break", "first"),
        campus_event=("campus_event", "first"),
        counter_count=("counter_id", "nunique"),
    )
    .sort_values(["date", "time_slot"])
    .reset_index(drop=True)
)

print("Aggregated dataset shape:", slot_data.shape)

if not (slot_data["counter_count"] == df["counter_id"].nunique()).all():
    raise ValueError(
        "Some date/time slots do not contain every counter. "
        "Check the aggregation before training."
    )


# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------

model_data = slot_data.copy()

start_time = model_data["time_slot"].str.split("-").str[0]
parsed_time = pd.to_datetime(start_time, format="%H:%M", errors="coerce")

if parsed_time.isna().any():
    raise ValueError("Some time_slot values could not be parsed.")

model_data["hour"] = parsed_time.dt.hour
model_data["minute"] = parsed_time.dt.minute
model_data["day_of_week"] = model_data["date"].dt.dayofweek
model_data["is_weekend"] = (
    model_data["day_of_week"] >= 5
).astype(int)

model_data = model_data.sort_values(
    ["date", "time_slot"]
).reset_index(drop=True)

if model_data[FEATURES + ["orders_served"]].isna().any().any():
    raise ValueError("Missing values found in features or target.")

print("Features:", FEATURES)
print("Target: orders_served")


# --------------------------------------------------
# 5. Chronological train/validation/test split
# --------------------------------------------------

unique_dates = np.array(
    sorted(model_data["date"].unique())
)

if len(unique_dates) < 3:
    raise ValueError("Not enough unique dates to split the dataset.")

train_end = int(len(unique_dates) * 0.70)
val_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]

train = model_data[model_data["date"].isin(train_dates)].copy()
val = model_data[model_data["date"].isin(val_dates)].copy()
test = model_data[model_data["date"].isin(test_dates)].copy()

print("\nDataset split:")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

print(
    "Training dates:",
    train["date"].min().date(),
    "to",
    train["date"].max().date(),
)
print(
    "Validation dates:",
    val["date"].min().date(),
    "to",
    val["date"].max().date(),
)
print(
    "Testing dates:",
    test["date"].min().date(),
    "to",
    test["date"].max().date(),
)


# --------------------------------------------------
# 6. Define demand categories using training data
# --------------------------------------------------

low_threshold = float(train["orders_served"].quantile(0.33))
high_threshold = float(train["orders_served"].quantile(0.67))

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

X_train = train[FEATURES]
y_train = train["demand_class"]

X_val = val[FEATURES]
y_val = val["demand_class"]

X_test = test[FEATURES]
y_test = test["demand_class"]

print("\nDemand thresholds:")
print(f"Low: below {low_threshold:.2f} orders")
print(f"Medium: {low_threshold:.2f} to {high_threshold:.2f} orders")
print(f"High: above {high_threshold:.2f} orders")


# --------------------------------------------------
# 7. Train Random Forest classifier
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed!")


# --------------------------------------------------
# 8. Evaluation helper
# --------------------------------------------------

def evaluate_model(name, X, y):
    predictions = model.predict(X)

    print(f"\n{'=' * 50}")
    print(f"{name} RESULTS")
    print("=" * 50)

    print(f"Accuracy:          {accuracy_score(y, predictions):.4f}")
    print(
        "Precision (macro): ",
        f"{precision_score(y, predictions, average='macro', zero_division=0):.4f}",
    )
    print(
        "Recall (macro):    ",
        f"{recall_score(y, predictions, average='macro', zero_division=0):.4f}",
    )
    print(
        "F1-score (macro):  ",
        f"{f1_score(y, predictions, average='macro', zero_division=0):.4f}",
    )

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y, predictions, labels=CLASSES))


# Use validation results to assess the model.
evaluate_model("VALIDATION", X_val, y_val)

# Final evaluation on later, unseen dates.
evaluate_model("TEST", X_test, y_test)


# --------------------------------------------------
# 9. Save model and configuration
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_bundle = {
    "model": model,
    "features": FEATURES,
    "low_threshold": low_threshold,
    "high_threshold": high_threshold,
    "classes": CLASSES,
    "target": "orders_served",
    "aggregation": "sum across counters per 15-minute slot",
}

joblib.dump(model_bundle, MODEL_PATH)

print(f"\nModel saved successfully:\n{MODEL_PATH}")