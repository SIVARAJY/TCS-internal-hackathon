
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "canteen_demand_model.pkl"

# Load the trained model
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

# Example input for one 15-minute slot:
# 10:15 AM, Monday, class break, no campus event
sample = pd.DataFrame([{
    "hour": 10,
    "minute": 15,
    "day_of_week": 0,
    "is_weekend": 0,
    "class_break": 1,
    "campus_event": 0
}])[features]

prediction = model.predict(sample)[0]

print("Smart Canteen Demand Prediction")
print("--------------------------------")
print("Time slot: 10:15 - 10:30")
print("Predicted demand:", prediction)
