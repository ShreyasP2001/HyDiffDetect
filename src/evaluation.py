# evaluation.py ✅ Predicts on test_features.csv, outputs test_predictions.csv with no accuracy evaluation

import pandas as pd
import numpy as np
import joblib
import argparse

# === Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="test_features.csv", help="CSV file with extracted features")
args = parser.parse_args()

# === Load input data
df = pd.read_csv(args.input_csv)
print("🧪 Columns found:", df.columns.tolist())

# Drop label columns if present
drop_cols = ["filename"]
if "label" in df.columns: drop_cols.append("label")
if "true_label" in df.columns: drop_cols.append("true_label")

X = df.drop(columns=drop_cols)
filenames = df["filename"]

# === Load model and scaler
model = joblib.load("xgb_model_3class.pkl")
scaler = joblib.load("scaler_3class.pkl")

# === Feature validation
expected_features = scaler.feature_names_in_
missing = set(expected_features) - set(X.columns)
extra = set(X.columns) - set(expected_features)
if missing or extra:
    print("❌ Feature mismatch:")
    if missing: print("🔺 Missing:", sorted(missing))
    if extra: print("🔻 Extra:", sorted(extra))
    exit(1)

# === Predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

label_map = {0: "Real", 1: "AI-Generated", 2: "Tampered"}
predicted_labels = [label_map[i] for i in y_pred]
confidences = [round(np.max(p), 4) for p in y_proba]

# === Save predictions
results = pd.DataFrame({
    "filename": filenames,
    "predicted_label": predicted_labels,
    "confidence": confidences
})

results.to_csv("test_predictions.csv", index=False)
print("\n✅ Saved: test_predictions.csv")
print("🔍 Prediction complete (no accuracy evaluation included).")
