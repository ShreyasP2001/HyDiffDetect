import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="test_predictions.csv", help="CSV with predicted labels")
parser.add_argument("--true_labels_csv", default=None, help="Optional: CSV with filename,true_label")
args = parser.parse_args()

# Load prediction file
df = pd.read_csv(args.input_csv)

# Load external true labels if given
if args.true_labels_csv:
    true_df = pd.read_csv(args.true_labels_csv)
    df = pd.merge(df, true_df, on="filename", how="left")

# Check columns
if "true_label" not in df.columns or "predicted_label" not in df.columns:
    print("❌ 'true_label' or 'predicted_label' column not found")
    exit()

# Evaluate
y_true = df["true_label"].astype(str).str.strip()
y_pred = df["predicted_label"].astype(str).str.strip()

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, labels=["Real", "AI-Generated", "Tampered"]))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=["Real", "AI-Generated", "Tampered"]))
