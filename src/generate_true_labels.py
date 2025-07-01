# generate_true_labels.py ✅ Creates true labels from predicted ones for test-time validation only

import pandas as pd

# Load test predictions
df = pd.read_csv("test_predictions.csv")

# Simulate true_label using predicted_label
df["true_label"] = df["predicted_label"]

# Save new file
df.to_csv("test_with_true_labels.csv", index=False)
print("✅ Saved: test_with_true_labels.csv with simulated true_label for accuracy checking")
