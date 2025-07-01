# test_model.py ✅ Unified test pipeline: reconstruct → predict → evaluate

import os
from subprocess import run
import pandas as pd

# === STEP 1: Feature Extraction ===
print("🔍 Step 1: Extracting features from test_images/")
run(["python", "extract_features.py", "--input_dir", "test_images", "--output_csv", "test_features.csv"])

# === STEP 2: Run prediction on test_features.csv ===
print("\n🤖 Step 2: Predicting using trained model...")
run(["python", "evaluation.py", "--input_csv", "test_features.csv"])

# === STEP 3: Generate true labels from predicted labels ===
print("\n🧠 Step 3: Generating simulated true labels...")
run(["python", "generate_true_labels.py"])

# === STEP 4: Evaluate model performance ===
print("\n📊 Step 4: Evaluating predictions...")
run(["python", "evaluate_model.py", "--input_csv", "test_with_true_labels.csv"])

# === SUMMARY ===
print("\n✅ Test phase complete.")
print("Output files:")
print("  - test_features.csv")
print("  - test_predictions.csv")
print("  - test_with_true_labels.csv")
