# train_model.py ✅ Unified training pipeline

import os
import pandas as pd
import joblib
import shap
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from subprocess import run

# ========== CONFIG ==========
TRAIN_IMAGE_DIR = "train_images"
FEATURES_CSV = "train_features.csv"
CHECKPOINTS_DIR = "checkpoints"
RECON_DIR = "reconstructed_images"
MODEL_PATH = "xgb_model_3class.pkl"
SCALER_PATH = "scaler_3class.pkl"
SHAP_PLOT_PATH = "shap_summary_plot.png"

# ========== STEP 1: Extract Features ==========
print("🔍 Step 1: Extracting features from training images...")
run(["python", "extract_features.py",
     "--input_dir", TRAIN_IMAGE_DIR,
     "--output_csv", FEATURES_CSV,
     "--checkpoints", CHECKPOINTS_DIR,
     "--recon_root", RECON_DIR])

# ========== STEP 2: Load Data ==========
print("📊 Step 2: Loading features from CSV...")
df = pd.read_csv(FEATURES_CSV)

if "label" not in df.columns:
    raise ValueError("❌ 'label' column not found in extracted features.")

y = df["label"]
X = df.drop(columns=["filename", "label"])
feature_names = X.columns.tolist()

# ========== STEP 3: Scale Features ==========
print("⚙️  Step 3: Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== STEP 4: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ========== STEP 5: Train Model ==========
print("🤖 Step 4: Training XGBoost classifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# ========== STEP 6: Evaluate ==========
print("\n✅ Classification Report:\n")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========== STEP 7: Save Model + Scaler ==========
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅ Saved model to {MODEL_PATH}")
print(f"✅ Saved scaler to {SCALER_PATH}")

# ========== STEP 8: SHAP Plot ==========
print("📈 Step 5: Generating SHAP summary plot...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(SHAP_PLOT_PATH)
print(f"✅ SHAP plot saved to {SHAP_PLOT_PATH}")
