# train_classifier.py ✅ Trains model on extracted features from train_images/

import pandas as pd
import joblib
import argparse
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="train_features.csv", help="Path to training feature CSV")
args = parser.parse_args()

# Load features
df = pd.read_csv(args.input_csv)

# Check if label exists
if "label" not in df.columns:
    raise ValueError("❌ 'label' column not found in training dataset.")

y = df["label"]
X = df.drop(columns=["filename", "label"])
feature_names = X.columns.tolist()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Train
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "xgb_model_3class.pkl")
joblib.dump(scaler, "scaler_3class.pkl")
print("\n✅ Saved: xgb_model_3class.pkl and scaler_3class.pkl")

# SHAP summary
print("\n🔍 Generating SHAP summary plot...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
print("✅ SHAP summary saved as shap_summary_plot.png")
