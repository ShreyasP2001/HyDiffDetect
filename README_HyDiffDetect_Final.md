# 🧠 HyDiffDetect – Hybrid AI Image Detection (Real | AI-Generated | Tampered)

**HyDiffDetect** is a high-accuracy hybrid AI image forensics system that classifies images into:
- ✅ Real (unaltered photos)
- 🤖 AI-Generated (e.g. Stable Diffusion, GANs)
- 🛠️ Tampered (manipulated or edited)

---

## 🚀 How It Works

This system combines:
- **DIRE** (diffusion-based image reconstruction)
- **CLIP ViT** embeddings (global and crop-level)
- **Image forensics**: entropy, PRNU, RGB, FFT, DCT, LBP
- **Fingerprinting** via embedding variance + max distance
- **XGBoost classifier** trained on rich 533-feature vectors
- **SHAP Explainability** for transparency

---

## 📁 Project Structure

```
HyDiffDetect/
├── checkpoints/                  # Pretrained DIRE models (.pth)
├── CLIP-main/                    # CLIP ViT feature extractor
├── data/
│   ├── train_images/
│   ├── test_images/
│   └── reconstructed_images/
├── networks/                     # Network modules
├── results/                      # Features, predictions, SHAP plots
├── src/
│   ├── app.py                    # Streamlit UI for image prediction
│   ├── batch_reconstruct_all.py # Reconstruct all input images
│   ├── extract_features.py      # Full 533-feature extractor
│   ├── train_model.py           # Train pipeline for classifier
│   ├── test_model.py            # Predict test set
│   ├── evaluate_model.py        # Accuracy/metrics on predictions
│   ├── train_classifier_new.py  # Optional classifier override
│   └── generate_true_labels.py  # Simulates truth for demo
└── requirements.txt
```

---

## 🔧 Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Also install manually (if needed):

```bash
pip install streamlit torch torchvision opencv-python xgboost scikit-learn pandas matplotlib shap
```

### 2. Prepare Data

- `data/train_images/` ← images with true labels
- `data/test_images/` ← unlabeled images to predict

Add pretrained checkpoints to:
```
checkpoints/
  celebahq_sdv2.pth
  imagenet_adm.pth
  isun_adm.pth
  isun_iddpm.pth
  isun_pndm.pth
  isun_stylegan.pth
```

---

## 🔄 Run the Pipeline

### 🛠 Step 1: Reconstruct

```bash
python src/batch_reconstruct_all.py
```

### 📊 Step 2: Extract Features

```bash
python src/extract_features.py
```

### 🧠 Step 3: Train Classifier

```bash
python src/train_model.py
```

### 🎯 Step 4: Predict on Test Set

```bash
python src/test_model.py
```

### 📈 Step 5: Evaluate Model

```bash
python src/evaluate_model.py
```

---

## ▶️ Streamlit App

To run the real-time GUI:

```bash
streamlit run src/app.py
```

You can:
- Upload any image
- Get predicted class
- See SHAP explanation
- View reconstruction side-by-side

---

## 🧪 Feature Vector per Image

**Total: 533 features**  
Includes:
- `clip_distance`, `dire_score`, `entropy`
- `mean_r`, `mean_g`, `mean_b`, `dct`, `fft`, `prnu`
- `clip_crop_var`, `clip_crop_max`
- `lbp_0` to `lbp_8`
- `clip_vit_0` to `clip_vit_511`

---

## 📊 Outputs (saved in `results/`)

- `train_features.csv`, `test_features.csv`
- `test_predictions.csv`, `test_with_true_labels.csv`
- `xgb_model_3class.pkl`, `scaler_3class.pkl`
- `shap_summary_plot.png`

---

## 👨‍💻 Author

- **Shreyas Prakash**
- GitHub: [ShreyasP2001](https://github.com/ShreyasP2001)

---

## 📄 License

MIT License © 2025 Shreyas Prakash
