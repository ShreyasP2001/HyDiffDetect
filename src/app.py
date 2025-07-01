# app.py ✅ Streamlit GUI for AI image prediction (Real / AI / Tampered)

import streamlit as st
import os
import torch
import joblib
import numpy as np
import pandas as pd
import clip
from PIL import Image
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct as scipy_dct
from numpy.fft import fft2
import cv2

# === Load model & scaler ===
model = joblib.load("xgb_model_3class.pkl")
scaler = joblib.load("scaler_3class.pkl")

# === Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)

# === DIRE dummy logic (replace with real recon if needed)
def reconstruct_image(image_tensor):
    return (image_tensor * 0.8).clamp(0, 1)

# === Feature extraction utils ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def get_clip_embedding(image):
    tensor = preprocess_clip(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(tensor).squeeze().cpu().numpy()

def get_crops(image, size=224):
    w, h = image.size
    return [TF.center_crop(image, size),
            image.crop((0, 0, size, size)),
            image.crop((w-size, 0, w, size)),
            image.crop((0, h-size, size, h)),
            image.crop((w-size, h-size, w, h))]

def compute_entropy(img):
    gray = img.convert("L")
    return entropy(np.array(gray), disk(5)).mean()

def compute_color_stats(img):
    arr = np.array(img.resize((224, 224)))
    return arr.mean(axis=(0, 1))

def compute_dct_energy(img):
    gray = img.convert("L").resize((224, 224))
    arr = np.array(gray)
    coeffs = scipy_dct(scipy_dct(arr.T, norm='ortho').T, norm='ortho')
    return np.abs(coeffs[:10, :10]).mean()

def compute_fft_energy(img):
    gray = np.array(img.convert("L").resize((224, 224)))
    fft_vals = np.abs(fft2(gray))
    return np.mean(fft_vals[:10, :10])

def compute_lbp_hist(img):
    gray = np.array(img.convert("L").resize((224, 224)))
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return (hist / (hist.sum() + 1e-6)).astype("float")

def compute_prnu(img):
    arr = np.array(img.resize((224, 224)).convert("L"))
    noise = arr - cv2.medianBlur(arr, 3)
    return np.std(noise)

def get_clip_crop_stats(image):
    crops = get_crops(image)
    crop_embeddings = np.array([get_clip_embedding(c) for c in crops])
    mean_embedding = crop_embeddings.mean(axis=0)
    max_dist = np.max([np.linalg.norm(mean_embedding - e) for e in crop_embeddings])
    var = np.var(crop_embeddings, axis=0).mean()
    return var, max_dist, mean_embedding

# === Streamlit UI ===
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("🧠 AI Image Prediction (Real / AI-Generated / Tampered)")

uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # === Feature extraction
    st.subheader("🔍 Extracting Features...")
    crop_var, crop_max, clip_vit = get_clip_crop_stats(image)
    emb1 = get_clip_embedding(image)
    recon_tensor = reconstruct_image(transform(image).unsqueeze(0).to(device))
    recon_image = transforms.ToPILImage()(recon_tensor.squeeze().cpu())
    emb2 = get_clip_embedding(recon_image)
    clip_distance = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    dire_score = 0.5  # Dummy placeholder
    ent = compute_entropy(image)
    r, g, b = compute_color_stats(image)
    dct_val = compute_dct_energy(image)
    fft_val = compute_fft_energy(image)
    prnu = compute_prnu(image)
    lbp_hist = compute_lbp_hist(image)

    # Combine features
    row = [clip_distance, dire_score, ent, r, g, b, dct_val, fft_val, prnu, crop_var, crop_max]
    row += lbp_hist.tolist() + clip_vit.tolist()

    df = pd.DataFrame([row], columns=[
        "clip_distance", "dire_score", "entropy", "mean_r", "mean_g", "mean_b", "dct", "fft", "prnu", "clip_crop_var", "clip_crop_max"
    ] + [f"lbp_{i}" for i in range(9)] + [f"clip_vit_{i}" for i in range(len(clip_vit))])

    # === Predict
    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label_map = {0: "Real", 1: "AI-Generated", 2: "Tampered"}
    predicted_label = label_map[pred]
    confidence = round(np.max(proba), 4)

    st.success(f"✅ **Prediction**: {predicted_label}")
    st.metric("🔒 Confidence", f"{confidence * 100:.2f} %")

    # === Show reconstructed image
    st.subheader("🔁 Simulated Reconstructed Image (DIRE Placeholder)")
    st.image(recon_image, caption="Reconstructed (simulated)", use_column_width=True)

    # === Save prediction
    output = pd.DataFrame({
        "filename": [uploaded_file.name],
        "predicted_label": [predicted_label],
        "confidence": [confidence]
    })
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction CSV", data=csv, file_name="prediction_result.csv")

