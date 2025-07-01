# extract_features.py ✅ Final version with true_label for test, label for train

import os, sys, argparse, torch, clip, numpy as np, pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct as scipy_dct
from numpy.fft import fft2
import cv2
from utils.utils import get_network
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# --------------- ARGUMENTS ---------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Folder with original input images (train_images or test_images)")
parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
parser.add_argument("--checkpoints", default="checkpoints", help="Path to DIRE .pth checkpoints")
parser.add_argument("--recon_root", default="reconstructed_images", help="Root path to reconstructed images")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

label_map = {0: "Real", 1: "AI-Generated", 2: "Tampered"}

# --------------- LOAD DIRE MODELS ---------------
models = {}
for ckpt_file in os.listdir(args.checkpoints):
    if ckpt_file.endswith(".pth"):
        model = get_network("resnet50")
        state = torch.load(os.path.join(args.checkpoints, ckpt_file), map_location=device)
        if "model" in state: state = state["model"]
        model.load_state_dict(state)
        model.eval().to(device)
        models[ckpt_file.replace(".pth", "")] = model

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
])

# --------------- FEATURE FUNCTIONS ---------------
def get_clip_embedding(image):
    tensor = preprocess(image).unsqueeze(0).to(device)
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

def detect_label(fname):
    name = fname.lower()
    if "tampered" in name or "tp" in name:
        return 2
    elif "ai" in name or "gen" in name:
        return 1
    elif "real" in name:
        return 0
    else:
        return None

# --------------- MAIN EXTRACTION LOOP ---------------
rows = []
input_type = "test" if "test" in args.input_dir.lower() else "train"

for fname in os.listdir(args.input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    try:
        original = Image.open(os.path.join(args.input_dir, fname)).convert("RGB")
    except:
        print(f"⚠️ Skipping unreadable: {fname}")
        continue

    crop_var, crop_max, clip_vit = get_clip_crop_stats(original)

    clip_scores, dire_scores = [], []
    for model_name, model in models.items():
        recon_path = os.path.join(args.recon_root, input_type, model_name, fname)
        if os.path.exists(recon_path):
            try:
                recon = Image.open(recon_path).convert("RGB")
                emb1 = get_clip_embedding(original)
                emb2 = get_clip_embedding(recon)
                clip_scores.append(1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                dire_tensor = transform(original)
                normed = TF.normalize(dire_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                with torch.no_grad():
                    dire_scores.append(model(normed.unsqueeze(0).to(device)).sigmoid().item())
            except:
                print(f"⚠️ Error loading recon for: {fname}")
                continue

    if not clip_scores or not dire_scores:
        print(f"⚠️ Skipping (no recon found): {fname}")
        continue

    ent = compute_entropy(original)
    r, g, b = compute_color_stats(original)
    dct_val = compute_dct_energy(original)
    fft_val = compute_fft_energy(original)
    prnu = compute_prnu(original)
    lbp_hist = compute_lbp_hist(original)

    row = [fname, np.mean(clip_scores), np.mean(dire_scores), ent, r, g, b, dct_val, fft_val, prnu, crop_var, crop_max]
    row += lbp_hist.tolist() + clip_vit.tolist()

    label = detect_label(fname)
    if label is not None:
        if input_type == "test":
            row += [label_map[label]]
        else:
            row += [label]

    rows.append(row)

# --------------- SAVE OUTPUT ---------------
base_cols = ["filename", "clip_distance", "dire_score", "entropy", "mean_r", "mean_g", "mean_b", "dct", "fft", "prnu", "clip_crop_var", "clip_crop_max"]
base_cols += [f"lbp_{i}" for i in range(9)]
base_cols += [f"clip_vit_{i}" for i in range(len(clip_vit))]

if any(detect_label(f) is not None for f in os.listdir(args.input_dir)):
    if input_type == "test":
        base_cols += ["true_label"]
    else:
        base_cols += ["label"]

df = pd.DataFrame(rows, columns=base_cols)
df.to_csv(args.output_csv, index=False)

print(f"\n✅ Saved features: {args.output_csv}")
print(f"✅ Processed {len(rows)} images from {args.input_dir}")
print(f"✅ Reconstructions used from: {os.path.join(args.recon_root, input_type)}")
