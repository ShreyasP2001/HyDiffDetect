import torch
import clip
from PIL import Image
import numpy as np
import os

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_features(img_path):
    """Extract CLIP embeddings from image."""
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        exit(1)

    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img)
    return embedding.cpu().numpy()

# Input paths
original_path = "../DIRE-main/original_images/test1.png"
reconstructed_path = "../DIRE-main/reconstructed_images/test1.png"

# Output path
print(f"📌 Original: {original_path}")
print(f"📌 Reconstructed: {reconstructed_path}")

# Extract features
original_embedding = extract_clip_features(original_path)
recon_embedding = extract_clip_features(reconstructed_path)

# Compute cosine distance
cosine_similarity = np.dot(original_embedding, recon_embedding.T) / (
    np.linalg.norm(original_embedding) * np.linalg.norm(recon_embedding)
)
semantic_distance = 1 - cosine_similarity

print("🔍 Semantic Distance:", semantic_distance.item())

# Optional: Save for classifier input
with open("clip_results.txt", "w") as f:
    f.write(f"{os.path.basename(original_path)}, {semantic_distance.item():.4f}\n")

print("✅ CLIP analysis done and saved to clip_results.txt.")
