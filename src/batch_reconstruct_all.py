import os
import torch
import argparse
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import sys

# ✅ Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dire_path = os.path.join(current_dir, "DIRE-main")
sys.path.append(dire_path)

from utils.utils import get_network

# ✅ CLI args
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    default="train_images",  # defaults to train_images inside HyDiffDetect/
    help="Input folder containing original images (train_images or test_images)"
)

parser.add_argument(
    "--output_root",
    default="reconstructed_images",  # inside HyDiffDetect/
    help="Folder to save reconstructed images"
)

parser.add_argument(
    "--checkpoint_folder",
    default="checkpoints",  # inside HyDiffDetect/
    help="Folder containing DIRE .pth model checkpoints"
)

args = parser.parse_args()

# ✅ Ensure output folder exists
os.makedirs(args.output_root, exist_ok=True)

# ✅ Load checkpoints
checkpoints = [f for f in os.listdir(args.checkpoint_folder) if f.endswith(".pth")]

# ✅ Transform for DIRE
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Placeholder reconstruction logic (can replace later)
def reconstruct_image(image_tensor):
    return (image_tensor * 0.8).clamp(0, 1)

# ✅ Loop through models
for ckpt_name in checkpoints:
    ckpt_path = os.path.join(args.checkpoint_folder, ckpt_name)
    model_name = ckpt_name.replace(".pth", "")
    input_type = "test" if "test" in args.input_dir.lower() else "train"
    save_dir = os.path.join(args.output_root, input_type, model_name)

    os.makedirs(save_dir, exist_ok=True)

    model = get_network("resnet50")
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    print(f"\n🔁 Using model: {model_name}")

    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        orig_path = os.path.join(args.input_dir, filename)
        save_path = os.path.join(save_dir, filename)

        try:
            image = Image.open(orig_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            if img_tensor.dim() != 4 or img_tensor.size(1) != 3:
                raise ValueError(f"Invalid shape: {img_tensor.shape}")

            with torch.no_grad():
                recon_tensor = reconstruct_image(img_tensor)

            recon_image = transforms.ToPILImage()(recon_tensor.squeeze().cpu())
            recon_image.save(save_path)
            print(f"✅ Saved: {model_name}/{filename}")

        except UnidentifiedImageError:
            print(f"❌ Unreadable image file: {filename}")
        except Exception as e:
            print(f"❌ Error: {filename} -> {e}")
