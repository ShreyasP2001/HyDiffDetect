import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.utils import get_network, str2bool, to_cuda

# Argument Parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f", "--file", default="data/test/lsun_adm/1_fake/0.png", type=str, help="Path to image file or directory of images"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth",
    help="Path to the pretrained model"
)
parser.add_argument("--use_cpu", action="store_true", help="Uses GPU by default, turn on to use CPU")
parser.add_argument("--arch", type=str, default="resnet50", help="Model architecture")
parser.add_argument("--aug_norm", type=str2bool, default=True, help="Apply normalization")
args = parser.parse_args()

# Check if file exists
if os.path.isfile(args.file):
    print(f"✅ Testing on image: {args.file}")
    file_list = [args.file]
elif os.path.isdir(args.file):
    file_list = sorted(glob.glob(os.path.join(args.file, "*.jpg")) + glob.glob(os.path.join(args.file, "*.png")) + glob.glob(os.path.join(args.file, "*.JPEG")))
    print(f"✅ Testing images from folder: {args.file}")
else:
    raise FileNotFoundError(f"❌ ERROR: Invalid file path: '{args.file}'")

# Load Model
print(f"📌 Loading model from: {args.model_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"❌ ERROR: Model file '{args.model_path}' not found!")

model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()
if not args.use_cpu:
    model.cuda()
print("✅ Model loaded successfully!")

# Image Transformations
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Process each image
for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
    print(f"🖼️ Processing image: {img_path}")
    
    img = Image.open(img_path).convert("RGB")
    img = trans(img)

    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    in_tens = img.unsqueeze(0)
    if not args.use_cpu:
        in_tens = in_tens.cuda()

    # Run model inference
    with torch.no_grad():
        output = model(in_tens).sigmoid().item()
    
    print(f"🔍 Probability of being synthetic: {output:.4f}")

    # Generate output file path
    output_filename = os.path.basename(img_path)
    output_path = os.path.join("reconstructed_images", output_filename)

    # Save the reconstructed image (Manual Fix)
    if img is not None:
        os.makedirs("reconstructed_images", exist_ok=True)
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(output_path)
        print(f"✅ Reconstructed image saved at: {output_path}")
    else:
        print(f"❌ ERROR: No reconstructed image generated for {img_path}")

print("🎉 Processing complete!")
