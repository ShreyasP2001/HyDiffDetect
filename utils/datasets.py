import os
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler

from utils.config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def binary_dataset(root: str, cfg: CONFIGCLASS):
    identity = transforms.Lambda(lambda img: img)

    rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg)) if cfg.aug_resize else identity
    crop_func = transforms.RandomCrop(cfg.cropSize) if cfg.isTrain else transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity
    flip_func = transforms.RandomHorizontalFlip() if cfg.isTrain and cfg.aug_flip else identity
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.aug_norm else identity

    transform = transforms.Compose([
        rz_func,
        transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        normalize,
    ])

    return datasets.ImageFolder(root=root, transform=transform)


def dataset_folder(root: str, cfg: CONFIGCLASS):
    if cfg.mode == "binary":
        return binary_dataset(root, cfg)
    raise ValueError("cfg.mode must be 'binary'")


def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp_method = sample_discrete(cfg.rz_interp)
    interpolation = {
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'nearest': Image.NEAREST
    }.get(interp_method, Image.BILINEAR)
    return TF.resize(img, cfg.loadSize, interpolation=interpolation)


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img = np.array(img)

    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sigma = sample_continuous(cfg.blur_sig)
            for c in range(3):
                gaussian_filter(img[:, :, c], sigma=sigma, output=img[:, :, c])

        if random() < cfg.jpg_prob:
            qual = sample_discrete(cfg.jpg_qual)
            method = sample_discrete(cfg.jpg_method)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_discrete(options: list):
    return choice(options) if len(options) > 1 else options[0]


def sample_continuous(range_list: list):
    if len(range_list) == 1:
        return range_list[0]
    elif len(range_list) == 2:
        return random() * (range_list[1] - range_list[0]) + range_list[0]
    else:
        raise ValueError("blur_sig or jpg_qual should be of length 1 or 2")


def jpeg_from_key(img: np.ndarray, quality: int, method: str) -> np.ndarray:
    if method == "cv2":
        img_bgr = img[:, :, ::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode(".jpg", img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:, :, ::-1]
    elif method == "pil":
        out = BytesIO()
        Image.fromarray(img).save(out, format="jpeg", quality=quality)
        out.seek(0)
        return np.array(Image.open(out))
    else:
        raise ValueError(f"Unsupported JPEG method: {method}")


def get_dataset(cfg: CONFIGCLASS):
    return dataset_folder(cfg.dataset_root, cfg)


def get_bal_sampler(dataset: torch.utils.data.Dataset):
    targets = dataset.targets
    class_counts = np.bincount(targets)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
