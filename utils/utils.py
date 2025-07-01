import argparse
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on", "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True


def to_cuda(data, device="cuda", exclude_keys: "list[str]" = None):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (tuple, list, set)):
        return [to_cuda(b, device) for b in data]
    elif isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        for k in data.keys():
            if k not in exclude_keys:
                data[k] = to_cuda(data[k], device)
    return data


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode="w"):
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0
        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass


def get_network(arch: str, isTrain=False, continue_train=False, init_gain=0.02, pretrained=True):
    from networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
    from torchvision.models import mobilenet_v2, densenet121, efficientnet_b0, convnext_tiny

    arch_map = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "mobilenet_v2": mobilenet_v2,
        "densenet121": densenet121,
        "efficientnet_b0": efficientnet_b0,
        "convnext_tiny": convnext_tiny,
    }

    model_fn = arch_map.get(arch.lower())
    if model_fn is None:
        raise ValueError(f"Unsupported architecture: {arch}")

    if "resnet" in arch.lower():
        model = model_fn(pretrained=pretrained if not continue_train else False, num_classes=1)
        if isTrain and not continue_train:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)
            nn.init.normal_(model.fc.weight.data, 0.0, init_gain)

    elif arch == "mobilenet_v2":
        model = model_fn(pretrained=pretrained if not continue_train else False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        if isTrain and not continue_train:
            nn.init.normal_(model.classifier[1].weight.data, 0.0, init_gain)

    elif arch == "densenet121":
        model = model_fn(pretrained=pretrained if not continue_train else False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
        if isTrain and not continue_train:
            nn.init.normal_(model.classifier.weight.data, 0.0, init_gain)

    elif arch == "efficientnet_b0":
        model = model_fn(pretrained=pretrained if not continue_train else False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        if isTrain and not continue_train:
            nn.init.normal_(model.classifier[1].weight.data, 0.0, init_gain)

    elif arch == "convnext_tiny":
        model = model_fn(pretrained=pretrained if not continue_train else False)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
        if isTrain and not continue_train:
            nn.init.normal_(model.classifier[2].weight.data, 0.0, init_gain)

    return model


def pad_img_to_square(img: np.ndarray):
    H, W = img.shape[:2]
    if H != W:
        new_size = max(H, W)
        img = np.pad(img, ((0, new_size - H), (0, new_size - W), (0, 0)), mode="constant")
        assert img.shape[0] == img.shape[1] == new_size
    return img
