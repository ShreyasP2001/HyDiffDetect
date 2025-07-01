import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

from utils.datasets import create_dataloader
from utils.utils import to_cuda


# 🔧 Required by train.py to configure validation dataloader
def get_val_cfg(cfg, split="val", copy=True):
    if copy:
        from copy import deepcopy
        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg

    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    val_cfg.datasets = cfg.datasets_test if hasattr(cfg, "datasets_test") else cfg.datasets
    val_cfg.isTrain = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]

    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_cfg


# 🔍 Evaluation with multi-class support
def validate(model, cfg):
    model.eval()
    data_loader = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true, y_pred_cls, y_pred_prob = [], [], []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="🔍 Validating", dynamic_ncols=True):
            img, label, meta = data if len(data) == 3 else (*data, None)
            img = to_cuda(img, device)
            meta = to_cuda(meta, device) if meta is not None else None

            output = model(img, meta)
            probs = torch.sigmoid(output) if output.shape[1] == 1 else torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(label.cpu().numpy().flatten().tolist())
            y_pred_cls.extend(preds.cpu().numpy().flatten().tolist())
            y_pred_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_cls = np.array(y_pred_cls)
    y_pred_prob = np.array(y_pred_prob)

    # Binarize true labels for AP
    classes = sorted(set(y_true))
    y_true_bin = label_binarize(y_true, classes=classes)

    if y_pred_prob.ndim == 1:
        y_pred_prob = y_pred_prob.reshape(-1, 1)

    try:
        if y_pred_prob.shape[1] == 1 and y_true_bin.shape[1] > 1:
            ap = average_precision_score(y_true_bin[:, 1], y_pred_prob.ravel())
        else:
            ap = average_precision_score(y_true_bin, y_pred_prob, average="macro")
    except Exception as e:
        print(f"⚠️ AP calculation error: {e}")
        ap = 0.0

    acc = accuracy_score(y_true, y_pred_cls)
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_cls))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred_cls))

    results = {
        "ACC": acc,
        "AP": ap,
        "confusion_matrix": confusion_matrix(y_true, y_pred_cls),
    }

    return results
