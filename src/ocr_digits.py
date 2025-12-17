from __future__ import annotations

"""
Train/infer a lightweight OCR model on EMNIST CSVs you already have locally.

Dataset layout (already in your repo):
  dataset/emnist/
    emnist-balanced-train.csv / emnist-balanced-mapping.txt
    emnist-byclass-train.csv  / emnist-byclass-mapping.txt
    emnist-bymerge-train.csv  / emnist-bymerge-mapping.txt
    emnist-digits-train.csv   / emnist-digits-mapping.txt
    emnist-letters-train.csv  / emnist-letters-mapping.txt
    emnist-mnist-train.csv    / emnist-mnist-mapping.txt

Train (example for balanced split, 8 epochs):
  python src/ocr_digits.py --train --split balanced --epochs 8 --batch-size 256

Train digits-only:
  python src/ocr_digits.py --train --split digits --epochs 6 --batch-size 256

Inference test:
  python src/ocr_digits.py --weights models/ocr_emnist_best.pth --mapping dataset/emnist/emnist-balanced-mapping.txt --image path/to/crop.png

In-app usage:
  from ocr_digits import EMNIST_OCR
  ocr = EMNIST_OCR("models/ocr_emnist_best.pth", "dataset/emnist/emnist-balanced-mapping.txt")
  label, conf = ocr.predict(pil_image_crop)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# -------------------- Mapping & CSV loaders -------------------- #
def load_mapping(mapping_path: Path) -> Tuple[Dict[int, str], List[str]]:
    """
    EMNIST mapping file lines: <label_int> <unicode_codepoint>
    Returns int->char dict and labels list (index-aligned).
    """
    int_to_char: Dict[int, str] = {}
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        idx = int(parts[0])
        codepoint = int(parts[1])
        int_to_char[idx] = chr(codepoint)
    max_idx = max(int_to_char.keys())
    labels = [""] * (max_idx + 1)
    for k, v in int_to_char.items():
        labels[k] = v
    return int_to_char, labels


def load_emnist_csv(csv_path: Path, cache_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load EMNIST CSV (label + 784 pixels). Optionally cache to NPZ."""
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        return data["images"], data["labels"]

    arr = np.loadtxt(csv_path, delimiter=",")
    labels = arr[:, 0].astype(np.int64)
    pixels = arr[:, 1:].astype(np.float32)
    # reshape to 28x28, fix EMNIST orientation: transpose, then horizontal flip
    images = pixels.reshape(-1, 28, 28).transpose(0, 2, 1)
    images = np.flip(images, axis=2)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, images=images, labels=labels)
    return images, labels


class EmnistDataset(Dataset):
    def __init__(self, csv_path: Path, cache_npz: Path | None = None, transform=None):
        self.images, self.labels = load_emnist_csv(csv_path, cache_npz)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx].astype(np.uint8), mode="L")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


# -------------------- Model -------------------- #
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mnist_polarity(gray: Image.Image) -> Image.Image:
    """
    Ensure MNIST-like polarity: white ink (high values) on black background.
    If the input looks like black ink on white background, invert it.
    """
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.mean() > 127:
        return ImageOps.invert(gray)
    return gray


def segment_digits(img: Image.Image, max_digits: int = 2) -> List[Image.Image]:
    """
    Best-effort segmentation for 1–2 handwritten digits.

    Uses a simple vertical projection heuristic (no OpenCV dependency).
    Returns a list of digit crops (grayscale, MNIST-like polarity), left-to-right.
    """
    gray = _mnist_polarity(img.convert("L"))
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.size == 0:
        return []

    thr = int(max(20, min(200, arr.mean() + arr.std() * 0.3)))
    mask = arr > thr  # True where "ink" is present (white on black)
    if int(mask.sum()) == 0:
        return []

    h, w = mask.shape
    col_sum = mask.sum(axis=0)
    min_col_ink = max(1, int(0.02 * h))
    active = col_sum >= min_col_ink

    runs: List[Tuple[int, int]] = []
    start = None
    for x, is_on in enumerate(active.tolist()):
        if is_on and start is None:
            start = x
        elif not is_on and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, w))

    bboxes: List[Tuple[int, int, int, int]] = []
    for x0, x1 in runs:
        sub = mask[:, x0:x1]
        rows = sub.sum(axis=1)
        ys = np.where(rows > 0)[0]
        if ys.size == 0:
            continue
        y0, y1 = int(ys[0]), int(ys[-1] + 1)
        bw = int(x1 - x0)
        bh = int(y1 - y0)
        if bw < 2 or bh < 6:
            continue
        pad = 2
        x0p = max(0, x0 - pad)
        x1p = min(w, x1 + pad)
        y0p = max(0, y0 - pad)
        y1p = min(h, y1 + pad)
        bboxes.append((x0p, y0p, x1p, y1p))

    if not bboxes:
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        bboxes = [(x0, y0, x1, y1)]

    # If there are many segments (e.g. symbol frame), try dropping "full-height thin" borders.
    if len(bboxes) > 1:
        filtered = []
        for x0, y0, x1, y1 in bboxes:
            bw = x1 - x0
            bh = y1 - y0
            if bh >= int(0.95 * h) and bw <= int(0.15 * w):
                continue
            filtered.append((x0, y0, x1, y1))
        if filtered:
            bboxes = filtered

    # Keep the most plausible boxes if too many remain.
    if len(bboxes) > max_digits:
        bboxes = sorted(
            bboxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )[:max_digits]

    bboxes = sorted(bboxes, key=lambda b: b[0])
    crops: List[Image.Image] = []
    for x0, y0, x1, y1 in bboxes:
        crop = gray.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        side = max(cw, ch)
        square = Image.new("L", (side, side), 0)
        square.paste(crop, ((side - cw) // 2, (side - ch) // 2))
        crops.append(square)
    return crops


class EMNIST_OCR:
    def __init__(
        self,
        weights_path: Path | str,
        mapping_path: Path | str,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _, self.labels = load_mapping(Path(mapping_path))
        num_classes = len(self.labels)
        self.model = SmallCNN(num_classes=num_classes).to(self.device)
        state = torch.load(weights_path, map_location="cpu")
        # unwrap common checkpoint wrappers
        if isinstance(state, dict) and not any(k.startswith("net.") or k.startswith("conv") for k in state.keys()):
            for key in ("state_dict", "model_state", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        # strip possible "module." prefixes
        if isinstance(state, dict):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def predict(self, img: Image.Image) -> Tuple[str, float]:
        gray = _mnist_polarity(img.convert("L"))
        x = self.transform(gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)
        return self.labels[idx.item()], float(conf)

    def predict_sequence(self, img: Image.Image, max_digits: int = 2) -> Tuple[str, float]:
        """
        Predict a 1–2 digit string (e.g., '7', '11', '42') from a single image.
        Uses `segment_digits()` then per-digit classification.
        """
        crops = segment_digits(img, max_digits=max_digits)
        if not crops:
            return self.predict(img)
        labels: List[str] = []
        confs: List[float] = []
        for c in crops:
            lab, conf = self.predict(c)
            labels.append(lab)
            confs.append(conf)
        return "".join(labels), float(min(confs) if confs else 0.0)


# -------------------- Training -------------------- #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, labels = load_mapping(Path(args.mapping))
    tfm = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    csv_path = Path(args.train_csv)
    cache_path = Path(args.cache_npz) if args.cache_npz else csv_path.with_suffix(".npz")
    full_ds = EmnistDataset(csv_path, cache_npz=cache_path, transform=tfm)

    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    ds_train, ds_val = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SmallCNN(num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, y in train_loader:
            imgs, y = imgs.to(device), y.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, y in val_loader:
                imgs, y = imgs.to(device), y.to(device)
                logits = model(imgs)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"Saved best weights to {out_path} | best val acc={best_acc:.4f}")


# -------------------- CLI -------------------- #
SPLIT_FILES = {
    "digits": ("emnist-digits-train.csv", "emnist-digits-mapping.txt"),
    "balanced": ("emnist-balanced-train.csv", "emnist-balanced-mapping.txt"),
    "byclass": ("emnist-byclass-train.csv", "emnist-byclass-mapping.txt"),
    "bymerge": ("emnist-bymerge-train.csv", "emnist-bymerge-mapping.txt"),
    "letters": ("emnist-letters-train.csv", "emnist-letters-mapping.txt"),
    "mnist": ("emnist-mnist-train.csv", "emnist-mnist-mapping.txt"),
}


def parse_args():
    ap = argparse.ArgumentParser(description="Train or run EMNIST OCR (CSV files, local).")
    ap.add_argument("--train", action="store_true", help="Run training mode.")
    ap.add_argument("--split", type=str, default="balanced", choices=list(SPLIT_FILES.keys()), help="Which EMNIST split to use.")
    ap.add_argument("--train-csv", type=str, help="Override train CSV path.")
    ap.add_argument("--mapping", type=str, help="Override mapping path.")
    ap.add_argument("--cache-npz", type=str, help="Path to cache NPZ (optional).")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", type=str, default="models/ocr_emnist_best.pth")
    ap.add_argument("--weights", type=str, default="models/ocr_emnist_best.pth", help="Weights for inference.")
    ap.add_argument("--image", type=str, help="Image path for inference test.")
    return ap.parse_args()


def main():
    args = parse_args()
    # resolve paths relative to repo root (parent of src)
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "dataset" / "emnist"
    if not args.train_csv or not args.mapping:
        csv_name, map_name = SPLIT_FILES[args.split]
        if not args.train_csv:
            args.train_csv = str(root / csv_name)
        if not args.mapping:
            args.mapping = str(root / map_name)
    if args.train or not args.image:
        train(args)
    else:
        ocr = EMNIST_OCR(args.weights, args.mapping)
        label, conf = ocr.predict(Image.open(args.image))
        print(f"Pred: {label} (conf={conf:.3f})")


if __name__ == "__main__":
    main()
