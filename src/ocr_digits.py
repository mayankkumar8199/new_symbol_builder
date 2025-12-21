from __future__ import annotations

"""
Unified digit OCR (0–9) + segmentation-based 0–99 inference.

This expects the unified digits dataset prepared by:
  python dataset/prepare_digits_dataset.py --overwrite

Dataset layout:
  dataset/digits_dataset/
    images/*.png
    labels.csv
    splits/train.csv
    splits/val.csv
    splits/test.csv

Train:
  python src/ocr_digits.py --train --epochs 10 --batch-size 256

Inference (single image, will try to segment up to 2 digits):
  python src/ocr_digits.py --weights models/ocr_digits_best.pth --mapping dataset/emnist/emnist-digits-mapping.txt --image path/to/crop.png

In-app usage:
  from ocr_digits import EMNIST_OCR
  ocr = EMNIST_OCR("models/ocr_digits_best.pth", "dataset/emnist/emnist-digits-mapping.txt")
  label, conf = ocr.predict_sequence(pil_image, max_digits=2)
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "dataset" / "digits_dataset"
DEFAULT_MAPPING = ROOT / "dataset" / "emnist" / "emnist-digits-mapping.txt"
DEFAULT_WEIGHTS = ROOT / "models" / "ocr_digits_best.pth"
DEFAULT_PLOTS_DIR = ROOT / "models" / "plots"
DEFAULT_LOGS_DIR = ROOT / "models" / "logs"

# For CLI inference convenience
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# -------------------- Mapping (for app compatibility) -------------------- #
def load_mapping(mapping_path: Path) -> Tuple[Dict[int, str], List[str]]:
    """
    EMNIST mapping file lines: <label_int> <unicode_codepoint>.
    For digits, mapping typically contains 10 rows: 0..9.
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
    if not int_to_char:
        raise ValueError(f"Invalid/empty mapping file: {mapping_path}")
    max_idx = max(int_to_char.keys())
    labels = [""] * (max_idx + 1)
    for k, v in int_to_char.items():
        labels[k] = v
    return int_to_char, labels


# -------------------- Dataset -------------------- #
@dataclass(frozen=True)
class DigitRow:
    filename: str
    label: int
    source: str
    source_split: str


def _read_rows(csv_path: Path) -> List[DigitRow]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[DigitRow] = []
        for row in reader:
            rows.append(
                DigitRow(
                    filename=row["filename"],
                    label=int(row["label"]),
                    source=row.get("source", "unknown"),
                    source_split=row.get("source_split", "n/a"),
                )
            )
        return rows


def _resolve_dataset_layout(dataset_arg: str) -> Tuple[Path, Path, Path]:
    """
    Accepts either:
      - dataset root: <...>/dataset/digits_dataset  (contains images/ + labels.csv)
      - images dir:   <...>/dataset/digits_dataset/images
    Returns: (dataset_root, images_dir, labels_csv)
    """
    p = Path(dataset_arg).resolve()

    if p.is_file() and p.name.lower() == "labels.csv":
        dataset_root = p.parent
        images_dir = dataset_root / "images"
        labels_csv = p
        return dataset_root, images_dir, labels_csv

    if p.is_dir() and p.name.lower() == "images":
        dataset_root = p.parent
        images_dir = p
        labels_csv = dataset_root / "labels.csv"
        return dataset_root, images_dir, labels_csv

    if p.is_dir():
        dataset_root = p
        images_dir = dataset_root / "images"
        labels_csv = dataset_root / "labels.csv"
        return dataset_root, images_dir, labels_csv

    raise FileNotFoundError(f"Dataset path not found: {p}")


def _stratified_split_indices(
    rows: Sequence[DigitRow],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("train/val fractions must be >0 and train+val < 1.0")
    by_label: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(int(r.label), []).append(i)
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _write_rows_csv(path: Path, rows: Sequence[DigitRow], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "source", "source_split"])
        for i in indices:
            r = rows[i]
            w.writerow([r.filename, r.label, r.source, r.source_split])


def _ensure_splits(dataset_root: Path, labels_csv: Path, seed: int, train_frac: float, val_frac: float, rebuild: bool) -> Path:
    splits_dir = dataset_root / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    if not rebuild and train_csv.exists() and val_csv.exists() and test_csv.exists():
        return splits_dir

    rows = _read_rows(labels_csv)
    train_idx, val_idx, test_idx = _stratified_split_indices(rows, train_frac=train_frac, val_frac=val_frac, seed=seed)
    _write_rows_csv(train_csv, rows, train_idx)
    _write_rows_csv(val_csv, rows, val_idx)
    _write_rows_csv(test_csv, rows, test_idx)
    return splits_dir


class DigitsDataset(Dataset[Tuple[torch.Tensor, int, int]]):
    def __init__(self, images_dir: Path, rows: Sequence[DigitRow], transform: transforms.Compose):
        self.images_dir = images_dir
        self.rows = list(rows)
        self.transform = transform
        sources = sorted({r.source for r in self.rows})
        self.source_to_idx = {s: i for i, s in enumerate(sources)}
        self.idx_to_source = {i: s for s, i in self.source_to_idx.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        r = self.rows[idx]
        img = Image.open(self.images_dir / r.filename).convert("L")
        return self.transform(img), int(r.label), int(self.source_to_idx.get(r.source, 0))


class CachedDigitsDataset(Dataset[Tuple[torch.Tensor, int, int]]):
    """
    Faster training on very large datasets (avoids decoding thousands of PNGs per epoch).
    Cache is a set of memory-mapped .npy arrays created from the split CSVs.
    """

    def __init__(self, images_path: Path, labels_path: Path, sources_path: Path, idx_to_source: Dict[int, str], transform):
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.sources = np.load(sources_path, mmap_mode="r")
        self.idx_to_source = dict(idx_to_source)
        self.source_to_idx = {v: int(k) for k, v in self.idx_to_source.items()}
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        arr = self.images[idx]
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L")
        x = self.transform(img) if self.transform else img
        y = int(self.labels[idx])
        s = int(self.sources[idx]) if self.sources is not None else 0
        return x, y, s


# -------------------- Preprocessing & Segmentation -------------------- #
def _border_mean(arr: np.ndarray, border: int = 4) -> float:
    if arr.ndim != 2:
        raise ValueError("expected 2D grayscale array")
    b = max(1, min(border, min(arr.shape) // 2))
    top = arr[:b, :]
    bottom = arr[-b:, :]
    left = arr[:, :b]
    right = arr[:, -b:]
    vals = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    return float(vals.mean())


def _mnist_polarity(gray: Image.Image) -> Image.Image:
    """Normalize polarity to MNIST-like: bright ink on dark background."""
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.size and _border_mean(arr) > 127:
        gray = ImageOps.invert(gray)
    return ImageOps.autocontrast(gray)


def _otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t) * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = t
    return int(thr)


def _binary_dilate(mask: np.ndarray, k: int = 3) -> np.ndarray:
    pad = k // 2
    h, w = mask.shape
    m = np.pad(mask, pad, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(k):
        for dx in range(k):
            out |= m[dy : dy + h, dx : dx + w]
    return out


def _binary_erode(mask: np.ndarray, k: int = 3) -> np.ndarray:
    pad = k // 2
    h, w = mask.shape
    m = np.pad(mask, pad, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(k):
        for dx in range(k):
            out &= m[dy : dy + h, dx : dx + w]
    return out


def _binary_close(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, k=k), k=k)


def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """
    Returns list of (x0,y0,x1,y1,area) for 8-connected components.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[int, int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            minx = maxx = x
            miny = maxy = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                minx = min(minx, cx)
                maxx = max(maxx, cx)
                miny = min(miny, cy)
                maxy = max(maxy, cy)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((minx, miny, maxx + 1, maxy + 1, area))
    return comps


def _median_int(values: Sequence[int], default: int) -> int:
    if not values:
        return default
    return int(np.median(np.asarray(values, dtype=np.float32)).item())


def _union_bbox(boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _group_digit_boxes_into_number_boxes(
    bboxes: Sequence[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    max_numbers: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Groups per-digit bboxes into multi-digit "number" bboxes.
    Returns number bboxes in reading order (top->bottom, left->right).
    """
    if not bboxes:
        return []

    w, h = image_size
    widths = [x1 - x0 for x0, y0, x1, y1 in bboxes]
    heights = [y1 - y0 for x0, y0, x1, y1 in bboxes]
    med_w = _median_int(widths, default=max(6, int(0.08 * w)))
    med_h = _median_int(heights, default=max(10, int(0.12 * h)))

    # 1) Group into rows by y-center proximity
    row_thresh = max(8, int(0.65 * med_h))
    boxes_by_y = sorted(bboxes, key=lambda b: (b[1] + b[3]) / 2)
    rows: List[Dict[str, object]] = []
    for b in boxes_by_y:
        cy = (b[1] + b[3]) / 2
        placed = False
        for row in rows:
            rcy = float(row["cy"])  # type: ignore[assignment]
            if abs(cy - rcy) <= row_thresh:
                row["boxes"].append(b)  # type: ignore[index]
                # update running center
                row["cy"] = (rcy + cy) / 2
                placed = True
                break
        if not placed:
            rows.append({"cy": float(cy), "boxes": [b]})

    # 2) Within each row, group left-to-right into numbers
    numbers: List[Tuple[int, int, int, int]] = []
    gap_thresh = max(4, int(1.6 * med_w))
    overlap_thresh = 0.25

    for row in rows:
        boxes = sorted(row["boxes"], key=lambda b: b[0])  # type: ignore[arg-type]
        if not boxes:
            continue
        group: List[Tuple[int, int, int, int]] = [boxes[0]]  # type: ignore[index]
        for b in boxes[1:]:  # type: ignore[index]
            px0, py0, px1, py1 = group[-1]
            x0, y0, x1, y1 = b
            gap = x0 - px1
            overlap_y = min(py1, y1) - max(py0, y0)
            min_h = max(1, min(py1 - py0, y1 - y0))
            overlap_ratio = overlap_y / min_h
            if gap <= gap_thresh and overlap_ratio >= overlap_thresh:
                group.append(b)
            else:
                numbers.append(_union_bbox(group))
                group = [b]
        if group:
            numbers.append(_union_bbox(group))

    # Pad and clamp
    padded: List[Tuple[int, int, int, int]] = []
    pad = max(2, int(0.15 * med_w))
    for x0, y0, x1, y1 in numbers:
        padded.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))

    padded = sorted(padded, key=lambda b: (b[1], b[0]))
    return padded[:max_numbers]


def segment_digits(img: Image.Image, max_digits: int = 2) -> List[Image.Image]:
    """
    Best-effort segmentation for 1–2 handwritten digits.

    Pure-Python (no OpenCV): threshold -> morphology -> connected components -> filtering.
    Returns digit crops (grayscale, MNIST-like polarity), left-to-right.
    """
    gray = _mnist_polarity(img.convert("L"))
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.size == 0:
        return []

    thr = _otsu_threshold(arr)
    thr = int(max(10, min(245, thr + 10)))
    mask = arr > thr
    if mask.sum() == 0:
        return []

    h, w = mask.shape
    # For long digit strings, avoid aggressive closing which can merge adjacent digits.
    close_k = 2 if max_digits >= 3 else 3
    mask = _binary_close(mask, k=close_k)
    comps = _connected_components(mask)
    if not comps:
        return []

    # Filter noise and long borders. Keep thresholds conservative because digits can be thin (e.g. "1").
    min_area = max(8, int(0.0008 * h * w))
    candidates: List[Tuple[int, int, int, int, int]] = []
    for x0, y0, x1, y1, area in comps:
        bw = x1 - x0
        bh = y1 - y0
        if area < min_area:
            continue
        if bh < int(0.12 * h):
            continue
        if bw < 2:
            continue
        if bh >= int(0.92 * h) and bw <= int(0.12 * w):
            continue
        candidates.append((x0, y0, x1, y1, area))

    if not candidates:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return []
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        candidates = [(x0, y0, x1, y1, int(mask.sum()))]

    # Keep the largest regions and sort left-to-right.
    # Keep enough regions for long digit strings while still trimming heavy noise.
    candidates = sorted(candidates, key=lambda b: b[4], reverse=True)[: max(24, max_digits * 6)]
    bboxes = sorted([(x0, y0, x1, y1) for x0, y0, x1, y1, _ in candidates], key=lambda b: b[0])

    # If digits are touching, components may merge. Iteratively split wide boxes using
    # vertical projection minima until we have enough digits (up to max_digits).
    if max_digits >= 2 and bboxes:
        widths = [max(1, x1 - x0) for x0, _y0, x1, _y1 in bboxes]
        med_w = int(np.median(np.asarray(widths, dtype=np.int32))) if widths else 1

        def _split_once(bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
            x0, y0, x1, y1 = bbox
            bw = x1 - x0
            bh = y1 - y0
            if bw <= 0 or bh <= 0:
                return [bbox]
            if bw < 12:
                return [bbox]
            ratio = bw / max(1.0, float(bh))
            wide_by_ratio = ratio >= 1.25
            wide_by_width = bw >= max(12, int(1.6 * med_w))
            if not wide_by_ratio and not wide_by_width:
                return [bbox]

            sub = mask[y0:y1, x0:x1]
            if sub.size == 0 or sub.shape[1] < 6:
                return [bbox]
            col = sub.sum(axis=0).astype(np.int32)
            lo = int(0.10 * col.size)
            hi = int(0.90 * col.size)
            if hi <= lo + 2:
                return [bbox]
            split = int(lo + np.argmin(col[lo:hi]))
            min_w = max(2, int(0.08 * bh))
            if split < min_w or (col.size - split) < min_w:
                return [bbox]

            valley = int(col[split])
            # Require a noticeable valley unless the box is *very* wide.
            valley_thresh = int(0.08 * bh)
            very_wide = ratio >= 2.0 or bw >= max(20, int(2.2 * med_w))
            if valley > valley_thresh and not very_wide:
                return [bbox]
            return [(x0, y0, x0 + split, y1), (x0 + split, y0, x1, y1)]

        max_iters = max(1, max_digits * 2)
        for _ in range(max_iters):
            if len(bboxes) >= max_digits:
                break
            # Split the box that looks most like a multi-digit merge.
            idx = max(
                range(len(bboxes)),
                key=lambda i: (bboxes[i][2] - bboxes[i][0]) / max(1, (bboxes[i][3] - bboxes[i][1])),
            )
            parts = _split_once(bboxes[idx])
            if len(parts) == 1:
                break
            bboxes = bboxes[:idx] + parts + bboxes[idx + 1 :]
            bboxes = sorted(bboxes, key=lambda b: b[0])

    crops: List[Image.Image] = []
    for x0, y0, x1, y1 in bboxes:
        pad = 2
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(w, x1 + pad)
        y1p = min(h, y1 + pad)
        crop = gray.crop((x0p, y0p, x1p, y1p))
        cw, ch = crop.size
        side = max(cw, ch, 1)
        square = Image.new("L", (side, side), 0)
        square.paste(crop, ((side - cw) // 2, (side - ch) // 2))
        crops.append(square)
        if len(crops) >= max_digits:
            break
    return crops


# -------------------- Model -------------------- #
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.10),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# -------------------- OCR class (used by the app) -------------------- #
class EMNIST_OCR:
    """
    Name kept for backward compatibility with the app.
    Supports single-digit predict() and 1–2 digit predict_sequence() via segmentation.
    """

    def __init__(self, weights_path: Path | str, mapping_path: Path | str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _, self.labels = load_mapping(Path(mapping_path))
        num_classes = len(self.labels)
        self.model = SmallCNN(num_classes=num_classes).to(self.device)

        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model_state", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if isinstance(state, dict):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Lambda(_mnist_polarity),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Tuple[str, float]:
        gray = img.convert("L")
        x = self.transform(gray).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        return self.labels[int(idx.item())], float(conf.item())

    @torch.no_grad()
    def predict_topk(self, img: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
        gray = img.convert("L")
        x = self.transform(gray).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confs, idxs = torch.topk(probs, k=min(k, int(probs.numel())))
        out: List[Tuple[str, float]] = []
        for c, i in zip(confs.tolist(), idxs.tolist()):
            out.append((self.labels[int(i)], float(c)))
        return out

    def predict_sequence(
        self,
        img: Image.Image,
        max_digits: int = 2,
        *,
        search_rois: bool = True,
    ) -> Tuple[str, float]:
        """
        Predict a digit string from a single image (up to `max_digits`).
        """
        w, h = img.size
        candidates: List[Image.Image] = [img]
        if search_rois and w * h >= 256 * 256:
            candidates = [
                img,
                img.crop((int(0.65 * w), int(0.65 * h), w, h)),  # bottom-right
                img.crop((int(0.65 * w), 0, w, int(0.35 * h))),  # top-right band
                img.crop((0, 0, w, int(0.35 * h))),  # top band
            ]

        best_label: Optional[str] = None
        best_conf = -1.0
        for cand in candidates:
            crops = segment_digits(cand, max_digits=max_digits)
            if not crops:
                lab, conf = self.predict(cand)
                if conf > best_conf:
                    best_label, best_conf = lab, conf
                continue
            labs: List[str] = []
            confs: List[float] = []
            for c in crops:
                lab, conf = self.predict(c)
                labs.append(lab)
                confs.append(conf)
            seq = "".join(labs)
            conf = float(min(confs) if confs else 0.0)
            if conf > best_conf:
                best_label, best_conf = seq, conf
        return (best_label or ""), float(best_conf if best_conf >= 0 else 0.0)

    def predict_numbers(
        self,
        img: Image.Image,
        *,
        max_numbers: int = 5,
        max_digits_per_number: int = 4,
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect multiple digit *groups* from an image and decode each group as a number.

        Returns: list of (digits_string, confidence, bbox) in reading order.
        """
        gray = _mnist_polarity(img.convert("L"))
        arr = np.asarray(gray, dtype=np.uint8)
        if arr.size == 0:
            return []

        thr = _otsu_threshold(arr)
        thr = int(max(10, min(245, thr + 10)))
        mask = arr > thr
        if mask.sum() == 0:
            return []
        mask = _binary_close(mask, k=3)

        # Tight-crop to the digit ink to speed up connected components on large canvases.
        ys, xs = np.where(mask)
        if ys.size == 0:
            return []
        pad = 4
        x0c = max(0, int(xs.min()) - pad)
        x1c = min(mask.shape[1], int(xs.max()) + 1 + pad)
        y0c = max(0, int(ys.min()) - pad)
        y1c = min(mask.shape[0], int(ys.max()) + 1 + pad)
        offset_x, offset_y = x0c, y0c
        mask = mask[y0c:y1c, x0c:x1c]

        comps = _connected_components(mask)
        if not comps:
            return []

        h, w = mask.shape
        min_area = max(8, int(0.0008 * h * w))
        digit_boxes: List[Tuple[int, int, int, int]] = []
        for x0, y0, x1, y1, area in comps:
            bw = x1 - x0
            bh = y1 - y0
            if area < min_area:
                continue
            if bh < int(0.12 * h):
                continue
            if bw < 2:
                continue
            if bh >= int(0.92 * h) and bw <= int(0.12 * w):
                continue
            digit_boxes.append((x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y))

        if not digit_boxes:
            # fall back to whole image as one candidate
            s, c = self.predict_sequence(img, max_digits=max_digits_per_number)
            digits_only = "".join(ch for ch in str(s) if ch.isdigit())
            return [(digits_only, float(c), (0, 0, img.size[0], img.size[1]))] if digits_only else []

        number_boxes = _group_digit_boxes_into_number_boxes(digit_boxes, img.size, max_numbers=max_numbers)
        results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        for bbox in number_boxes:
            crop = img.crop(bbox)
            # For an already-localized number crop, don't search alternative ROIs:
            # it can accidentally pick a partial window with higher confidence.
            s, c = self.predict_sequence(crop, max_digits=max_digits_per_number, search_rois=False)
            digits_only = "".join(ch for ch in str(s) if ch.isdigit())
            if not digits_only:
                continue
            results.append((digits_only, float(c), bbox))

        return results


# -------------------- Training -------------------- #
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _default_num_workers() -> int:
    if sys.platform.startswith("win"):
        return 0
    return min(8, (os.cpu_count() or 4))


def _make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Lambda(_mnist_polarity),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=12,
                            translate=(0.10, 0.10),
                            scale=(0.85, 1.15),
                            shear=8,
                            fill=0,
                        )
                    ],
                    p=0.9,
                ),
                transforms.RandomPerspective(distortion_scale=0.25, p=0.3, fill=0),
                transforms.RandomInvert(p=0.05),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), value=0.0),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(_mnist_polarity),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_sources: int,
) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    confmat = np.zeros((10, 10), dtype=np.int64)
    src_total = [0 for _ in range(n_sources)]
    src_correct = [0 for _ in range(n_sources)]
    for x, y, src in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.item()) * int(y.size(0))
        pred = logits.argmax(dim=1)
        total += int(y.size(0))
        correct += int((pred == y).sum().item())
        y_np = y.cpu().numpy()
        p_np = pred.cpu().numpy()
        for t, p in zip(y_np.tolist(), p_np.tolist()):
            if 0 <= t < 10 and 0 <= p < 10:
                confmat[t, p] += 1
        for s, is_ok in zip(src.cpu().numpy().tolist(), (pred == y).cpu().numpy().tolist()):
            if 0 <= s < n_sources:
                src_total[s] += 1
                src_correct[s] += int(bool(is_ok))
    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    return float(avg_loss), float(acc), confmat, src_total, src_correct


def _cache_paths(cache_dir: Path, split: str) -> Tuple[Path, Path, Path, Path]:
    return (
        cache_dir / f"{split}_images.npy",
        cache_dir / f"{split}_labels.npy",
        cache_dir / f"{split}_sources.npy",
        cache_dir / f"{split}_meta.json",
    )


def _build_cache(images_dir: Path, rows: Sequence[DigitRow], cache_dir: Path, split: str, rebuild: bool) -> Dict[int, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    images_path, labels_path, sources_path, meta_path = _cache_paths(cache_dir, split)
    if not rebuild and images_path.exists() and labels_path.exists() and sources_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        idx_to_source = {int(k): v for k, v in meta.get("idx_to_source", {}).items()}
        return idx_to_source

    from tqdm import tqdm  # local import

    sources = sorted({r.source for r in rows})
    source_to_idx = {s: i for i, s in enumerate(sources)}
    idx_to_source = {i: s for s, i in source_to_idx.items()}

    n = len(rows)
    images_mm = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(n, 28, 28))
    labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(n,))
    sources_mm = np.lib.format.open_memmap(sources_path, mode="w+", dtype=np.int16, shape=(n,))

    for i, r in enumerate(tqdm(rows, desc=f"[digits] caching {split}", unit="img")):
        with Image.open(images_dir / r.filename) as im:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.uint8)
            if arr.shape != (28, 28):
                im = im.resize((28, 28), resample=Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
        images_mm[i] = arr
        labels_mm[i] = int(r.label)
        sources_mm[i] = int(source_to_idx.get(r.source, 0))

    meta_path.write_text(json.dumps({"idx_to_source": idx_to_source}, indent=2), encoding="utf-8")
    return idx_to_source


def _load_cached_dataset(cache_dir: Path, split: str, transform) -> CachedDigitsDataset:
    images_path, labels_path, sources_path, meta_path = _cache_paths(cache_dir, split)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idx_to_source = {int(k): v for k, v in meta.get("idx_to_source", {}).items()}
    return CachedDigitsDataset(images_path, labels_path, sources_path, idx_to_source, transform)


def train(args: argparse.Namespace) -> None:
    from tqdm import tqdm  # local import: keeps app startup lighter

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root, images_dir, labels_csv = _resolve_dataset_layout(args.dataset)
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_csv}")

    splits_dir = _ensure_splits(
        dataset_root,
        labels_csv,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        rebuild=args.rebuild_splits,
    )
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    rows_train = _read_rows(train_csv)
    rows_val = _read_rows(val_csv)
    rows_test = _read_rows(test_csv) if test_csv.exists() else []

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (dataset_root / "cache")
    if args.cache:
        _build_cache(images_dir, rows_train, cache_dir, "train", rebuild=args.rebuild_cache)
        _build_cache(images_dir, rows_val, cache_dir, "val", rebuild=args.rebuild_cache)
        if rows_test:
            _build_cache(images_dir, rows_test, cache_dir, "test", rebuild=args.rebuild_cache)
        ds_train = _load_cached_dataset(cache_dir, "train", transform=_make_transforms(train=True))
        ds_val = _load_cached_dataset(cache_dir, "val", transform=_make_transforms(train=False))
        ds_test = _load_cached_dataset(cache_dir, "test", transform=_make_transforms(train=False)) if rows_test else None
    else:
        ds_train = DigitsDataset(images_dir, rows_train, transform=_make_transforms(train=True))
        ds_val = DigitsDataset(images_dir, rows_val, transform=_make_transforms(train=False))
        ds_test = DigitsDataset(images_dir, rows_test, transform=_make_transforms(train=False)) if rows_test else None

    num_workers = args.num_workers if args.num_workers is not None else _default_num_workers()
    pin = device.type == "cuda"
    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    loader_test = (
        DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
        if ds_test is not None
        else None
    )

    model = SmallCNN(num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_acc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, float]] = []
    patience_left = args.patience

    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0
            correct = 0
            loss_sum = 0.0
            pbar = tqdm(loader_train, desc=f"[digits] epoch {epoch}/{args.epochs}", unit="batch")
            for x, y, _src in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * int(y.size(0))
                pred = logits.argmax(dim=1)
                total += int(y.size(0))
                correct += int((pred == y).sum().item())
                pbar.set_postfix(loss=loss_sum / max(1, total), acc=correct / max(1, total))
            sched.step()

            train_loss = loss_sum / max(1, total)
            train_acc = correct / max(1, total)
            val_loss, val_acc, _confmat, src_total, src_correct = _evaluate(
                model, loader_val, device, n_sources=len(ds_val.source_to_idx)
            )

            entry = {
                "epoch": float(epoch),
                "loss": float(train_loss),
                "acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            history.append(entry)
            print(
                f"[digits] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            for sid, tot in enumerate(src_total):
                if tot <= 0:
                    continue
                name = ds_val.idx_to_source.get(sid, str(sid))
                acc_s = src_correct[sid] / max(1, tot)
                print(f"  [val] {name}: {acc_s:.4f} ({src_correct[sid]}/{tot})")

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if args.patience and patience_left <= 0:
                    print(f"[digits] early stopping (best val_acc={best_acc:.4f})")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("[digits] training interrupted (saving best weights so far).")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), out_path)

    from utils.training_plots import save_training_plot  # local import: matplotlib is train-only

    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_LOGS_DIR / "ocr_digits_history.json").write_text(json.dumps(history, indent=2))
    save_training_plot(history, DEFAULT_PLOTS_DIR / "ocr_digits_training.png", "Digit OCR Training")

    metrics: Dict[str, float] = {"best_val_acc": float(best_acc), "epochs_ran": float(len(history))}
    if interrupted:
        metrics["interrupted"] = 1.0
    if loader_test is not None and best_state is not None:
        model.load_state_dict(best_state, strict=True)
        test_loss, test_acc, confmat, src_total, src_correct = _evaluate(
            model, loader_test, device, n_sources=len(ds_test.source_to_idx)
        )
        metrics.update({"test_loss": float(test_loss), "test_acc": float(test_acc)})
        (DEFAULT_LOGS_DIR / "ocr_digits_confmat.json").write_text(json.dumps(confmat.tolist()))
        per_source = {}
        for sid, tot in enumerate(src_total):
            if tot <= 0:
                continue
            name = ds_test.idx_to_source.get(sid, str(sid))
            per_source[name] = float(src_correct[sid] / max(1, tot))
        (DEFAULT_LOGS_DIR / "ocr_digits_test_per_source.json").write_text(json.dumps(per_source, indent=2))
    (DEFAULT_LOGS_DIR / "ocr_digits_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[digits] saved best weights -> {out_path} (best val_acc={best_acc:.4f})")


# -------------------- CLI -------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train or run digit OCR (0–9) with segmentation-based 0–99 inference.")
    ap.add_argument("--train", action="store_true", help="Run training mode.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to dataset root (digits_dataset) OR its images/ folder.",
    )
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction when creating splits")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction when creating splits")
    ap.add_argument("--rebuild-splits", action="store_true", help="Recreate splits from labels.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3, help="Early stop after N bad val epochs (0 disables).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--output", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="Use a memory-mapped cache to speed up training (recommended).",
    )
    ap.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache",
        help="Disable dataset cache (will read PNGs every epoch).",
    )
    ap.add_argument("--cache-dir", type=str, default="", help="Cache directory (default: <dataset>/cache)")
    ap.add_argument("--rebuild-cache", action="store_true", help="Rebuild cache even if it exists")

    ap.add_argument("--mapping", type=str, default=str(DEFAULT_MAPPING), help="Mapping for inference (digits).")
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Weights for inference.")
    ap.add_argument(
        "--image",
        type=str,
        help="Image path for inference; can also be a directory (will sample from it).",
    )
    ap.add_argument(
        "--infer-count",
        type=int,
        default=1,
        help="If --image is a directory, run inference on N sampled images.",
    )
    ap.add_argument("--max-digits", type=int, default=2)
    ap.add_argument("--topk", type=int, default=0, help="Print top-k predictions (0 disables).")
    return ap.parse_args()


def _reservoir_sample(iterable, k: int, rng: random.Random) -> List[Path]:
    if k <= 0:
        return []
    sample: List[Path] = []
    for i, p in enumerate(iterable, start=1):
        if len(sample) < k:
            sample.append(p)
            continue
        j = rng.randrange(i)
        if j < k:
            sample[j] = p
    return sample


def _iter_images_in_dir(dir_path: Path):
    if (dir_path / "images").is_dir():
        base = dir_path / "images"
        it = base.iterdir()
    else:
        it = dir_path.rglob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def _resolve_image_paths(arg: str, count: int, seed: int) -> List[Path]:
    p = Path(arg)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"--image path not found: {p}")
    if not p.is_dir():
        raise ValueError(f"--image must be a file or directory: {p}")
    rng = random.Random(seed)
    candidates_it = _iter_images_in_dir(p)
    if count <= 1:
        try:
            return [next(candidates_it)]
        except StopIteration:
            raise RuntimeError(f"No image files found under: {p}") from None
    sample = _reservoir_sample(candidates_it, count, rng)
    if not sample:
        raise RuntimeError(f"No image files found under: {p}")
    return sample


def main() -> None:
    args = parse_args()
    if args.train:
        train(args)
        return

    if not args.image:
        # Default behavior: train when run without args (matches earlier workflow).
        print("[digits] No --image provided; defaulting to training. (Use --image for inference.)")
        train(args)
        return

    ocr = EMNIST_OCR(args.weights, args.mapping)
    paths = _resolve_image_paths(args.image, args.infer_count, args.seed)
    for path in paths:
        img = Image.open(path).convert("RGB")
        label, conf = ocr.predict_sequence(img, max_digits=args.max_digits)
        print(f"{path}: {label} (conf={conf:.3f})")
        if args.topk:
            topk = ocr.predict_topk(img, k=args.topk)
            print("  Top-k:", ", ".join([f"{l}:{c:.3f}" for l, c in topk]))


if __name__ == "__main__":
    main()
