from __future__ import annotations

import random
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from rules_parser import available_classes, canonical_label, load_rules


DATASET_DIR = Path(__file__).resolve().parent
NORMAL_DIR = DATASET_DIR / "normal_images"
SKETCH_DIR = DATASET_DIR / "sketches"
AUG_DIR = DATASET_DIR / "augmented"
FEEDBACK_LOG = ROOT_DIR / "models" / "logs" / "prediction_feedback.jsonl"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LABELS_JSON = ROOT_DIR / "models" / "primitive_labels_kfold.json"

# Expanded target to combat model confusion
TARGET_COUNT = 100_000
OUTPUT_SIZE = (256, 256)

RNG = random.Random(1337)
NP_RNG = np.random.default_rng(1337)

RULES_CONFIG = load_rules(ROOT_DIR / "rules.yaml")
CANONICAL_SET = set(RULES_CONFIG.get("canonical_classes", []))

# Load labels from json (if present) and merge/replace canonical set
def _labels_from_json(path: Path) -> set[str]:
    try:
        import json
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return set(data.values())
        if isinstance(data, list):
            return set(data)
    except Exception:
        return set()
    return set()

JSON_LABELS = _labels_from_json(LABELS_JSON)
if JSON_LABELS:
    CANONICAL_SET = CANONICAL_SET.union(JSON_LABELS) if CANONICAL_SET else JSON_LABELS.copy()

# Treat all canonical labels as priority to ensure coverage; if none specified, will fall back to observed stems
PRIORITY_CLASSES = set(CANONICAL_SET)


def _collect_observed_labels() -> tuple[list[str], list[str], list[str]]:
    observed = set()
    for path in list(NORMAL_DIR.glob("*")) + list(SKETCH_DIR.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = canonical_label(path, RULES_CONFIG)
        stem_label = path.stem
        if label:
            observed.add(label)
        else:
            observed.add(stem_label)
        if CANONICAL_SET and label in CANONICAL_SET:
            continue
    missing = sorted(CANONICAL_SET - observed) if CANONICAL_SET else []
    missing_priority = [c for c in PRIORITY_CLASSES if c not in observed] if PRIORITY_CLASSES else []
    return sorted(observed), missing, missing_priority


AVAILABLE_CLASSES, MISSING_CLASSES, MISSING_PRIORITY = _collect_observed_labels()
CLASS_SET = set(AVAILABLE_CLASSES)

if MISSING_CLASSES:
    print(f"[augment] Warning: {len(MISSING_CLASSES)} canonical classes lack images in normal_images or sketches.")
if MISSING_PRIORITY:
    print(f"[augment] Priority classes missing sources: {', '.join(MISSING_PRIORITY)}")


@dataclass
class SourceImage:
    path: Path
    label: str
    stem: str
    domain: str  # "normal" or "sketch" or "feedback"


Transform = Callable[[Image.Image, random.Random], Image.Image]


def _ensure_dirs() -> None:
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f"Missing normal images directory: {NORMAL_DIR}")
    if not SKETCH_DIR.exists():
        raise FileNotFoundError(f"Missing sketches directory: {SKETCH_DIR}")
    if not AVAILABLE_CLASSES:
        print("[augment] No canonical classes found; proceeding with all observed stems.")
    AUG_DIR.mkdir(exist_ok=True)


def _canonical_label(path: Path) -> str | None:
    label = canonical_label(path, RULES_CONFIG)
    if label:
        return label
    return path.stem


def _load_sources() -> List[SourceImage]:
    sources: List[SourceImage] = []
    for path in NORMAL_DIR.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _canonical_label(path)
        if not label:
            continue
        sources.append(SourceImage(path=path, label=label, stem=path.stem, domain="normal"))

    for path in SKETCH_DIR.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _canonical_label(path)
        if not label:
            continue
        sources.append(SourceImage(path=path, label=label, stem=path.stem, domain="sketch"))

    # feedback images (labeled by user during app usage)
    if FEEDBACK_LOG.exists():
        try:
            for line in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = entry.get("selected")
                img_path = entry.get("image_path")
                if not label or not img_path:
                    continue
                p = Path(img_path)
                if not p.exists() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # keep only labels we know, if a canonical set is defined
                if CANONICAL_SET and label not in CANONICAL_SET:
                    continue
                sources.append(SourceImage(path=p, label=label, stem=p.stem, domain="feedback"))
        except Exception:
            pass

    return sources


def _pil_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)


def _array_to_pil(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)


# --- Geometric / photometric transforms ---
def _affine_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-25, 25)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    max_shift_x = int(rotated.width * 0.1)
    max_shift_y = int(rotated.height * 0.1)
    dx = rng.randint(-max_shift_x, max_shift_x)
    dy = rng.randint(-max_shift_y, max_shift_y)
    return rotated.transform(
        rotated.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.BICUBIC,
        fillcolor=255,
    )


def _random_crop_pad(img: Image.Image, rng: random.Random) -> Image.Image:
    zoom = rng.uniform(0.8, 1.1)
    new_size = (max(1, int(img.width * zoom)), max(1, int(img.height * zoom)))
    resized = img.resize(new_size, Image.BICUBIC)
    return ImageOps.pad(
        resized,
        img.size,
        method=Image.BICUBIC,
        color=255,
        centering=(rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)),
    )


def _contrast_and_brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    contrast = ImageEnhance.Contrast(img).enhance(rng.uniform(0.65, 1.55))
    return ImageEnhance.Brightness(contrast).enhance(rng.uniform(0.7, 1.35))


def _sharpen_or_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 2.0)))
    percent = int(rng.uniform(100, 190))
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent))


def _elastic(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _pil_to_array(img)
    shape = arr.shape
    dx = NP_RNG.normal(0, 1.5, size=shape)
    dy = NP_RNG.normal(0, 1.5, size=shape)
    return _array_to_pil(arr + dx + dy)


def _grid_distort(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _pil_to_array(img)
    h, w = arr.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distort = NP_RNG.normal(0, 1.2, size=(h, w))
    x = x + distort
    y = y + distort
    x = np.clip(x, 0, w - 1).astype(np.int32)
    y = np.clip(y, 0, h - 1).astype(np.int32)
    warped = arr[y, x]
    return _array_to_pil(warped)


# --- Noise / strokes ---
def _add_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = _pil_to_array(img).astype(np.float32)
    noise_strength = rng.uniform(5, 20)
    noise = NP_RNG.normal(0, noise_strength, size=arr.shape)
    if rng.random() < 0.4:
        salt = NP_RNG.uniform(0, 255, size=arr.shape)
        mask = NP_RNG.random(arr.shape) < 0.025
        arr = np.where(mask, salt, arr)
    return _array_to_pil(arr + noise)


def _stroke_thicken(img: Image.Image, rng: random.Random) -> Image.Image:
    size = rng.choice([3, 5])
    return img.filter(ImageFilter.MaxFilter(size=size))


def _stroke_erode(img: Image.Image, rng: random.Random) -> Image.Image:
    size = rng.choice([3, 5])
    return img.filter(ImageFilter.MinFilter(size=size))


def _invert_chance(img: Image.Image, rng: random.Random) -> Image.Image:
    return ImageOps.invert(img) if rng.random() < 0.2 else img


def _posterize(img: Image.Image, rng: random.Random) -> Image.Image:
    return ImageOps.posterize(img, bits=rng.randint(3, 6))


NORMAL_TRANSFORMS: List[Transform] = [
    _affine_jitter,
    _random_crop_pad,
    _contrast_and_brightness,
    _sharpen_or_blur,
    _invert_chance,
    _elastic,
    _grid_distort,
    _posterize,
]

SKETCH_EXTRAS: List[Transform] = [
    _stroke_thicken,
    _stroke_erode,
]

UNIVERSAL_TRANSFORMS: List[Transform] = [
    _add_noise,
]


def _select_transforms(domain: str, rng: random.Random) -> List[Transform]:
    pool: List[Transform] = NORMAL_TRANSFORMS + UNIVERSAL_TRANSFORMS
    if domain == "sketch":
        pool += SKETCH_EXTRAS
    steps = rng.randint(3, min(6, len(pool)))
    return rng.sample(pool, k=steps)


def _apply_transforms(img: Image.Image, domain: str, rng: random.Random) -> Image.Image:
    transformed = img
    for transform in _select_transforms(domain, rng):
        transformed = transform(transformed, rng)
    return transformed


def _normalize_canvas(img: Image.Image) -> Image.Image:
    return ImageOps.pad(
        img,
        OUTPUT_SIZE,
        method=Image.BICUBIC,
        color=255,
        centering=(0.5, 0.5),
    )


def _outline_color(img: Image.Image, color: str) -> Image.Image:
    # Color only the strokes; keep background white
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    mask = arr < 200  # strokes/dark pixels
    base = Image.new("RGB", img.size, "white")
    color_img = Image.new("RGB", img.size, color)
    base.paste(color_img, mask=Image.fromarray(mask.astype(np.uint8) * 255))
    return base


def _color_variants(img: Image.Image, label: str) -> List[tuple[str, Image.Image]]:
    variants = [("default", img.convert("RGB"))]
    # Hostile cues or hostile frame
    if "hostile" in label or "frame_hostile_rect" in label:
        variants.append(("red", _outline_color(img, "red")))
    # Specific green symbol
    if label == "cm_mine_field_area":
        variants.append(("green", _outline_color(img, "green")))
    # Friendly/other: add blue and black outline versions
    variants.append(("blue", _outline_color(img, "blue")))
    variants.append(("black", _outline_color(img, "black")))
    return variants


def _distribution_counts(labels: List[str]) -> List[int]:
    weights = []
    for lbl in labels:
        w = 1
        if lbl in PRIORITY_CLASSES:
            w = 4
        weights.append(w)
    total_weight = max(1, sum(weights))
    # initial proportional counts
    counts = [max(1, int(TARGET_COUNT * (w / total_weight))) for w in weights]
    # adjust to hit TARGET_COUNT exactly
    diff = TARGET_COUNT - sum(counts)
    if diff > 0:
        for i in range(diff):
            counts[i % len(counts)] += 1
    elif diff < 0:
        for i in range(-diff):
            idx = i % len(counts)
            if counts[idx] > 1:
                counts[idx] -= 1
    return counts


def _augment_all(sources: List[SourceImage]) -> None:
    labels = [s.label for s in sources]
    counts = _distribution_counts(labels)
    total_written = 0
    for source, target_copies in zip(sources, counts):
        base = Image.open(source.path).convert("L")
        for copy_idx in range(target_copies):
            augmented = _apply_transforms(base, source.domain, RNG)
            normalized = _normalize_canvas(augmented)
            total_written += 1
            for color_tag, variant in _color_variants(normalized, source.label):
                filename = (
                    f"{source.label}_{source.domain}_{color_tag}_"
                    f"{source.stem}_aug_{copy_idx + 1:02d}_{total_written:05d}.png"
                )
                variant.save(AUG_DIR / filename)
    print(f"[augment] wrote {total_written} images to {AUG_DIR}")


def main() -> None:
    _ensure_dirs()
    sources = _load_sources()
    if not sources:
        raise RuntimeError("No labeled images discovered for augmentation.")
    _augment_all(sources)


if __name__ == "__main__":
    main()
