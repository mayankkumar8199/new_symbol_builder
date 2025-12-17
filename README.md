# new_symbol_builder

# Symbols-AI

End-to-end pipeline for **military-style symbol primitives**: dataset organization, augmentation, model training, and a **desktop Symbol Builder** app with **primitive detection + digit OCR (0–99)**.

This README is derived from the repository’s **Implementation & Training Report**. :contentReference[oaicite:0]{index=0}

---

## What’s implemented

- **Datasets + taxonomy**
  - Canonical primitive classes + groups are defined in `rules.yaml`
  - Training/inference class order is anchored in `models/primitive_labels_kfold.json`
  - Filename → canonical label normalization via `rules_parser.py`

- **Augmentation pipeline**
  - Generates a large, diversified training set from **normal images + sketches + user feedback**
  - Adds doctrinally relevant **outline color cues** (e.g., hostile red, control-measure green)

- **Primitive training**
  - Baseline (`src/train_augmented.py`)
  - Advanced/experimental (`src/train_augmented_advanced.py`)
  - Robust stratified **k-fold** training with “best fold” export (`src/train_augmented_kfold.py`)

- **Desktop application**
  - `symbol_builder_appV11.py` Tkinter app: palette + canvas + layers + detection + feedback logging
  - Rule/doctrine-based grouping and “Doctrine Analyze” flow

- **Digit OCR integration**
  - EMNIST-based OCR training + inference (`src/ocr_digits.py`)
  - App button “Detect Digits” supports **1–2 digit decoding (0–99)**

---

## Repository layout (key paths)

| Path | Purpose |
|------|---------|
| `dataset/normal_images/` | Canonical symbol primitives (one file per primitive) |
| `dataset/sketches/` | Hand-drawn sketches for primitives (multiple variants) |
| `dataset/augmented/` | Generated augmented set (output of `dataset/augment.py`) |
| `dataset/emnist/` | EMNIST CSV + mapping files (OCR training) |
| `dataset/augment.py` | Augmentation pipeline → writes to `dataset/augmented/` |
| `rules.yaml` | Canonical class definitions + groupings + normalizer rules |
| `rules_parser.py` | Helpers for canonical label mapping / normalization |
| `models/primitive_labels_kfold.json` | Label order used by training + app inference mapping |
| `src/train_augmented.py` | Baseline primitive training (EfficientNet by default) |
| `src/train_augmented_advanced.py` | Advanced training (MaxViT) |
| `src/train_augmented_kfold.py` | Main robust training (ConvNeXt + stratified folds) |
| `src/ocr_digits.py` | EMNIST OCR training/inference + digit segmentation (0–99) |
| `symbol_builder_appV11.py` | Desktop Symbol Builder app (Tkinter) |
| `src/serve_app.py` | Optional Gradio server attempt |
| `src/utils/training_plots.py` | Saves training curves to `models/plots/` |

---

## Datasets and class taxonomy

### Primitive classes
- Defined via:
  - `rules.yaml` (`canonical_classes` + `groups`)
  - `models/primitive_labels_kfold.json` (explicit index→label mapping used by k-fold training and app inference)
- `rules_parser.py` normalizes names (cleanup, standardizing parentheses, typos, etc.) so filenames map to canonical labels.

### Augmented dataset
`dataset/augment.py` merges:
- `dataset/normal_images/`
- `dataset/sketches/`
- feedback images referenced in: `models/logs/prediction_feedback.jsonl`

Outputs:
- `dataset/augmented/*.png` (normalized to **256×256**, padded with white background)
- Filenames encode:
  - canonical label
  - domain/source (`normal`, `sketch`, `feedback`)
  - color tag (`default`, `red`, `green`, `blue`, `black`)
  - augmentation/global indices

### Digit OCR dataset (EMNIST)
Stored under `dataset/emnist/` as CSV + mapping files. Includes multiple splits:
- **digits** (10 classes: 0–9) — recommended for digit-only OCR
- **balanced** (47 classes) — requires `emnist-balanced-mapping.txt`
- other splits (bymerge/byclass/letters/mnist)

---

## Augmentation pipeline details

Goals:
- Increase diversity (rotation/shift/warp/noise)
- Reduce confusion with many per-class variants
- Incorporate user feedback examples
- Add doctrinally relevant outline color variants:
  - hostile frames → red outline
  - friendly/neutral → black/blue outline
  - specific control measures (e.g., mine field area) → green outline

Implementation highlights:
- Uses `rules.yaml` + `rules_parser.canonical_label()` for mapping
- Uses `primitive_labels_kfold.json` to ensure label coverage
- Reads feedback from `models/logs/prediction_feedback.jsonl`
- Applies randomized geometric/photometric/noise/stroke-morphology transforms
- Targets a global `TARGET_COUNT` and boosts priority classes

---

## Training: primitive classifier

### 1) Baseline training (`src/train_augmented.py`)
- Simple “train on everything” loop (no train/val split)
- Default model: `efficientnet_b0` (pretrained)
- Writes:
  - checkpoint `.pth`
  - label map JSON
  - training plot

### 2) Advanced training (`src/train_augmented_advanced.py`)
- MaxViT experiment
- Default: `maxvit_large_tf_384.in21k_ft_in1k`
- Techniques:
  - AMP, MixUp, label smoothing
  - AdamW + cosine annealing + warmup
  - gradient clipping
  - plots/logs to `models/plots/`, `models/logs/`

### 3) Robust training (`src/train_augmented_kfold.py`)
- Stratified k-fold training + evaluation → copies best fold to canonical best file
- Default: `convnext_xxlarge.clip_laion2b_soup_ft_in1k`
- Techniques:
  - stratified folds by label index
  - strong torchvision aug (AutoAugment, ColorJitter, RandomErasing)
  - MixUp, label smoothing
  - AdamW + cosine LR + warmup
  - dropout / drop-path in timm model config
  - fold histories saved to JSON + best fold summary

---

## Desktop app: `symbol_builder_appV11.py`

UI:
- Left palette auto-populated from `dataset/normal_images/`
- Center canvas:
  - drag-drop primitives
  - sketch/drawing layer
  - multiple map layers (overlay UI)
- Right panel (Notebook tabs): map layers + inspector/doctrine analysis
- Toolbar actions: folder selection, reload, upload symbols, delete, clear board, layer mgmt, draw toggle, color selection, detection actions

Primitive detection:
- Loads timm model arch (`DEFAULT_MODEL_NAME`) + `.pth` weights
- Loads label mapping from `models/primitive_labels_kfold.json`
- “Detect Symbol” runs inference on rendered canvas and shows top-k predictions
- Logs user confirmations/rejections to `models/logs/prediction_feedback.jsonl`

Doctrine/rules integration:
- Loads `rules.yaml` (via `DoctrineEngine` if PyYAML installed)
- Uses rule groups to categorize items (Frames, Echelon, Status, Roles, Mobility, etc.)
- “Doctrine Analyze” validates symbol composition constraints

---

## Digit OCR (0–99)

### OCR model: `src/ocr_digits.py`
- Loads EMNIST CSV and caches to NPZ for faster reruns
- Fixes EMNIST orientation (transpose + horizontal flip)
- Small CNN (`SmallCNN`) classifier
- Polarity normalization (black-on-white or white-on-black input)
- `segment_digits()` + `predict_sequence(...)` for **1–2 digit decoding (0–99)**

### OCR in app
- “Detect Digits” button runs OCR on current board render (or prompts for an image)
- Auto-loads weights from `models/ocr_digits_best.pth`
- Selects mapping file based on checkpoint output size:
  - 10-class → digits mapping
  - 47-class → balanced mapping
  - fallback: tries mapping files until one loads
- Uses `predict_sequence(max_digits=2)` when available

---

## How to reproduce (local)

### 1) Generate augmented dataset
```bash
python dataset/augment.py
