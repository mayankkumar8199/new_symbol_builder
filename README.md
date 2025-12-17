# new_symbol_builder

Symbol Builder + training utilities for military symbology primitives, augmentation, and handwritten OCR integration.

## What’s in this repo
- `symbol_builder_appV11.py`: desktop Tkinter app (palette + canvas + detect).
- `dataset/augment.py`: builds `dataset/augmented/` from `dataset/normal_images/`, `dataset/sketches/`, and optional feedback logs.
- `src/train_augmented_kfold.py`: k-fold training for the primitive classifier (timm models).
- `src/ocr_digits.py`: lightweight EMNIST OCR (0–9 / 0–99 via segmentation).
- `rules.yaml` + `rules_parser.py`: canonical class names and normalization.

## Large files
This repo ignores:
- `dataset/augmented/` (hundreds of thousands of images)
- `dataset/emnist/*.csv` (multi-GB EMNIST CSVs)
- `models/*.pth` (model checkpoints can be multi-GB)

If you need to version weights/datasets, use Git LFS or a separate artifact store (Releases/Drive/S3).

