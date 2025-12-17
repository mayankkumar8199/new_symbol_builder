from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.training_plots import save_training_plot

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from rules_parser import available_classes, load_rules


DATASET_ROOT = ROOT_DIR / "dataset" / "augmented"
NORMAL_DIR = ROOT_DIR / "dataset" / "normal_images"
MODEL_DIR = ROOT_DIR / "models"
PLOTS_DIR = MODEL_DIR / "plots"
LABELS_PATH = MODEL_DIR / "primitive_labels.json"

DOMAIN_SPLIT_RE = re.compile(r"_(normal|sketch)_", re.IGNORECASE)
RULES_CONFIG = load_rules(ROOT_DIR / "rules.yaml")
AVAILABLE_CLASSES, _ = available_classes(RULES_CONFIG, NORMAL_DIR)
CLASS_ORDER = AVAILABLE_CLASSES
CLASS_SET = set(CLASS_ORDER)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


class AugmentedDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, class_labels: List[str], transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.all_labels = class_labels
        self.samples: List[Sample] = self._collect_samples()
        seen = {sample.label for sample in self.samples}
        self.class_labels = [label for label in class_labels if label in seen]
        if not self.class_labels:
            raise RuntimeError(f"No augmented images matched canonical classes in {root}")
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.class_labels)}

    def _collect_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for path in sorted(self.root.glob("*.png")):
            label = self._extract_label(path)
            if label:
                samples.append(Sample(path=path, label=label))
        return samples

    def _extract_label(self, path: Path) -> str | None:
        match = DOMAIN_SPLIT_RE.split(path.stem, maxsplit=1)
        if len(match) < 3:
            return None
        label = match[0]
        return label if label in CLASS_SET else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        label_idx = self.label_to_idx[sample.label]
        return tensor, label_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a primitive classifier using all augmented images (no split)."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        help="Timm model name with pretrained weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "primitive_classifier.pth",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker threads"
    )
    return parser.parse_args()


def create_dataloader(batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = AugmentedDataset(DATASET_ROOT, CLASS_ORDER, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset.class_labels


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            progress.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

        avg_loss = epoch_loss / total
        accuracy = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "acc": accuracy, "val_acc": accuracy})
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={accuracy:.4f}")

    return history


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    loader, labels = create_dataloader(args.batch_size, args.num_workers)

    model = timm.create_model(
        args.model, pretrained=True, num_classes=len(labels), in_chans=3
    )
    history = train(model, loader, args.epochs, args.lr)

    torch.save(model.state_dict(), args.output)
    LABELS_PATH.write_text(json.dumps({idx: label for idx, label in enumerate(labels)}, indent=2))
    plot_path = PLOTS_DIR / "train_augmented.png"
    save_training_plot(history, plot_path, "Baseline Training Metrics")
    print(f"Saved model to {args.output}")
    print(f"Saved label map to {LABELS_PATH}")
    print(f"Saved training plot to {plot_path}")


if __name__ == "__main__":
    main()
