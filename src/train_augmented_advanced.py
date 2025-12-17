from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from torch import nn
from torch.cuda import amp
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
LOGS_DIR = MODEL_DIR / "logs"
LABELS_PATH = MODEL_DIR / "primitive_labels_advanced.json"

LABEL_SPLIT_RE = re.compile(r"_(normal|sketch)_", re.IGNORECASE)
RULES_CONFIG = load_rules(ROOT_DIR / "rules.yaml")
AVAILABLE_CLASSES, _ = available_classes(RULES_CONFIG, NORMAL_DIR)
CLASS_SET = set(AVAILABLE_CLASSES)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


class AugmentedDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, class_labels: List[str], transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.all_labels = class_labels
        self.samples: List[Sample] = self._scan()
        seen = {sample.label for sample in self.samples}
        self.class_labels = [label for label in class_labels if label in seen]
        if not self.class_labels:
            raise RuntimeError(f"No augmented images found under {root}")
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.class_labels)}

    def _scan(self) -> List[Sample]:
        samples: List[Sample] = []
        for image_path in sorted(self.root.glob("*.png")):
            label = self._label_from_stem(image_path.stem)
            if label:
                samples.append(Sample(path=image_path, label=label))
        return samples

    @staticmethod
    def _label_from_stem(stem: str) -> str | None:
        match = LABEL_SPLIT_RE.split(stem, maxsplit=1)
        if len(match) < 3:
            return None
        label = match[0]
        return label if label in CLASS_SET else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        target = self.label_to_idx[sample.label]
        return tensor, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an advanced primitive classifier using the augmented dataset."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for CE loss")
    parser.add_argument("--mixup-alpha", type=float, default=0.1, help="Mixup alpha (0 to disable)")
    parser.add_argument(
        "--model",
        type=str,
        default="maxvit_large_tf_384.in21k_ft_in1k",
        help="Advanced timm model name (pretrained) to fine-tune",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=384,
        help="Input resolution for training",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker threads"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "primitive_classifier_advanced.pth",
        help="Path to save fine-tuned weights",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=LABELS_PATH,
        help="Where to save label index mapping",
    )
    return parser.parse_args()


def create_loader(batch_size: int, num_workers: int, img_size: int) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = AugmentedDataset(DATASET_ROOT, AVAILABLE_CLASSES, transform)
    pin = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return loader, dataset.class_labels


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def _maybe_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    if alpha <= 0.0:
        return inputs, (targets, targets, 0.0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, (targets_a, targets_b, lam)


def _mixup_criterion(
    criterion: nn.Module, preds: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]
) -> torch.Tensor:
    targets_a, targets_b, lam = targets
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad: float,
    label_smoothing: float,
    mixup_alpha: float,
    warmup_epochs: int,
) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*torch.cuda.amp.*deprecated.*",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"[MaxViT] Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            mixed_inputs, mix_targets = _maybe_mixup(inputs, targets, mixup_alpha)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(mixed_inputs)
                loss = _mixup_criterion(loss_fn, outputs, mix_targets) if mixup_alpha > 0 else loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            progress.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / max(1, warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        avg_loss = running_loss / total
        acc = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, "lr": current_lr})
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f} lr={current_lr:.6f}")

    final_acc = evaluate(model, loader, device)
    history.append({"epoch": epochs + 1, "loss": history[-1]["loss"], "acc": history[-1]["acc"], "val_acc": final_acc})
    print(f"[Advanced] Final evaluation accuracy={final_acc:.4f}")
    return history


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    loader, labels = create_loader(args.batch_size, args.num_workers, args.img_size)
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(labels),
        in_chans=3,
        drop_rate=0.4,
        drop_path_rate=0.25,
    )

    history = train(
        model,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        warmup_epochs=args.warmup_epochs,
    )
    torch.save(model.state_dict(), args.output)
    torch.save(model.state_dict(), MODEL_DIR / "primitive_classifier_advanced_latest.pth")
    args.label_output.write_text(json.dumps({idx: label for idx, label in enumerate(labels)}, indent=2))
    plot_path = PLOTS_DIR / "train_augmented_advanced.png"
    save_training_plot(history, plot_path, "Advanced Training Metrics")
    history_path = LOGS_DIR / "train_augmented_advanced_history.json"
    metrics_path = LOGS_DIR / "train_augmented_advanced_metrics.json"
    history_path.write_text(json.dumps(history, indent=2))
    final_acc = history[-1].get("val_acc", None)
    metrics_path.write_text(
        json.dumps(
            {
                "final_acc": final_acc,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "model": args.model,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            indent=2,
        )
    )
    print(f"Saved advanced model to {args.output}")
    print(f"Saved latest checkpoint to {MODEL_DIR / 'primitive_classifier_advanced_latest.pth'}")
    print(f"Saved labels to {args.label_output}")
    print(f"Saved training plot to {plot_path}")
    print(f"Saved history to {history_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
