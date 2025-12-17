from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import timm
import torch
from PIL import Image
from torch import nn
import torch.amp as torch_amp
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "augmented"
NORMAL_DIR = ROOT / "dataset" / "normal_images"
MODEL_DIR = ROOT / "models"
PLOTS_DIR = MODEL_DIR / "plots"
LOGS_DIR = MODEL_DIR / "logs"
LABELS_PATH = MODEL_DIR / "primitive_labels_kfold.json"

DEFAULT_MODEL = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
DEFAULT_IMG_SIZE = 384


# --------------------- Utilities --------------------- #
def load_label_list(path: Path) -> List[str]:
    """Load labels from the kfold JSON (dict or list)."""
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return list(data)


def stratified_splits(labels: List[int], folds: int) -> List[Tuple[List[int], List[int]]]:
    """Return indices for (train_idx, val_idx) for each fold."""
    from collections import defaultdict
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        buckets[y].append(idx)

    splits: List[Tuple[List[int], List[int]]] = []
    for fold in range(folds):
        train_idx: List[int] = []
        val_idx: List[int] = []
        for cls, idxs in buckets.items():
            if len(idxs) < folds:
                # too few samples -> simple split: last item per fold
                split = len(idxs) * fold // folds
                next_split = len(idxs) * (fold + 1) // folds
            else:
                split = len(idxs) * fold // folds
                next_split = len(idxs) * (fold + 1) // folds
            val_idx.extend(idxs[split:next_split])
            train_idx.extend(idxs[:split] + idxs[next_split:])
        splits.append((train_idx, val_idx))
    return splits


# --------------------- Dataset --------------------- #
@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


class AugmentedDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, label_list: Sequence[str], transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.labels = list(label_list)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.samples: List[Sample] = self._collect()
        if not self.samples:
            raise RuntimeError(f"No augmented images found in {root}")

    def _collect(self) -> List[Sample]:
        out: List[Sample] = []
        for p in sorted(self.root.glob("*.png")):
            stem = p.stem.split("_normal_")[0].split("_sketch_")[0]
            if stem in self.label_to_idx:
                out.append(Sample(p, stem))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        return self.transform(img), self.label_to_idx[s.label]


# --------------------- Transforms --------------------- #
def build_transforms(img_size: int, train: bool = True) -> transforms.Compose:
    aug = [
        transforms.Resize((img_size, img_size)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    if train:
        aug.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    return transforms.Compose(aug)


# --------------------- Training --------------------- #
def mixup_data(inputs, targets, alpha: float):
    if alpha <= 0:
        return inputs, (targets, targets, 1.0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    return lam * inputs + (1 - lam) * inputs[idx], (targets, targets[idx], lam)


def mixup_criterion(criterion, preds, targets):
    y_a, y_b, lam = targets
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    scaler: torch_amp.GradScaler,
    device: torch.device,
    mixup_alpha: float,
    clip_grad: float,
    train: bool = True,
    desc: str | None = None,
):
    model.train(train)
    total, correct, running_loss = 0, 0, 0.0
    progress = tqdm(loader, leave=False, desc=desc)
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            mixed, mix_t = mixup_data(imgs, labels, mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch_amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(mixed)
                loss = mixup_criterion(criterion, logits, mix_t) if mixup_alpha > 0 else criterion(logits, labels)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(imgs)
                loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_fold(
    fold_id: int,
    train_ds: Dataset,
    val_ds: Dataset,
    labels: List[str],
    args,
    device: torch.device,
):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(labels),
        in_chans=3,
        drop_rate=0.3,
        drop_path_rate=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    scaler = torch_amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

    best_acc = 0.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs:
            warm_lr = args.lr * epoch / max(1, args.warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = warm_lr
        else:
            scheduler.step()

        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.mixup_alpha,
            args.clip_grad,
            train=True,
            desc=f"Fold {fold_id} Epoch {epoch}/{args.epochs} [train]",
        )
        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            scaler,
            device,
            mixup_alpha=0.0,
            clip_grad=0.0,
            train=False,
            desc=f"Fold {fold_id} Epoch {epoch}/{args.epochs} [val]",
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    fold_ckpt = MODEL_DIR / f"primitive_classifier_fold{fold_id}.pth"
    torch.save(best_state, fold_ckpt)
    return best_acc, history, fold_ckpt


# --------------------- CLI --------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Robust k-fold training on augmented images.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", type=Path, default=MODEL_DIR / "primitive_classifier_best.pth")
    return ap.parse_args()


def main():
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_label_list(LABELS_PATH)
    tf_train = build_transforms(args.img_size, train=True)
    tf_val = build_transforms(args.img_size, train=False)
    full_ds = AugmentedDataset(DATASET_DIR, labels, tf_train)

    # Build stratified folds on the label indices.
    label_indices = [full_ds.label_to_idx[s.label] for s in full_ds.samples]
    splits = stratified_splits(label_indices, args.folds)

    best_overall = 0.0
    best_ckpt = None
    all_histories = {}

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        train_subset = Subset(full_ds, train_idx)
        val_subset = Subset(AugmentedDataset(DATASET_DIR, labels, tf_val), val_idx)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_acc, history, ckpt = train_fold(fold_id, train_subset, val_subset, labels, args, device)
        all_histories[f"fold_{fold_id}"] = history

        if best_acc > best_overall:
            best_overall = best_acc
            best_ckpt = ckpt

        # save history per fold
        LOGS_DIR.joinpath(f"train_kfold_fold{fold_id}_history.json").write_text(json.dumps(history, indent=2))

    # Save label map actually used
    label_map = {i: lbl for i, lbl in enumerate(labels)}
    LABELS_PATH.write_text(json.dumps(label_map, indent=2))

    # Copy best checkpoint to common name
    if best_ckpt:
        (MODEL_DIR / "primitive_classifier_best.pth").write_bytes(best_ckpt.read_bytes())

    # Global metrics file
    LOGS_DIR.joinpath("train_kfold_summary.json").write_text(
        json.dumps(
            {
                "best_acc": best_overall,
                "folds": args.folds,
                "epochs": args.epochs,
                "model": args.model,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "mixup_alpha": args.mixup_alpha,
            },
            indent=2,
        )
    )

    print(f"Best fold acc={best_overall:.4f}. Checkpoint saved to {MODEL_DIR / 'primitive_classifier_best.pth'}")


if __name__ == "__main__":
    main()
