from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import timm
import torch
from PIL import Image
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOGS_DIR = MODEL_DIR / "logs"
FEEDBACK_LOG = LOGS_DIR / "prediction_feedback.jsonl"
LABELS_PATH = MODEL_DIR / "primitive_labels_kfold.json"
DEFAULT_WEIGHTS = MODEL_DIR / "primitive_classifier_newbest.pth"
OUTPUT_WEIGHTS = MODEL_DIR / "primitive_classifier_feedback_ft.pth"
DEFAULT_MODEL = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
IMG_SIZE = 384


def load_labels(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return list(data)


class FeedbackDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, entries: List[dict], labels: List[str], img_size: int):
        self.entries = []
        label_to_idx = {l: i for i, l in enumerate(labels)}
        for e in entries:
            path = e.get("image_path")
            sel = e.get("selected")
            if not path or sel not in label_to_idx:
                continue
            p = Path(path)
            if p.exists():
                self.entries.append((p, label_to_idx[sel]))
        if not self.entries:
            raise RuntimeError("No valid feedback entries with existing images.")
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label_idx = self.entries[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label_idx


def load_feedback(path: Path) -> List[dict]:
    entries = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune model on logged feedback images.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", type=Path, default=OUTPUT_WEIGHTS)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


def train(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in tqdm(loader, leave=False, desc="FT train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False, desc="FT eval"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


def main():
    args = parse_args()
    labels = load_labels(LABELS_PATH)
    entries = load_feedback(FEEDBACK_LOG)
    dataset = FeedbackDataset(entries, labels, IMG_SIZE)

    # simple split: 90/10 train/val
    split = max(1, int(0.9 * len(dataset)))
    train_subset = torch.utils.data.Subset(dataset, range(0, split))
    val_subset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=len(labels),
        in_chans=3,
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc >= best_acc:
            best_acc = va_acc
            best_state = model.state_dict()
        print(f"[FT] Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, val_loss={va_loss:.4f}, val_acc={va_acc:.4f}")

    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, args.output)
    LOGS_DIR.joinpath("finetune_feedback_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved fine-tuned weights to {args.output} (best val acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
