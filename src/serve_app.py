from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Dict, Any
import os

import gradio as gr
import timm
import torch
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "primitive_classifier_best.pth"
DEFAULT_LABELS = ROOT / "models" / "primitive_labels_kfold.json"
DEFAULT_MODEL_NAME = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
DEFAULT_IMG_SIZE = 384
FEEDBACK_LOG = ROOT / "models" / "logs" / "prediction_feedback.jsonl"
PALETTE_DIR = ROOT / "dataset" / "normal_images"
PALETTE_LIMIT = 120  # cap for gallery to keep UI snappy
GRADIO_CACHE_DIR = ROOT / "gradio_cache"

# Ensure Gradio temp/cache is writable (avoid /tmp permission issues on some hosts)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_CACHE_DIR))
GRADIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(path: pathlib.Path) -> List[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return list(data)


def build_model(weights: pathlib.Path, labels: List[str], model_name: str, img_size: int):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(labels),
        in_chans=3,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, device, transform


def predict(model, device, transform, labels: List[str], image: Image.Image, skip: List[str] = None, topk: int = 5) -> List[Dict[str, Any]]:
    if image is None:
        return []
    skip = set(skip or [])
    with torch.no_grad():
        t = transform(image.convert("RGB")).unsqueeze(0).to(device)
        logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        values, idxs = torch.topk(probs, k=min(topk + len(skip), len(labels)))
    preds = []
    for conf, idx in zip(values, idxs):
        label = labels[idx.item()]
        if label in skip:
            continue
        preds.append({"label": label, "confidence": float(conf)})
        if len(preds) >= topk:
            break
    return preds


def append_feedback(kind: str, source: str, candidates: List[Dict[str, Any]], selected: str, rejected: List[str]):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "kind": kind,
        "source": source,
        "selected": selected,
        "rejected": rejected,
        "candidates": candidates,
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_palette(limit: int = PALETTE_LIMIT) -> List[pathlib.Path]:
    items = []
    if PALETTE_DIR.exists():
        for p in sorted(PALETTE_DIR.glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                items.append(p)
                if len(items) >= limit:
                    break
    return items


def blank_canvas(size: int = 768) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def paste_symbol(canvas: Image.Image, symbol_path: pathlib.Path) -> Image.Image:
    base = canvas.copy() if canvas is not None else blank_canvas(max(canvas.size) if canvas else DEFAULT_IMG_SIZE * 2)
    if not symbol_path or not symbol_path.exists():
        return base
    try:
        icon = Image.open(symbol_path).convert("RGBA")
    except Exception:
        return base
    # scale icon to ~25% of canvas and paste at center
    scale = 0.25
    target_w = int(base.width * scale)
    target_h = int(icon.height * (target_w / icon.width))
    icon = icon.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
    x = (base.width - icon.width) // 2
    y = (base.height - icon.height) // 2
    base.paste(icon, (x, y), icon)
    return base.convert("RGB")


def choose_source(canvas_img: Image.Image, sketch_img: Image.Image) -> Image.Image | None:
    if sketch_img is not None:
        return sketch_img
    return canvas_img


def build_interface(model, device, transform, labels: List[str], img_size: int):
    palette = load_palette()
    palette_labels = [p.stem for p in palette]

    with gr.Blocks() as demo:
        gr.Markdown("# Symbol Builder (Gradio)")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Palette (from normal_images)")
                gallery = gr.Gallery(value=palette, label="Select symbol", columns=2, allow_preview=True, height=400)
                selected = gr.Textbox(label="Selected symbol", interactive=False)
                hint = gr.Textbox(label="Hint", value="Pick a symbol from the palette, optionally draw, then click Detect.", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### Canvas & Drawing")
                canvas = gr.ImageEditor(type="pil", image_mode="RGB", label="Canvas (drag/drop + edits)", height=img_size * 2, width=img_size * 2, value=blank_canvas(img_size * 2))
                sketch = gr.Sketchpad(label="Hand draw", height=img_size, width=img_size)
                with gr.Row():
                    add_btn = gr.Button("Add selected to canvas")
                    reset_canvas = gr.Button("Clear canvas")
                with gr.Row():
                    detect_btn = gr.Button("Detect drawing")
                    wrong = gr.Button("Wrong -> next guess")
                    confirm = gr.Button("Confirm & Log")
                preds_box = gr.JSON(label="Predictions (top-5)")
                select = gr.Radio(choices=[], label="Select label")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                rejected_state = gr.State([])
                preds_state = gr.State([])
                gr.Markdown("Detection uses either the sketch (if drawn) or the canvas image.")

        def on_select(evt: gr.SelectData):
            idx = evt.index
            if idx is None or idx >= len(palette):
                return ""
            return palette_labels[idx]

        def on_add(symbol_name, current_canvas):
            if not symbol_name:
                return current_canvas, "No palette symbol selected."
            try:
                idx = palette_labels.index(symbol_name)
            except ValueError:
                return current_canvas, "Symbol not in palette."
            updated = paste_symbol(current_canvas, palette[idx])
            return updated, f"Pasted {symbol_name}."

        def on_reset():
            return blank_canvas(img_size * 2), "Canvas cleared."

        def on_detect(canvas_img, sketch_img):
            source = choose_source(canvas_img, sketch_img)
            preds = predict(model, device, transform, labels, source, skip=[])
            choices = [p["label"] for p in preds]
            return preds, preds, choices, [], "Detected." if preds else "No predictions."

        def on_wrong(canvas_img, sketch_img, preds, rejected):
            source = choose_source(canvas_img, sketch_img)
            if not preds:
                return preds, preds, [""], rejected, "No predictions available."
            rej = rejected + [preds[0]["label"]]
            new_preds = predict(model, device, transform, labels, source, skip=rej)
            choices = [p["label"] for p in new_preds] or [""]
            return new_preds, new_preds, choices, rej, f"Skipped: {rej[-1]}"

        def on_confirm(canvas_img, sketch_img, preds, rejected, selection):
            source = "sketch" if sketch_img is not None else "canvas"
            if not selection:
                return "No selection made."
            append_feedback(source, source, preds, selection, rejected)
            return f"Saved: {selection} (rejected {len(rejected)})"

        gallery.select(on_select, None, selected)
        add_btn.click(on_add, [selected, canvas], [canvas, status])
        reset_canvas.click(on_reset, None, [canvas, status])
        detect_btn.click(on_detect, [canvas, sketch], [preds_box, preds_state, select, rejected_state, status])
        wrong.click(on_wrong, [canvas, sketch, preds_state, rejected_state], [preds_box, preds_state, select, rejected_state, status])
        confirm.click(on_confirm, [canvas, sketch, preds_state, rejected_state, select], status)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Serve symbol detector via Gradio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing link")
    args = parser.parse_args()

    weights = DEFAULT_WEIGHTS
    labels_path = DEFAULT_LABELS
    if not weights.exists():
        raise FileNotFoundError(f"Required weights not found: {weights}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Required labels not found: {labels_path}")

    labels = load_labels(labels_path)
    model, device, transform = build_model(weights, labels, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE)
    demo = build_interface(model, device, transform, labels, DEFAULT_IMG_SIZE)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
