import os
import re
import glob
import json
import shutil
import pathlib
import subprocess
import sys
import time
import threading
import tkinter as tk
from collections import defaultdict
from difflib import get_close_matches
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Dict, List
from PIL import Image, ImageTk, ImageDraw
import torch
import timm
from torchvision import transforms

try:
    import yaml
except ImportError:
    yaml = None

# Ensure src is on path for local imports (ocr_digits, etc.)
ROOT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from ocr_digits import EMNIST_OCR as DigitOCR
except ImportError:
    DigitOCR = None

# ---------- Config ----------
DEFAULT_SYMBOLS_DIR = str((ROOT_DIR / "dataset" / "normal_images").resolve())
DEFAULT_MODEL_WEIGHTS = pathlib.Path(__file__).parent / "models" / "primitive_classifier_newbest.pth"
DEFAULT_MODEL_LABELS = pathlib.Path(__file__).parent / "models" / "primitive_labels_kfold.json"
DEFAULT_MODEL_NAME = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
DEFAULT_IMG_SIZE = 384
FEEDBACK_LOG = pathlib.Path(__file__).parent / "models" / "logs" / "prediction_feedback.jsonl"
FEEDBACK_IMG_DIR = pathlib.Path(__file__).parent / "models" / "logs" / "feedback_images"
# OCR weights/mapping (EMNIST digits by default; change to balanced if needed)
DEFAULT_OCR_WEIGHTS = (pathlib.Path(__file__).parent / "models" / "ocr_digits_best.pth").resolve()
DEFAULT_OCR_MAPPING = (pathlib.Path(__file__).parent / "dataset" / "emnist" / "emnist-digits-mapping.txt").resolve()
# Auto-finetune disabled; keep these None to avoid background training
AUTO_FT_SCRIPT = None
AUTO_FT_OUTPUT = None
_AUTO_FT_RUNNING = False
PALETTE_WIDTH = 340
PALETTE_COLLAPSED_WIDTH = 150
RIGHT_PANEL_WIDTH = 360
CANVAS_SIZE = (1100, 720)
THUMB_SIZE = (96, 96)
BG = "#f6f7fb"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---------- Doctrine Engine ----------
class DoctrineEngine:
    def __init__(self, project_root: pathlib.Path):
        self.project_root = project_root
        # Use root-level rules.yaml added to project
        self.rules_path = project_root / "rules.yaml"
        self.rules = {}
        self.available = False
        self.load_error = None
        self._rules_mtime = None
        self._load_rules()

    def _load_rules(self):
        self.load_error = None
        if yaml is None:
            self.available = False
            self.rules = {}
            self._rules_mtime = None
            self.load_error = "PyYAML is not installed. Install PyYAML to enable doctrine analysis."
            return
        try:
            text = self.rules_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.available = False
            self.rules = {}
            self._rules_mtime = None
            self.load_error = f"Doctrine rules file not found: {self.rules_path}"
            return
        try:
            self.rules = yaml.safe_load(text) or {}
        except Exception as exc:
            self.available = False
            self.rules = {}
            self._rules_mtime = None
            self.load_error = f"Failed to parse {self.rules_path.name}: {exc}"
            return
        try:
            self._rules_mtime = self.rules_path.stat().st_mtime
        except OSError:
            self._rules_mtime = None
        self.available = True
        self.load_error = None

    def ensure_rules_loaded(self):
        self._ensure_rules_up_to_date()
        return self.available

    def _ensure_rules_up_to_date(self):
        if yaml is None:
            return
        try:
            mtime = self.rules_path.stat().st_mtime
        except FileNotFoundError:
            if self.available or not self.load_error or "not found" not in (self.load_error or ""):
                self.available = False
                self.rules = {}
                self._rules_mtime = None
                self.load_error = f"Doctrine rules file not found: {self.rules_path}"
            return
        except OSError:
            mtime = None
        if self._rules_mtime is None or mtime != self._rules_mtime:
            self._load_rules()

    def analyze(self, board) -> dict:
        self._ensure_rules_up_to_date()
        if not self.available or not self.rules:
            errors = []
            info = []
            if self.load_error:
                errors.append(self.load_error)
            else:
                info.append("Doctrine rules.yaml not available or PyYAML missing.")
            return {
                "available": False,
                "errors": errors,
                "warnings": [],
                "info": info,
            }

        items = self._collect_items(board)
        result = {
            "affiliation": None,
            "frame_bbox": None,
            "role": None,
            "echelon": None,
            "status": None,
            "mobility": None,
            "capabilities": [],
            "amplifiers": [],
            "unit_name": None,
            "canonical_name": None,
        }
        notes = {"errors": [], "warnings": [], "info": []}

        if not items:
            return {"available": True, "result": result, **notes}

        frame_item = self._find_frame(items)
        if not frame_item:
            notes["errors"].append("Place a Friendly Unit or Hostile Unit frame to anchor analysis.")
            return {"available": True, "result": result, **notes}

        frame_bbox = frame_item["bbox"]
        result["affiliation"] = frame_item.get("affiliation")
        result["frame_bbox"] = frame_bbox

        zones = self._zones_pixels(frame_bbox)

        components = {
            "role": [],
            "echelon": [],
            "status": [],
            "mobility": [],
            "capability": [],
            "amplifier": [],
        }

        for item in items:
            if item is frame_item:
                continue
            kind, label = self._classify_label(item["name"])
            cx = (item["bbox"][0] + item["bbox"][2]) / 2.0
            cy = (item["bbox"][1] + item["bbox"][3]) / 2.0

            def inside(zone_name):
                bbox = zones.get(zone_name)
                if not bbox:
                    return False
                x1, y1, x2, y2 = bbox
                return x1 <= cx <= x2 and y1 <= cy <= y2

            if kind in components and inside(kind if kind != "capability" else "capability"):
                components[kind].append(label)
            elif kind in ("role", "echelon", "status", "mobility", "capability"):
                notes["warnings"].append(f"{kind.title()} '{label}' appears outside its expected zone.")
            elif kind == "amplifier":
                components[kind].append(label)

        role_list = components["role"]
        if not role_list:
            notes["errors"].append("No role detected inside the frame zone.")
        else:
            if len(role_list) > 1:
                notes["warnings"].append("Multiple roles detected; using the first.")
            result["role"] = role_list[0]

        echelon_list = components["echelon"]
        if echelon_list:
            if len(echelon_list) > 1:
                notes["warnings"].append("Multiple echelons detected; using the first.")
            result["echelon"] = self._resolve_echelon_wording(echelon_list[0], result["role"], components["capability"])

        status_list = components["status"]
        if status_list:
            if len(status_list) > 1:
                plus = any(s.lower() == "reinforced" for s in status_list)
                minus = any(s.lower() == "reduced" for s in status_list)
                if plus and minus:
                    result["status"] = "Reinforced and Reduced"
                else:
                    notes["warnings"].append("Multiple status markers; using the first.")
                    result["status"] = status_list[0]
            else:
                result["status"] = status_list[0]

        mobility_list = components["mobility"]
        if mobility_list:
            if len(mobility_list) > 1:
                notes["warnings"].append("Multiple mobility markers; using the first.")
            result["mobility"] = mobility_list[0]

        chosen_caps = self._reduce_capabilities_by_family(components["capability"])
        result["capabilities"] = [{"family": fam, "name": name} for fam, name in chosen_caps]
        result["amplifiers"] = components["amplifier"]

        result["canonical_name"] = self._format_canonical(result)

        return {"available": True, "result": result, **notes}

    def _collect_items(self, board):
        items = []
        for cid, rec in board.placed.items():
            x, y = board.coords(cid)
            tk_img = rec.get("tk")
            if tk_img is None:
                continue
            w = tk_img.width()
            h = tk_img.height()
            bbox = (x - w/2, y - h/2, x + w/2, y + h/2)
            items.append({"cid": cid, "name": rec.get("name", ""), "bbox": bbox})
        return items

    def _find_frame(self, items):
        for item in items:
            name = item["name"].lower()
            if "friendly" in name and "unit" in name:
                item["affiliation"] = "Friendly"
                return item
            if "hostile" in name and "unit" in name:
                item["affiliation"] = "Hostile"
                return item
        return None

    def _zones_pixels(self, bbox):
        zones = {}
        try:
            zdefs = (self.rules.get("zones") or {})
            x1, y1, x2, y2 = bbox
            fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)

            def rel_to_abs(rel_bbox):
                rx1, ry1, rx2, ry2 = rel_bbox
                return (
                    x1 + rx1 * fw,
                    y1 + ry1 * fh,
                    x1 + rx2 * fw,
                    y1 + ry2 * fh,
                )

            for key in ("role", "echelon", "status", "mobility", "capability"):
                z = zdefs.get(key)
                if z and "rel_bbox" in z:
                    zones[key] = rel_to_abs(z["rel_bbox"])
        except Exception:
            pass
        return zones

    def _canonical_choices(self, section: str):
        ontology = self.rules.get("ontology") or {}
        entries = ontology.get(section)
        if not entries:
            return []
        if isinstance(entries, dict):
            choices = []
            for key, meta in entries.items():
                if isinstance(meta, dict):
                    choices.append(meta.get("canonical", key))
                else:
                    choices.append(str(meta))
            return choices
        return list(entries)

    def _fuzzy_match(self, value: str, choices, cutoff: float = 0.82):
        if not value or not choices:
            return None
        lowered = {choice.lower(): choice for choice in choices}
        matches = get_close_matches(value.lower(), list(lowered.keys()), n=1, cutoff=cutoff)
        if matches:
            return lowered[matches[0]]
        return None

    def _match_section(self, section: str, label: str):
        choices = self._canonical_choices(section)
        if not choices:
            return None
        lab = label.lower()
        for choice in choices:
            if lab == choice.lower():
                return choice
        if len(lab) >= 3:
            for choice in choices:
                c = choice.lower()
                if c in lab or lab in c:
                    return choice
        return self._fuzzy_match(label, choices)

    def _match_status(self, label: str):
        status_def = (self.rules.get("ontology") or {}).get("status") or {}
        if not status_def:
            return None
        lab = label.strip().lower()
        canonical_choices = []
        for key, meta in status_def.items():
            meta = meta or {}
            canonical = meta.get("canonical", key)
            mark = meta.get("mark")
            canonical_choices.append(canonical)
            if lab == canonical.lower():
                return canonical
            if mark and label.strip() == mark:
                return canonical
        match = self._fuzzy_match(label, canonical_choices)
        return match
    def _normalize_label(self, label: str) -> str:
        norm = label.replace("_", " ").strip()
        norm = re.sub(r"\s+", " ", norm)
        normalize_section = self.rules.get("normalize") or {}
        for section in ("roles", "echelons", "capabilities"):
            mapping = normalize_section.get(section) or {}
            for key, value in mapping.items():
                if norm.lower() == key.lower():
                    return value
        return norm

    def _classify_label(self, name: str):
        label = self._normalize_label(name)

        role = self._match_section("roles", label)
        if role:
            return ("role", role)
        echelon = self._match_section("echelons", label)
        if echelon:
            return ("echelon", echelon)

        status = self._match_status(label)
        if status:
            return ("status", status)

        mobility = self._match_section("mobility", label)
        if mobility:
            return ("mobility", mobility)
        capability = self._match_section("capabilities", label)
        if capability:
            return ("capability", capability)
        amplifier = self._match_section("amplifiers", label)
        if amplifier:
            return ("amplifier", amplifier)
        graphic = self._match_section("graphics", label)
        if graphic:
            return ("graphic", graphic)
        return ("unknown", label)

    def _capability_family_for(self, name: str):
        families = self.rules.get("capability_families") or {}
        for fam, meta in families.items():
            for member in meta.get("members", []):
                if name.lower() == member.lower():
                    return fam
        return "Other"

    def _reduce_capabilities_by_family(self, caps):
        by_family = defaultdict(list)
        for cap in caps:
            fam = self._capability_family_for(cap)
            by_family[fam].append(cap)
        families = self.rules.get("capability_families") or {}
        chosen = []
        for fam, names in by_family.items():
            priorities = (families.get(fam) or {}).get("priority") or []
            choice = None
            for p in priorities:
                if any(p.lower() == n.lower() for n in names):
                    choice = p
                    break
            if not choice:
                choice = names[0]
            chosen.append((fam, choice))
        return chosen

    def _resolve_echelon_wording(self, echelon: str, role: str, caps: list) -> str:
        overrides = self.rules.get("echelon_word_overrides") or {}
        if not echelon:
            return echelon
        canon = echelon.lower()
        if canon == "company/battery/squadron":
            ov = overrides.get("Company/Battery/Squadron") or {}
            role_list = [role] if role else []
            fams = {self._capability_family_for(c) for c in caps}
            if any(r.lower() in {x.lower() for x in ov.get("roles_imply_battery", [])} for r in role_list):
                return "Battery"
            if any(fam in {x for x in ov.get("capabilities_imply_battery_families", [])} for fam in fams):
                return "Battery"
            if any(r.lower() in {x.lower() for x in ov.get("roles_imply_squadron", [])} for r in role_list):
                return "Squadron"
            return ov.get("default", "Company")
        if canon == "regiment/battalion":
            ov = overrides.get("Regiment/Battalion") or {}
            role_list = [role] if role else []
            fams = {self._capability_family_for(c) for c in caps}
            if any(r.lower() in {x.lower() for x in ov.get("roles_imply_regiment", [])} for r in role_list):
                return "Regiment"
            if any(fam in {x for x in ov.get("capabilities_imply_regiment_families", [])} for fam in fams):
                return "Regiment"
            return ov.get("default", "Battalion")
        return echelon

    def _format_canonical(self, result: dict) -> str:
        formatting = (self.rules.get("formatting") or {}).get("unit") or {}
        template = formatting.get("template") or "{affiliation} {role} {echelon}{status_suffix}{mobility_suffix}{capability_suffix}{unit_suffix}"
        status_map = formatting.get("status_suffix") or {}
        mobility_map = formatting.get("mobility_suffix") or {}
        cap_joiner = formatting.get("capability_joiner", ", ")
        cap_prefix = formatting.get("capability_prefix", " with ")
        unit_prefix = formatting.get("unit_prefix", " — ")

        status_suffix = status_map.get(result.get("status"), "")
        mobility_suffix = mobility_map.get(result.get("mobility"), "")
        caps = result.get("capabilities") or []
        capability_suffix = ""
        if caps:
            names = [c["name"] for c in caps]
            capability_suffix = f"{cap_prefix}{cap_joiner.join(names)}"
        unit_suffix = ""
        if result.get("unit_name"):
            unit_suffix = f"{unit_prefix}{result['unit_name']}"

        rendered = template.format(
            affiliation=result.get("affiliation") or "",
            role=result.get("role") or "",
            echelon=result.get("echelon") or "",
            status_suffix=status_suffix,
            mobility_suffix=mobility_suffix,
            capability_suffix=capability_suffix,
            unit_suffix=unit_suffix,
        )
        rendered = re.sub(r"\s+", " ", rendered).strip()
        return rendered

# ---------- Model Inference ----------
class SymbolPredictor:
    def __init__(self, project_root: pathlib.Path, weights_path: pathlib.Path, labels_path: pathlib.Path, img_size: int = DEFAULT_IMG_SIZE, model_name: str = DEFAULT_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels(labels_path)
        state = torch.load(weights_path, map_location="cpu")
        num_classes_ckpt = None
        for key in ("head.fc.weight", "head.weight"):
            if key in state:
                num_classes_ckpt = state[key].shape[0]
                break
        if num_classes_ckpt is None:
            raise RuntimeError("Could not infer num_classes from checkpoint.")
        if len(self.labels) != num_classes_ckpt:
            # Trim or fallback to checkpoint count to avoid size mismatch.
            self.labels = self.labels[:num_classes_ckpt]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes_ckpt,
            in_chans=3,
        )
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def _load_labels(self, labels_path: pathlib.Path):
        data = json.loads(labels_path.read_text())
        if isinstance(data, dict):
            return [data[str(i)] for i in range(len(data))]
        return list(data)

    def _predict_tensor(self, tensor):
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        return probs

    def predict_path(self, path: pathlib.Path, topk: int = 5):
        img = Image.open(path).convert("RGB")
        return self.predict_image(img, topk=topk)

    def predict_image(self, img: Image.Image, topk: int = 5):
        tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self._predict_tensor(tensor)
        values, idxs = torch.topk(probs, k=min(topk, len(self.labels)))
        results = []
        for v, i in zip(values, idxs):
            results.append({"label": self.labels[i.item()], "confidence": float(v)})
        return results


def append_feedback(entry: dict):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_feedback_image(img: Image.Image, source_path: str | None = None) -> str:
    FEEDBACK_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = pathlib.Path(source_path).suffix if source_path else ".png"
    out_path = FEEDBACK_IMG_DIR / f"fb_{ts}_{int(time.time()*1e6)%1_000_000:06d}{suffix}"
    img.save(out_path, format="PNG")
    return str(out_path)


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        # strip 'module.' prefixes
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _infer_num_classes_from_ocr_weights(weights_path: pathlib.Path) -> int | None:
    try:
        state = torch.load(weights_path, map_location="cpu")
    except Exception:
        return None
    state = _unwrap_state_dict(state)
    if not isinstance(state, dict):
        return None
    for k in (
        "net.9.weight",
        "fc.weight",
        "classifier.weight",
        "classifier.4.weight",  # our digit CNN head
        "head.weight",
        "head.fc.weight",
    ):
        if k in state and hasattr(state[k], "shape") and len(state[k].shape) == 2:
            return int(state[k].shape[0])
    return None


def _infer_num_classes_from_emnist_mapping(mapping_path: pathlib.Path) -> int | None:
    try:
        max_idx = -1
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            max_idx = max(max_idx, idx)
        return max_idx + 1 if max_idx >= 0 else None
    except Exception:
        return None


def _select_emnist_mapping_for_weights(weights_path: pathlib.Path) -> pathlib.Path | None:
    """Choose a mapping file under dataset/emnist that matches the checkpoint class count."""
    num_classes = _infer_num_classes_from_ocr_weights(weights_path)
    if num_classes is None:
        return DEFAULT_OCR_MAPPING if DEFAULT_OCR_MAPPING.exists() else None

    emnist_dir = (ROOT_DIR / "dataset" / "emnist").resolve()
    candidates = sorted(emnist_dir.glob("emnist-*-mapping.txt"))
    # Prefer the default mapping first if it matches.
    if DEFAULT_OCR_MAPPING.exists():
        n = _infer_num_classes_from_emnist_mapping(DEFAULT_OCR_MAPPING)
        if n == num_classes:
            return DEFAULT_OCR_MAPPING

    matches = []
    for p in candidates:
        n = _infer_num_classes_from_emnist_mapping(p)
        if n == num_classes:
            matches.append(p)
    if not matches:
        return DEFAULT_OCR_MAPPING if DEFAULT_OCR_MAPPING.exists() else None
    # Prefer digits mapping when num_classes == 10
    if num_classes == 10:
        for p in matches:
            if "digits" in p.name:
                return p
    # Prefer balanced mapping when num_classes == 47
    if num_classes == 47:
        for p in matches:
            if "balanced" in p.name:
                return p
    return matches[0]


def _try_load_digit_ocr(weights_path: pathlib.Path):
    """Try loading OCR with a best-guess mapping, then fallback across all mapping files."""
    if DigitOCR is None:
        return None, None, None
    if not weights_path.exists():
        return None, None, None
    emnist_dir = (ROOT_DIR / "dataset" / "emnist").resolve()
    mapping_files = sorted(emnist_dir.glob("emnist-*-mapping.txt"))
    preferred = _select_emnist_mapping_for_weights(weights_path)
    candidates: List[pathlib.Path] = []
    if preferred:
        candidates.append(preferred)
    for p in mapping_files:
        if p not in candidates:
            candidates.append(p)

    last_exc = None
    for mp in candidates:
        try:
            if mp.exists():
                return DigitOCR(weights_path, mp), mp, None
        except Exception as exc:
            last_exc = exc
            continue
    return None, None, last_exc


def trigger_background_finetune(weights_path: pathlib.Path):
    # Disabled: no background finetune in-app
    return
# ---------- Helpers ----------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def filename_to_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_\-]+", " ", base).strip()

def list_symbol_files(folder: str):
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files, key=natural_key)

def safe_copy_to_folder(src_path: str, dest_folder: str) -> str:
    os.makedirs(dest_folder, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(src_path))
    ext = ext.lower()
    base = re.sub(r"[^\w\-]+", "_", name).strip("_") or "symbol"
    candidate = os.path.join(dest_folder, base + ext)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_folder, f"{base}_{i}{ext}")
        i += 1
    shutil.copy2(src_path, candidate)
    return candidate

# ---------- Palette ----------
class SymbolPalette(ttk.Frame):
    def __init__(self, master, on_start_drag, on_expand_change=None, **kw):
        super().__init__(master, **kw)
        self.on_start_drag = on_start_drag
        self.on_expand_change = on_expand_change
        self._imgrefs = {}
        self.folder = None
        self.files = []
        self._grouped = {}
        self._weapons = {}
        self._active_group = None
        self._tab_buttons = {}
        self._expanded = True

        # header (non-scroll)
        self.header = ttk.Label(self, text="Palette", font=("Segoe UI", 12, "bold"))
        self.header.pack(anchor="w", padx=12, pady=(10, 6))
        ttk.Separator(self).pack(fill="x", padx=12, pady=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        # left "vertical tabs"
        self._tab_width = 132
        self.tab_frame = ttk.Frame(body, width=self._tab_width)
        self.tab_frame.pack(side="left", fill="y", padx=(8, 4), pady=(0, 8))
        self.tab_frame.pack_propagate(False)

        # right scrollable list
        self.content = ttk.Frame(body)
        self.content.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=(0, 8))

        self.canvas = tk.Canvas(self.content, bg=BG, highlightthickness=0)
        self.sb = ttk.Scrollbar(self.content, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._inner_win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._inner_win, width=e.width))
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.configure(yscrollincrement=18)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")

        # Mouse-wheel scrolling for the palette list (Windows/macOS/Linux).
        self.canvas.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.inner.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind("<Button-5>", self._on_mousewheel, add="+")
        self.inner.bind("<Button-4>", self._on_mousewheel, add="+")
        self.inner.bind("<Button-5>", self._on_mousewheel, add="+")

    def _recolor_frame_if_needed(self, pil: Image.Image, name: str) -> Image.Image:
        lname = name.lower()
        if "frame_hostile_rect" in lname:
            pil = pil.convert("RGBA")
            alpha = pil.split()[3] if pil.mode == "RGBA" else None
            if alpha is None:
                alpha = pil.convert("L")
            red = Image.new("RGBA", pil.size, (220, 0, 0, 0))
            red.putalpha(alpha)
            return red
        if "frame_friendly_rect" in lname:
            pil = pil.convert("RGBA")
            alpha = pil.split()[3] if pil.mode == "RGBA" else None
            if alpha is None:
                alpha = pil.convert("L")
            blk = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            blk.putalpha(alpha)
            return blk
        return pil

    def _on_mousewheel(self, event):
        if not self._expanded:
            return "break"
        try:
            # Linux uses Button-4/5
            if getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
            else:
                delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
                if delta == 0 and getattr(event, "delta", 0):
                    delta = -1 if event.delta > 0 else 1
                self.canvas.yview_scroll(delta, "units")
        except Exception:
            pass
        return "break"

    def _update_tab_styles(self) -> None:
        active = self._active_group if self._expanded else None
        for key, btn in (self._tab_buttons or {}).items():
            try:
                if key == active:
                    btn.configure(relief="sunken", bg="#e9eef6")
                else:
                    btn.configure(relief="raised", bg=BG)
            except Exception:
                continue

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        if expanded:
            # restore content area
            self.content.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=(0, 8))
            if self.files:
                self.header.configure(text=f"Palette ({len(self.files)} symbols)")
        else:
            # hide list to free horizontal space for the canvas
            try:
                self.content.pack_forget()
            except Exception:
                pass
            self.header.configure(text=f"Palette ({len(self.files)})" if self.files else "Palette")
        if callable(self.on_expand_change):
            try:
                self.on_expand_change(expanded)
            except Exception:
                pass
        self._update_tab_styles()

    def load_folder(self, folder: str):
        self.folder = folder
        self.files = list_symbol_files(folder)
        self._imgrefs = {}
        self._tab_buttons = {}

        self.header.configure(text=f"Palette ({len(self.files)} symbols)")
        for w in self.inner.winfo_children():
            w.destroy()
        for w in self.tab_frame.winfo_children():
            w.destroy()

        if not self.files:
            ttk.Label(
                self.inner,
                text="No images found.\nAdd PNG/JPG symbols to the folder.",
                foreground="#666",
            ).pack(anchor="w", padx=12, pady=8)
            return

        def normalize_key(name: str) -> str:
            key = name.lower().replace("-", "_")
            key = re.sub(r"\s+", "_", key)
            key = re.sub(r"_+", "_", key).strip("_")
            # common filename typos observed in dataset
            key = key.replace("intrest", "interest")
            key = key.replace("regement", "regiment")
            key = key.replace("secion", "section")
            return key

        def load_rules_groups() -> Dict[str, str]:
            groups: Dict[str, str] = {}
            if yaml is None:
                return groups
            rules_path = pathlib.Path(__file__).parent / "rules.yaml"
            try:
                data = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
            except Exception:
                return groups

            def add(items, group_key: str):
                for item in items or []:
                    groups[normalize_key(str(item))] = group_key

            add(data.get("frames", []), "frames")
            add(data.get("echelons", []), "echelons")
            add(data.get("status", []), "status")
            add(data.get("roles", []), "core_glyphs")
            add(data.get("mobility", []), "mobility")
            add(data.get("control_measures", []), "control_measures")
            # Axes are treated as control measures for UI grouping
            add(data.get("axes_of_advance", []), "control_measures")
            add(data.get("capabilities_weapons", []), "weapons_missiles")
            add(data.get("missiles", []), "weapons_missiles")
            add(data.get("boundaries_misc", []), "other")
            add(data.get("atomic_glyphs", []), "other")
            return groups

        group_map = load_rules_groups()

        def weapon_subcategory(key: str) -> str:
            # Exactly the PDF-driven subcategories.
            if "howitzer" in key or key == "howitzer_base":
                return "Howitzers"
            if key in {"lmg", "mmg", "hmg"} or "machine_gun" in key or key.endswith("_mg") or "grenade_launcher" in key:
                return "Machine Gun"
            if "air_defense" in key or key == "primitive_air_defense_base":
                return "Air Defense Missile"
            if "anti_tank_rocket_launcher" in key or "rocket_launcher" in key:
                return "Anti-Tank Rocket Launcher"
            if "anti_tank_gun" in key or "recoilless" in key or "armored_wheeled_medium_gun_system" in key:
                return "Anti-Tank Gun"
            if "missile" in key or key == "missile_base":
                return "Missiles"
            # Fallback: keep inside Weapons \& Missiles tab
            return "Missiles"

        def group_for(key: str) -> str:
            # First, rules.yaml mapping (normalized)
            if key in group_map:
                return group_map[key]
            # Then, common prefixes
            if key.startswith("frame_"):
                return "frames"
            if key.startswith("echelon_"):
                return "echelons"
            if key.startswith("status_"):
                return "status"
            if key.startswith("role_"):
                return "core_glyphs"
            if key in {"tracked", "towed", "wheeled_high_mobility", "wheeled_limited_mobility"}:
                return "mobility"
            if key.startswith("cm_") or key.endswith("_axis_of_advance") or key.endswith("axis_of_advance"):
                return "control_measures"
            if any(t in key for t in ("missile", "howitzer", "anti_tank", "grenade_launcher", "lmg", "mmg", "hmg")):
                return "weapons_missiles"
            return "other"

        # Group definitions (vertical tabs)
        ui_groups = [
            ("Frames / Affiliation", "frames"),
            ("Echelon", "echelons"),
            ("Status", "status"),
            ("Core Glyphs", "core_glyphs"),
            ("Mobility Symbols", "mobility"),
            ("Control Measures", "control_measures"),
            ("Weapons & Missiles", "weapons_missiles"),
            ("Other Symbols", "other"),
        ]

        grouped: Dict[str, List[pathlib.Path]] = {k: [] for _label, k in ui_groups}
        weapons: Dict[str, List[pathlib.Path]] = {
            "Howitzers": [],
            "Machine Gun": [],
            "Missiles": [],
            "Air Defense Missile": [],
            "Anti-Tank Rocket Launcher": [],
            "Anti-Tank Gun": [],
        }

        for p in self.files:
            key = normalize_key(pathlib.Path(p).stem)
            # force key bases into weapons/missiles tab
            if key in {"missile_base", "howitzer_base", "primitive_air_defense_base"}:
                g = "weapons_missiles"
            else:
                g = group_for(key)
            if g == "weapons_missiles":
                weapons[weapon_subcategory(key)].append(p)
            else:
                grouped.setdefault(g, []).append(p)

        for k in grouped:
            grouped[k] = sorted(grouped[k], key=lambda p: natural_key(str(p)))
        for k in weapons:
            weapons[k] = sorted(weapons[k], key=lambda p: natural_key(str(p)))

        self._grouped = grouped
        self._weapons = weapons

        def render_items(items: List[pathlib.Path], *, start_idx: int = 1):
            idx = start_idx
            for path in items:
                row = ttk.Frame(self.inner)
                row.pack(fill="x", padx=8, pady=4, anchor="w")

                lbl_img = ttk.Label(row)
                lbl_img.grid(row=0, column=0, rowspan=2, sticky="w")
                try:
                    im = Image.open(path).convert("RGBA")
                    im = self._recolor_frame_if_needed(im, name=filename_to_name(path))
                    im.thumbnail(THUMB_SIZE, Image.LANCZOS)
                    tkimg = ImageTk.PhotoImage(im)
                except Exception:
                    im = Image.new("RGBA", THUMB_SIZE, (230, 230, 230, 255))
                    tkimg = ImageTk.PhotoImage(im)
                self._imgrefs[path] = tkimg
                lbl_img.configure(image=tkimg)

                name = filename_to_name(path)
                wrap = max(160, PALETTE_WIDTH - self._tab_width - 150)
                lbl_txt = ttk.Label(row, text=f"{idx}. {name}", wraplength=wrap, justify="left")
                lbl_txt.grid(row=0, column=1, sticky="w", padx=(8, 0))

                def begin(ev, n=name, p=path):
                    self.on_start_drag(n, p, ev)

                lbl_img.bind("<Button-1>", begin)
                lbl_txt.bind("<Button-1>", begin)
                idx += 1

        def show_group(group_key: str):
            self._active_group = group_key
            self.canvas.yview_moveto(0.0)
            for w in self.inner.winfo_children():
                w.destroy()

            label = next((lbl for lbl, k in ui_groups if k == group_key), group_key)
            ttk.Label(self.inner, text=label, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(6, 2))

            if group_key == "status":
                ttk.Label(
                    self.inner,
                    text="Placement note: Status is shown at the bottom-right of the frame.",
                    foreground="#555",
                    wraplength=PALETTE_WIDTH - self._tab_width - 24,
                ).pack(anchor="w", padx=12, pady=(0, 6))

            if group_key == "weapons_missiles":
                order = [
                    "Howitzers",
                    "Machine Gun",
                    "Missiles",
                    "Air Defense Missile",
                    "Anti-Tank Rocket Launcher",
                    "Anti-Tank Gun",
                ]
                any_items = False
                idx = 1
                for sub in order:
                    items = weapons.get(sub, [])
                    if not items:
                        continue
                    any_items = True
                    ttk.Label(self.inner, text=sub, font=("Segoe UI", 10, "bold"), foreground="#1B4F72").pack(
                        anchor="w", padx=12, pady=(10, 2)
                    )
                    render_items(items, start_idx=idx)
                    idx += len(items)
                if not any_items:
                    ttk.Label(self.inner, text="No symbols in this group.", foreground="#666").pack(
                        anchor="w", padx=12, pady=8
                    )
                return

            items = grouped.get(group_key, [])
            if not items:
                ttk.Label(self.inner, text="No symbols in this group.", foreground="#666").pack(anchor="w", padx=12, pady=8)
                return
            render_items(items, start_idx=1)
            self._update_tab_styles()

        def on_tab_click(group_key: str) -> None:
            # Toggle / accordion behavior:
            # - click once -> expand
            # - click again on the same tab -> collapse
            # - clicking a different tab replaces the visible list
            if self._expanded and self._active_group == group_key:
                self.set_expanded(False)
                return
            if not self._expanded:
                self.set_expanded(True)
            show_group(group_key)

        # Build vertical tabs as buttons (always clickable, even if already selected).
        for label, key in ui_groups:
            cnt = len(grouped.get(key, []))
            if key == "weapons_missiles":
                cnt = sum(len(v) for v in weapons.values())
            txt = f"{label} ({cnt})"
            b = tk.Button(
                self.tab_frame,
                text=txt,
                anchor="w",
                padx=8,
                pady=8,
                bg=BG,
                fg="#111",
                activebackground="#e9eef6",
                activeforeground="#111",
                relief="raised",
                bd=1,
                command=lambda k=key: on_tab_click(k),
            )
            b.pack(fill="x", pady=2)
            self._tab_buttons[key] = b

        # Default group: first with content, else frames.
        default_key = "frames"
        for _lbl, k in ui_groups:
            c = len(grouped.get(k, []))
            if k == "weapons_missiles":
                c = sum(len(v) for v in weapons.values())
            if c:
                default_key = k
                break

        # Start collapsed to maximize canvas; expand when user clicks a tab.
        self._active_group = default_key
        self.set_expanded(False)

# ---------- Canvas ----------
class BoardCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.configure(bg="white", width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        self.placed = {}        # id -> {name, path, pil, tk, scale}
        self.selected_id = None
        self._drag = {"item": None, "x": 0, "y": 0}
        self.map_layers = []    # bottom -> top
        self._map_layer_counter = 0
        self._last_size = (CANVAS_SIZE[0], CANVAS_SIZE[1])
        self.draw_mode = False
        self.draw_color = "#000000"
        self._draw_last = None
        self.sketch_dirty = False
        self._sketch_lines = []

        # interactions
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind_all("<Delete>", self._delete_selected)
        self.bind_all("<plus>", lambda e: self.resize_selected(1.15))
        self.bind_all("<minus>", lambda e: self.resize_selected(1/1.15))
        self.bind_all("<KP_Add>", lambda e: self.resize_selected(1.15))
        self.bind_all("<KP_Subtract>", lambda e: self.resize_selected(1/1.15))
        self.bind_all("<Control-MouseWheel>", self._wheel_resize)
        # nudge with arrow keys
        self.bind_all("<Left>", lambda e: self.nudge(-5, 0))
        self.bind_all("<Right>", lambda e: self.nudge(5, 0))
        self.bind_all("<Up>", lambda e: self.nudge(0, -5))
        self.bind_all("<Down>", lambda e: self.nudge(0, 5))
        self.bind("<Configure>", self._on_configure)

        # sketch layer for freehand drawing
        self._init_sketch_layer()

        # context menu
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Duplicate", command=lambda: self.duplicate_selected())
        self.menu.add_command(label="Bring to Front", command=lambda: self._raise_selected())
        self.menu.add_command(label="Send to Back", command=lambda: self._lower_selected())
        self.menu.add_separator()
        self.menu.add_command(label="Delete", command=lambda: self._delete_selected())
        self.bind("<Button-3>", self._show_menu)  # right-click

        # hint
        self.hint = self.create_text(
            CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2,
            text="Drag from palette + drop here",
            fill="#8b8b8b", font=("Segoe UI", 14, "italic")
        )

    def _init_sketch_layer(self):
        # Use a modest-sized RGB layer to avoid Tk memory issues
        try_w = self.winfo_width() or CANVAS_SIZE[0]
        try_h = self.winfo_height() or CANVAS_SIZE[1]
        w = max(1, int(try_w))
        h = max(1, int(try_h))
        try:
            self.sketch_image = Image.new("RGB", (w, h), (255, 255, 255))
            self.sketch_tk = ImageTk.PhotoImage(self.sketch_image)
        except MemoryError:
            # Fallback to a smaller buffer if memory is constrained
            self.sketch_image = Image.new("RGB", (800, 600), (255, 255, 255))
            self.sketch_tk = ImageTk.PhotoImage(self.sketch_image)
        self.sketch_item = self.create_image(0, 0, image=self.sketch_tk, anchor="nw")
        self.tag_lower(self.sketch_item)
        self.sketch_dirty = False

    def set_draw_mode(self, enabled: bool):
        self.draw_mode = enabled
        self._draw_last = None
        self.config(cursor="pencil" if enabled else "")

    def set_draw_color(self, color: str):
        self.draw_color = color

    def clear_sketch(self):
        # clear any OCR overlays tied to the sketch layer
        try:
            self.delete("ocr_digits_overlay")
        except Exception:
            pass
        if hasattr(self, "sketch_item"):
            self.delete(self.sketch_item)
        self._init_sketch_layer()
        # remove drawn canvas lines
        for lid in getattr(self, "_sketch_lines", []):
            try:
                self.delete(lid)
            except Exception:
                pass
        self._sketch_lines = []
        self.event_generate("<<SketchCleared>>")

    def _draw_to_sketch(self, x0, y0, x1, y1):
        draw = ImageDraw.Draw(self.sketch_image)
        draw.line((x0, y0, x1, y1), fill=self.draw_color, width=4)
        try:
            # Update in-place to avoid allocating a new PhotoImage on every stroke.
            if getattr(self, "sketch_tk", None) is not None and hasattr(self.sketch_tk, "paste"):
                self.sketch_tk.paste(self.sketch_image)
            else:
                self.sketch_tk = ImageTk.PhotoImage(self.sketch_image)
                self.itemconfig(self.sketch_item, image=self.sketch_tk)
        except Exception:
            self.sketch_tk = ImageTk.PhotoImage(self.sketch_image)
            self.itemconfig(self.sketch_item, image=self.sketch_tk)
        # also draw visible line on canvas for immediate feedback
        line_id = self.create_line(
            x0,
            y0,
            x1,
            y1,
            fill=self.draw_color,
            width=3,
            capstyle="round",
            tags=("sketch_line",),
        )
        self._sketch_lines.append(line_id)
        self.sketch_dirty = True

    # placement
    def place_symbol(self, name, img_path, x, y, base_px=160):
        try:
            pil = Image.open(img_path).convert("RGBA")
            # recolor hostile/friendly frames
            if "frame_hostile_rect" in name.lower() or "frame_friendly_rect" in name.lower():
                alpha = pil.split()[3] if pil.mode == "RGBA" else pil.convert("L")
                color = (220, 0, 0, 0) if "hostile" in name.lower() else (0, 0, 0, 0)
                base = Image.new("RGBA", pil.size, color)
                base.putalpha(alpha)
                pil = base
        except Exception:
            pil = Image.new("RGBA", (base_px, int(base_px*0.7)), (0, 0, 0, 0))

        scale = min(1.0, base_px / max(1, max(pil.size)))
        w = max(1, int(pil.width * scale))
        h = max(1, int(pil.height * scale))
        tkimg = ImageTk.PhotoImage(pil.resize((w, h), Image.LANCZOS))

        cid = self.create_image(x, y, image=tkimg)
        self.placed[cid] = {"name": name, "path": img_path, "pil": pil, "scale": scale, "tk": tkimg}
        if self.hint:
            self.delete(self.hint)
            self.hint = None

        # select ONLY the newly placed symbol
        self._update_selection(cid)
        self.event_generate("<<SymbolPlaced>>")
        return cid

    # selection / move
    def _on_click(self, ev):
        if self.draw_mode:
            self._draw_last = (ev.x, ev.y)
            return
        # Prefer selecting real symbols even if sketch/overlay items are above them.
        for item in reversed(self.find_overlapping(ev.x - 1, ev.y - 1, ev.x + 1, ev.y + 1)):
            if item in self.placed:
                # Only start dragging if clicking the already selected symbol.
                if self.selected_id == item:
                    self._drag["item"] = item
                    self._drag["x"], self._drag["y"] = ev.x, ev.y
                else:
                    # Just select (no drag yet)
                    self._update_selection(item)
                return
        # click on empty area clears selection
        self._update_selection(None)

    def _on_drag(self, ev):
        if self.draw_mode and self._draw_last:
            x0, y0 = self._draw_last
            self._draw_to_sketch(x0, y0, ev.x, ev.y)
            self._draw_last = (ev.x, ev.y)
            return
        if self._drag["item"]:
            dx, dy = ev.x - self._drag["x"], ev.y - self._drag["y"]
            self.move(self._drag["item"], dx, dy)
            self._drag["x"], self._drag["y"] = ev.x, ev.y
            self.event_generate("<<SymbolMoved>>")

    def _on_release(self, ev):
        if self.draw_mode:
            self._draw_last = None
            return
        self._drag["item"] = None

    def _update_selection(self, cid_or_none):
        # remove old selection box
        for it in self.find_withtag("selbox"):
            self.delete(it)
        self.selected_id = cid_or_none
        if cid_or_none:
            x, y = self.coords(cid_or_none)
            rec = self.placed[cid_or_none]
            w, h = rec["tk"].width(), rec["tk"].height()
            box = self.create_rectangle(x-w/2, y-h/2, x+w/2, y+h/2,
                                        dash=(3, 2), outline="#4A90E2", tags=("selbox",))
            self.tag_lower(box, cid_or_none)
        self.event_generate("<<SelectionChanged>>")

    def has_content(self):
        if self.placed:
            return True
        if getattr(self, "sketch_dirty", False):
            return True
        # fallback: check bbox to detect any drawing
        if hasattr(self, "sketch_image") and self.sketch_image.getbbox() is not None:
            return True
        return False

    def render_to_image(self):
        """Composite all placed symbols into a PIL image sized to the canvas."""
        w = self.winfo_width() or CANVAS_SIZE[0]
        h = self.winfo_height() or CANVAS_SIZE[1]
        base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        if getattr(self, "sketch_image", None):
            try:
                sk = self.sketch_image.resize((w, h), Image.LANCZOS).convert("RGBA")
                base.alpha_composite(sk)
            except Exception:
                pass
        for cid in self.find_all():
            if cid not in self.placed:
                continue
            rec = self.placed[cid]
            x, y = self.coords(cid)
            pil = rec["pil"]
            scale = rec.get("scale", 1.0)
            rw = max(1, int(pil.width * scale))
            rh = max(1, int(pil.height * scale))
            icon = pil.resize((rw, rh), Image.LANCZOS)
            bx = int(x - rw / 2)
            by = int(y - rh / 2)
            base.alpha_composite(icon, dest=(bx, by))
        return base.convert("RGB")

    def get_item_bbox(self, cid):
        if cid not in self.placed:
            return None
        x, y = self.coords(cid)
        rec = self.placed[cid]
        tkimg = rec.get("tk")
        if tkimg is None:
            return None
        w = tkimg.width()
        h = tkimg.height()
        return (x - w/2, y - h/2, x + w/2, y + h/2)

    # delete/clear

    def _delete_selected(self, ev=None):
        if self.selected_id and self.selected_id in self.placed:
            self.delete(self.selected_id)
            del self.placed[self.selected_id]
            self.selected_id = None
            if not self.placed and not self.hint:
                w = self.winfo_width() or CANVAS_SIZE[0]
                h = self.winfo_height() or CANVAS_SIZE[1]
                self.hint = self.create_text(
                    w//2, h//2,
                    text="Drag from palette + drop here",
                    fill="#8b8b8b", font=("Segoe UI", 14, "italic")
                )
            self.event_generate("<<SymbolRemoved>>")
            self.event_generate("<<SelectionChanged>>")

    def clear_board(self):
        for cid in list(self.placed.keys()):
            self.delete(cid)
        self.placed.clear()
        self.selected_id = None
        self.clear_sketch()
        if not self.hint:
            w = self.winfo_width() or CANVAS_SIZE[0]
            h = self.winfo_height() or CANVAS_SIZE[1]
            self.hint = self.create_text(
                w//2, h//2,
                text="Drag from palette + drop here",
                fill="#8b8b8b", font=("Segoe UI", 14, "italic")
            )
        self.event_generate("<<SymbolRemoved>>")
        self.event_generate("<<SelectionChanged>>")

    # resize
    def resize_selected(self, factor):
        cid = self.selected_id
        if not cid or cid not in self.placed: return
        rec = self.placed[cid]
        new_scale = max(0.2, min(4.0, rec["scale"] * factor))
        if abs(new_scale - rec["scale"]) < 1e-3: return
        rec["scale"] = new_scale
        self._apply_scale(cid, rec)

    def set_selected_scale_abs(self, scale_abs):
        cid = self.selected_id
        if not cid or cid not in self.placed: return
        rec = self.placed[cid]
        rec["scale"] = max(0.2, min(4.0, scale_abs))
        self._apply_scale(cid, rec)

    def get_selected_scale(self):
        cid = self.selected_id
        if not cid or cid not in self.placed: return None
        return self.placed[cid]["scale"]

    def _apply_scale(self, cid, rec):
        w = max(1, int(rec["pil"].width * rec["scale"]))
        h = max(1, int(rec["pil"].height * rec["scale"]))
        tkimg = ImageTk.PhotoImage(rec["pil"].resize((w, h), Image.LANCZOS))
        rec["tk"] = tkimg
        self.itemconfig(cid, image=tkimg)
        self._update_selection(cid)

    def _wheel_resize(self, ev):
        if self.selected_id:
            self.resize_selected(1.15 if ev.delta > 0 else 1/1.15)

    def duplicate_selected(self):
        cid = self.selected_id
        if not cid or cid not in self.placed: return
        rec = self.placed[cid]
        x, y = self.coords(cid)
        nid = self.place_symbol(rec["name"], rec["path"], x + 25, y + 25, base_px=max(rec["pil"].size))
        self.placed[nid]["scale"] = rec["scale"]
        self._apply_scale(nid, self.placed[nid])

    def nudge(self, dx, dy):
        if self.selected_id:
            self.move(self.selected_id, dx, dy)
            self.event_generate("<<SymbolMoved>>")

    def _show_menu(self, ev):
        hit = self.find_closest(ev.x, ev.y)
        if hit and self.type(hit) == "image":
            self._update_selection(hit[0])
            try:
                self.menu.tk_popup(ev.x_root, ev.y_root)
            finally:
                self.menu.grab_release()

    def _raise_selected(self):
        if self.selected_id:
            self.tag_raise(self.selected_id)

    def _lower_selected(self):
        if self.selected_id:
            self.tag_lower(self.selected_id)

    # background map layers
    def add_map_layer(self, path: str, name: str = None, fit_mode: str = "fit"):
        pil = Image.open(path).convert("RGBA")
        self.update_idletasks()
        width = max(1, self.winfo_width() or CANVAS_SIZE[0])
        height = max(1, self.winfo_height() or CANVAS_SIZE[1])
        self._map_layer_counter += 1
        layer_id = self._map_layer_counter
        cid = self.create_image(width // 2, height // 2, tags=("map_layer", f"map_layer_{layer_id}"))
        fit_mode = fit_mode if fit_mode in {"fit", "cover"} else "fit"
        layer = {
            "layer_id": layer_id,
            "id": cid,
            "name": name or os.path.basename(path),
            "path": path,
            "pil": pil,
            "tk": None,
            "fit_mode": fit_mode,
            "visible": True,
            "scale": 1.0,
            "rotation": 0.0,
            "offset": [0.0, 0.0],
            "last_render_dims": pil.size,
        }
        self.map_layers.append(layer)
        self._render_map_layer(layer, width, height)
        self._reapply_map_layer_zorder()
        self._notify_map_layers_changed()
        return layer_id

    def remove_map_layer(self, layer_id: int):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        self.delete(layer["id"])
        self.map_layers = [l for l in self.map_layers if l["layer_id"] != layer_id]
        self._notify_map_layers_changed()
        return True

    def clear_map_layers(self):
        for layer in self.map_layers:
            self.delete(layer["id"])
        self.map_layers.clear()
        self._notify_map_layers_changed()

    def raise_map_layer(self, layer_id: int):
        idx = self._map_layer_index(layer_id)
        if idx is None or idx >= len(self.map_layers) - 1:
            return False
        self.map_layers[idx], self.map_layers[idx + 1] = self.map_layers[idx + 1], self.map_layers[idx]
        self._reapply_map_layer_zorder()
        self._notify_map_layers_changed()
        return True

    def lower_map_layer(self, layer_id: int):
        idx = self._map_layer_index(layer_id)
        if idx is None or idx <= 0:
            return False
        self.map_layers[idx], self.map_layers[idx - 1] = self.map_layers[idx - 1], self.map_layers[idx]
        self._reapply_map_layer_zorder()
        self._notify_map_layers_changed()
        return True

    def set_map_layer_visibility(self, layer_id: int, visible: bool):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        state = "normal" if visible else "hidden"
        self.itemconfigure(layer["id"], state=state)
        layer["visible"] = visible
        self._notify_map_layers_changed()
        return True

    def set_map_layer_scale(self, layer_id: int, scale_abs: float):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        scale_abs = max(0.05, min(8.0, float(scale_abs)))
        layer["scale"] = scale_abs
        layer["fit_mode"] = "manual"
        self._render_map_layer(layer)
        self._notify_map_layers_changed()
        return True

    def step_map_layer_scale(self, layer_id: int, factor: float):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        current = layer.get("scale", 1.0)
        target = current * float(factor)
        return self.set_map_layer_scale(layer_id, target)

    def set_map_layer_rotation(self, layer_id: int, rotation_deg: float):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        rotation = float(rotation_deg) % 360.0
        if rotation > 180:
            rotation -= 360
        layer["rotation"] = rotation
        layer["fit_mode"] = "manual"
        self._render_map_layer(layer)
        self._notify_map_layers_changed()
        return True

    def rotate_map_layer(self, layer_id: int, delta_deg: float):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        return self.set_map_layer_rotation(layer_id, layer["rotation"] + delta_deg)

    def move_map_layer(self, layer_id: int, dx: float, dy: float):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        layer["offset"][0] += dx
        layer["offset"][1] += dy
        self._apply_map_layer_position(layer)
        self._notify_map_layers_changed()
        return True

    def reset_map_layer_transform(self, layer_id: int, mode: str = "fit"):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return False
        mode = mode if mode in {"fit", "cover"} else "fit"
        layer["fit_mode"] = mode
        layer["scale"] = 1.0
        layer["rotation"] = 0.0
        layer["offset"] = [0.0, 0.0]
        self._render_map_layer(layer)
        self._notify_map_layers_changed()
        return True

    def get_map_layers(self):
        layers = []
        for order, layer in enumerate(self.map_layers):
            layers.append({
                "layer_id": layer["layer_id"],
                "name": layer["name"],
                "path": layer["path"],
                "visible": layer["visible"],
                "fit_mode": layer["fit_mode"],
                "scale": layer["scale"],
                "rotation": layer["rotation"],
                "offset": tuple(layer["offset"]),
                "size": layer.get("last_render_dims"),
                "order": order,
            })
        return layers

    def get_map_layer(self, layer_id: int):
        layer = self._find_map_layer(layer_id)
        if not layer:
            return None
        return {
            "layer_id": layer["layer_id"],
            "name": layer["name"],
            "path": layer["path"],
            "visible": layer["visible"],
            "fit_mode": layer["fit_mode"],
            "scale": layer["scale"],
            "rotation": layer["rotation"],
            "offset": tuple(layer["offset"]),
            "size": layer.get("last_render_dims"),
        }

    # helpers
    def _find_map_layer(self, layer_id: int):
        for layer in self.map_layers:
            if layer["layer_id"] == layer_id:
                return layer
        return None

    def _map_layer_index(self, layer_id: int):
        for idx, layer in enumerate(self.map_layers):
            if layer["layer_id"] == layer_id:
                return idx
        return None

    def _reapply_map_layer_zorder(self):
        for layer in reversed(self.map_layers):
            self.tag_lower(layer["id"])
        if self.hint:
            w = self.winfo_width() or CANVAS_SIZE[0]
            h = self.winfo_height() or CANVAS_SIZE[1]
            self.coords(self.hint, w // 2, h // 2)

    def _notify_map_layers_changed(self):
        self.event_generate("<<MapLayersChanged>>")

    def _on_configure(self, ev):
        width, height = ev.width, ev.height
        if (width, height) != self._last_size:
            self._last_size = (width, height)
            self._resize_map_layers(width, height)
            if self.hint:
                self.coords(self.hint, width // 2, height // 2)

    def _resize_map_layers(self, width: int, height: int):
        width = max(1, width)
        height = max(1, height)
        for layer in self.map_layers:
            self._render_map_layer(layer, width, height)

    def _compute_map_scale(self, pil_img: Image.Image, width: int, height: int, fit_mode: str):
        if pil_img.width == 0 or pil_img.height == 0:
            return 1.0
        if fit_mode == "cover":
            return max(width / pil_img.width, height / pil_img.height)
        return min(width / pil_img.width, height / pil_img.height)

    def _render_map_layer(self, layer: dict, width: int = None, height: int = None):
        width = max(1, width or self.winfo_width() or CANVAS_SIZE[0])
        height = max(1, height or self.winfo_height() or CANVAS_SIZE[1])
        mode = layer.get("fit_mode", "fit")
        if mode != "manual":
            layer["scale"] = self._compute_map_scale(layer["pil"], width, height, mode)
        scale = max(0.05, layer.get("scale", 1.0))
        new_w = max(1, int(round(layer["pil"].width * scale)))
        new_h = max(1, int(round(layer["pil"].height * scale)))
        resized = layer["pil"].resize((new_w, new_h), Image.LANCZOS)
        rotation = layer.get("rotation", 0.0)
        if abs(rotation) > 1e-3:
            rendered = resized.rotate(rotation, expand=True, resample=Image.BICUBIC)
        else:
            rendered = resized
        tkimg = ImageTk.PhotoImage(rendered)
        layer["tk"] = tkimg
        layer["last_render_dims"] = rendered.size
        self.itemconfigure(layer["id"], image=tkimg)
        self._apply_map_layer_position(layer, width, height)

    def _apply_map_layer_position(self, layer: dict, width: int = None, height: int = None):
        width = max(1, width or self.winfo_width() or CANVAS_SIZE[0])
        height = max(1, height or self.winfo_height() or CANVAS_SIZE[1])
        ox, oy = layer.get("offset", (0.0, 0.0))
        self.coords(layer["id"], width // 2 + ox, height // 2 + oy)


# ---------- Map layer panel ----------
class MapLayerPanel(ttk.Frame):
    def __init__(self, master, board: BoardCanvas, **kw):
        super().__init__(master, **kw)
        self.board = board
        self.selected_layer_id = None
        self._row_vars = {}
        self._display_layers = []
        self._suspend_controls = False
        self._suspend_select = False
        self._selected_var = tk.IntVar(value=0)

        header = ttk.Label(self, text="Layer List", font=("Segoe UI", 11, "bold"))
        header.pack(anchor="w", padx=6, pady=(6, 0))

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=6, pady=(6, 2))
        self.add_btn = ttk.Button(actions, text="Add Layer...", command=self._add_layer)
        self.add_btn.pack(side="left")
        self.remove_btn = ttk.Button(actions, text="Remove", command=self._remove_layer)
        self.remove_btn.pack(side="left", padx=(6, 0))
        self.up_btn = ttk.Button(actions, text="Move Up", command=self._move_up)
        self.up_btn.pack(side="left", padx=(12, 0))
        self.down_btn = ttk.Button(actions, text="Move Down", command=self._move_down)
        self.down_btn.pack(side="left", padx=(6, 0))
        self.clear_btn = ttk.Button(actions, text="Clear All", command=self._clear_layers)
        self.clear_btn.pack(side="right")

        self.rows_frame = ttk.Frame(self)
        self.rows_frame.pack(fill="both", expand=True, padx=6, pady=(2, 4))

        controls = ttk.LabelFrame(self, text="Layer Transform")
        controls.pack(fill="both", expand=False, padx=6, pady=(4, 6))

        self.current_label = ttk.Label(controls, text="(no layer selected)")
        self.current_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(4, 6))

        ttk.Label(controls, text="Scale (%)").grid(row=1, column=0, sticky="w")
        self.scale_slider = ttk.Scale(controls, from_=10, to=400, orient="horizontal",
                                      command=self._on_scale_change)
        self.scale_slider.grid(row=1, column=1, columnspan=2, sticky="we", padx=(6, 6))
        self.scale_value = ttk.Label(controls, width=6, text="100%")
        self.scale_value.grid(row=1, column=3, sticky="e")

        ttk.Label(controls, text="Rotation (deg)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.rotation_slider = ttk.Scale(controls, from_=-180, to=180, orient="horizontal",
                                         command=self._on_rotation_change)
        self.rotation_slider.grid(row=2, column=1, columnspan=2, sticky="we", padx=(6, 6), pady=(6, 0))
        self.rotation_value = ttk.Label(controls, width=6, text="0")
        self.rotation_value.grid(row=2, column=3, sticky="e", pady=(6, 0))

        rotate_frame = ttk.Frame(controls)
        rotate_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Button(rotate_frame, text="Rotate -15", command=lambda: self._rotate_selected(-15)).pack(side="left", padx=(0, 4))
        ttk.Button(rotate_frame, text="Rotate +15", command=lambda: self._rotate_selected(15)).pack(side="left")

        nudge_frame = ttk.Frame(controls)
        nudge_frame.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(nudge_frame, text="Move").grid(row=0, column=1)
        ttk.Button(nudge_frame, text="Up", width=4, command=lambda: self._nudge(0, -20)).grid(row=1, column=1, pady=(2, 2))
        ttk.Button(nudge_frame, text="Left", width=4, command=lambda: self._nudge(-20, 0)).grid(row=2, column=0, padx=(0, 2))
        ttk.Button(nudge_frame, text="Right", width=4, command=lambda: self._nudge(20, 0)).grid(row=2, column=2, padx=(2, 0))
        ttk.Button(nudge_frame, text="Down", width=4, command=lambda: self._nudge(0, 20)).grid(row=3, column=1, pady=(2, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=5, column=0, columnspan=4, sticky="we", pady=(8, 4))
        ttk.Button(buttons, text="Fit to Canvas", command=lambda: self._apply_reset('fit')).pack(side="left")
        ttk.Button(buttons, text="Cover Canvas", command=lambda: self._apply_reset('cover')).pack(side="left", padx=6)
        ttk.Button(buttons, text="Reset Transform", command=lambda: self._apply_reset('fit')).pack(side="left", padx=(6, 0))

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.board.bind("<<MapLayersChanged>>", lambda e: self.refresh(), add="+")
        self.refresh()

    def refresh(self, select_layer_id=None):
        if select_layer_id is not None:
            self.selected_layer_id = select_layer_id

        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._row_vars.clear()

        layers = list(reversed(self.board.get_map_layers()))
        self._display_layers = layers

        if not layers:
            ttk.Label(self.rows_frame, text="No map layers yet. Use Add Layer... to load maps.",
                      foreground="#666").pack(fill="x", pady=8)
            self.selected_layer_id = None
            self._selected_var.set(0)
        else:
            id_set = {layer["layer_id"] for layer in layers}
            for layer in layers:
                row = ttk.Frame(self.rows_frame)
                row.pack(fill="x", pady=2)
                vis_var = tk.BooleanVar(value=layer["visible"])
                self._row_vars[layer["layer_id"]] = vis_var
                ttk.Checkbutton(row, variable=vis_var,
                                command=lambda lid=layer["layer_id"], var=vis_var: self._toggle_visibility(lid, var.get())).pack(side="left")
                ttk.Radiobutton(row, text=layer["name"], variable=self._selected_var, value=layer["layer_id"],
                                command=self._on_select).pack(side="left", padx=(6, 0))
                ttk.Label(row, text=layer["fit_mode"]).pack(side="right")

            if self.selected_layer_id not in id_set:
                self.selected_layer_id = layers[0]["layer_id"]
            self._suspend_select = True
            self._selected_var.set(self.selected_layer_id or 0)
            self._suspend_select = False

        self._update_button_state()
        self._sync_controls()

    def _add_layer(self):
        path = filedialog.askopenfilename(
            title="Select map layer image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
        )
        if not path:
            return
        try:
            layer_id = self.board.add_map_layer(path)
        except Exception as exc:
            messagebox.showerror("Add Map Layer", f"Failed to add map layer:\n{exc}")
            return
        self.refresh(select_layer_id=layer_id)

    def _remove_layer(self):
        layer_id = self.selected_layer_id
        if not layer_id:
            return
        if not self.board.remove_map_layer(layer_id):
            messagebox.showwarning("Remove Map Layer", "Could not remove selected layer.")
            return
        self.refresh()

    def _move_up(self):
        if self.selected_layer_id and self.board.raise_map_layer(self.selected_layer_id):
            self.refresh(select_layer_id=self.selected_layer_id)

    def _move_down(self):
        if self.selected_layer_id and self.board.lower_map_layer(self.selected_layer_id):
            self.refresh(select_layer_id=self.selected_layer_id)

    def _clear_layers(self):
        if not self.board.get_map_layers():
            return
        self.board.clear_map_layers()
        self.refresh()

    def _toggle_visibility(self, layer_id: int, visible: bool):
        self.board.set_map_layer_visibility(layer_id, visible)

    def _on_select(self):
        if self._suspend_select:
            return
        sel = self._selected_var.get()
        self.selected_layer_id = sel or None
        self._update_button_state()
        self._sync_controls()

    def _get_selected_layer_data(self):
        if not self.selected_layer_id:
            return None
        return self.board.get_map_layer(self.selected_layer_id)

    def _sync_controls(self):
        data = self._get_selected_layer_data()
        control_widgets = [self.scale_slider, self.rotation_slider]
        if not data:
            self.current_label.configure(text="(no layer selected)")
            for widget in control_widgets:
                widget.state(["disabled"])
            self.scale_value.configure(text="-")
            self.rotation_value.configure(text="-")
            self.remove_btn.state(["disabled"])
            self.up_btn.state(["disabled"])
            self.down_btn.state(["disabled"])
            if self.board.get_map_layers():
                self.clear_btn.state(["!disabled"])
            else:
                self.clear_btn.state(["disabled"])
            return

        for widget in control_widgets:
            widget.state(["!disabled"])
        self.clear_btn.state(["!disabled"])

        self.current_label.configure(text=f"{data['name']} [{os.path.basename(data['path'])}]")
        self._suspend_controls = True
        self.scale_slider.set(max(10, min(400, data['scale'] * 100)))
        self.rotation_slider.set(data['rotation'])
        self._suspend_controls = False

        self.scale_value.configure(text=f"{int(round(data['scale'] * 100))}%")
        self.rotation_value.configure(text=f"{int(round(data['rotation']))}")
        self.remove_btn.state(["!disabled"])

        order_index = None
        for idx, layer in enumerate(self._display_layers):
            if layer['layer_id'] == data['layer_id']:
                order_index = idx
                break
        if order_index is None:
            self.up_btn.state(["disabled"])
            self.down_btn.state(["disabled"])
        else:
            if order_index == 0:
                self.up_btn.state(["disabled"])
            else:
                self.up_btn.state(["!disabled"])
            if order_index == len(self._display_layers) - 1:
                self.down_btn.state(["disabled"])
            else:
                self.down_btn.state(["!disabled"])

    def _on_scale_change(self, _value):
        if self._suspend_controls:
            return
        if not self.selected_layer_id:
            return
        value = float(self.scale_slider.get())
        self.scale_value.configure(text=f"{int(round(value))}%")
        self.board.set_map_layer_scale(self.selected_layer_id, value / 100.0)

    def _on_rotation_change(self, _value):
        if self._suspend_controls:
            return
        if not self.selected_layer_id:
            return
        value = float(self.rotation_slider.get())
        self.rotation_value.configure(text=f"{int(round(value))}")
        self.board.set_map_layer_rotation(self.selected_layer_id, value)

    def _rotate_selected(self, delta):
        if not self.selected_layer_id:
            return
        self.board.rotate_map_layer(self.selected_layer_id, delta)

    def _nudge(self, dx, dy):
        if not self.selected_layer_id:
            return
        self.board.move_map_layer(self.selected_layer_id, dx, dy)

    def _apply_reset(self, mode):
        if not self.selected_layer_id:
            return
        self.board.reset_map_layer_transform(self.selected_layer_id, mode)

    def _update_button_state(self):
        data = self._get_selected_layer_data()
        if not data:
            self.remove_btn.state(["disabled"])
            self.up_btn.state(["disabled"])
            self.down_btn.state(["disabled"])
            if self.board.get_map_layers():
                self.clear_btn.state(["!disabled"])
            else:
                self.clear_btn.state(["disabled"])
            return
        self.remove_btn.state(["!disabled"])
        self.clear_btn.state(["!disabled"])

        order_index = None
        for idx, layer in enumerate(self._display_layers):
            if layer['layer_id'] == data['layer_id']:
                order_index = idx
                break
        if order_index is None:
            self.up_btn.state(["disabled"])
            self.down_btn.state(["disabled"])
        else:
            if order_index == 0:
                self.up_btn.state(["disabled"])
            else:
                self.up_btn.state(["!disabled"])
            if order_index == len(self._display_layers) - 1:
                self.down_btn.state(["disabled"])
            else:
                self.down_btn.state(["!disabled"])


class MapLayerOverlay(ttk.Frame):
    def __init__(self, master: BoardCanvas, board: BoardCanvas, on_manage=None, **kw):
        super().__init__(master, **kw)
        self.board = board
        self.on_manage = on_manage
        self.configure(relief="raised", borderwidth=1)
        self.header = ttk.Frame(self)
        self.header.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(self.header, text="Map Layers", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(self.header, text="Manage…", command=self._open_manager).pack(side="right")

        self.inner = ttk.Frame(self)
        self.inner.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._vars = {}
        self._no_layers_lbl = ttk.Label(self.inner, text="No layers", foreground="#666")

        board.bind("<<MapLayersChanged>>", lambda e: self.refresh(), add="+")
        self.refresh()

    def refresh(self):
        for child in self.inner.winfo_children():
            child.destroy()
        self._vars.clear()

        layers = list(reversed(self.board.get_map_layers()))
        if not layers:
            self._no_layers_lbl = ttk.Label(self.inner, text="No layers", foreground="#666")
            self._no_layers_lbl.pack(anchor="w")
            return

        for layer in layers:
            row = ttk.Frame(self.inner)
            row.pack(fill="x", anchor="w", pady=1)
            var = tk.BooleanVar(value=layer["visible"])
            self._vars[layer["layer_id"]] = var
            ttk.Checkbutton(row, variable=var,
                            command=lambda lid=layer["layer_id"], v=var: self._toggle(lid, v)).pack(side="left")
            ttk.Label(row, text=layer["name"], width=16, anchor="w").pack(side="left", padx=(4, 0))
            ttk.Button(row, text="+", width=3,
                       command=lambda lid=layer["layer_id"]: self._scale(lid, 1.2)).pack(side="right")
            ttk.Button(row, text="-", width=3,
                       command=lambda lid=layer["layer_id"]: self._scale(lid, 1/1.2)).pack(side="right", padx=(0, 4))
            ttk.Label(row, text=f"{int(round(layer['scale'] * 100))}%").pack(side="right", padx=(0, 4))

    def _toggle(self, layer_id, var):
        self.board.set_map_layer_visibility(layer_id, var.get())

    def _open_manager(self):
        if callable(self.on_manage):
            self.on_manage()

    def _scale(self, layer_id, factor):
        self.board.step_map_layer_scale(layer_id, factor)
        self.refresh()

# ---------- Drag ghost ----------
class DragGhost:
    def __init__(self, root, name, img_path, on_drop):
        self.root, self.name, self.img_path, self.on_drop = root, name, img_path, on_drop
        self.top = tk.Toplevel(root); self.top.overrideredirect(True)
        self.top.attributes("-alpha", 0.85); self.top.attributes("-topmost", True)
        try:
            im = Image.open(img_path).convert("RGBA"); im.thumbnail((120, 120), Image.LANCZOS)
            self.tkimg = ImageTk.PhotoImage(im)
        except Exception:
            ph = Image.new("RGBA", (120, 90), (0, 0, 0, 0)); self.tkimg = ImageTk.PhotoImage(ph)
        ttk.Label(self.top, image=self.tkimg, padding=0).pack()
        self.cid_move = root.bind_all("<Motion>", self._follow)
        self.cid_up = root.bind_all("<ButtonRelease-1>", self._drop)
    def _follow(self, ev): self.top.geometry(f"+{ev.x_root+6}+{ev.y_root+6}")
    def _drop(self, ev):
        widget = self.root.winfo_containing(ev.x_root, ev.y_root)
        self._cleanup()
        if isinstance(widget, BoardCanvas):
            x = widget.canvasx(ev.x); y = widget.canvasy(ev.y)
            self.on_drop(widget, self.name, self.img_path, x, y)
    def _cleanup(self):
        try:
            self.root.unbind_all("<Motion>", self.cid_move)
            self.root.unbind_all("<ButtonRelease-1>", self.cid_up)
        except Exception:
            pass
        self.top.destroy()

# ---------- Inspector ----------
class Inspector(ttk.Frame):
    def __init__(self, master, board: BoardCanvas, doctrine_engine=None, on_doctrine_summary=None, **kw):
        super().__init__(master, **kw)
        self.board = board
        self.doctrine_engine = doctrine_engine
        self.on_doctrine_summary = on_doctrine_summary
        self._suspend_slider_cb = False
        self._last_report = None

        ttk.Label(self, text="Inspector", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 6))
        ttk.Separator(self).pack(fill="x", padx=10)

        self.txt = tk.Text(self, height=10, wrap="word")
        self.txt.pack(fill="both", expand=False, padx=10, pady=10)

        ttk.Label(self, text="Selected Symbol Controls", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10)
        ctr = ttk.Frame(self)
        ctr.pack(fill="x", padx=10, pady=(6, 10))

        self.sel_name = ttk.Label(ctr, text="(none selected)")
        self.sel_name.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(ctr, text="Size (%):").grid(row=1, column=0, sticky="w")
        self.scale_slider = ttk.Scale(ctr, from_=20, to=400, orient="horizontal",
                                      command=self._on_slider, length=180)
        self.scale_slider.grid(row=1, column=1, sticky="we", padx=6)
        ctr.columnconfigure(1, weight=1)

        self.lbl_scale = ttk.Label(ctr, width=5, text="100")
        self.lbl_scale.grid(row=1, column=2, sticky="e")

        self._suspend_slider_cb = True
        self.scale_slider.set(100)
        self._suspend_slider_cb = False

        btns = ttk.Frame(ctr)
        btns.grid(row=2, column=0, columnspan=3, sticky="we", pady=(8, 0))
        ttk.Button(btns, text="Smaller (-)", command=lambda: self._nudge_scale(1/1.15)).pack(side="left", padx=2)
        ttk.Button(btns, text="Bigger (+)", command=lambda: self._nudge_scale(1.15)).pack(side="left", padx=2)
        ttk.Button(btns, text="Duplicate", command=self.board.duplicate_selected).pack(side="left", padx=8)
        ttk.Button(btns, text="Delete Selected", command=self.board._delete_selected).pack(side="left", padx=2)

        doctrine_frame = ttk.LabelFrame(self, text="Doctrine Analysis")
        doctrine_frame.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        self.doctrine_txt = tk.Text(doctrine_frame, height=8, wrap="word")
        self.doctrine_txt.pack(fill="both", expand=True, padx=6, pady=(6, 2))
        self.doctrine_txt.configure(state="disabled")

        self.identify_btn = ttk.Button(doctrine_frame, text="Identify Symbol", command=self._identify_symbol)
        self.identify_btn.pack(fill="x", padx=6, pady=(2, 6))
        if not getattr(self.doctrine_engine, "available", False):
            self.identify_btn.configure(state="disabled")

        for ev in ("<<SymbolPlaced>>", "<<SymbolMoved>>", "<<SymbolRemoved>>", "<<SelectionChanged>>"):
            self.board.bind(ev, self.refresh)
        self.refresh()

    def run_doctrine_analysis(self):
        if self.doctrine_engine:
            self.doctrine_engine.ensure_rules_loaded()
        self._identify_symbol()

    def _on_slider(self, _val):
        if self._suspend_slider_cb:
            return
        val = float(self.scale_slider.get())
        self.lbl_scale.configure(text=str(int(val)))
        if self.board.selected_id:
            self.board.set_selected_scale_abs(val/100.0)

    def _nudge_scale(self, factor):
        scale = self.board.get_selected_scale()
        if scale is None:
            return
        scale = max(0.2, min(4.0, scale * factor))
        self._suspend_slider_cb = True
        try:
            self.scale_slider.set(int(scale * 100))
            self.lbl_scale.configure(text=str(int(scale * 100)))
        finally:
            self._suspend_slider_cb = False
        self.board.set_selected_scale_abs(scale)

    def refresh(self, _ev=None):
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "Placed symbols:\n\n")
        for i, cid in enumerate(self.board.placed.keys(), 1):
            x, y = self.board.coords(cid)
            name = self.board.placed[cid]["name"]
            self.txt.insert("end", f"{i}. {name} @ ({int(x)}, {int(y)})\n")

        if self.board.selected_id and self.board.selected_id in self.board.placed:
            rec = self.board.placed[self.board.selected_id]
            self.sel_name.configure(text=f"Selected: {rec['name']}")
            self._suspend_slider_cb = True
            try:
                self.scale_slider.set(int(rec["scale"] * 100))
                self.lbl_scale.configure(text=str(int(rec["scale"] * 100)))
            finally:
                self._suspend_slider_cb = False
        else:
            self.sel_name.configure(text="(none selected)")
            self._suspend_slider_cb = True
            try:
                self.scale_slider.set(100)
                self.lbl_scale.configure(text="100")
            finally:
                self._suspend_slider_cb = False

        report = None
        try:
            if self.doctrine_engine:
                report = self.doctrine_engine.analyze(self.board)
        except Exception as exc:
            report = {"available": False, "errors": [f"Doctrine error: {exc}"], "warnings": [], "info": []}
        self._last_report = report
        self.show_doctrine_report(report)

        if callable(self.on_doctrine_summary):
            summary = None
            if not report or not report.get('available', True):
                if report:
                    msgs = report.get('errors') or report.get('info')
                    if msgs:
                        summary = msgs[0]
                if not summary:
                    summary = 'Doctrine engine not available'
            else:
                res = report.get('result') or {}
                summary = res.get('canonical_name')
                if not summary:
                    bits = [res.get('affiliation') or '', res.get('role') or '', res.get('echelon') or '']
                    summary = ' '.join(b for b in bits if b).strip() or None
            self.on_doctrine_summary(summary)

    def show_doctrine_report(self, report):
        if not hasattr(self, "doctrine_txt"):
            return
        widget = self.doctrine_txt
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if not report or not report.get('available', True):
            message = None
            if report:
                message = (report.get('errors') or report.get('info'))
                if message:
                    message = message[0]
            widget.insert("end", message or "Doctrine engine not available.\n")
            widget.configure(state="disabled")
            return
        result = report.get('result', {})
        canonical = result.get('canonical_name')
        if canonical:
            widget.insert("end", f"Canonical: {canonical}\n")
        affiliation = result.get('affiliation')
        role = result.get('role')
        if affiliation or role:
            widget.insert("end", f"Role: {affiliation or 'Unknown'} {role or ''}\n")
        if result.get('echelon'):
            widget.insert("end", f"Echelon: {result['echelon']}\n")
        if result.get('status'):
            widget.insert("end", f"Status: {result['status']}\n")
        if result.get('mobility'):
            widget.insert("end", f"Mobility: {result['mobility']}\n")
        caps = result.get('capabilities') or []
        if caps:
            caps_text = ', '.join(f"{c['family']}: {c['name']}" for c in caps)
            widget.insert("end", f"Capabilities: {caps_text}\n")
        amps = result.get('amplifiers') or []
        if amps:
            widget.insert("end", f"Amplifiers: {', '.join(amps)}\n")
        widget.insert("end", "\n")
        for msg in report.get('errors', []):
            widget.insert("end", f"[ERROR] {msg}\n")
        for msg in report.get('warnings', []):
            widget.insert("end", f"[WARN] {msg}\n")
        for msg in report.get('info', []):
            widget.insert("end", f"[INFO] {msg}\n")
        widget.configure(state="disabled")

    def _identify_symbol(self):
        if not self.doctrine_engine:
            messagebox.showwarning("Doctrine Identification", "Doctrine engine not available.")
            return
        try:
            report = self.doctrine_engine.analyze(self.board)
        except Exception as exc:
            messagebox.showerror("Doctrine Identification", f"Failed to analyze symbol:\n{exc}")
            return
        self._last_report = report
        self.show_doctrine_report(report)
        if not report or not report.get('available', True):
            msgs = report.get('errors') or report.get('info') if report else None
            message = msgs[0] if msgs else "Doctrine engine not available."
            messagebox.showwarning("Doctrine Identification", message)
            return
        result = report.get('result') or {}
        lines = []
        canonical = result.get('canonical_name')
        if canonical:
            lines.append(f"Canonical: {canonical}")
        lines.append(f"Affiliation: {result.get('affiliation') or 'Unknown'}")
        lines.append(f"Role: {result.get('role') or '—'}")
        if result.get('echelon'):
            lines.append(f"Echelon: {result['echelon']}")
        if result.get('status'):
            lines.append(f"Status: {result['status']}")
        if result.get('mobility'):
            lines.append(f"Mobility: {result['mobility']}")
        caps = result.get('capabilities') or []
        if caps:
            caps_text = ', '.join(f"{c['family']}: {c['name']}" for c in caps)
            lines.append(f"Capabilities: {caps_text}")
        amps = result.get('amplifiers') or []
        if amps:
            lines.append(f"Amplifiers: {', '.join(amps)}")
        warnings = report.get('warnings') or []
        errors = report.get('errors') or []
        info = report.get('info') or []
        if errors:
            lines.append('')
            lines.extend(f"Error: {msg}" for msg in errors)
        if warnings:
            lines.extend(f"Warning: {msg}" for msg in warnings)
        if info:
            lines.extend(f"Info: {msg}" for msg in info)
        messagebox.showinfo("Doctrine Identification", "\n".join(lines))
# ---------- App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Symbol Builder (single-select drag + delete)")
        self.configure(bg=BG)
        self.geometry(f"{PALETTE_WIDTH+CANVAS_SIZE[0]+RIGHT_PANEL_WIDTH}x{CANVAS_SIZE[1]+90}")

        self._build_toolbar()

        left = ttk.Frame(self, width=PALETTE_WIDTH)
        center = ttk.Frame(self)
        right = ttk.Frame(self, width=RIGHT_PANEL_WIDTH)
        left.pack(side="left", fill="y")
        center.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y")

        self._left_panel = left
        # Control palette width manually (collapsed/expanded)
        try:
            self._left_panel.pack_propagate(False)
        except Exception:
            pass

        self.palette = SymbolPalette(
            left,
            on_start_drag=self._on_palette_drag_start,
            on_expand_change=self._on_palette_expand_change,
        )
        self.palette.pack(fill="both", expand=True)

        folder = DEFAULT_SYMBOLS_DIR
        if not os.path.isdir(folder):
            folder = filedialog.askdirectory(title="Select symbols folder")
            if not folder:
                messagebox.showerror("No folder", "No folder selected. Exiting.")
                self.destroy()
                return
        self.current_folder = folder
        self.palette.load_folder(folder)

        self.board = BoardCanvas(center)
        self.board.pack(fill="both", expand=True, padx=8, pady=8)
        self.board.focus_set()

        self.map_overlay = MapLayerOverlay(
            self.board,
            self.board,
            on_manage=lambda: self.sidebar.select(self.map_layers_panel) if hasattr(self, "sidebar") else None,
        )
        self.map_overlay.place(relx=1.0, rely=0.0, x=-12, y=12, anchor="ne")
        self.board.bind("<Configure>", lambda e: self._position_overlay(), add="+")

        self.sidebar = ttk.Notebook(right)
        self.sidebar.pack(fill="both", expand=True, padx=6, pady=6)

        self.map_layers_panel = MapLayerPanel(self.sidebar, self.board)
        self.sidebar.add(self.map_layers_panel, text="Map Layers")

        self._folder_status = f"Folder: {self.current_folder}"
        self._doctrine_summary = None
        self.status_var = tk.StringVar()
        self.status = ttk.Label(self, textvariable=self.status_var)
        self.status.pack(fill="x", side="bottom")
        self._update_status_label()

        app_root = pathlib.Path(__file__).parent.resolve()
        self.doctrine = DoctrineEngine(app_root)
        self.predictor = None
        self._predictor_weights_path = None
        self._predictor_mtime = 0.0
        self.digit_ocr = None
        try:
            if DEFAULT_MODEL_WEIGHTS.exists() and DEFAULT_MODEL_LABELS.exists():
                self.predictor = SymbolPredictor(app_root, DEFAULT_MODEL_WEIGHTS, DEFAULT_MODEL_LABELS, img_size=DEFAULT_IMG_SIZE)
                self._predictor_weights_path = DEFAULT_MODEL_WEIGHTS
                try:
                    self._predictor_mtime = DEFAULT_MODEL_WEIGHTS.stat().st_mtime
                except OSError:
                    self._predictor_mtime = 0.0
        except Exception as exc:
            messagebox.showwarning("Model Load", f"Failed to load model: {exc}")
        # load digit OCR if available (auto-select mapping that matches weights)
        self._ocr_mapping_path = None
        if DigitOCR and DEFAULT_OCR_WEIGHTS.exists():
            ocr, mapping, exc = _try_load_digit_ocr(DEFAULT_OCR_WEIGHTS)
            if ocr is not None:
                self.digit_ocr = ocr
                self._ocr_mapping_path = mapping
            elif exc is not None:
                print(f"[OCR] Failed to load default OCR: {exc}")
        self._update_doctrine_button_state()

        self.inspector = Inspector(
            self.sidebar,
            self.board,
            doctrine_engine=self.doctrine,
            on_doctrine_summary=self._update_doctrine_status,
        )
        self.sidebar.add(self.inspector, text="Inspector")
        self.sidebar.select(self.map_layers_panel)

    def _build_toolbar(self):
        tb = ttk.Frame(self)
        ttk.Button(tb, text="Choose Folder…", command=self._choose_folder).pack(side="left", padx=4, pady=6)
        ttk.Button(tb, text="Reload", command=self._reload).pack(side="left", padx=4)
        ttk.Button(tb, text="Upload Symbol(s)…", command=self._upload_symbols).pack(side="left", padx=4)
        ttk.Button(tb, text="Delete Selected", command=lambda: self.board._delete_selected()).pack(side="left", padx=4)
        ttk.Button(tb, text="Clear Board", command=self._clear_board).pack(side="left", padx=4)
        ttk.Button(tb, text="Add Map Layer…", command=self._add_map_layer_from_toolbar).pack(side="left", padx=4)
        ttk.Button(tb, text="Draw (toggle)", command=self._toggle_draw_mode).pack(side="left", padx=4)
        color_box = ttk.Frame(tb)
        ttk.Label(color_box, text="Color:").pack(side="left")
        for c, label in [("#000000", "Black"), ("#1e6be3", "Blue"), ("#d12b2b", "Red"), ("#1fa35c", "Green")]:
            ttk.Button(color_box, text=label, command=lambda col=c: self._set_draw_color(col)).pack(side="left", padx=1)
        color_box.pack(side="left", padx=4)
        ttk.Button(tb, text="Clear Draw", command=self._clear_sketch_only).pack(side="left", padx=4)
        detect_frame = ttk.Frame(tb)
        detect_frame.pack(side="left", padx=6)
        ttk.Button(detect_frame, text="Detect Symbol", command=self._detect_symbol_with_model).pack(side="left", padx=2)
        ttk.Button(detect_frame, text="Detect Digits", command=self._detect_digits_only).pack(side="left", padx=2)
        self.doctrine_btn = ttk.Button(tb, text="Doctrine Analyze", command=self._analyze_doctrine)
        self.doctrine_btn.pack(side="left", padx=4)
        ttk.Button(tb, text="Exit", command=self.destroy).pack(side="right", padx=4)
        tb.pack(fill="x")

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Select symbols folder", initialdir=getattr(self, "current_folder", None))
        if not folder:
            return
        self.current_folder = folder
        self.palette.load_folder(folder)
        self._folder_status = f"Folder: {self.current_folder}"
        self._update_status_label("Folder changed")

    def _reload(self):
        if not hasattr(self, "current_folder"):
            self._choose_folder()
            return
        self.palette.load_folder(self.current_folder)
        self._update_status_label(f"Reloaded: {self.current_folder}")

    def _toggle_draw_mode(self):
        self.draw_enabled = not getattr(self, "draw_enabled", False)
        if hasattr(self, "board"):
            # clear any selection so drawing starts immediately
            self.board._update_selection(None)
            self.board.set_draw_mode(self.draw_enabled)
            # ensure the canvas has focus to receive mouse events
            try:
                self.board.focus_set()
            except Exception:
                pass
        self._update_status_label(f"Draw mode {'on' if self.draw_enabled else 'off'}")

    def _on_palette_expand_change(self, expanded: bool) -> None:
        width = PALETTE_WIDTH if expanded else PALETTE_COLLAPSED_WIDTH
        try:
            panel = getattr(self, "_left_panel", None)
            if panel is not None:
                panel.configure(width=width)
        except Exception:
            return
        try:
            self.update_idletasks()
        except Exception:
            pass

    def _detect_digits_only(self):
        # ensure OCR is loaded; if default missing, prompt user for weights
        if self.digit_ocr is None:
            if DigitOCR and DEFAULT_OCR_WEIGHTS.exists():
                ocr, mapping, exc = _try_load_digit_ocr(DEFAULT_OCR_WEIGHTS)
                if ocr is not None:
                    self.digit_ocr = ocr
                    self._ocr_mapping_path = mapping
                elif exc is not None:
                    messagebox.showwarning("OCR", f"Failed to load default OCR weights/mapping:\n{exc}")
                    self.digit_ocr = None
            if self.digit_ocr is None and DigitOCR:
                wpath = filedialog.askopenfilename(
                    title="Select digit OCR weights (.pth)",
                    filetypes=[("PyTorch weights", "*.pth;*.pt")],
                )
                mpath = filedialog.askopenfilename(
                    title="Select EMNIST mapping (.txt)",
                    filetypes=[("Mapping", "*.txt")],
                )
                if wpath and mpath:
                    try:
                        self.digit_ocr = DigitOCR(wpath, mpath)
                    except Exception as exc:
                        messagebox.showerror("OCR", f"Failed to load selected OCR weights/mapping:\n{exc}")
                        self.digit_ocr = None
        if not getattr(self, "digit_ocr", None):
            missing = []
            if not DEFAULT_OCR_WEIGHTS.exists():
                missing.append(f"Weights missing: {DEFAULT_OCR_WEIGHTS}")
            if DEFAULT_OCR_WEIGHTS.exists():
                mapping = _select_emnist_mapping_for_weights(DEFAULT_OCR_WEIGHTS)
                if not mapping or not mapping.exists():
                    missing.append("Mapping not found (tried defaults under dataset/emnist).")
            if not missing:
                missing.append("Could not load digit OCR. Check console output for details.")
            messagebox.showwarning("OCR", "\n".join(missing))
            return
        img = None
        from_canvas = False
        if getattr(self, "board", None) and self.board.has_content():
            try:
                # Prefer the freehand sketch layer when present to avoid symbol clutter.
                if getattr(self.board, "sketch_dirty", False) and getattr(self.board, "sketch_image", None) is not None:
                    img = self.board.sketch_image.copy().convert("RGB")
                    from_canvas = True
                else:
                    img = self.board.render_to_image()
                    from_canvas = True
            except Exception:
                img = None
        if img is None:
            path = filedialog.askopenfilename(
                title="Select image to OCR",
                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
            )
            if not path:
                return
            img = Image.open(path).convert("RGB")

        try:
            # Prefer multi-number decoding when available.
            results = None
            if hasattr(self.digit_ocr, "predict_numbers"):
                results = self.digit_ocr.predict_numbers(img, max_numbers=6, max_digits_per_number=7)
            if results:
                if from_canvas and getattr(self, "board", None) is not None:
                    try:
                        self.board.delete("ocr_digits_overlay")
                    except Exception:
                        pass
                    cw = self.board.winfo_width() or CANVAS_SIZE[0]
                    ch = self.board.winfo_height() or CANVAS_SIZE[1]
                    iw, ih = img.size
                    sx = cw / max(1, iw)
                    sy = ch / max(1, ih)
                    for digits, conf, bbox in results:
                        x0, y0, x1, y1 = bbox
                        cx0, cy0 = x0 * sx, y0 * sy
                        cx1, cy1 = x1 * sx, y1 * sy
                        self.board.create_rectangle(
                            cx0,
                            cy0,
                            cx1,
                            cy1,
                            outline="#1B4F72",
                            width=2,
                            tags=("ocr_digits_overlay",),
                        )
                        ty = cy0 - 6
                        anchor = "sw"
                        if ty < 14:
                            ty = cy1 + 6
                            anchor = "nw"
                        self.board.create_text(
                            cx0 + 2,
                            ty,
                            text=f"{digits} ({conf:.2f})",
                            fill="#1B4F72",
                            font=("Segoe UI", 10, "bold"),
                            anchor=anchor,
                            tags=("ocr_digits_overlay",),
                        )

                msg = "\n".join([f"{s} (conf={c:.3f})" for s, c, _ in results])
                joined = ", ".join([s for s, _, _ in results])
                messagebox.showinfo("OCR", f"Detected number(s):\n{msg}")
                self._update_status_label(f"OCR: {joined}")
                return

            if hasattr(self.digit_ocr, "predict_sequence"):
                label, conf = self.digit_ocr.predict_sequence(img, max_digits=7)
            else:
                label, conf = self.digit_ocr.predict(img)
        except Exception as exc:
            messagebox.showerror("OCR", f"Digit OCR failed:\n{exc}")
            return
        digits_only = "".join(ch for ch in str(label) if ch.isdigit())
        if digits_only:
            try:
                value = int(digits_only)
            except ValueError:
                value = None
        else:
            value = None
        if value is not None:
            if from_canvas and getattr(self, "board", None) is not None:
                try:
                    self.board.delete("ocr_digits_overlay")
                except Exception:
                    pass
                cw = self.board.winfo_width() or CANVAS_SIZE[0]
                ch = self.board.winfo_height() or CANVAS_SIZE[1]
                iw, ih = img.size
                sx = cw / max(1, iw)
                sy = ch / max(1, ih)
                # Whole-image fallback bbox
                cx0, cy0 = 4, 4
                cx1, cy1 = cw - 4, ch - 4
                self.board.create_rectangle(
                    cx0,
                    cy0,
                    cx1,
                    cy1,
                    outline="#1B4F72",
                    width=2,
                    dash=(4, 2),
                    tags=("ocr_digits_overlay",),
                )
                self.board.create_text(
                    cx0 + 2,
                    cy0 + 2,
                    text=f"{value} ({conf:.2f})",
                    fill="#1B4F72",
                    font=("Segoe UI", 10, "bold"),
                    anchor="nw",
                    tags=("ocr_digits_overlay",),
                )
            messagebox.showinfo("OCR", f"Detected number: {value} (conf={conf:.3f})")
            self._update_status_label(f"OCR: {value} ({conf:.2f})")
        else:
            messagebox.showinfo("OCR", f"Detected: {label} (conf={conf:.3f})")
            self._update_status_label(f"OCR: {label} ({conf:.2f})")

    def _set_draw_color(self, color: str):
        if hasattr(self, "board"):
            self.board.set_draw_color(color)
        self._update_status_label(f"Draw color set")

    def _clear_sketch_only(self):
        if hasattr(self, "board"):
            try:
                self.board.delete("ocr_digits_overlay")
            except Exception:
                pass
            self.board.clear_sketch()
        self._update_status_label("Cleared drawing")

    def _upload_symbols(self):
        if not hasattr(self, "current_folder"):
            self._choose_folder()
            if not hasattr(self, "current_folder"):
                return
        paths = filedialog.askopenfilenames(
            title="Select image files to add",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.svg")]
        )
        if not paths:
            return
        copied = 0
        for p in paths:
            try:
                safe_copy_to_folder(p, self.current_folder)
                copied += 1
            except Exception as exc:
                messagebox.showwarning("Copy failed", f"Could not add {os.path.basename(p)}:\n{exc}")
        if copied:
            self.palette.load_folder(self.current_folder)
            self._update_status_label(f"Uploaded {copied} file(s)")

    def _clear_board(self):
        try:
            self.board.delete("ocr_digits_overlay")
        except Exception:
            pass
        self.board.clear_board()
        self._update_status_label("Board cleared")

    def _add_map_layer_from_toolbar(self):
        path = filedialog.askopenfilename(
            title="Select map layer image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
        )
        if not path:
            return
        try:
            layer_id = self.board.add_map_layer(path)
        except Exception as exc:
            messagebox.showerror("Add Map Layer", f"Failed to add map layer:\n{exc}")
            return
        panel = getattr(self, "map_layers_panel", None)
        if panel:
            panel.refresh(select_layer_id=layer_id)
        self._update_status_label(f"Map layer added: {os.path.basename(path)}")

    def _on_palette_drag_start(self, name, img_path, event):
        DragGhost(self, name, img_path, on_drop=self._on_drop_to_canvas)

    def _on_drop_to_canvas(self, canvas: BoardCanvas, name, img_path, x, y):
        # If dropping a Status symbol, prefer snapping it to the bottom-right of a frame.
        selected_before = getattr(canvas, "selected_id", None)
        placed_id = canvas.place_symbol(name, img_path, x, y)

        try:
            stem = pathlib.Path(str(img_path)).stem.lower()
        except Exception:
            stem = str(name).lower()
        is_status = stem.startswith("status_") or str(name).lower().strip().startswith("status")

        def is_frame_symbol(cid) -> bool:
            rec = getattr(canvas, "placed", {}).get(cid) if hasattr(canvas, "placed") else None
            if not rec:
                return False
            nm = str(rec.get("name", "")).lower()
            return "frame_" in nm or nm.startswith("frame ")

        frame_id = selected_before if (selected_before and is_frame_symbol(selected_before)) else None
        if frame_id is None and is_status and getattr(canvas, "placed", None):
            # pick nearest frame (if user didn't have one selected)
            best = None
            best_d = None
            for cid, rec in canvas.placed.items():
                nm = str(rec.get("name", "")).lower()
                if "frame_" not in nm and not nm.startswith("frame "):
                    continue
                bbox = canvas.get_item_bbox(cid)
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                d = (cx - x) ** 2 + (cy - y) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    best = cid
            # only snap if reasonably close
            if best is not None and best_d is not None and best_d <= (260**2):
                frame_id = best

        if is_status and frame_id is not None:
            bbox = canvas.get_item_bbox(frame_id)
            if bbox and placed_id in canvas.placed:
                fx0, fy0, fx1, fy1 = bbox
                tkimg = canvas.placed[placed_id].get("tk")
                sw = tkimg.width() if tkimg is not None else 0
                sh = tkimg.height() if tkimg is not None else 0
                margin = 6
                nx = fx1 - (sw / 2) - margin
                ny = fy1 - (sh / 2) - margin
                try:
                    canvas.coords(placed_id, nx, ny)
                    canvas.tag_raise(placed_id)
                except Exception:
                    pass
        self._update_status_label(f"Placed: {name}")

    def _analyze_doctrine(self):
        inspector = getattr(self, "inspector", None)
        if not inspector:
            messagebox.showwarning("Doctrine Analysis", "Inspector not ready yet.")
            return
        engine = getattr(self, "doctrine", None)
        if engine:
            engine.ensure_rules_loaded()
        self._update_doctrine_button_state()
        if not engine or not getattr(engine, "available", False):
            message = getattr(engine, "load_error", "Doctrine rules.yaml not available.") if engine else "Doctrine rules.yaml not available."
            messagebox.showwarning("Doctrine Analysis", message)
            return
        inspector.run_doctrine_analysis()

    def _detect_symbol_with_model(self):
        if not self.predictor:
            messagebox.showwarning("Model", "Model not loaded. Check weights/labels in models/.")
            return

        # Prefer current canvas if it has content; otherwise ask for a file.
        img = None
        img_src = None
        if getattr(self, "board", None) and self.board.has_content():
            try:
                img = self.board.render_to_image()
            except Exception:
                img = None
        if img is None:
            path = filedialog.askopenfilename(
                title="Select symbol image to classify",
                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
            )
            if not path:
                return
            img = Image.open(path).convert("RGB")
            img_src = path

        try:
            preds = self.predictor.predict_image(img, topk=5)
        except Exception as exc:
            messagebox.showerror("Model Prediction", f"Failed to classify symbol:\n{exc}")
            return

        if not preds:
            messagebox.showinfo("Model Prediction", "No prediction produced.")
            return

        # interactive selection from top-5
        opts = [f"{i+1}. {p['label']} ({p['confidence']:.3f})" for i, p in enumerate(preds)]
        choice = simpledialog.askstring(
            "Model Prediction",
            "Top predictions:\n" + "\n".join(opts) + "\n\nType the correct label (or pick a number 1-5).",
        )
        selected = preds[0]["label"]
        if choice:
            choice = choice.strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(preds):
                    selected = preds[idx]["label"]
            else:
                selected = choice

        # optional digit OCR on bottom-right region to capture unit numbers
        ocr_text = ""
        if self.digit_ocr and img is not None:
            try:
                w, h = img.size
                # crop bottom-right quarter as a simple heuristic for unit text
                crop = img.crop((w * 0.5, h * 0.5, w, h))
                ocr_label, ocr_conf = self.digit_ocr.predict(crop)
                ocr_text = ocr_label
            except Exception:
                ocr_text = ""

        img_path_saved = save_feedback_image(img, img_src)
        append_feedback(
            {
                "selected": selected,
                "predictions": preds,
                "accepted_best": selected == preds[0]["label"],
                "source": "canvas" if self.board and self.board.has_content() else "file",
                "image_path": img_path_saved,
                "ocr": ocr_text,
            }
        )
        # background finetune disabled; just log feedback
        self._update_status_label(f"Predicted {preds[0]['label']} ({preds[0]['confidence']:.2f}); chosen: {selected}")

    def _update_doctrine_status(self, summary):
        self._doctrine_summary = summary.strip() if summary else None
        self._update_status_label()
        self._update_doctrine_button_state()

    def _update_doctrine_button_state(self):
        btn = getattr(self, "doctrine_btn", None)
        if not btn:
            return
        if getattr(self, "doctrine", None) and getattr(self.doctrine, "available", False):
            btn.state(["!disabled"])
        else:
            btn.state(["disabled"])

    def _update_status_label(self, transient=None):
        parts = []
        if transient:
            parts.append(transient)
        if self._folder_status:
            parts.append(self._folder_status)
        if self._doctrine_summary:
            parts.append(f"Doctrine: {self._doctrine_summary}")
        self.status_var.set(" | ".join(parts))

    def _maybe_load_auto_weights(self):
        if not AUTO_FT_OUTPUT.exists():
            return
        try:
            mtime = AUTO_FT_OUTPUT.stat().st_mtime
        except OSError:
            return
        if mtime <= getattr(self, "_predictor_mtime", 0.0):
            return
        app_root = pathlib.Path(__file__).parent.resolve()
        try:
            self.predictor = SymbolPredictor(app_root, AUTO_FT_OUTPUT, DEFAULT_MODEL_LABELS, img_size=DEFAULT_IMG_SIZE, model_name=DEFAULT_MODEL_NAME)
            self._predictor_weights_path = AUTO_FT_OUTPUT
            self._predictor_mtime = mtime
            self._update_status_label("Auto-loaded latest feedback-tuned weights")
        except Exception as exc:
            self._update_status_label(f"Auto-load failed: {exc}")

    def _position_overlay(self):
        if getattr(self, "map_overlay", None) and getattr(self, "board", None):
            self.map_overlay.lift()
            self.map_overlay.place_configure(relx=1.0, rely=0.0, x=-12, y=12, anchor="ne")

if __name__ == "__main__":
    App().mainloop()
