from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
RULES_PATH = PROJECT_ROOT / "rules.yaml"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SERIAL_PATTERN = re.compile(r"_[0-9]+$")
SPACE_HYPHEN_PATTERN = re.compile(r"[\\s-]+")
UNDERSCORE_PATTERN = re.compile(r"_+")


def load_rules(path: Path | None = None) -> Dict:
    rules_path = path or RULES_PATH
    with open(rules_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_normalizer_step(normalizer: Dict, key: str, default=None):
    for step in normalizer.get("steps", []):
        if key in step:
            return step[key]
    return default


def _normalize_stem(stem: str, config: Dict) -> str:
    normalizer = config.get("normalizer", {})
    value = stem

    if _extract_normalizer_step(normalizer, "lower_case", False):
        value = value.lower()

    std_parentheses = _extract_normalizer_step(normalizer, "standardize_parentheses", {}) or {}
    replacements: Dict[str, str] = {}
    if isinstance(std_parentheses, dict):
        replacements = {k.lower(): v for k, v in std_parentheses.items()}
    elif isinstance(std_parentheses, list):
        for item in std_parentheses:
            if isinstance(item, dict):
                for k, v in item.items():
                    replacements[k.lower()] = v
    temp_value = value
    for raw, replacement in replacements.items():
        temp_value = temp_value.replace(raw, replacement)
    value = temp_value

    if _extract_normalizer_step(normalizer, "strip_extension", False):
        value = Path(value).stem

    if _extract_normalizer_step(normalizer, "replace_spaces_hyphens_with_underscore", False):
        value = SPACE_HYPHEN_PATTERN.sub("_", value)

    if _extract_normalizer_step(normalizer, "collapse_multiple_underscores", False):
        value = UNDERSCORE_PATTERN.sub("_", value)

    strip_serial = _extract_normalizer_step(normalizer, "strip_trailing_serial")
    if strip_serial == "_<digits>":
        value = SERIAL_PATTERN.sub("", value)

    if _extract_normalizer_step(normalizer, "trim_edge_underscores", False):
        value = value.strip("_")

    value = value.replace("(", "_").replace(")", "_")
    value = UNDERSCORE_PATTERN.sub("_", value)
    value = value.strip("_")

    return value


def _apply_fixes(label: str, fixes: Dict[str, str]) -> str:
    if not fixes:
        return label

    tokens = label.split("_")
    tokens = [fixes.get(token, token) for token in tokens]
    updated = "_".join(tokens)
    return fixes.get(updated, updated)


def canonical_label(path: Path, config: Dict) -> Optional[str]:
    explicit = config.get("explicit", {})
    canonical_classes = set(config.get("canonical_classes", []))
    fixes = config.get("fixes", {})
    stem_equiv = config.get("stem_equiv", {})

    if path.name in explicit:
        label = explicit[path.name]
    else:
        label = _normalize_stem(path.name, config)
        label = _apply_fixes(label, fixes)
        label = stem_equiv.get(label, label)

    if canonical_classes and label not in canonical_classes:
        return None
    return label


def available_classes(config: Dict, normal_dir: Path) -> Tuple[List[str], List[str]]:
    canonical = config.get("canonical_classes", [])
    canonical_set = set(canonical)
    seen = set()

    if normal_dir.exists():
        for path in normal_dir.iterdir():
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = canonical_label(path, config)
            if label:
                seen.add(label)

    available = [label for label in canonical if label in seen]
    missing = [label for label in canonical if label not in seen]
    return available, missing
