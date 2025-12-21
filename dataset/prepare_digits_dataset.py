from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "dataset" / "digits_raw"
DEFAULT_OUT_ROOT = ROOT / "dataset" / "digits_dataset"
DEFAULT_ARDIS_RAR = ROOT / "dataset" / "ARDIS_DATASET_III.rar"
DEFAULT_ARDIS_EXTRACT = ROOT / "dataset" / "ardis_extracted"
DEFAULT_ARDIS_FOLDER = "ARDIS_DATASET_3"
DEFAULT_IMG_SIZE = 28
DEFAULT_SEED = 1337


@dataclass(frozen=True)
class SampleRow:
    filename: str
    label: int
    source: str
    source_split: str


def _configure_ssl_certifi() -> None:
    """Fix SSL issues on some conda/python installs (Windows especially)."""
    try:
        import certifi  # type: ignore
    except Exception:
        return
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)

    try:
        ssl._create_default_https_context = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)
        )
    except Exception:
        return


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - trusted public dataset/tool URL
    tmp.replace(dest)


def _find_on_path(names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = shutil.which(name)
        if p:
            return Path(p)
    return None


def _ensure_7z(tools_dir: Path) -> Path:
    """
    Returns a path to a working 7-zip CLI (7z/7zz).
    - Uses system PATH if available.
    - Otherwise installs a local copy into tools/7zip/installed/.
    """
    on_path = _find_on_path(["7z", "7zz", "7za"])
    if on_path:
        return on_path

    install_dir = tools_dir / "7zip" / "installed"

    if sys.platform.startswith("win"):
        exe = install_dir / "7z.exe"
        if exe.exists():
            return exe

        setup = tools_dir / "7zip" / "7z-setup.exe"
        url = "https://www.7-zip.org/a/7z2501-x64.exe"
        if not setup.exists():
            print(f"[digits] downloading 7-Zip installer: {url}")
            _download(url, setup)

        install_dir.mkdir(parents=True, exist_ok=True)
        print(f"[digits] installing 7-Zip to: {install_dir}")
        subprocess.run(  # noqa: S603,S607 - trusted installer URL
            [str(setup), "/S", f"/D={install_dir}"],
            check=True,
        )
        if not exe.exists():
            raise RuntimeError(f"7-Zip install completed but {exe} not found.")
        return exe

    if sys.platform.startswith("linux"):
        seven = install_dir / "7zz"
        if seven.exists():
            return seven

        tarball = tools_dir / "7zip" / "7z-linux.tar.xz"
        url = "https://www.7-zip.org/a/7z2501-linux-x64.tar.xz"
        if not tarball.exists():
            print(f"[digits] downloading 7-Zip for linux: {url}")
            _download(url, tarball)

        install_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, mode="r:xz") as tf:
            tf.extractall(path=install_dir)  # noqa: S202 - local tarball, known contents

        found = list(install_dir.rglob("7zz"))
        if not found:
            raise RuntimeError(f"Could not find 7zz under {install_dir} after extract.")
        seven = found[0]
        seven.chmod(seven.stat().st_mode | 0o111)
        return seven

    raise RuntimeError(
        "No 7-zip CLI found. Install 7z/7zz or extract ARDIS manually and re-run."
    )


def _ensure_ardis_extracted(rar_path: Path, extract_root: Path) -> Path:
    """
    Ensures ARDIS Dataset III is extracted and returns the digit-root folder:
    .../ARDIS_DATASET_3/{0..9}/*.png
    """
    target = extract_root / DEFAULT_ARDIS_FOLDER
    if target.exists():
        return target
    if not rar_path.exists():
        raise FileNotFoundError(f"ARDIS archive not found: {rar_path}")

    tools_dir = ROOT / "tools"
    seven = _ensure_7z(tools_dir)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"[digits] extracting ARDIS: {rar_path} -> {extract_root}")
    subprocess.run(  # noqa: S603 - trusted local executable
        [str(seven), "x", "-y", str(rar_path), f"-o{extract_root}"],
        check=True,
    )
    if not target.exists():
        # Some archives wrap an extra folder; search for it.
        matches = list(extract_root.rglob(DEFAULT_ARDIS_FOLDER))
        if matches:
            return matches[0]
        raise RuntimeError(f"ARDIS extracted but {DEFAULT_ARDIS_FOLDER} not found.")
    return target


def _emnist_fix(img: Image.Image) -> Image.Image:
    """
    EMNIST images are stored transposed relative to MNIST.
    Transpose is equivalent to rotate(-90) + mirror.
    """
    try:
        return img.transpose(Image.Transpose.TRANSPOSE)  # Pillow >= 9
    except Exception:
        return img.transpose(Image.TRANSPOSE)  # type: ignore[attr-defined]


def _border_mean(arr: np.ndarray, border: int = 4) -> float:
    if arr.ndim != 2:
        raise ValueError("expected 2D grayscale array")
    b = max(1, min(border, min(arr.shape) // 2))
    top = arr[:b, :]
    bottom = arr[-b:, :]
    left = arr[:, :b]
    right = arr[:, -b:]
    border_vals = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    return float(border_vals.mean())


def _standardize_image(img: Image.Image, *, invert_if_light_bg: bool = True) -> Image.Image:
    """
    Standardize to 28x28 grayscale, MNIST-like polarity (bright digit on dark bg).
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    if invert_if_light_bg and _border_mean(arr) > 127:
        gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray)
    out = ImageOps.pad(
        gray,
        (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
        method=Image.BILINEAR,
        color=0,
        centering=(0.5, 0.5),
    )
    return out


def _iter_ardis(ardis_root: Path) -> Iterable[Tuple[Image.Image, int, str]]:
    for digit_dir in sorted([p for p in ardis_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not digit_dir.name.isdigit():
            continue
        label = int(digit_dir.name)
        for p in sorted(digit_dir.glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            with Image.open(p) as img:
                yield img.copy(), label, "ardis"


def _iter_torchvision_digits(
    raw_root: Path, include_emnist: bool, include_mnist: bool, include_usps: bool
) -> Iterable[Tuple[Image.Image, int, str, str]]:
    _configure_ssl_certifi()

    if include_mnist:
        from torchvision.datasets import MNIST

        for is_train, split_name in [(True, "train"), (False, "test")]:
            ds = MNIST(root=str(raw_root), train=is_train, download=True)
            for img, label in ds:
                yield img, int(label), "mnist", split_name

    if include_emnist:
        from torchvision.datasets import EMNIST

        for is_train, split_name in [(True, "train"), (False, "test")]:
            ds = EMNIST(root=str(raw_root), split="digits", train=is_train, download=True)
            for img, label in ds:
                yield _emnist_fix(img), int(label), "emnist_digits", split_name

    if include_usps:
        from torchvision.datasets import USPS

        for is_train, split_name in [(True, "train"), (False, "test")]:
            ds = USPS(root=str(raw_root), train=is_train, download=True)
            for img, label in ds:
                yield img, int(label), "usps", split_name


def _stratified_split(
    rows: Sequence[SampleRow],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("train/val fractions must be >0 and train+val < 1.0")
    test_frac = 1.0 - train_frac - val_frac

    by_label: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r.label, []).append(i)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label, indices in by_label.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
        n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    assert len(set(train_idx) & set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(rows)
    print(
        f"[digits] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (test_frac={test_frac:.2f})"
    )
    return train_idx, val_idx, test_idx


def _write_split_csv(path: Path, rows: Sequence[SampleRow], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "source", "source_split"])
        for i in indices:
            r = rows[i]
            writer.writerow([r.filename, r.label, r.source, r.source_split])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare a unified 0–9 handwritten digits dataset (MNIST/EMNIST/USPS + ARDIS) as PNGs + CSV."
    )
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="torchvision download/cache root")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT, help="output dataset folder")
    ap.add_argument("--overwrite", action="store_true", help="delete existing --out folder first")

    ap.add_argument(
        "--no-mnist",
        action="store_false",
        dest="include_mnist",
        default=True,
        help="skip MNIST",
    )
    ap.add_argument(
        "--no-emnist",
        action="store_false",
        dest="include_emnist",
        default=True,
        help="skip EMNIST(digits)",
    )
    ap.add_argument(
        "--no-usps",
        action="store_false",
        dest="include_usps",
        default=True,
        help="skip USPS",
    )
    ap.add_argument(
        "--no-ardis",
        action="store_false",
        dest="include_ardis",
        default=True,
        help="skip ARDIS Dataset III",
    )

    ap.add_argument("--ardis-rar", type=Path, default=DEFAULT_ARDIS_RAR)
    ap.add_argument("--ardis-extract", type=Path, default=DEFAULT_ARDIS_EXTRACT)

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument(
        "--external-test",
        type=str,
        default="",
        help="comma-separated sources to use as test set (e.g. 'usps,ardis').",
    )
    ap.add_argument(
        "--limit-per-source",
        type=int,
        default=0,
        help="debug: limit number of samples per source (0 = no limit)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite and args.out.exists():
        shutil.rmtree(args.out)

    images_dir = args.out / "images"
    splits_dir = args.out / "splits"
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ARDIS
    ardis_root: Optional[Path] = None
    if args.include_ardis:
        ardis_root = _ensure_ardis_extracted(args.ardis_rar, args.ardis_extract)

    rows: List[SampleRow] = []
    labels_csv = args.out / "labels.csv"
    counter_by_source: Dict[str, int] = {}

    def next_name(source: str) -> str:
        counter_by_source[source] = counter_by_source.get(source, 0) + 1
        return f"{source}_{counter_by_source[source]:08d}.png"

    external_test_sources = {s.strip().lower() for s in args.external_test.split(",") if s.strip()}

    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "source", "source_split"])

        # Torchvision datasets
        iterable = _iter_torchvision_digits(
            args.raw_root, args.include_emnist, args.include_mnist, args.include_usps
        )
        per_source_seen: Dict[str, int] = {}
        for img, label, source, src_split in tqdm(iterable, desc="[digits] exporting torchvision", unit="img"):
            if args.limit_per_source:
                per_source_seen[source] = per_source_seen.get(source, 0) + 1
                if per_source_seen[source] > args.limit_per_source:
                    continue
            out_img = _standardize_image(img)
            filename = next_name(source)
            out_img.save(images_dir / filename, format="PNG", optimize=True)
            row = SampleRow(filename=filename, label=label, source=source, source_split=src_split)
            rows.append(row)
            writer.writerow([row.filename, row.label, row.source, row.source_split])

        # ARDIS
        if ardis_root is not None:
            per_source_seen = per_source_seen if "per_source_seen" in locals() else {}
            for img, label, source in tqdm(_iter_ardis(ardis_root), desc="[digits] exporting ARDIS", unit="img"):
                if args.limit_per_source:
                    per_source_seen[source] = per_source_seen.get(source, 0) + 1
                    if per_source_seen[source] > args.limit_per_source:
                        continue
                out_img = _standardize_image(img)
                filename = next_name(source)
                out_img.save(images_dir / filename, format="PNG", optimize=True)
                row = SampleRow(filename=filename, label=label, source=source, source_split="n/a")
                rows.append(row)
                writer.writerow([row.filename, row.label, row.source, row.source_split])

    if not rows:
        raise RuntimeError("No samples exported. Check dataset availability / flags.")

    # Splits
    if external_test_sources:
        test_idx = [i for i, r in enumerate(rows) if r.source.lower() in external_test_sources]
        remain = [i for i in range(len(rows)) if i not in set(test_idx)]
        remain_rows = [rows[i] for i in remain]
        train_idx_rel, val_idx_rel, _ = _stratified_split(
            remain_rows, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
        )
        train_idx = [remain[i] for i in train_idx_rel]
        val_idx = [remain[i] for i in val_idx_rel]
        print(f"[digits] external-test sources={sorted(external_test_sources)} => test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = _stratified_split(
            rows, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
        )

    _write_split_csv(splits_dir / "train.csv", rows, train_idx)
    _write_split_csv(splits_dir / "val.csv", rows, val_idx)
    _write_split_csv(splits_dir / "test.csv", rows, test_idx)

    # Summary
    counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r.source}:{r.label}"
        counts[key] = counts.get(key, 0) + 1
    summary_path = args.out / "summary_counts.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "label", "count"])
        for key in sorted(counts.keys()):
            source, label_str = key.split(":")
            w.writerow([source, int(label_str), counts[key]])

    print(f"[digits] wrote {len(rows)} images -> {images_dir}")
    print(f"[digits] labels -> {labels_csv}")
    print(f"[digits] splits -> {splits_dir}")
    print(f"[digits] summary -> {summary_path}")


if __name__ == "__main__":
    main()
