#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List

import numpy as np


DEFAULT_URLS = [
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "https://archive.ics.uci.edu/static/public/464/superconductivty%2Bdata.zip",
]


def _download(urls: List[str], out_zip: Path) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"[INFO] downloading: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                out_zip.write_bytes(response.read())
            print(f"[INFO] downloaded to: {out_zip}")
            return url
        except Exception as exc:  # pragma: no cover - only triggered by network issues
            last_error = exc
            print(f"[WARN] failed URL: {url}\n       {exc}")
    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def _find_file(root: Path, name: str) -> Path:
    matches = sorted([p for p in root.rglob(name) if p.is_file()])
    if not matches:
        raise FileNotFoundError(f"Could not find {name} after extracting under {root}")
    return matches[0]


def _read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _load_train_csv(train_csv: Path) -> tuple[np.ndarray, np.ndarray, List[str], str]:
    header = _read_header(train_csv)
    data = np.loadtxt(train_csv, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected train.csv shape: {data.shape}")

    feature_names = header[:-1]
    target_name = header[-1]
    X = data[:, :-1].astype(np.float32, copy=False)
    y = data[:, -1].astype(np.float32, copy=False).reshape(-1, 1)

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Header/data mismatch: X has {X.shape[1]} columns but header has {len(feature_names)} features."
        )
    return X, y, feature_names, target_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the UCI Superconductivity dataset and convert it into a compact .npz "
            "that is fast to load during experiment-2 training."
        )
    )
    parser.add_argument("--out_dir", type=str, default="dataset/superconductivity")
    parser.add_argument("--url", type=str, default=None, help="Optional custom URL for the UCI zip file.")
    parser.add_argument("--force", action="store_true", help="Re-download/recreate files even if they already exist.")
    parser.add_argument("--keep_zip", action="store_true", help="Keep the downloaded zip file under raw/.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_npz = processed_dir / "superconductivity.npz"
    metadata_json = processed_dir / "metadata.json"

    if out_npz.exists() and metadata_json.exists() and not args.force:
        print(f"[DONE] processed file already exists: {out_npz}")
        print("       use --force to recreate it")
        return

    zip_path = raw_dir / "superconductivity_uci.zip"
    urls = [args.url] if args.url else DEFAULT_URLS

    with tempfile.TemporaryDirectory(prefix="superconductivity_") as tmp:
        tmp_dir = Path(tmp)
        tmp_zip = tmp_dir / "dataset.zip"

        if zip_path.exists() and not args.force:
            print(f"[INFO] using existing zip: {zip_path}")
            tmp_zip.write_bytes(zip_path.read_bytes())
            source_url = "existing local zip"
        else:
            source_url = _download(urls, tmp_zip)
            if args.keep_zip:
                shutil.copy2(tmp_zip, zip_path)

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(extract_dir)

        train_csv = _find_file(extract_dir, "train.csv")
        unique_m_csv = _find_file(extract_dir, "unique_m.csv")

        raw_train = raw_dir / "train.csv"
        raw_unique = raw_dir / "unique_m.csv"
        shutil.copy2(train_csv, raw_train)
        shutil.copy2(unique_m_csv, raw_unique)

    X, y, feature_names, target_name = _load_train_csv(raw_dir / "train.csv")

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        target_name=np.array(target_name),
        target_units=np.array("K"),
        source=np.array("UCI Machine Learning Repository, dataset id 464"),
    )

    metadata = {
        "name": "superconductivity",
        "source_url": source_url,
        "uci_dataset_id": 464,
        "raw_train_csv": str((raw_dir / "train.csv").resolve()),
        "raw_unique_m_csv": str((raw_dir / "unique_m.csv").resolve()),
        "processed_npz": str(out_npz.resolve()),
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "target_name": str(target_name),
        "target_units": "K",
        "feature_names": feature_names,
        "notes": (
            "X contains the 81 numerical composition-derived features from train.csv; "
            "y contains critical_temp as a column vector. unique_m.csv is kept only as raw metadata."
        ),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[DONE] raw train.csv:     {raw_dir / 'train.csv'}")
    print(f"[DONE] raw unique_m.csv:  {raw_dir / 'unique_m.csv'}")
    print(f"[DONE] processed npz:     {out_npz}")
    print(f"[DONE] metadata:          {metadata_json}")
    print(f"[INFO] X shape={X.shape} y shape={y.shape} target={target_name} units=K")


if __name__ == "__main__":
    main()
