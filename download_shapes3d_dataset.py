#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np

DEFAULT_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
TARGET_TO_COL: Dict[str, int] = {
    "floor_hue": 0,
    "wall_hue": 1,
    "object_hue": 2,
    "scale": 3,
    "shape": 4,
    "orientation": 5,
}


def _download(url: str, out_path: Path, *, force: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[INFO] raw file already exists: {out_path}", flush=True)
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[INFO] downloading {out_path}", flush=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, int(block_num * block_size * 100 / total_size))
        # Print only every ~5 percent to avoid huge logs.
        if pct == 100 or pct % 5 == 4:
            print(f"[DOWNLOAD] {pct:3d}%", flush=True)

    urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
    tmp_path.replace(out_path)
    print(f"[DONE] downloaded raw h5: {out_path}", flush=True)


def _short_n(n: int) -> str:
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Download/process DeepMind 3D Shapes into a small .npz for old-schema "
            "BBP_in_SGD regression experiments. The processor is chunked and stores "
            "X as uint8 by default, avoiding the memory spike that kills the old script."
        )
    )
    p.add_argument("--out-dir", type=str, default="dataset/shapes3d")
    p.add_argument("--url", type=str, default=DEFAULT_URL)
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--target", type=str, default="orientation", choices=sorted(TARGET_TO_COL))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--selection", type=str, default="random", choices=["random", "first"])
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-process", action="store_true")
    p.add_argument("--save-float32", action="store_true", help="Save X as float32 in [0,1]. Not recommended on low-RAM nodes.")
    args = p.parse_args()

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("This script requires h5py. Install it in the rhm environment.") from exc

    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_path = out_dir / "raw" / "3dshapes.h5"
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_npz = processed_dir / f"shapes3d_{_short_n(int(args.num_images))}_{args.target}.npz"
    metadata_path = processed_dir / f"shapes3d_{_short_n(int(args.num_images))}_{args.target}_metadata.json"

    _download(args.url, raw_path, force=bool(args.force_download))

    if out_npz.exists() and metadata_path.exists() and not args.force_process:
        print(f"[DONE] processed file already exists: {out_npz}", flush=True)
        print("       use --force-process to recreate it", flush=True)
        return

    num_images = int(args.num_images)
    if num_images <= 0:
        raise ValueError("--num-images must be positive")
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    target_col = TARGET_TO_COL[str(args.target)]

    print(f"[INFO] opening raw h5: {raw_path}", flush=True)
    with h5py.File(raw_path, "r") as f:
        if "images" not in f or "labels" not in f:
            raise KeyError("Expected h5 keys 'images' and 'labels'.")
        images_ds = f["images"]
        labels_ds = f["labels"]
        n_total = int(images_ds.shape[0])
        image_shape = tuple(int(v) for v in images_ds.shape[1:])
        flat_dim = int(np.prod(image_shape))

        if num_images > n_total:
            raise ValueError(f"Requested {num_images} images but dataset has only {n_total}.")

        print(f"[INFO] raw images shape={images_ds.shape} dtype={images_ds.dtype}", flush=True)
        print(f"[INFO] raw labels shape={labels_ds.shape} dtype={labels_ds.dtype}", flush=True)

        if args.selection == "first":
            selected = np.arange(num_images, dtype=np.int64)
        else:
            rng = np.random.default_rng(int(args.seed))
            selected = rng.choice(n_total, size=num_images, replace=False).astype(np.int64)
            selected.sort()

        X_dtype = np.float32 if args.save_float32 else np.uint8
        X = np.empty((num_images, flat_dim), dtype=X_dtype)
        y = np.empty((num_images, 1), dtype=np.float32)

        print(
            f"[INFO] processing {num_images} images in chunks of {chunk_size}; "
            f"saving X dtype={np.dtype(X_dtype)} flat_dim={flat_dim}",
            flush=True,
        )

        write_pos = 0
        for start in range(0, num_images, chunk_size):
            end = min(start + chunk_size, num_images)
            idx = selected[start:end]

            imgs = images_ds[idx]  # [B, 64, 64, 3], usually uint8
            labs = labels_ds[idx, target_col]

            imgs = np.asarray(imgs)
            imgs = imgs.reshape(imgs.shape[0], -1)
            if args.save_float32:
                imgs = imgs.astype(np.float32, copy=False) / 255.0

            X[write_pos:write_pos + imgs.shape[0]] = imgs
            y[write_pos:write_pos + imgs.shape[0], 0] = np.asarray(labs, dtype=np.float32)
            write_pos += imgs.shape[0]

            if end == num_images or end % max(chunk_size * 10, 1) == 0:
                print(f"[PROCESS] {end}/{num_images}", flush=True)

    # IMPORTANT: numpy.savez_compressed appends ".npz" automatically when
    # the target is a string/path not ending in .npz. Therefore the temporary
    # file itself must end in .npz, otherwise numpy would create
    # "...npz.tmp.npz" and tmp_npz.replace(out_npz) would fail.
    tmp_npz = out_npz.with_name(out_npz.name + ".tmp.npz")
    if tmp_npz.exists():
        tmp_npz.unlink()

    # Clean up the old buggy temporary name, in case it exists from a previous run.
    old_buggy_tmp = out_npz.with_suffix(out_npz.suffix + ".tmp.npz")
    if old_buggy_tmp.exists() and old_buggy_tmp != tmp_npz:
        old_buggy_tmp.unlink()

    print(f"[INFO] writing npz: {out_npz}", flush=True)
    np.savez_compressed(
        str(tmp_npz),
        X=X,
        y=y,
        target=np.array(str(args.target), dtype=object),
        target_col=np.array(target_col, dtype=np.int64),
        image_shape=np.array(image_shape, dtype=np.int64),
        flatten=np.array(True),
        x_storage=np.array("float32_0_1" if args.save_float32 else "uint8_0_255", dtype=object),
        selected_indices=selected,
    )
    tmp_npz.replace(out_npz)

    metadata = {
        "name": "shapes3d",
        "raw_h5": str(raw_path),
        "processed_npz": str(out_npz),
        "num_images": int(num_images),
        "target": str(args.target),
        "target_col": int(target_col),
        "target_columns": TARGET_TO_COL,
        "image_shape": list(image_shape),
        "input_dim": int(flat_dim),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "selection": str(args.selection),
        "seed": int(args.seed),
        "notes": "X is flattened. If X_dtype is uint8, set dataset.normalize_images=true in the training JSON.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[DONE] processed npz: {out_npz}", flush=True)
    print(f"[DONE] metadata:      {metadata_path}", flush=True)
    print("[JSON] use this path:", flush=True)
    print(f'  "processed_path": "{out_npz.relative_to(Path.cwd()).as_posix() if out_npz.is_relative_to(Path.cwd()) else str(out_npz)}"', flush=True)


if __name__ == "__main__":
    main()
