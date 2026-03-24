#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="dataset/UTKFace")
    parser.add_argument("--cache_dir", type=str, default="dataset/hf_cache")
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] cache dir:  {cache_dir}")

    ds = load_dataset(
        "nu-delta/utkface",
        split="train",
        cache_dir=str(cache_dir),
    )

    print(f"[INFO] dataset loaded with {len(ds)} rows")

    n_saved = 0
    n_skipped = 0

    for i, ex in enumerate(ds):
        if args.max_items is not None and i >= args.max_items:
            break

        img = ex["image"]
        file_name = ex["file_name"]

        if not file_name:
            # fallback that still matches your parser
            age = int(ex["age"])
            gender = 0 if str(ex["gender"]).lower().startswith("male") else 1
            ethnicity_map = {
                "white": 0,
                "black": 1,
                "asian": 2,
                "indian": 3,
                "others": 4,
                "other": 4,
            }
            ethnicity = ethnicity_map.get(str(ex["ethnicity"]).lower(), 4)
            date = str(ex.get("date", i))
            file_name = f"{age}_{gender}_{ethnicity}_{date}.jpg"

        out_path = out_dir / file_name

        if out_path.exists():
            n_skipped += 1
            continue

        img.save(out_path)
        n_saved += 1

        if (n_saved + n_skipped) % 1000 == 0:
            print(f"[INFO] processed {n_saved + n_skipped} items | saved={n_saved} skipped={n_skipped}")

    print(f"[DONE] saved={n_saved} skipped={n_skipped}")
    print(f"[DONE] files are in: {out_dir}")


if __name__ == "__main__":
    main()