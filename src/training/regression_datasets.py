from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass
class RegressionData:
    train_dataset: TensorDataset
    test_dataset: TensorDataset
    info: Dict[str, Any]


def _as_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(x)


def _cfg_get(cfg: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if isinstance(cfg, dict) and k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def _safe_json_loads(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode('utf-8')
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def _first_existing(*paths) -> Optional[Path]:
    for p in paths:
        if p is None:
            continue
        pp = Path(str(p)).expanduser()
        if pp.exists():
            return pp
    return None

def _load_npz_common(path: Path, test_fraction=0.1, seed=0):
    """
    Load both possible processed .npz formats.

    Format A: already split
        x_train / X_train / train_x / train_X / x_train_u8
        x_test  / X_test  / test_x  / test_X  / x_test_u8
        y_train / Y_train / train_y / train_Y
        y_test  / Y_test  / test_y  / test_Y

    Format B: unsplit supervised dataset
        X, y

    This is needed because your superconductivity file has:
        X, y, feature_names, target_name, target_units, source
    """
    z = np.load(path, allow_pickle=True)

    def pick(*names):
        for name in names:
            if name in z:
                return z[name]
        raise KeyError(
            f"None of {names} found in {path}; available keys={list(z.keys())}"
        )

    def has_any(*names):
        return any(name in z for name in names)

    def clean_meta_value(v):
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                v = v.tolist()
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        return v

    meta = _safe_json_loads(z["metadata"]) if "metadata" in z else {}

    # Keep useful metadata from files like superconductivity.npz
    for key in ["feature_names", "target_name", "target_units", "source"]:
        if key in z:
            meta[key] = clean_meta_value(z[key])

    # Case A: already split file
    if has_any("x_train", "X_train", "train_x", "train_X", "x_train_u8"):
        x_train = pick("x_train", "X_train", "train_x", "train_X", "x_train_u8")
        x_test = pick("x_test", "X_test", "test_x", "test_X", "x_test_u8")
        y_train = pick("y_train", "Y_train", "train_y", "train_Y")
        y_test = pick("y_test", "Y_test", "test_y", "test_Y")
        meta["npz_format"] = "split"
        return x_train, x_test, y_train, y_test, meta

    # Case B: unsplit file with X, y
    if "X" in z and "y" in z:
        x = z["X"]
        y = z["y"]
        x_train, x_test, y_train, y_test = _split_arrays(
            x,
            y,
            test_fraction=test_fraction,
            seed=seed,
        )
        meta["npz_format"] = "unsplit_X_y"
        meta["test_fraction"] = float(test_fraction)
        meta["split_seed"] = int(seed)
        return x_train, x_test, y_train, y_test, meta

    # Case C: unsplit file with lowercase x, y
    if "x" in z and "y" in z:
        x = z["x"]
        y = z["y"]
        x_train, x_test, y_train, y_test = _split_arrays(
            x,
            y,
            test_fraction=test_fraction,
            seed=seed,
        )
        meta["npz_format"] = "unsplit_x_y"
        meta["test_fraction"] = float(test_fraction)
        meta["split_seed"] = int(seed)
        return x_train, x_test, y_train, y_test, meta

    raise KeyError(
        f"Unsupported npz format in {path}; available keys={list(z.keys())}. "
        "Expected either split keys x_train/x_test/y_train/y_test "
        "or unsplit keys X/y."
    )


def _split_arrays(x, y, test_fraction=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_test = int(round(n * float(test_fraction)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def _prep_x_image(x_train, x_test, flatten=True, normalize_images=True):
    x_train = np.asarray(x_train).astype(np.float32, copy=False)
    x_test = np.asarray(x_test).astype(np.float32, copy=False)
    if normalize_images and max(float(np.nanmax(x_train)), float(np.nanmax(x_test))) > 1.5:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, x_test


def _prep_y(y_train, y_test):
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]
    return y_train, y_test


def _standardize_train_test(x_train, x_test, eps=1e-8):
    mu = x_train.mean(axis=0, keepdims=True)
    sig = x_train.std(axis=0, keepdims=True)
    sig = np.where(sig < eps, 1.0, sig)
    return (x_train - mu) / sig, (x_test - mu) / sig, mu, sig


def _maybe_standardize_targets(y_train, y_test, enabled):
    if not enabled:
        return y_train, y_test, None, None
    mu = y_train.mean(axis=0, keepdims=True)
    sig = y_train.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return (y_train - mu) / sig, (y_test - mu) / sig, mu, sig


def _to_dataset(x, y):
    return TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))


def load_shapes3d(dataset_cfg: Dict[str, Any], seed: int = 0) -> RegressionData:
    path = _first_existing(
        _cfg_get(dataset_cfg, 'processed_path', 'path', 'dataset_path', 'npz_path'),
        'dataset/shapes3d/processed/shapes3d_50k_orientation.npz',
    )
    if path is None:
        raise FileNotFoundError(
            'Shapes3D processed file not found. Create it with:\n'
            '  python download_shapes3d_dataset.py --num-images 50000 --target orientation'
        )
    xtr, xte, ytr, yte, meta = _load_npz_common(path)
    xtr, xte = _prep_x_image(
        xtr, xte,
        flatten=_as_bool(dataset_cfg.get('flatten'), True),
        normalize_images=_as_bool(dataset_cfg.get('normalize_images'), True),
    )
    ytr, yte = _prep_y(ytr, yte)
    ytr, yte, ymu, ysig = _maybe_standardize_targets(
        ytr, yte, _as_bool(dataset_cfg.get('standardize_targets'), False)
    )
    info = {
        'dataset_name': 'shapes3d',
        'target': dataset_cfg.get('target', meta.get('target', 'orientation')),
        'processed_path': str(path),
        'input_dim': int(xtr.shape[1]),
        'output_dim': int(ytr.shape[1]),
        'num_train': int(xtr.shape[0]),
        'num_test': int(xte.shape[0]),
        'image_shape': meta.get('image_shape', [64, 64, 3]),
        'target_standardized': _as_bool(dataset_cfg.get('standardize_targets'), False),
        'target_mean': None if ymu is None else ymu.astype(float).tolist(),
        'target_std': None if ysig is None else ysig.astype(float).tolist(),
        'metadata': meta,
    }
    return RegressionData(_to_dataset(xtr, ytr), _to_dataset(xte, yte), info)


def _load_images_from_folder_for_age(root: Path, image_size=64, max_images=None, seed=0):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError('Pillow is required to load UTKFace images from folders') from exc

    items = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            age = float(p.name.split('_')[0])
        except Exception:
            continue
        items.append((p, age))
    if not items:
        raise RuntimeError(f'No UTKFace-style images found in {root}')
    rng = random.Random(seed)
    rng.shuffle(items)
    if max_images is not None:
        items = items[:int(max_images)]
    xs, ys = [], []
    for i, (p, age) in enumerate(items):
        im = Image.open(p).convert('RGB').resize((int(image_size), int(image_size)))
        xs.append(np.asarray(im, dtype=np.uint8))
        ys.append(age)
        if (i + 1) % 5000 == 0:
            print(f'[DATA] loaded {i+1} face images', flush=True)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32)[:, None]


def load_utkface_age(dataset_cfg: Dict[str, Any], seed: int = 0) -> RegressionData:
    path = _first_existing(_cfg_get(dataset_cfg, 'processed_path', 'path', 'dataset_path', 'npz_path'))
    if path is not None and path.suffix.lower() == '.npz':
        xtr, xte, ytr, yte, meta = _load_npz_common(
            path,
            test_fraction=dataset_cfg.get('test_fraction', 0.1),
            seed=seed,
        )
    else:
        root = _first_existing(
            _cfg_get(dataset_cfg, 'root_dir', 'image_dir', 'images_dir'),
            'dataset/utkface', 'dataset/UTKFace', 'datasets/utkface', 'datasets/UTKFace',
        )
        if root is None:
            raise FileNotFoundError('Could not find UTKFace. Set dataset.root_dir or dataset.processed_path.')
        x, y = _load_images_from_folder_for_age(
            root,
            image_size=int(dataset_cfg.get('image_size', 64)),
            max_images=dataset_cfg.get('max_images', None),
            seed=seed,
        )
        xtr, xte, ytr, yte = _split_arrays(x, y, dataset_cfg.get('test_fraction', 0.1), seed)
        meta = {'root_dir': str(root)}
    xtr, xte = _prep_x_image(
        xtr, xte,
        flatten=_as_bool(dataset_cfg.get('flatten'), True),
        normalize_images=_as_bool(dataset_cfg.get('normalize_images'), True),
    )
    ytr, yte = _prep_y(ytr, yte)
    ytr, yte, ymu, ysig = _maybe_standardize_targets(
        ytr, yte, _as_bool(dataset_cfg.get('standardize_targets'), False)
    )
    info = {
        'dataset_name': 'utkface_age', 'target': 'age',
        'processed_path': None if path is None else str(path),
        'input_dim': int(xtr.shape[1]), 'output_dim': int(ytr.shape[1]),
        'num_train': int(xtr.shape[0]), 'num_test': int(xte.shape[0]),
        'target_standardized': _as_bool(dataset_cfg.get('standardize_targets'), False),
        'target_mean': None if ymu is None else ymu.astype(float).tolist(),
        'target_std': None if ysig is None else ysig.astype(float).tolist(),
        'metadata': meta,
    }
    return RegressionData(_to_dataset(xtr, ytr), _to_dataset(xte, yte), info)


def load_superconductivity(dataset_cfg: Dict[str, Any], seed: int = 0) -> RegressionData:
    npz_path = _first_existing(_cfg_get(dataset_cfg, 'processed_path', 'path', 'dataset_path', 'npz_path'))
    csv_path = _first_existing(
        _cfg_get(dataset_cfg, 'csv_path'),
        'dataset/superconductivity/train.csv',
        'dataset/superconductivity/superconductivity.csv',
        'datasets/superconductivity/train.csv',
    )
    if npz_path is not None and npz_path.suffix.lower() == '.npz':
        xtr, xte, ytr, yte, meta = _load_npz_common(
            npz_path,
            test_fraction=dataset_cfg.get('test_fraction', 0.1),
            seed=seed,
        )
    elif csv_path is not None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError('pandas is required to load superconductivity CSV') from exc
        df = pd.read_csv(csv_path)
        target_col = dataset_cfg.get('target_column') or dataset_cfg.get('target')
        if target_col is None or target_col == 'critical_temperature':
            for cand in ['critical_temp', 'critical_temperature', 'target', 'y']:
                if cand in df.columns:
                    target_col = cand
                    break
        if target_col not in df.columns:
            raise KeyError(f'Could not find target column {target_col}; columns={list(df.columns)}')
        y = df[target_col].to_numpy(dtype=np.float32)[:, None]
        x = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
        xtr, xte, ytr, yte = _split_arrays(x, y, dataset_cfg.get('test_fraction', 0.1), seed)
        meta = {'csv_path': str(csv_path), 'target_column': target_col}
    else:
        raise FileNotFoundError('Could not find superconductivity data. Set dataset.csv_path or dataset.processed_path.')

    xtr = np.asarray(xtr, dtype=np.float32).reshape(len(xtr), -1)
    xte = np.asarray(xte, dtype=np.float32).reshape(len(xte), -1)
    ytr, yte = _prep_y(ytr, yte)
    if _as_bool(dataset_cfg.get('standardize_features'), True):
        xtr, xte, _, _ = _standardize_train_test(xtr, xte)
    ytr, yte, ymu, ysig = _maybe_standardize_targets(
        ytr, yte, _as_bool(dataset_cfg.get('standardize_targets'), True)
    )
    info = {
        'dataset_name': 'superconductivity',
        'target': meta.get('target_column', 'critical_temp'),
        'processed_path': None if npz_path is None else str(npz_path),
        'csv_path': None if csv_path is None else str(csv_path),
        'input_dim': int(xtr.shape[1]), 'output_dim': int(ytr.shape[1]),
        'num_train': int(xtr.shape[0]), 'num_test': int(xte.shape[0]),
        'features_standardized': _as_bool(dataset_cfg.get('standardize_features'), True),
        'target_standardized': _as_bool(dataset_cfg.get('standardize_targets'), True),
        'target_mean': None if ymu is None else ymu.astype(float).tolist(),
        'target_std': None if ysig is None else ysig.astype(float).tolist(),
        'metadata': meta,
    }
    return RegressionData(_to_dataset(xtr, ytr), _to_dataset(xte, yte), info)


def load_regression_dataset(dataset_cfg: Dict[str, Any], seed: int = 0) -> RegressionData:
    if dataset_cfg is None:
        dataset_cfg = {}
    name = str(_cfg_get(dataset_cfg, 'name', 'dataset', 'dataset_name', default='utkface_age')).lower()
    if name in {'utkface', 'utkface_age', 'faces', 'face_age', 'age', 'age_regression'}:
        return load_utkface_age(dataset_cfg, seed)
    if name in {'superconductivity', 'superconductor', 'superconductors', 'conductivity'}:
        return load_superconductivity(dataset_cfg, seed)
    if name in {'shapes3d', '3dshapes', '3d_shapes', 'shapes3d_orientation'}:
        return load_shapes3d(dataset_cfg, seed)
    raise ValueError(f"Unknown dataset '{name}'. Supported: utkface_age, superconductivity, shapes3d")


def make_regression_dataloaders(dataset_cfg: Dict[str, Any], batch_size: int, *, test_batch_size=None,
                                num_workers=0, pin_memory=True, seed=0, shuffle_train=True):
    data = load_regression_dataset(dataset_cfg, seed)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if test_batch_size is None:
        test_batch_size = batch_size
    train_loader = DataLoader(
        data.train_dataset, batch_size=int(batch_size), shuffle=shuffle_train,
        num_workers=int(num_workers), pin_memory=pin_memory, generator=gen,
    )
    test_loader = DataLoader(
        data.test_dataset, batch_size=int(test_batch_size), shuffle=False,
        num_workers=int(num_workers), pin_memory=pin_memory,
    )
    return train_loader, test_loader, data.info
