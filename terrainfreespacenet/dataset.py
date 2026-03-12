from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import prepare_points, maybe_augment_points


DatasetItem = Tuple[str, int]  # (csv_path, frame_id)


def discover_items(
    data_dir: str | Path,
    frame_col: str = "frame_id",
    label_col: str = "free_space",
    xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
) -> tuple[List[DatasetItem], Dict[DatasetItem, float]]:
    """
    Discover frame-level items from CSV files and extract one label per frame.
    Each CSV may contain many frames. Each frame becomes one training sample.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    items: List[DatasetItem] = []
    labels: Dict[DatasetItem, float] = {}

    required_cols = [frame_col, *xyz_cols, label_col]

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        for c in xyz_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=required_cols)
        df[frame_col] = df[frame_col].astype(int)

        grouped = df.groupby(frame_col, sort=True)

        for frame_id, g in grouped:
            item = (str(csv_path), int(frame_id))
            label = float(g[label_col].iloc[0])
            items.append(item)
            labels[item] = label

    return items, labels


def split_items(
    items: List[DatasetItem],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[DatasetItem], List[DatasetItem]]:
    """
    Split at item level.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]
    return train_items, val_items


class MultiCsvFramesDataset(Dataset):
    def __init__(
        self,
        items: List[DatasetItem],
        y_map: Dict[DatasetItem, float],
        num_points: int = 2048,
        frame_col: str = "frame_id",
        xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
        augment: bool = False,
        rotate_z: bool = False,
        jitter_std: float = 0.0,
        cache_files: bool = True,
    ):
        self.items = items
        self.y_map = y_map
        self.num_points = num_points
        self.frame_col = frame_col
        self.xyz_cols = list(xyz_cols)
        self.augment = augment
        self.rotate_z = rotate_z
        self.jitter_std = jitter_std
        self.cache_files = cache_files
        self._cache: dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load_csv(self, path: str) -> pd.DataFrame:
        if self.cache_files and path in self._cache:
            return self._cache[path]

        df = pd.read_csv(path, usecols=[self.frame_col] + self.xyz_cols)
        df[self.frame_col] = pd.to_numeric(df[self.frame_col], errors="coerce")
        for c in self.xyz_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[self.frame_col] + self.xyz_cols)
        df[self.frame_col] = df[self.frame_col].astype(int)

        if self.cache_files:
            self._cache[path] = df

        return df

    def __getitem__(self, idx: int):
        csv_path, frame_id = self.items[idx]
        df = self._load_csv(csv_path)
        frame_df = df[df[self.frame_col] == frame_id]

        points = frame_df[self.xyz_cols].to_numpy(dtype=np.float32)
        points = prepare_points(points, num_points=self.num_points)

        if self.augment:
            points = maybe_augment_points(
                points,
                rotate_z=self.rotate_z,
                jitter_std=self.jitter_std,
            )

        x = torch.from_numpy(points).transpose(0, 1)  # [3, N]
        y = torch.tensor([self.y_map[(csv_path, frame_id)]], dtype=torch.float32)
        return x, y