from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import discover_items, split_items, MultiCsvFramesDataset
from .infer import load_model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
    }


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy().reshape(-1)
        gt = y.numpy().reshape(-1)

        y_pred_all.append(pred)
        y_true_all.append(gt)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return regression_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TerrainFreeSpaceNet")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.checkpoint)

    items, y_map = discover_items(args.data_dir)
    _, val_items = split_items(items, train_ratio=args.train_ratio, seed=args.seed)

    val_ds = MultiCsvFramesDataset(
        val_items,
        y_map=y_map,
        num_points=int(checkpoint.get("num_points", 2048)),
        augment=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate(model, val_loader, device=device)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()