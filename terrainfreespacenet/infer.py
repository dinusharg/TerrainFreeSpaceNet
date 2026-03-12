from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from .model import PointNetRegressor
from .preprocess import prepare_points
from .utils import get_device


def load_points_from_csv(
    csv_path: str,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> torch.Tensor:
    df = pd.read_csv(csv_path, usecols=[x_col, y_col, z_col])
    for c in [x_col, y_col, z_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, z_col])
    points = df[[x_col, y_col, z_col]].to_numpy(dtype="float32")
    return points


def load_model(checkpoint_path: str, device: str | None = None):
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint.get("input_dim", 3))
    model = PointNetRegressor(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


@torch.no_grad()
def predict(
    csv_path: str,
    checkpoint_path: str,
    num_points: int | None = None,
    device: str | None = None,
) -> dict:
    model, checkpoint, device = load_model(checkpoint_path, device=device)

    if num_points is None:
        num_points = int(checkpoint.get("num_points", 2048))

    points = load_points_from_csv(csv_path)
    points = prepare_points(points, num_points=num_points)

    x = torch.from_numpy(points).transpose(0, 1).unsqueeze(0).to(device)  # [1, 3, N]
    pred = model(x).item()

    return {
        "free_space_score": float(pred),
        "num_points_used": int(num_points),
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference for TerrainFreeSpaceNet")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_points", type=int, default=None, help="Override checkpoint num_points")
    args = parser.parse_args()

    result = predict(
        csv_path=args.input_csv,
        checkpoint_path=args.checkpoint,
        num_points=args.num_points,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()