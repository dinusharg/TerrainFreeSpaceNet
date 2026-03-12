from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import discover_items, split_items, MultiCsvFramesDataset
from .model import PointNetRegressor
from .utils import set_seed, get_device, save_checkpoint, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train TerrainFreeSpaceNet")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--save_path", type=str, default="checkpoints/terrainfreespacenet_best.pt")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--rotate_z", action="store_true")
    parser.add_argument("--jitter_std", type=float, default=0.0)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)  # [B, 3, N]
        y = y.to(device)  # [B, 1]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, y)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    items, y_map = discover_items(args.data_dir)
    train_items, val_items = split_items(items, train_ratio=args.train_ratio, seed=args.seed)

    train_ds = MultiCsvFramesDataset(
        train_items,
        y_map=y_map,
        num_points=args.num_points,
        augment=True,
        rotate_z=args.rotate_z,
        jitter_std=args.jitter_std,
    )

    val_ds = MultiCsvFramesDataset(
        val_items,
        y_map=y_map,
        num_points=args.num_points,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PointNetRegressor(input_dim=3, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "model_name": "PointNetRegressor",
                "epoch": epoch,
                "val_loss": best_val,
                "num_points": args.num_points,
                "input_dim": 3,
                "frame_col": "frame_id",
                "xyz_cols": ["x", "y", "z"],
                "label_col": "free_space",
                "normalization": "center_and_unit_sphere",
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "data_dir": str(Path(args.data_dir).resolve()),
            }
            save_checkpoint(checkpoint, args.save_path)
            print(f"Saved best checkpoint to: {args.save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()