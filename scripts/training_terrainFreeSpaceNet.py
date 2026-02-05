import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# HARD CODED PATHS / PARAMS
# =========================
DATA_DIR  = "/home/your_user/data/csv_frames_folder"   # <-- folder with many CSVs
SAVE_PATH = "/home/your_user/models/output_model_name.pt" # <-- where to save best model checkpoint

# CSV column names
FRAME_COL = "frame_id"
X_COL, Y_COL, Z_COL = "x", "y", "z"
LABEL_COL = "free_space"

# Split
SPLIT_TRAIN = 0.80 # fraction of frames for training; rest for validation
SPLIT_SEED  = 42 # random seed for reproducible train/val splits

# Training
NUM_POINTS = 2048 # number of points to sample per frame; can experiment with this (e.g. 1024, 4096) for speed/memory tradeoffs
BATCH_SIZE = 16 # can experiment with this (e.g. 8, 32) for speed/memory tradeoffs
EPOCHS     = 50 # can experiment with this (e.g. 30, 100) for better convergence; watch out for overfitting
LR         = 1e-3 # learning rate; can experiment with this (e.g. 1e-4, 5e-4) for better convergence
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Speed/memory tradeoffs
PRELOAD_INDEX_ONLY = True   # True = only index (file,frame_id) pairs; load frame rows on demand
NUM_WORKERS = 4 # number of DataLoader workers for parallel data loading; can experiment with this (e.g. 0, 2, 8) for speed tradeoffs; set to 0 if encounter issues on Windows


# =========================
# HELPERS
# =========================
def sample_points(pts: np.ndarray, num_points: int) -> np.ndarray:
    n = pts.shape[0]
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    return pts[idx]

def normalize_points(pts: np.ndarray) -> np.ndarray:
    centroid = np.mean(pts, axis=0, keepdims=True)
    pts = pts - centroid
    furthest = np.max(np.sqrt(np.sum(pts**2, axis=1)))
    if furthest > 1e-9:
        pts = pts / furthest
    return pts

def list_csvs(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")
    return files


# =========================
# INDEX BUILDER
# =========================
def build_frame_index_from_folder(data_dir: str):

    csv_files = list_csvs(data_dir)

    items = []
    y_map = {}  # (csv_path, frame_id) -> y

    for path in csv_files:
        df = pd.read_csv(path, usecols=[FRAME_COL, LABEL_COL]) 
        df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
        df = df.dropna(subset=[FRAME_COL]) 
        df[FRAME_COL] = df[FRAME_COL].astype(int)

        for fid, g in df.groupby(FRAME_COL):
            lbl = g[LABEL_COL].dropna()
            if len(lbl) == 0:
                raise ValueError(f"No {LABEL_COL} found for frame_id={fid} in file={path}")
            y = float(lbl.iloc[0])
            key = (path, int(fid))
            items.append(key)
            y_map[key] = y

    if len(items) == 0:
        raise ValueError("No frames found across CSV files.")
    return items, y_map


# =========================
# DATASET
# =========================
class MultiCsvFramesDataset(Dataset):
    """
    Each dataset item corresponds to one frame from one CSV.
    """
    def __init__(self, items, y_map, num_points=2048, do_normalize=True):
        self.items = list(items) 
        self.y_map = y_map
        self.num_points = num_points
        self.do_normalize = do_normalize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        csv_path, fid = self.items[idx]
        y = self.y_map[(csv_path, fid)]

        df = pd.read_csv(csv_path, usecols=[FRAME_COL, X_COL, Y_COL, Z_COL])
        df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
        df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
        df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
        df[Z_COL] = pd.to_numeric(df[Z_COL], errors="coerce")
        df = df.dropna(subset=[FRAME_COL, X_COL, Y_COL, Z_COL])
        df[FRAME_COL] = df[FRAME_COL].astype(int)

        frame_df = df[df[FRAME_COL] == fid]
        pts = frame_df[[X_COL, Y_COL, Z_COL]].to_numpy(dtype=np.float32)

        pts = sample_points(pts, self.num_points)
        if self.do_normalize:
            pts = normalize_points(pts)

        pts_t = torch.from_numpy(pts).transpose(0, 1)         # (3, P)
        y_t   = torch.tensor([y], dtype=torch.float32)        # (1,)
        return pts_t, y_t


# =========================
# POINTNET REGRESSION MODEL
# =========================
class PointNetRegressor(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # regression head: 512, 512, 256
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = torch.max(x, dim=2)[0]  # (B, 1024)
        return self.head(x)


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for pts, y in loader:
        pts = pts.to(DEVICE)
        y   = y.to(DEVICE)
        pred = model(pts)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * pts.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total = 0.0
    for pts, y in loader:
        pts = pts.to(DEVICE)
        y   = y.to(DEVICE)
        pred = model(pts)
        loss = loss_fn(pred, y)
        total += loss.item() * pts.size(0)
    return total / len(loader.dataset)


def main():
    rng = np.random.default_rng(SPLIT_SEED)

    # 1) Build global frame index across ALL csvs
    items, y_map = build_frame_index_from_folder(DATA_DIR)

    # 2) Random split 80/20 (by items; each item is (csv_path, frame_id))
    items = np.array(items, dtype=object)
    rng.shuffle(items)

    n_total = len(items)
    n_train = int(round(SPLIT_TRAIN * n_total))
    train_items = items[:n_train].tolist()
    val_items   = items[n_train:].tolist()

    print(f"Total frames: {n_total} | Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"Device: {DEVICE}")

    train_ds = MultiCsvFramesDataset(train_items, y_map, num_points=NUM_POINTS, do_normalize=True)
    val_ds   = MultiCsvFramesDataset(val_items,   y_map, num_points=NUM_POINTS, do_normalize=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = PointNetRegressor(dropout=0.3).to(DEVICE)

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn)
        va = eval_one_epoch(model, val_loader, loss_fn)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{EPOCHS} | train={tr:.6f} | val={va:.6f}")

        if va < best_val:
            best_val = va
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "num_points": NUM_POINTS,
                "split_seed": SPLIT_SEED,
                "train_ratio": SPLIT_TRAIN,
                "data_dir": DATA_DIR
            }, SAVE_PATH)
            print(f"saved best -> {SAVE_PATH}")

    print("Done.")

if __name__ == "__main__":
    main()
