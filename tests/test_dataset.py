from pathlib import Path

import pandas as pd

from terrainfreespacenet.dataset import discover_items, MultiCsvFramesDataset


def test_discover_items(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "frame_id": [1, 1, 2, 2],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
            "z": [0.1, 0.2, 0.3, 0.4],
            "free_space": [0.9, 0.9, 0.3, 0.3],
        }
    )
    df.to_csv(csv_path, index=False)

    items, y_map = discover_items(tmp_path)
    assert len(items) == 2
    assert len(y_map) == 2


def test_dataset_getitem(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "frame_id": [1, 1, 1],
            "x": [0.1, 0.2, 0.3],
            "y": [0.1, 0.2, 0.3],
            "z": [0.1, 0.2, 0.3],
            "free_space": [0.8, 0.8, 0.8],
        }
    )
    df.to_csv(csv_path, index=False)

    items, y_map = discover_items(tmp_path)
    ds = MultiCsvFramesDataset(items, y_map, num_points=16)
    x, y = ds[0]

    assert x.shape == (3, 16)
    assert y.shape == (1,)