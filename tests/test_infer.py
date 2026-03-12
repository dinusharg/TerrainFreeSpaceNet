import torch
import pandas as pd

from terrainfreespacenet.model import PointNetRegressor
from terrainfreespacenet.infer import predict


def test_predict(tmp_path):
    csv_path = tmp_path / "sample.csv"
    ckpt_path = tmp_path / "model.pt"

    df = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df.to_csv(csv_path, index=False)

    model = PointNetRegressor()
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": 3,
            "num_points": 32,
        },
        ckpt_path,
    )

    result = predict(str(csv_path), str(ckpt_path))
    assert "free_space_score" in result
    assert 0.0 <= result["free_space_score"] <= 1.0