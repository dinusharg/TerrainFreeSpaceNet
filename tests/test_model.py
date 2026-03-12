import torch

from terrainfreespacenet.model import PointNetRegressor


def test_model_output_shape():
    model = PointNetRegressor(input_dim=3)
    x = torch.randn(4, 3, 1024)
    y = model(x)
    assert y.shape == (4, 1)


def test_model_output_range():
    model = PointNetRegressor(input_dim=3)
    x = torch.randn(2, 3, 512)
    y = model(x)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)