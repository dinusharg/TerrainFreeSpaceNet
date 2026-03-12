import numpy as np

from terrainfreespacenet.preprocess import sample_points, normalize_points, prepare_points


def test_sample_points_output_size():
    pts = np.random.rand(100, 3).astype(np.float32)
    out = sample_points(pts, 64)
    assert out.shape == (64, 3)


def test_normalize_points_shape():
    pts = np.random.rand(64, 3).astype(np.float32)
    out = normalize_points(pts)
    assert out.shape == (64, 3)


def test_prepare_points_empty_input():
    pts = np.zeros((0, 3), dtype=np.float32)
    out = prepare_points(pts, 32)
    assert out.shape == (32, 3)