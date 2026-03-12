from __future__ import annotations

import numpy as np


def sample_points(points: np.ndarray, num_points: int, seed: int | None = None) -> np.ndarray:
    """
    Randomly sample `num_points` from input points.
    If input has fewer points, sampling is done with replacement.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    replace = n < num_points
    indices = rng.choice(n, size=num_points, replace=replace)
    return points[indices].astype(np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalize points by centering at origin and scaling to unit sphere.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid

    furthest_distance = np.max(np.linalg.norm(points, axis=1))
    if furthest_distance > 1e-12:
        points = points / furthest_distance

    return points.astype(np.float32)


def prepare_points(points: np.ndarray, num_points: int, seed: int | None = None) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. sample/fill to fixed point count
      2. normalize
    """
    points = sample_points(points, num_points=num_points, seed=seed)
    points = normalize_points(points)
    return points


def maybe_augment_points(
    points: np.ndarray,
    rotate_z: bool = False,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Optional augmentation for training.
    """
    rng = np.random.default_rng(seed)
    augmented = points.copy()

    if rotate_z:
        theta = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(
            [[c, -s, 0.0],
             [s,  c, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        augmented = augmented @ rot.T

    if jitter_std > 0.0:
        noise = rng.normal(0.0, jitter_std, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise

    return augmented.astype(np.float32)