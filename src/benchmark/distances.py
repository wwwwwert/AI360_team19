from __future__ import annotations

from typing import Any, Literal

import numpy as np
import stumpy


DistanceScale = Literal["sqrt_m", "none"]


def mass_subsequence_distance(
    a: np.ndarray,
    b: np.ndarray,
    normalize: bool = True,
    distance_scale: DistanceScale = "sqrt_m",
) -> dict[str, Any]:
    """Minimum MASS distance between the shorter series and any window of the longer."""
    a = as_float_array(a)
    b = as_float_array(b)

    if len(a) <= len(b):
        query = a
        target = b
        query_source = "test"
        target_source = "train"
    else:
        query = b
        target = a
        query_source = "train"
        target_source = "test"

    profile = stumpy.mass(query, target, normalize=normalize)
    profile = np.asarray(profile, dtype=float)
    best_idx = int(np.nanargmin(profile))
    raw_distance = float(profile[best_idx])
    distance = scale_distance(raw_distance, len(query), distance_scale)
    return {
        "distance": distance,
        "raw_distance": raw_distance,
        "query_length": int(len(query)),
        "target_length": int(len(target)),
        "match_start_idx": best_idx,
        "match_end_idx": best_idx + len(query) - 1,
        "query_source": query_source,
        "target_source": target_source,
    }


def equal_length_mass_distance(
    a: np.ndarray,
    b: np.ndarray,
    normalize: bool = True,
    distance_scale: DistanceScale = "sqrt_m",
) -> dict[str, Any]:
    a = as_float_array(a)
    b = as_float_array(b)
    if len(a) != len(b):
        raise ValueError("equal_length_mass_distance expects arrays of the same length")

    profile = stumpy.mass(a, b, normalize=normalize)
    raw_distance = float(np.asarray(profile, dtype=float)[0])
    return {
        "distance": scale_distance(raw_distance, len(a), distance_scale),
        "raw_distance": raw_distance,
        "query_length": int(len(a)),
        "target_length": int(len(b)),
        "match_start_idx": 0,
        "match_end_idx": len(a) - 1,
        "query_source": "test",
        "target_source": "train",
    }


def euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    normalize: bool = True,
    distance_scale: DistanceScale = "sqrt_m",
) -> dict[str, Any]:
    a = as_float_array(a)
    b = as_float_array(b)
    if len(a) != len(b):
        raise ValueError("euclidean_distance expects arrays of the same length")
    if normalize:
        a = z_normalize(a)
        b = z_normalize(b)
    raw_distance = float(np.linalg.norm(a - b))
    return {
        "distance": scale_distance(raw_distance, len(a), distance_scale),
        "raw_distance": raw_distance,
        "query_length": int(len(a)),
        "target_length": int(len(b)),
        "match_start_idx": 0,
        "match_end_idx": len(a) - 1,
        "query_source": "test",
        "target_source": "train",
    }


def as_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Time series is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Time series contains NaN or infinite values")
    return arr


def resample_values(values: np.ndarray, output_length: int) -> np.ndarray:
    values = as_float_array(values)
    if output_length <= 1:
        raise ValueError("output_length must be greater than 1")
    if len(values) == output_length:
        return values.copy()
    old_x = np.linspace(0.0, 1.0, num=len(values))
    new_x = np.linspace(0.0, 1.0, num=output_length)
    return np.interp(new_x, old_x, values)


def z_normalize(values: np.ndarray) -> np.ndarray:
    values = as_float_array(values)
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def scale_distance(
    distance: float,
    query_length: int,
    distance_scale: DistanceScale,
) -> float:
    if distance_scale == "none":
        return float(distance)
    if distance_scale == "sqrt_m":
        return float(distance) / float(np.sqrt(query_length))
    raise ValueError("distance_scale must be 'sqrt_m' or 'none'")
