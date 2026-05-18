from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


MethodName = Literal["mass_subsequence", "mass_resample", "euclidean_resample"]


@dataclass(frozen=True)
class BenchmarkResult:
    """Container for one benchmark run."""

    method: str
    k: int
    predictions: pd.DataFrame
    neighbors: pd.DataFrame
    metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
