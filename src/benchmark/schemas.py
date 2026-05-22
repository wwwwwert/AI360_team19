from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


METHOD_NAMES = ("mass_subsequence",)
MethodName = Literal["mass_subsequence"]


@dataclass(frozen=True)
class BenchmarkResult:
    """Container for one benchmark run."""

    method: str
    resample: bool
    k: int
    predictions: pd.DataFrame
    neighbors: pd.DataFrame
    metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
