from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


CLASS_NAMES = {
    "1": "hh",
    "2": "hu",
    "3": "ud",
    "4": "hud",
    "5": "hh2",
    "6": "hu2",
}

Split = Literal["train", "test"]


@dataclass(frozen=True)
class GestureRecord:
    """One labeled gesture recording from GesturePebbleZ1."""

    series_id: str
    split: Split
    label: str
    label_name: str
    values: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.values))

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=pd.RangeIndex(self.length), name=self.series_id)


class GesturePebbleZ1Dataset:
    """Parser and convenience wrapper for the GesturePebbleZ1 UCR dataset."""

    def __init__(self, root: str | Path = "data/GesturePebbleZ1") -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self._records: dict[Split, list[GestureRecord]] = {}

    def load(self, file_format: Literal["ts", "txt"] = "ts") -> "GesturePebbleZ1Dataset":
        self._records = {
            "train": self._load_split("train", file_format=file_format),
            "test": self._load_split("test", file_format=file_format),
        }
        return self

    @property
    def train(self) -> list[GestureRecord]:
        self._require_loaded()
        return self._records["train"]

    @property
    def test(self) -> list[GestureRecord]:
        self._require_loaded()
        return self._records["test"]

    @property
    def records(self) -> list[GestureRecord]:
        self._require_loaded()
        return self.train + self.test

    def summary(self) -> pd.DataFrame:
        self._require_loaded()
        rows = []
        for split, records in self._records.items():
            lengths = np.array([record.length for record in records], dtype=int)
            labels = pd.Series([record.label for record in records], dtype=str)
            row: dict[str, Any] = {
                "split": split,
                "n_series": len(records),
                "min_length": int(lengths.min()),
                "median_length": float(np.median(lengths)),
                "max_length": int(lengths.max()),
                "n_classes": int(labels.nunique()),
            }
            for label, count in labels.value_counts().sort_index().items():
                row[f"class_{label}_{CLASS_NAMES.get(label, label)}"] = int(count)
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)

    def to_long_dataframe(self, split: Split | None = None) -> pd.DataFrame:
        """Return tidy data: one row per time point, no padded values."""
        self._require_loaded()
        records = self._select_records(split)
        rows = []
        for record in records:
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": record.series_id,
                        "split": record.split,
                        "label": record.label,
                        "label_name": record.label_name,
                        "sample_idx": np.arange(record.length, dtype=int),
                        "value": record.values,
                        "length": record.length,
                    }
                )
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "split",
                    "label",
                    "label_name",
                    "sample_idx",
                    "value",
                    "length",
                ]
            )
        return pd.concat(rows, ignore_index=True)

    def to_wide_dataframe(self, split: Split) -> pd.DataFrame:
        """Return UCR-like wide rows: metadata + padded value_0 ... value_n."""
        self._require_loaded()
        records = self._select_records(split)
        max_len = max(record.length for record in records)
        rows = []
        for record in records:
            padded = np.full(max_len, np.nan, dtype=float)
            padded[: record.length] = record.values
            row: dict[str, Any] = {
                "series_id": record.series_id,
                "split": record.split,
                "label": record.label,
                "label_name": record.label_name,
                "length": record.length,
            }
            row.update({f"value_{idx}": value for idx, value in enumerate(padded)})
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(
        self,
        output_dir: str | Path,
        long: bool = True,
        wide: bool = False,
    ) -> dict[str, Path]:
        """Export parsed data to CSV."""
        self._require_loaded()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        if long:
            path = output / "gesture_pebble_z1_long.csv"
            self.to_long_dataframe().to_csv(path, index=False)
            written["long"] = path

        if wide:
            for split in ("train", "test"):
                path = output / f"gesture_pebble_z1_{split}_wide.csv"
                self.to_wide_dataframe(split).to_csv(path, index=False)
                written[f"{split}_wide"] = path

        return written

    def _load_split(
        self,
        split: Split,
        file_format: Literal["ts", "txt"],
    ) -> list[GestureRecord]:
        suffix = split.upper()
        path = self.root / f"GesturePebbleZ1_{suffix}.{file_format}"
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} file: {path}")

        if file_format == "ts":
            return self._parse_ts(path, split=split)
        return self._parse_txt(path, split=split)

    def _parse_ts(self, path: Path, split: Split) -> list[GestureRecord]:
        records = []
        in_data = False

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data or line.startswith("#"):
                continue

            parts = line.split(":")
            if len(parts) < 2:
                raise ValueError(f"Expected '<values>:<label>' in {path}: {line[:80]}")

            label = parts[-1].strip()
            values = self._parse_value_list(parts[0])
            records.append(self._make_record(split, label, len(records), values))

        return records

    def _parse_txt(self, path: Path, split: Split) -> list[GestureRecord]:
        matrix = np.genfromtxt(path, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError(f"Unexpected txt shape in {path}: {matrix.shape}")

        records = []
        for row_idx, row in enumerate(matrix):
            label = str(int(row[0]))
            values = self._trim_trailing_nan(row[1:].astype(float))
            records.append(self._make_record(split, label, row_idx, values))
        return records

    @staticmethod
    def _parse_value_list(values_text: str) -> np.ndarray:
        values = []
        for token in values_text.split(","):
            token = token.strip()
            if token in {"", "?", "NaN", "nan"}:
                values.append(np.nan)
            else:
                values.append(float(token))
        return GesturePebbleZ1Dataset._trim_trailing_nan(np.array(values, dtype=float))

    @staticmethod
    def _trim_trailing_nan(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        finite_positions = np.flatnonzero(~np.isnan(values))
        if finite_positions.size == 0:
            return np.array([], dtype=float)
        return values[: finite_positions[-1] + 1]

    @staticmethod
    def _make_record(
        split: Split,
        label: str,
        row_idx: int,
        values: np.ndarray,
    ) -> GestureRecord:
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{split} row {row_idx} contains empty or non-finite values")
        series_id = f"{split}_{row_idx:04d}"
        return GestureRecord(
            series_id=series_id,
            split=split,
            label=label,
            label_name=CLASS_NAMES.get(label, label),
            values=values.astype(float),
        )

    def _select_records(self, split: Split | None) -> list[GestureRecord]:
        if split is None:
            return self.train + self.test
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train', 'test', or None")
        return self._records[split]

    def _require_loaded(self) -> None:
        if not self._records:
            raise RuntimeError("Call load() before accessing dataset records")
