from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.benchmark.schemas import METHOD_NAMES
from src.metrics import DEFAULT_AVERAGES, DEFAULT_METRIC_NAMES, METRIC_REGISTRY


FileFormat = Literal["ts", "txt"]
DistanceScaleName = Literal["sqrt_m", "none"]


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "data/GesturePebbleZ1"
    file_format: FileFormat = "ts"
    methods: tuple[str, ...] = ("mass_subsequence",)
    k_values: tuple[int, ...] = (1, 5, 10)
    metric_names: tuple[str, ...] = DEFAULT_METRIC_NAMES
    metric_averages: tuple[str, ...] = DEFAULT_AVERAGES
    normalize: bool = True
    resample: bool = False
    distance_scale: DistanceScaleName = "sqrt_m"
    resample_length: int | Literal["max", "median"] = "max"
    max_test: int | None = None
    output_dir: str = "gesture_pebble_z1_benchmark"
    export_parsed_csv: bool = False
    export_wide_csv: bool = False


def load_benchmark_config(path: str | Path | None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark config does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Benchmark config must be a YAML mapping")

    return _parse_config(raw)


def _parse_config(raw: Mapping[str, Any]) -> BenchmarkConfig:
    defaults = BenchmarkConfig()
    metrics = _mapping(raw.get("metrics"), "metrics")
    exports = _mapping(raw.get("exports"), "exports")

    config = BenchmarkConfig(
        dataset=str(raw.get("dataset", defaults.dataset)),
        file_format=str(
            raw.get("format", raw.get("file_format", defaults.file_format))
        ),  # type: ignore[arg-type]
        methods=_str_tuple(raw.get("methods", defaults.methods), "methods"),
        k_values=_int_tuple(raw.get("k_values", defaults.k_values), "k_values"),
        metric_names=_str_tuple(
            metrics.get("names", raw.get("metric_names", defaults.metric_names)),
            "metrics.names",
        ),
        metric_averages=_str_tuple(
            metrics.get("averages", raw.get("metric_averages", defaults.metric_averages)),
            "metrics.averages",
        ),
        normalize=_bool(raw.get("normalize", defaults.normalize), "normalize"),
        resample=_bool(raw.get("resample", defaults.resample), "resample"),
        distance_scale=str(
            raw.get("distance_scale", defaults.distance_scale)
        ),  # type: ignore[arg-type]
        resample_length=_resample_length(raw.get("resample_length", defaults.resample_length)),
        max_test=_optional_positive_int(raw.get("max_test", defaults.max_test), "max_test"),
        output_dir=str(raw.get("output_dir", defaults.output_dir)),
        export_parsed_csv=_bool(
            exports.get("parsed_csv", raw.get("export_parsed_csv", defaults.export_parsed_csv)),
            "exports.parsed_csv",
        ),
        export_wide_csv=_bool(
            exports.get("wide_csv", raw.get("export_wide_csv", defaults.export_wide_csv)),
            "exports.wide_csv",
        ),
    )
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    _require_non_empty(config.methods, "methods")
    _require_subset(config.methods, METHOD_NAMES, "methods")

    _require_non_empty(config.k_values, "k_values")
    for value in config.k_values:
        if value <= 0:
            raise ValueError("k_values must contain only positive integers")

    _require_non_empty(config.metric_names, "metrics.names")
    _require_subset(config.metric_names, tuple(METRIC_REGISTRY), "metrics.names")

    _require_non_empty(config.metric_averages, "metrics.averages")
    _require_subset(config.metric_averages, ("micro", "macro"), "metrics.averages")

    if config.file_format not in {"ts", "txt"}:
        raise ValueError("format must be either 'ts' or 'txt'")
    if config.distance_scale not in {"sqrt_m", "none"}:
        raise ValueError("distance_scale must be either 'sqrt_m' or 'none'")
    if not isinstance(config.resample, bool):
        raise ValueError("resample must be true or false")
    if config.max_test is not None and config.max_test <= 0:
        raise ValueError("max_test must be a positive integer or null")
    if not config.output_dir:
        raise ValueError("output_dir must not be empty")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _iterable(value, name))


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    values = []
    for item in _iterable(value, name):
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain only integers")
        values.append(int(item))
    return tuple(values)


def _iterable(value: Any, name: str) -> Iterable[Any]:
    if (
        isinstance(value, str)
        or isinstance(value, Mapping)
        or not isinstance(value, Iterable)
    ):
        raise ValueError(f"{name} must be a list")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _resample_length(value: Any) -> int | Literal["max", "median"]:
    if value in {"max", "median"}:
        return value
    if isinstance(value, bool):
        raise ValueError(
            "resample_length must be 'max', 'median', or an integer greater than 1"
        )
    length = int(value)
    if length <= 1:
        raise ValueError("resample_length must be greater than 1")
    return length


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or null")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return result


def _require_non_empty(values: tuple[Any, ...], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _require_subset(values: tuple[str, ...], allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown {name}: {', '.join(unknown)}. Known values: {known}")
