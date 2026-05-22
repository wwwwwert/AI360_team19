from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.benchmark.config import (
    BenchmarkConfig,
    load_benchmark_config,
    validate_benchmark_config,
)
from src.benchmark.inferencer import GesturePebbleZ1Benchmark
from src.benchmark.io import write_benchmark_outputs
from src.benchmark.schemas import METHOD_NAMES
from src.datasets.gesture_pebble_z1 import GesturePebbleZ1Dataset
from src.metrics import METRIC_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark subsequence similarity methods on GesturePebbleZ1."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. CLI flags override values from the config.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to GesturePebbleZ1 directory.",
    )
    parser.add_argument("--format", choices=["ts", "txt"], default=None, help="Input file format.")
    parser.add_argument(
        "--method",
        choices=METHOD_NAMES,
        action="append",
        dest="methods",
        help="Method to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        dest="k_values",
        help="k for kNN voting. Can be passed multiple times. Defaults to 1, 5, 10.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_REGISTRY),
        action="append",
        dest="metric_names",
        help="Metric to compute. Can be passed multiple times.",
    )
    parser.add_argument(
        "--average",
        choices=["micro", "macro"],
        action="append",
        dest="metric_averages",
        help="Metric averaging mode. Can be passed multiple times.",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=None,
        help="Use z-normalization.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Use raw values instead of z-normalization.",
    )
    parser.add_argument(
        "--resample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resample every series to a fixed length before computing distances.",
    )
    parser.add_argument(
        "--distance-scale",
        choices=["sqrt_m", "none"],
        default=None,
        help="Scale distances by sqrt(query length) for variable-length comparability.",
    )
    parser.add_argument(
        "--resample-length",
        default=None,
        help="'max', 'median', or an integer length used when resample is enabled.",
    )
    parser.add_argument("--max-test", type=int, default=None, help="Evaluate only the first N test cases.")
    parser.add_argument("--output-dir", default=None, help="Where to write CSV outputs.")
    parser.add_argument(
        "--export-parsed-csv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also export parsed long CSV.",
    )
    parser.add_argument(
        "--export-wide-csv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also export padded wide train/test CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = _resolve_config(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    resample_length: int | Literal["max", "median"]
    if config.resample_length in {"max", "median"}:
        resample_length = config.resample_length
    else:
        resample_length = int(config.resample_length)

    dataset = GesturePebbleZ1Dataset(config.dataset).load(file_format=config.file_format)
    print("Dataset summary:")
    print(dataset.summary().to_string(index=False))

    if config.export_parsed_csv or config.export_wide_csv:
        written = dataset.export_csv(
            config.output_dir,
            long=config.export_parsed_csv,
            wide=config.export_wide_csv,
        )
        print("Exported parsed CSV:")
        print(json.dumps({key: str(path) for key, path in written.items()}, indent=2))

    benchmark = GesturePebbleZ1Benchmark(
        dataset=dataset,
        normalize=config.normalize,
        resample=config.resample,
        distance_scale=config.distance_scale,
        resample_length=resample_length,
        metric_names=config.metric_names,
        metric_averages=config.metric_averages,
    )
    results = benchmark.run(
        methods=config.methods,
        k_values=config.k_values,
        max_test=config.max_test,
    )
    written = write_benchmark_outputs(results, config.output_dir)
    print("Benchmark outputs:")
    print(json.dumps(written, indent=2))

    summary = pd.read_csv(Path(config.output_dir) / "benchmark_summary.csv")
    print("Benchmark summary:")
    print(summary.to_string(index=False))


def _resolve_config(args: argparse.Namespace) -> BenchmarkConfig:
    base = load_benchmark_config(args.config)
    config = BenchmarkConfig(
        dataset=args.dataset if args.dataset is not None else base.dataset,
        file_format=args.format if args.format is not None else base.file_format,
        methods=tuple(args.methods) if args.methods is not None else base.methods,
        k_values=tuple(args.k_values) if args.k_values is not None else base.k_values,
        metric_names=tuple(args.metric_names)
        if args.metric_names is not None
        else base.metric_names,
        metric_averages=tuple(args.metric_averages)
        if args.metric_averages is not None
        else base.metric_averages,
        normalize=args.normalize if args.normalize is not None else base.normalize,
        resample=args.resample if args.resample is not None else base.resample,
        distance_scale=args.distance_scale if args.distance_scale is not None else base.distance_scale,
        resample_length=_parse_resample_length(args.resample_length)
        if args.resample_length is not None
        else base.resample_length,
        max_test=args.max_test if args.max_test is not None else base.max_test,
        output_dir=args.output_dir if args.output_dir is not None else base.output_dir,
        export_parsed_csv=args.export_parsed_csv
        if args.export_parsed_csv is not None
        else base.export_parsed_csv,
        export_wide_csv=args.export_wide_csv
        if args.export_wide_csv is not None
        else base.export_wide_csv,
    )
    validate_benchmark_config(config)
    return config


def _parse_resample_length(value: str) -> int | Literal["max", "median"]:
    if value in {"max", "median"}:
        return value
    length = int(value)
    if length <= 1:
        raise ValueError("resample_length must be greater than 1")
    return length


if __name__ == "__main__":
    main()
