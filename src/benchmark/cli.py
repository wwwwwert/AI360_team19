from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.benchmark.inferencer import GesturePebbleZ1Benchmark
from src.benchmark.io import write_benchmark_outputs
from src.datasets.gesture_pebble_z1 import GesturePebbleZ1Dataset
from src.metrics import DEFAULT_AVERAGES, DEFAULT_METRIC_NAMES, METRIC_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark subsequence similarity methods on GesturePebbleZ1."
    )
    parser.add_argument(
        "--dataset",
        default="data/GesturePebbleZ1",
        help="Path to GesturePebbleZ1 directory.",
    )
    parser.add_argument("--format", choices=["ts", "txt"], default="ts", help="Input file format.")
    parser.add_argument(
        "--method",
        choices=["mass_subsequence", "mass_resample", "euclidean_resample"],
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
        help="Metric to compute. Can be passed multiple times. Defaults to all metrics.",
    )
    parser.add_argument(
        "--average",
        choices=["micro", "macro"],
        action="append",
        dest="metric_averages",
        help="Metric averaging mode. Can be passed multiple times. Defaults to micro and macro.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Use raw values instead of z-normalization.")
    parser.add_argument(
        "--distance-scale",
        choices=["sqrt_m", "none"],
        default="sqrt_m",
        help="Scale distances by sqrt(query length) for variable-length comparability.",
    )
    parser.add_argument(
        "--resample-length",
        default="max",
        help="'max', 'median', or an integer length for resample-based methods.",
    )
    parser.add_argument("--max-test", type=int, default=None, help="Evaluate only the first N test cases.")
    parser.add_argument("--output-dir", default="gesture_pebble_z1_benchmark", help="Where to write CSV outputs.")
    parser.add_argument("--export-parsed-csv", action="store_true", help="Also export parsed long CSV.")
    parser.add_argument("--export-wide-csv", action="store_true", help="Also export padded wide train/test CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.methods or ["mass_subsequence", "mass_resample", "euclidean_resample"]
    k_values = args.k_values or [1, 5, 10]
    metric_names = args.metric_names or DEFAULT_METRIC_NAMES
    metric_averages = args.metric_averages or DEFAULT_AVERAGES
    resample_length: int | Literal["max", "median"]
    if args.resample_length in {"max", "median"}:
        resample_length = args.resample_length
    else:
        resample_length = int(args.resample_length)

    dataset = GesturePebbleZ1Dataset(args.dataset).load(file_format=args.format)
    print("Dataset summary:")
    print(dataset.summary().to_string(index=False))

    if args.export_parsed_csv or args.export_wide_csv:
        written = dataset.export_csv(
            args.output_dir,
            long=args.export_parsed_csv,
            wide=args.export_wide_csv,
        )
        print("Exported parsed CSV:")
        print(json.dumps({key: str(path) for key, path in written.items()}, indent=2))

    benchmark = GesturePebbleZ1Benchmark(
        dataset=dataset,
        normalize=not args.no_normalize,
        distance_scale=args.distance_scale,
        resample_length=resample_length,
        metric_names=metric_names,
        metric_averages=metric_averages,
    )
    results = benchmark.run(methods=methods, k_values=k_values, max_test=args.max_test)
    written = write_benchmark_outputs(results, args.output_dir)
    print("Benchmark outputs:")
    print(json.dumps(written, indent=2))

    summary = pd.read_csv(Path(args.output_dir) / "benchmark_summary.csv")
    print("Benchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
