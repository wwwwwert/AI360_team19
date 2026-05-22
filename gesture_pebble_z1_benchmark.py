from src.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    GesturePebbleZ1Benchmark,
    MethodName,
    Predictor,
    as_float_array,
    classification_metrics,
    confusion_matrix,
    equal_length_mass_distance,
    euclidean_distance,
    load_benchmark_config,
    mass_subsequence_distance,
    resample_values,
    scale_distance,
    vote_label,
    write_benchmark_outputs,
    z_normalize,
)
from src.benchmark.cli import main, parse_args
from src.datasets.gesture_pebble_z1 import (
    CLASS_NAMES,
    GesturePebbleZ1Dataset,
    GestureRecord,
    Split,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CLASS_NAMES",
    "GesturePebbleZ1Benchmark",
    "GesturePebbleZ1Dataset",
    "GestureRecord",
    "MethodName",
    "Predictor",
    "Split",
    "as_float_array",
    "classification_metrics",
    "confusion_matrix",
    "equal_length_mass_distance",
    "euclidean_distance",
    "load_benchmark_config",
    "main",
    "mass_subsequence_distance",
    "parse_args",
    "resample_values",
    "scale_distance",
    "vote_label",
    "write_benchmark_outputs",
    "z_normalize",
]


if __name__ == "__main__":
    main()
