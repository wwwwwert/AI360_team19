#!/bin/bash
set -euo pipefail

CONFIG_PATH="${BENCHMARK_CONFIG:-benchmark_config.yaml}"
PYTHON_BIN="${BENCHMARK_PYTHON:-python3}"
if [[ -z "${BENCHMARK_PYTHON:-}" && -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" ./src/benchmark/cli.py \
    --config "$CONFIG_PATH" \
    "$@"
