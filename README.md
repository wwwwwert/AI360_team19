# AI360_team19


Install environment: `uv sync`

Run the GesturePebbleZ1 benchmark:

```bash
./benchmark.sh
```

Edit `benchmark_config.yaml` to choose methods, k values, metrics, and outputs.
The default config runs only `mass_subsequence` and reports macro metrics.
To reproduce the old `mass_resample` behavior, set `resample: true`.
CLI flags override YAML values, for example:

```bash
./benchmark.sh --method mass_subsequence --resample --k 1
```
