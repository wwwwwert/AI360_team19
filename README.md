# AI360_team19


Install environment: `uv sync`

Datasets should be placed like: `./data/Yahoo/...`, `./data/AIOPS/...`

Run benchmark:
```bash
uv run run_benchmark.py \
  --datasets "NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5
```

Run web app:
```bash
uv run streamlit run web_app.py
```
