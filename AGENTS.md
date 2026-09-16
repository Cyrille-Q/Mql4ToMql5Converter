# AGENTS.md — mql4-to-mql5-converter

## Project state

**Early scaffold.** Core logic files (`dataset.py`, `preprocessing.py`, `model.py`, `trainer.py`, `converter.py`) do not exist yet — only their `__init__.py` stubs with `__all__` exports. Active branch: `dataset_init`.

## Architecture

```
MQL4 code → [preprocess_code_pair] → [MqlDataset] → [T5 model] → [MqlConverter] → MQL5 code
                                            ↑
                                      [train()]
```

All hyperparameters in `configs/default.yaml`. Override via `configs/local/*.yaml` (gitignored).

## Key entrypoints

| Module | Exports | Purpose |
|---|---|---|
| `src.inference` | `MqlConverter` | End-user API: MQL4 → MQL5 |
| `src.training` | `train` | Training entrypoint |
| `src.models` | `create_model`, `load_model` | Model factory / checkpoint loader |
| `src.data` | `MqlDataset`, `preprocess_code_pair` | JSONL dataset + tokenization |

## Data

| Path | Contents |
|---|---|
| `data/raw/mql_dataset_manual.jsonl` | 36 hand-crafted MQL4→MQL5 pairs |
| `data/raw/` | 72 `.mq4`/`.mq5` example files (30 pairs across 10 categories) |
| `data/raw/collect_mql_dataset.py` | GitHub code search collector — requires `GITHUB_TOKEN` env var. `--dry-run` to preview, `--sleep 1.5` to throttle. |
| `data/processed/` | Empty — expects `train.jsonl`, `val.jsonl`, `test.jsonl` |

## Commands

```bash
# Install
pip install -e ".[dev]"

# Lint & format
ruff check .        # line-length=100, select E,F,I,W
ruff format . --check

# Typecheck
mypy src/

# Test (no tests exist yet — create under tests/)
pytest

# Train
python -c "from src.training import train; train()"

# GitHub dataset collection
GITHUB_TOKEN=ghp_xxx python data/raw/collect_mql_dataset.py
```

Order: `ruff check → mypy → pytest` (once tests exist).

## Conventions

- Ruff `line-length=100`, `target-version=py310`, rules `E,F,I,W`
- mypy `strict = true`, `python_version = "3.10"`
- Git LFS tracks: `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `data/raw/**`, `data/processed/**`, `outputs/**`
- GPL-3.0 licensed — check licenses of collected GitHub code before redistribution