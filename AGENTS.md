# AGENTS.md — mql4-to-mql5-converter

## Project state

**Prototype GPT character-level fonctionnel.** Les anciens modules T5 (`src.data`, `src.inference`, `src.models`, `src.training`) ont été supprimés. Le projet implémente maintenant un GPT de type nanoGPT (Karpathy) qui apprend à convertir MQL4→MQL5 via un format de séquence textuel avec vocabulaire au niveau caractère. Active branch: `dataset_init` (3 commits ahead of origin).

## Architecture

```
mql_dataset_manual.jsonl (36 paires MQL4↔MQL5)
         ↓
[prepare_conversion_data.py]   90/10 split, seed 42
         ↓
train.txt / val.txt    format: MQL4: ...\n\nMQL5: ...###
         ↓
[src/train.py → src/GPT/gpt.py]    GPT char-level
         ↓
checkpoints/*.pt    best_model.pt + checkpoint_iter_NNNNNN.pt
         ↓
[src/scripts/convert_mql4.py]    CLI d'inférence
         ↓
MQL5 output
```

Hyperparamètres **codés en dur** dans `src/train.py` (batch=2, block=512, lr=1e-3, n_embd=128, n_head=4, n_layer=3, dropout=0.2, max_iters=5000, eval_interval=50, device=cuda/cpu). `configs/default.yaml` est legacy T5 — inutilisé pour l'instant.

## Key entrypoints

| Module | Export(s) | Purpose |
|---|---|---|
| `src.GPT.gpt` | `GPTLanguageModel` | Modèle GPT caractère (embeddings + blocs transformer, forward + generate) |
| `src.train` | *(exécutable)* | Entraînement : `python src/train.py` (doit être lancé depuis `src/` ou sys.path) |
| `src.scripts.convert_mql4` | `convert_mql4()` + CLI | Chargement checkpoint → génération MQL5 pour un fichier .mq4 / code entré |
| `src.scripts.prepare_conversion_data` | *(exécutable)* | Convertit `data/raw/mql_dataset_manual.jsonl` → `train.txt`/`val.txt` |

**Note import/package** : `from GPT import gpt` n'est pas un import relatif. Les scripts utilisent `sys.path.insert(0, SRC_DIR)` ou requièrent un lancement depuis le dossier `src/`.

## Data

| Path | Contents |
|---|---|
| `data/raw/mql_dataset_manual.jsonl` | 36 paires MQL4→MQL5 faites à la main |
| `data/raw/mql_dataset_collected.jsonl` | Données collectées via GitHub — **vide** (0 lignes) |
| `data/raw/` | 72 `.mq4`/`.mq5` example files (30 paires across 10 categories) |
| `data/raw/collect_mql_dataset.py` | GitHub code search collector — requires `GITHUB_TOKEN` env var |
| `data/processed/train.txt` | ~218 lignes, format `MQL4: …\n\nMQL5: …###` (LFS) |
| `data/processed/val.txt` | ~46 lignes, idem (LFS) |

## Commands

```bash
# Install (dev optional)
pip install -e ".[dev]"

# Préparer les données (depuis la racine du projet)
python src/scripts/prepare_conversion_data.py

# Entraînement
python src/train.py

# Inférence
python src/scripts/convert_mql4.py checkpoints/best_model.pt fichier.mq4
python src/scripts/convert_mql4.py checkpoints/best_model.pt "#property strict\nextern int P=14;"

# Génération de données synthétiques (CWD)
python src/scripts/mql4_generate.py
python src/scripts/mql4_convert.py

# Lint & typecheck (⚠️ requis: pip install -e ".[dev]", non installé dans venv_mq4mq5/ par défaut)
ruff check .
ruff format . --check
mypy src/
pytest
```

**Remarque** : le venv actif est `venv_mq4mq5/` (non encore dans `.gitignore`). Les dépendances `[dev]` (ruff, mypy, pytest) n'y sont pas installées. Les outils de lint/typecheck ne sont pas exécutables avant leur installation.

## Conventions

- Ruff `line-length=100`, `target-version=py310`, rules `E,F,I,W`
- mypy `strict = true`, `python_version = "3.10"`
- Git LFS tracks: `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `data/raw/**`, `data/processed/**`, `outputs/**`
- GPL-3.0 licensed — check licenses of collected GitHub code before redistribution
- Checkpoints sauvegardés dans `checkpoints/` (`*.pt` gitignoré, mais pas le dossier)
- Le code actuel **ne passe pas** `ruff check` (lignes >100 dans `src/train.py`) ni `mypy strict` (pas de typage dans la plupart des fichiers) — état transitoire accepté en phase prototype.