# AGENTS.md — mql4-to-mql5-converter

## Project state

**Deux architectures disponibles :** GPT char-level (baseline) et Seq2Seq token-level (expérimental).  
Les anciens modules T5 (`src.data`, `src.inference`, `src.models`, `src.training`) ont été supprimés.  
Active branch: `dataset_init` (3 commits ahead of origin).

## Architecture

### Système Seq2Seq token-level (nouveau)

```
configs/seq2seq.yaml  ← hyperparamètres
         ↓
mql_all_936.jsonl (936 paires MQL4↔MQL5)
         ↓
[src/scripts/prepare_seq2seq_data.py --config]   tokenizer regex + encode
         ↓
data/processed/seq2seq_dataset.pkl      enc_inputs / dec_inputs / labels
         ↓
[src/train_seq2seq.py --config → src/models/seq2seq.py]    Seq2Seq (encodeur-décodeur)
         ↓
checkpoints/*.pt    seq2seq_best.pt + seq2seq_epoch_NNNN.pt
         ↓
[src/scripts/convert_mql4_seq2seq.py --config]    CLI d'inférence
         ↓
MQL5 output
```

### Ancien système GPT char-level (conservé)

```
configs/gpt.yaml  ← hyperparamètres
         ↓
mql_dataset_manual.jsonl (36 paires MQL4↔MQL5)
         ↓
[prepare_conversion_data.py --config]   90/10 split, seed 42
         ↓
train.txt / val.txt    format: MQL4: ...\n\nMQL5: ...###
         ↓
[src/train.py --config → src/models/gpt.py]    GPT char-level
         ↓
checkpoints/*.pt    best_model.pt + checkpoint_iter_NNNNNN.pt
         ↓
[src/scripts/convert_mql4.py --config]    CLI d'inférence
         ↓
MQL5 output
```

## Key entrypoints

| Module | Export(s) | Purpose |
|---|---|---|
| `src.models.gpt` | `GPTLanguageModel` | Modèle GPT caractère (baseline) |
| `src.models.seq2seq` | `Seq2SeqTransformer` | Modèle Seq2Seq token-level (encodeur-décodeur) |
| `src.tokenizers.mql_tokenizer` | `MQLTokenizer` | Tokenizer regex niveau token (343 tokens) |
| `src.utils.config` | `load_config()`, `resolve_device()` | Chargeur de configuration YAML + résolution device |
| `src.train` | *(exécutable)* | Entraînement GPT : `python src/train.py --config configs/gpt.yaml` |
| `src.train_seq2seq` | *(exécutable)* | Entraînement Seq2Seq : `python src/train_seq2seq.py --config configs/seq2seq.yaml` |
| `src.scripts.convert_mql4` | `convert_mql4()` + CLI | GPT inférence (legacy) |
| `src.scripts.convert_mql4_seq2seq` | `convert_mql4()` + CLI | Seq2Seq inférence |
| `src.scripts.prepare_seq2seq_data` | *(exécutable)* | Prépare les données pour Seq2Seq |

**Note import/package** : Le dossier `src/GPT/` a ééenommé e `rc/models/`. Les imorts anciens (`frm GPT impr gpt`) ont étémis jor.

## Data

| Path | Contents |
|---|---|
| `data/raw/mql_dataset_manual.jsonl` | 36 paires MQL4→MQL5 faites à la main |
| `data/raw/mql_dataset_collected.jsonl` | Données collectées via GitHub — **vide** (0 lignes) |
| `data/raw/` | 72 `.mq4`/`.mq5` example files (30 paires across 10 categories) |
| `data/processed/seq2seq_dataset.pkl` | tokenizer + enc_inputs / dec_inputs / labels (Seq2Seq) |
| `data/processed/train.txt` | ~218 lignes, format `MQL4: …\n\nMQL5: …###` (LFS, GPT legacy) |
| `data/processed/val.txt` | ~46 lignes, idem (LFS, GPT legacy) |
| `configs/seq2seq.yaml` | Configuration YAML du pipeline Seq2Seq |
| `configs/gpt.yaml` | Configuration YAML du pipeline GPT (legacy) |
| `src.scripts.debug_seq2seq_data` | *(exécutable)* | Debug dataset : inspecte les tokens de chaque paire |

## Commands

```bash
# Install (dev optional)
pip install -e ".[dev]"

# === GPT char-level (legacy) ===
python src/scripts/prepare_conversion_data.py --config configs/gpt.yaml
python src/train.py --config configs/gpt.yaml
python src/train.py --config configs/gpt.yaml --resume checkpoints/checkpoint_iter_002000.pt  # reprise d'entraînement
python src/scripts/convert_mql4.py checkpoints/best_model.pt fichier.mq4 --config configs/gpt.yaml

# === Seq2Seq token-level ===
python src/scripts/prepare_seq2seq_data.py --config configs/seq2seq.yaml
python src/train_seq2seq.py --config configs/seq2seq.yaml
python src/train_seq2seq.py --config configs/seq2seq.yaml --resume checkpoints/seq2seq_epoch_0040.pt  # reprise d'entraînement
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt fichier.mq4 --config configs/seq2seq.yaml
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt "#property strict\nextern int P=14;" --config configs/seq2seq.yaml

# Génération de données synthétiques (CWD)
python src/scripts/mql4_generate.py
python src/scripts/mql4_convert.py

# === Debug / Inspection du Dataset ===
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --index 0 --show-meta --show-code --raw-ids
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --range 100 105 --mode enc --raw-ids
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --index 0 --alignment --show-meta

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
- Hyperparamètres dans `configs/seq2seq.yaml` et `configs/gpt.yaml` (lus via `--config` par tous les scripts)
- Le code actuel **ne passe pas** `ruff check` ni `mypy strict` — état transitoire accepté en phase prototype.