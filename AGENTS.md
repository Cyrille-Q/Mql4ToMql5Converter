# AGENTS.md — mql4-to-mql5-converter

## Project state

**Deux architectures disponibles :** GPT char-level (baseline) et Seq2Seq token-level (actuelle, recommandée).  
Les anciens modules T5 (`src.data`, `src.inference`, `src.training`) ont été supprimés.  
Active branch: `dataset_init` (à jour avec `origin/dataset_init`). Branche locale `main` conservée.

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
| `src.scripts.debug_seq2seq_data` | *(exécutable)* | Debug dataset : inspecte les tokens/alignment de chaque paire |
| `src.scripts.inspect_seq2seq_vocab` | *(exécutable)* | Rapport sur le vocabulaire du tokenizer depuis le `.pkl` |
| `src.scripts.mql4_generate` / `mql4_convert` | *(exécutables)* | Génération + conversion de données synthétiques (expérimental) |
| `src.utils.metrics` | `write_history_csv/json()`, `plot_history()` | Export CSV/JSON + courbe de loss (matplotlib optionnel) |

**Note import/package** : le dossier `src/GPT/` a été renommé en `src/models/`. Les imports anciens (`from GPT import gpt`) ont été mis à jour — le package s'importe via `src.models.gpt` / `src.models.seq2seq`.

## Data

| Path | Contents |
|---|---|
| `data/raw/mql_all_936.jsonl` | 936 paires MQL4→MQL5 — **dataset principal du pipeline Seq2Seq** |
| `data/raw/mql_synthetic_900.jsonl` | 900 paires synthétiques (génération) |
| `data/raw/mql_dataset_manual.jsonl` | 36 paires MQL4→MQL5 faites à la main (GPT legacy) |
| `data/raw/mql_dataset_collected.jsonl` | Données collectées via GitHub — **vide** (0 lignes) |
| `data/raw/` | 72 `.mq4`/`.mq5` example files (36 paires across 10 categories) + `collect_mql_dataset.py` |
| `data/processed/seq2seq_dataset.pkl` | tokenizer + enc_inputs / dec_inputs / labels + indices train/val (Seq2Seq) |
| `data/processed/train.txt` | ~218 lignes, format `MQL4: …\n\nMQL5: …###` (LFS, GPT legacy) |
| `data/processed/val.txt` | ~46 lignes, idem (LFS, GPT legacy) |
| `configs/seq2seq.yaml` | Configuration YAML du pipeline Seq2Seq |
| `configs/gpt.yaml` | Configuration YAML du pipeline GPT (legacy) |
| `configs/default.yaml` | Obsolète (config T5) — non utilisé |
| `docs/scripts/*.md` | Documentation par script (CLI, options, exemples) |

## Commands

```bash
# Install (dev optional)
pip install -e ".[dev]"

# === GPT char-level (legacy) ===
python src/scripts/prepare_conversion_data.py --config configs/gpt.yaml
python src/train.py --config configs/gpt.yaml
python src/train.py --config configs/gpt.yaml --verbose           # + test de génération final
python src/train.py --config configs/gpt.yaml --resume checkpoints/checkpoint_iter_002000.pt  # reprise d'entraînement
python src/scripts/convert_mql4.py checkpoints/best_model.pt fichier.mq4 --config configs/gpt.yaml

# === Seq2Seq token-level ===
python src/scripts/prepare_seq2seq_data.py --config configs/seq2seq.yaml
python src/train_seq2seq.py --config configs/seq2seq.yaml
python src/train_seq2seq.py --config configs/seq2seq.yaml --verbose           # + test de génération à chaque checkpoint
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
python src/scripts/inspect_seq2seq_vocab.py                    # rapport vocabulaire depuis le .pkl
python src/scripts/inspect_seq2seq_vocab.py --pkl <chemin.pkl> # variante avec chemin explicite

# Lint & typecheck (⚠️ requis: pip install -e ".[dev]", non installé dans .venv/ par défaut)
ruff check .
ruff format . --check
mypy src/
pytest
```

**Remarque** : le venv actif est `.venv/` (déjà dans `.gitignore`). Les dépendances `[dev]` (ruff, mypy, pytest) n'y sont pas installées — les outils de lint/typecheck ne sont donc pas exécutables avant leur installation. `tqdm` est installé (barre de progression) ; `matplotlib` est optionnel (courbe de loss générée seulement s'il est présent). Gestion des dépendances via `uv` (`uv.lock` présent à la racine).

## Suivi de l'apprentissage (affichage écran)

Les deux entraîneurs suivent l'évolution de la loss via :

- **Barre de progression `tqdm`** sur la boucle d'entraînement (loss courante en `postfix`). Désactivée automatiquement si `tqdm` n'est pas installé.
- **Ligne de résumé** : loss de train affichée à **chaque époque** (ou itération), loss de val + sauvegarde checkpoint tous les `eval_interval`.
  - Seq2Seq : `epoch  N | train loss X | val loss Y | step N` (val/checkpoint aux époques `% eval_interval`).
  - GPT : `step N: train loss X, val loss Y` (toutes les `eval_interval` itérations).
- **`--verbose`** : détail supplémentaire — test de génération à chaque checkpoint (Seq2Seq) ou final (GPT).
- **Export d'historique** en fin d'entraînement (dans `checkpoints/`) :
  - `seq2seq_loss_history.csv/json` + `seq2seq_loss_curve.png` (Seq2Seq)
  - `gpt_loss_history.csv/json` + `gpt_loss_curve.png` (GPT)
  - La courbe (`png`) n'est produite que si `matplotlib` est installé.

Implémentation : `src/utils/metrics.py` (export CSV/JSON + courbe, dépendances optionnelles).

## Conventions

- Ruff `line-length=100`, `target-version=py310`, rules `E,F,I,W`
- mypy `strict = true`, `python_version = "3.10"`
- Git LFS tracks: `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `data/raw/**`, `data/processed/**`, `outputs/**`
- GPL-3.0 licensed — check licenses of collected GitHub code before redistribution
- Checkpoints sauvegardés dans `checkpoints/` (`*.pt` gitignoré, mais pas le dossier)
- Hyperparamètres dans `configs/seq2seq.yaml` et `configs/gpt.yaml` (lus via `--config` par tous les scripts)
- Le code actuel **ne passe pas** `ruff check` ni `mypy strict` — état transitoire accepté en phase prototype.