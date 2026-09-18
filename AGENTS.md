# AGENTS.md — mql4-to-mql5-converter

## Project state

**Deux architectures disponibles :** GPT char-level (baseline) et Seq2Seq token-level (expérimental).  
Les anciens modules T5 (`src.data`, `src.inference`, `src.models`, `src.training`) ont été supprimés.  
Active branch: `dataset_init` (3 commits ahead of origin).

## Architecture

### Système Seq2Seq token-level (nouveau)

```
mql_dataset_manual.jsonl (36 paires MQL4↔MQL5)
         ↓
[src/scripts/prepare_seq2seq_data.py]   tokenizer regex + encode
         ↓
data/processed/seq2seq_dataset.pkl      enc_inputs / dec_inputs / labels
         ↓
[src/train_seq2seq.py → src/models/seq2seq.py]    Seq2Seq (encodeur-décodeur)
         ↓
checkpoints/*.pt    seq2seq_best.pt + seq2seq_epoch_NNNN.pt
         ↓
[src/scripts/convert_mql4_seq2seq.py]    CLI d'inférence
         ↓
MQL5 output
```

### Ancien système GPT char-level (conservé)

```
mql_dataset_manual.jsonl (36 paires MQL4↔MQL5)
         ↓
[prepare_conversion_data.py]   90/10 split, seed 42
         ↓
train.txt / val.txt    format: MQL4: ...\n\nMQL5: ...###
         ↓
[src/train.py → src/models/gpt.py]    GPT char-level
         ↓
checkpoints/*.pt    best_model.pt + checkpoint_iter_NNNNNN.pt
         ↓
[src/scripts/convert_mql4.py]    CLI d'inférence
         ↓
MQL5 output
```

## Key entrypoints

| Module | Export(s) | Purpose |
|---|---|---|
| `src.models.gpt` | `GPTLanguageModel` | Modèle GPT caractère (baseline) |
| `src.models.seq2seq` | `Seq2SeqTransformer` | Modèle Seq2Seq token-level (encodeur-décodeur) |
| `src.tokenizers.mql_tokenizer` | `MQLTokenizer` | Tokenizer regex niveau token (343 tokens) |
| `src.train` | *(éxécutale)* | Enraînement GPT : `python rc/rain.py` | | `sc.tain_seq2seq` | *(éxécutable)* | Enraînement Seq2Seq : `python rc/train_seq2seq.py` |
| `src.scripts.convet_mql4` | `convet_mql4()` + CLI | GPT inférec (legacy) | | `sc.scripts.convert_mql4_seq2seq` | `cnvert_mql4()` + CLI | Seq2Seq inférence |
| `src.scripts.preare_seq2seq_daa` | *(éxécutale)* | Prepare es doméepour Seq2Seq |

**Note import/package** : Le dossier `src/GPT/` a ééenommé e `rc/models/`. Les imorts anciens (`frm GPT impr gpt`) ont étémis jor.

## Data

| Path | Contents |
|---|---|
| `data/aw/mql_dataset_manual.jonl` | 36 paires MQL4→MQL5 faites al main |
| `data/raw/mql_dataset_cllected.jsonl` | Données colletes via GitHb — **ide** (0 ligns) |
| `data/raw/` | 72 `.mq4`/`.mq5` example files (30 paires across 10 categories) |
| `data/processed/seq2seq_dataset.pkl`  tokenizer + en_inputs,/dec_inputs,/lbls (Seq2Seq) |
| `data/processed/train.txt` | ~218 lignes, format `MQL4: …\n\nMQL5: …###` (LFS, GPT legacy) |
| `data/processed/val.txt` | ~46 lignes, idem (LFS, GPT legacy) |

## Commands

```bash
# Install (dev optional)
pip install -e ".[dev]"

# === GPT char-level (legacy) ===
python src/scripts/prepare_conversion_data.py
python src/train.py
python src/scripts/convert_mql4.py checkpoints/best_model.pt fichier.mq4

# === Seq2Seq token-level ===
python src/scripts/prepare_seq2seq_data.py
python src/train_seq2seq.py
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt fichier.mq4
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt "#property strict\nextern int P=14;"

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
- Le code actuel **ne passe pas** `ruff check` ni `mypy strict` — état transitoire accepté en phase prototype.