# `src/scripts/convert_mql4_seq2seq.py` — Seq2Seq Inference

## Description

Exécute l'inférence avec un modèle Seq2Seq entraîné. Charge un checkpoint,
reconstruit le tokenizer depuis le dataset `.pkl`, tokenize l'entrée MQL4, exécute
l'encodeur-décodeur, et génère le code MQL5 correspondant.

## Arguments

| Argument          | Type       | Requis | Défaut               | Description                              |
|-------------------|------------|--------|----------------------|------------------------------------------|
| `checkpoint_path` | `str` (pos) | **Oui** | —                    | Chemin vers le checkpoint (`.pt`)        |
| `mql4_input`      | `str` (pos) | Non    | `None` (fallback)    | Code MQL4 ou chemin vers `.mq4`          |
| `--config`        | `str`      | Non    | `configs/seq2seq.yaml` | Chemin vers le fichier YAML            |

L'argument `mql4_input` est optionnel : s'il est omis, un exemple MQL4 codé en dur
est utilisé comme fallback. La détection automatique détermine s'il s'agit d'un
chemin de fichier ou d'une chaîne de code inline.

## Exemples

```bash
# Conversion depuis un fichier .mq4
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt fichier.mq4 --config configs/seq2seq.yaml

# Conversion depuis une chaîne inline
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt "#property strict\nextern int P=14;" --config configs/seq2seq.yaml

# Utilisation du fallback (exemple codén dur)
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt --config configs/seq2seq.yaml
```

## Dépendances

- **Checkpoint :** `checkpoints/seq2seq_best.pt` (généré par `src/train_seq2seq.py`)
- **Config :** `configs/seq2seq.yaml`
- **Dataset :** `data/processed/seq2seq_dataset.pkl` (pour reconstruire le tokenizer)

## Pipeline

```
prepare_seq2seq_data.py → train_seq2seq.py → convert_mql4_seq2seq.py
```