# `src/scripts/convert_mql4_t5.py` — T5 Inference

## Description

Exécute l'inférence avec un modèle Seq2Seq entraîné. Charge un checkpoint,
reconstruit le tokenizer depuis le dataset `.pkl`, tokenize l'entrée MQL4, exécute
l'encodeur-décodeur, et génère le code MQL5 correspondant.

## Arguments

| Argument          | Type       | Requis | Défaut               | Description                              |
|-------------------|------------|--------|----------------------|------------------------------------------|
| `checkpoint_path` | `str` (pos) | **Oui** | —                    | Chemin vers le checkpoint (`.pt`)        |
| `mql4_input`      | `str` (pos) | Non    | `None` (fallback)    | Code MQL4 ou chemin vers `.mq4`          |
| `--config`        | `str`      | Non    | `configs/t5.yaml` | Chemin vers le fichier YAML            |

L'argument `mql4_input` est optionnel : s'il est omis, un exemple MQL4 codé en dur
est utilisé comme fallback. La détection automatique détermine s'il s'agit d'un
chemin de fichier ou d'une chaîne de code inline.

## Exemples

```bash
# Conversion depuis un fichier .mq4
python src/scripts/convert_mql4_t5.py checkpoints/t5_best.pt fichier.mq4 --config configs/t5.yaml

# Conversion depuis une chaîne inline
python src/scripts/convert_mql4_t5.py checkpoints/t5_best.pt "#property strict\nextern int P=14;" --config configs/t5.yaml

# Utilisation du fallback (exemple codén dur)
python src/scripts/convert_mql4_t5.py checkpoints/t5_best.pt --config configs/t5.yaml
```

## Dépendances

- **Checkpoint :** `checkpoints/t5_best.pt` (généré par `src/train_t5.py`)
- **Config :** `configs/t5.yaml`
- **Dataset :** `data/processed/t5_dataset.pkl` (pour reconstruire le tokenizer)

## Pipeline

```
prepare_t5_data.py → train_t5.py → convert_mql4_t5.py
```