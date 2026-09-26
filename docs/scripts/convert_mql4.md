# `src/scripts/convert_mql4.py` — GPT Inference (Legacy)

## Description

Exécute l'inférence avec un modèle GPT char-level entraîné. Charge un checkpoint,
reconstruit le vocabulaire depuis les données d'entraînement, prend une entrée MQL4
(chemin de fichier ou chaîne inline) et génère le code MQL5 correspondant.

## Arguments

| Argument          | Type       | Requis | Défaut               | Description                              |
|-------------------|------------|--------|----------------------|------------------------------------------|
| `checkpoint_path` | `str` (pos) | **Oui** | —                    | Chemin vers le checkpoint (`.pt`)        |
| `mql4_input`      | `str` (pos) | Non    | `None` (fallback)    | Code MQL4 ou chemin vers `.mq4`          |
| `--config`        | `str`      | Non    | `configs/gpt.yaml`   | Chemin vers le fichier YAML              |

L'argument `mql4_input` est optionnel : s'il est omis, un exemple MQL4 codé en dur
est utilisé comme fallback. La détection automatique détermine s'il s'agit d'un
chemin de fichier ou d'une chaîne de code inline.

## Exemples

```bash
# Conversion depuis un fichier .mq4
python src/scripts/convert_mql4.py checkpoints/best_model.pt fichier.mq4 --config configs/gpt.yaml

# Conversion depuis une chaîne inline
python src/scripts/convert_mql4.py checkpoints/best_model.pt "#property strict" --config configs/gpt.yaml

# Utilisation du fallback (exemple codé en dur)
python src/scripts/convert_mql4.py checkpoints/best_model.pt --config configs/gpt.yaml
```

## Dépendances

- **Checkpoint :** `checkpoints/best_model.pt` (généré par `src/train.py`)
- **Config :** `configs/gpt.yaml`
- **Données d'entraînement :** `data/processed/train.txt` (pour reconstruire le vocabulaire)

## Pipeline

```
prepare_conversion_data.py → train.py → convert_mql4.py
```