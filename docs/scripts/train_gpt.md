# `src/train.py` — GPT Char-Level Training

## Description

Entraîne le modèle GPT char-level (legacy) sur des paires MQL4→MQL5. Charge les données
préparées par `prepare_conversion_data.py`, encode au niveau caractère, entraîne un
transformer de type nanoGPT, et sauvegarde les checkpoints. Termine par une génération
de test sur un prompt codé en dur.

## Arguments

| Argument   | Type   | Requis | Défaut             | Description                     |
|------------|--------|--------|--------------------|---------------------------------|
| `--config` | `str`  | Non    | `configs/gpt.yaml` | Chemin vers le fichier YAML     |

Tous les hyperparamètres (`batch_size`, `block_size`, `max_iters`, `learning_rate`,
`n_embd`, `n_head`, `n_layer`, `dropout`, `eval_interval`, etc.) sont lus depuis le
fichier de configuration YAML.

## Exemples

```bash
# Entraînement avec la config par défaut
python src/train.py --config configs/gpt.yaml

# Entraînement avec une config personnalisée
python src/train.py --config configs/gpt_custom.yaml
```

## Dépendances

- **Données :** `data/processed/train.txt` et `data/processed/val.txt` (générés par
  `src/scripts/prepare_conversion_data.py`)
- **Config :** `configs/gpt.yaml`
- **Sorties :** `checkpoints/best_model.pt`, `checkpoints/checkpoint_iter_NNNNNN.pt`

## Pipeline

```
prepare_conversion_data.py → train.py → convert_mql4.py
```