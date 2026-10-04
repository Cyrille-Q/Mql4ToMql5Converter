# `src/train_t5.py` — T5 Token-Level Training

## Description

Entraîne le modèle Seq2Seq transformer (encodeur-décodeur) sur des paires MQL4→MQL5
tokenizées. Charge le dataset depuis un fichier `.pkl` préparé par
`prepare_t5_data.py`, entraîne l'architecture encodeur-décodeur, et sauvegarde
les checkpoints par époque ainsi que le meilleur modèle. Effectue périodiquement un
test de génération sur un exemple de validation.

## Arguments

| Argument   | Type   | Requis | Défaut               | Description                     |
|------------|--------|--------|----------------------|---------------------------------|
| `--config` | `str`  | Non    | `configs/t5.yaml` | Chemin vers le fichier YAML   |

Tous les hyperparamètres (`batch_size`, `block_size`, `max_epochs`, `learning_rate`,
`n_embd`, `n_head`, `n_layer`, `dropout`, `eval_interval`, `checkpoint_interval`)
sont lus depuis le fichier de configuration YAML.

## Exemples

```bash
# Entraînement avec la config par défaut
python src/train_t5.py --config configs/t5.yaml

# Entraînement avec une config personnalisée
python src/train_t5.py --config configs/seq2seq_custom.yaml
```

## Dépendances

- **Données :** `data/processed/t5_dataset.pkl` (généré par
  `src/scripts/prepare_t5_data.py`)
- **Config :** `configs/t5.yaml`
- **Sorties :** `checkpoints/t5_best.pt`, `checkpoints/t5_epoch_NNNN.pt`

## Pipeline

```
prepare_t5_data.py → train_t5.py → convert_mql4_t5.py
```