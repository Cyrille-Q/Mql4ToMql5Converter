# `src/train_seq2seq.py` — Seq2Seq Token-Level Training

## Description

Entraîne le modèle Seq2Seq transformer (encodeur-décodeur) sur des paires MQL4→MQL5
tokenizées. Charge le dataset depuis un fichier `.pkl` préparé par
`prepare_seq2seq_data.py`, entraîne l'architecture encodeur-décodeur, et sauvegarde
les checkpoints par époque ainsi que le meilleur modèle. Effectue périodiquement un
test de génération sur un exemple de validation.

## Arguments

| Argument   | Type   | Requis | Défaut               | Description                     |
|------------|--------|--------|----------------------|---------------------------------|
| `--config` | `str`  | Non    | `configs/seq2seq.yaml` | Chemin vers le fichier YAML   |

Tous les hyperparamètres (`batch_size`, `block_size`, `max_epochs`, `learning_rate`,
`n_embd`, `n_head`, `n_layer`, `dropout`, `eval_interval`, `checkpoint_interval`)
sont lus depuis le fichier de configuration YAML.

## Exemples

```bash
# Entraînement avec la config par défaut
python src/train_seq2seq.py --config configs/seq2seq.yaml

# Entraînement avec une config personnalisée
python src/train_seq2seq.py --config configs/seq2seq_custom.yaml
```

## Dépendances

- **Données :** `data/processed/seq2seq_dataset.pkl` (généré par
  `src/scripts/prepare_seq2seq_data.py`)
- **Config :** `configs/seq2seq.yaml`
- **Sorties :** `checkpoints/seq2seq_best.pt`, `checkpoints/seq2seq_epoch_NNNN.pt`

## Pipeline

```
prepare_seq2seq_data.py → train_seq2seq.py → convert_mql4_seq2seq.py
```