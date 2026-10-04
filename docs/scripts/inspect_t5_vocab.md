# `src/scripts/inspect_t5_vocab.py` — T5 Vocabulary Inspection

## Description

Charge le dataset Seq2Seq `.pkl` et affiche un rapport complet du vocabulaire :
chaque token avec son index, sa représentation et son statut (token spécial ou non).

## Arguments

| Argument | Type   | Requis | Défaut                               | Description              |
|----------|--------|--------|--------------------------------------|--------------------------|
| `--pkl`  | `str`  | Non    | `data/processed/t5_dataset.pkl` | Chemin vers le `.pkl`    |

## Exemples

```bash
# Inspection du vocabulaire par défaut
python src/scripts/inspect_t5_vocab.py

# Inspection avec un chemin personnalisé
python src/scripts/inspect_t5_vocab.py --pkl data/processed/t5_dataset.pkl
```

## Dépendances

- **Dataset :** `data/processed/t5_dataset.pkl` (généré par `prepare_t5_data.py`)