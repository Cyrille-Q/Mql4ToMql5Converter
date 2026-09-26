# `src/scripts/debug_seq2seq_data.py` — Seq2Seq Dataset Debug

## Description

Outil d'inspection et de débogage pour le dataset Seq2Seq. Charge le fichier `.pkl`
et affiche les paires tokenizées avec des tables de tokens détaillées, le code
brut optionnel, les métadonnées, une visualisation d'alignement et les IDs bruts.

## Arguments

| Argument       | Type       | Requis | Défaut               | Description                                  |
|----------------|------------|--------|----------------------|----------------------------------------------|
| `--config`     | `str`      | Non    | `configs/seq2seq.yaml` | Chemin vers le fichier YAML                |
| `--index`      | `int`      | Non    | `None`               | Affiche une seule paire par index            |
| `--range`      | `int int`  | Non    | `None`               | Affiche les paires de START à END (inclus)   |
| `--max`        | `int`      | Non    | `3`                  | Nombre maximum de paires à afficher          |
| `--all`        | `flag`     | Non    | `False`              | Affiche toutes les paires (outrepasse `--max`) |
| `--mode`       | `str`      | Non    | `all`                | Séquence à afficher : `enc`, `dec`, `labels`, ou `all` |
| `--show-code`  | `flag`     | Non    | `False`              | Affiche le code MQL4/MQL5 brut               |
| `--show-meta`  | `flag`     | Non    | `False`              | Affiche les métadonnées (catégorie, complexité) |
| `--raw-ids`    | `flag`     | Non    | `False`              | Affiche les listes d'IDs bruts               |
| `--alignment`  | `flag`     | Non    | `False`              | Affiche l'alignement teacher-forcing         |

## Exemples

```bash
# Affichage par défaut (3 premières paires, mode all)
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml

# Affiche une paire spécifique avec métadonnées et code brut
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --index 0 --show-meta --show-code --raw-ids

# Affiche une plage de paires en mode encoder uniquement
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --range 100 105 --mode enc --raw-ids

# Affiche l'alignement teacher-forcing
python src/scripts/debug_seq2seq_data.py --config configs/seq2seq.yaml --index 0 --alignment --show-meta
```

## Dépendances

- **Dataset :** `data/processed/seq2seq_dataset.pkl`
- **Config :** `configs/seq2seq.yaml`