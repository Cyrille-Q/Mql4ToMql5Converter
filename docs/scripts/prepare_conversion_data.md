# `src/scripts/prepare_conversion_data.py` — GPT Data Preparation (Legacy)

## Description

Prépare les fichiers d'entraînement et de validation pour le modèle GPT char-level.
Lit le fichier `mql_dataset_manual.jsonl`, formate chaque paire au format
`MQL4: ...\n\nMQL5: ...###`, mélange, split 90/10, et écrit `train.txt` et `val.txt`.

## Arguments

| Argument   | Type   | Requis | Défaut             | Description                     |
|------------|--------|--------|--------------------|---------------------------------|
| `--config | `str`  | Non    | `configs/gpt.yaml` | Chemin vers le fichier YAML   |

Paramètres lus depuis la config:
- `cfg.data.input_jsonl` — fichier d'entrée (`.jsonl`)
- `cfg.data.processed_dir` — dossier de sortie
- `cfg.data.train_file` — nom du fichier train
- `cfg.data.val_file` — nom du fichier val
- `cf.preprocessing.train_split` — proportion d'entrainement (défaut 0.9)
- `cfg.preprocessing.end_token` — token de fin (défaut `###`)

## Exemples

```bash
python src/scripts/prepare_conversion_data.py --config configs/gpt.yaml
```

## Dépendances

- **Données :** `data/raw/mql_dataset_manual.jsonl` (36 paires MQL4→MQL5)
- **Config :** `configs/gpt.yaml`
- **Sorties :** `data/processed/train.txt`, `data/processed/val.txt`

## Pipeline

```
prepare_conversion_data.py → train.py → convert_mql4.py
```