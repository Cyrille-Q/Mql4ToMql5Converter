# `src/scripts/prepare_seq2seq_data.py` — Seq2Seq Data Preparation

## Description

Prépare le dataset tokenizé pour l'entraînement Seq2Seq. Lit `mql_all_936.jsonl`,
instancie et fit le `MQLTokenizer` sur tous les textes, tokenize chaque paire
(encoder inputs, decoder inputs avec `<SOS>`, labels avec `<EOS>`), split
train/validation, et sauvegarde le tout dans un fichier `.pkl`.

## Arguments

| Argument   | Type   | Requis | Défaut               | Description                     |
|------------|--------|--------|----------------------|---------------------------------|
| `--config` | `str`  | Non    | `configs/seq2seq.yaml` | Chemin vers le fichier YAML   |

Paramètres lus depuis la config :
- `cfg.data.input_jsonl` — fichier d'entrée (`.jsonl`)
- `cfg.data.output_pkl` — fichier de sortie (`.pkl`)
- `cfg.preprocessing.train_split` — proportion d'entraînement
- `cfg.seed` — graine aléatoire pour le split

## Exemples

```bash
python src/scripts/prepare_seq2seq_data.py --config configs/seq2seq.yaml
```

## Dépendances

- **Données :** `data/raw/mql_all_936.jsonl` (936 paires MQL4↔MQL5)
- **Config :** `configs/seq2seq.yaml`
- **Tokenizer :** `src/tokenizers/mql_tokenizer.py` (MQLTokenizer)
- **Sortie :** `data/processed/seq2seq_dataset.pkl`

## Pipeline

```
prepare_seq2seq_data.py → train_seq2seq.py → convert_mql4_seq2seq.py
```