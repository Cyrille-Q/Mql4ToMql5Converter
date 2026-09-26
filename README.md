# Mql4ToMql5Converter

Convertit automatiquement du code **MQL4** (MetaTrader 4) en **MQL5** (MetaTrader 5) grâce à un modèle de langage neuronal entraîné sur des paires de scripts équivalents.

Deux approches coexistent dans le dépôt :

| Approche | Type | Statut |
|---|---|---|
| **Seq2Seq token-level** | Transformer encodeur-décodeur (`src/models/seq2seq.py`) | Architecture actuelle, à utiliser |
| **GPT char-level** | GPT caractère (`src/models/gpt.py`) | Baseline legacy, conservée pour comparaison |

---

## Installation

Prérequis : Python ≥ 3.10, PyTorch (CUDA optionnel).

```bash
# Depuis la racine du projet
pip install -e .

# Pour le développement (ruff, mypy, pytest) :
pip install -e ".[dev]"
```

Le projet embarque un venv prêt à l'emploi : `.venv/` (sans les dépendances de dev).

---

## Données

Les paires d'entraînement sont au format JSONL avec les champs `mql4` et `mql5`.

| Fichier | Contenu | Utilisé par |
|---|---|---|
| `data/raw/mql_all_936.jsonl` | 936 paires MQL4→MQL5 | Pipeline Seq2Seq |
| `data/raw/mql_synthetic_900.jsonl` | 900 paires synthétiques | Réserve / augmentation |
| `data/raw/mql_dataset_manual.jsonl` | 36 paires contrôlées faites main | Pipeline GPT legacy |
| `data/raw/mql_dataset_collected.jsonl` | Collecte GitHub (vide) | — |
| `data/raw/*.mq4` / `*.mq5` | 72 fichiers d'exemple (36 paires, 10 catégories) | Lecture / validation |

---

## Pipeline Seq2Seq (recommandé)

```
mql_all_936.jsonl
        │  src/scripts/prepare_seq2seq_data.py   (tokenise + split 90/10, seed 42)
        ▼
data/processed/seq2seq_dataset.pkl   enc_inputs / dec_inputs / labels + tokenizer
        │  src/train_seq2seq.py
        ▼
checkpoints/seq2seq_best.pt
        │  src/scripts/convert_mql4_seq2seq.py
        ▼
MQL5 généré
```

### 1. Préparer les données

```bash
python src/scripts/prepare_seq2seq_data.py
```

Construit `data/processed/seq2seq_dataset.pkl` (entrées encodées, tokenizer, indices train/val).

### 2. Entraîner

```bash
# Utiliser la configuration par défaut
python src/train_seq2seq.py

# Ou spécifier une configuration personnalisée
python src/train_seq2seq.py --config configs/ma_config.yaml
```

Les hyperparamètres sont définis dans `configs/seq2seq.yaml` (modifiable).

| Paramètre clé | Valeur par défaut | Description |
|---|---|---|
| `model.n_embd` / `n_head` / `n_layer` | 64 / 4 / 4 | Architecture du Transformer |
| `model.block_size` | 1024 | Taille du contexte maximal |
| `training.batch_size` | 5 | Séquences en parallèle |
| `training.max_epochs` | 200 | Nombre d'epochs |
| `training.eval_interval` | 10 | Évaluer la loss toutes les N époques |
| `training.checkpoint_interval` | 10 | Sauvegarder un checkpoint toutes les N époques |
| `training.learning_rate` | 0.001 | Taux d'apprentissage (AdamW) |

Checkpoints : `checkpoints/seq2seq_epoch_NNNN.pt` (selon `training.checkpoint_interval`) et `checkpoints/seq2seq_best.pt` (meilleure loss de validation).

### 3. Convertir du MQL4

```bash
# À partir d'un fichier
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt mon_indicateur.mq4

# Ou directement depuis une chaîne
python src/scripts/convert_mql4_seq2seq.py checkpoints/seq2seq_best.pt "extern int Period=14; void OnTick() { double ma = iMA(Symbol(),0,Period,0,MODE_SMA,PRICE_CLOSE,0); }"
```

---

## Pipeline GPT char-level (legacy)

```
mql_dataset_manual.jsonl (36 paires)
        │  src/scripts/prepare_conversion_data.py   (format "MQL4: …\n\nMQL5: …###", split 90/10)
        ▼
data/processed/train.txt / val.txt
        │  src/train.py
        ▼
checkpoints/best_model.pt
        │  src/scripts/convert_mql4.py
        ▼
MQL5 généré
```

```bash
python src/scripts/prepare_conversion_data.py
python src/train.py
python src/scripts/convert_mql4.py checkpoints/best_model.pt mon_indicateur.mq4
```

---

## Structure du projet

```
src/
├── models/
│   ├── seq2seq.py          # Seq2SeqTransformer (encodeur-décodeur, attention croisée)
│   └── gpt.py              # GPT char-level (baseline)
├── tokenizers/
│   └── mql_tokenizer.py    # Tokenizer regex niveau token (~343 tokens, <PAD>/<SOS>/<EOS>/<UNK>)
├── utils/
│   ├── config.py           # Chargeur de configuration YAML + résolution device
│   └── metrics.py          # Export CSV/JSON de la loss + courbe matplotlib (optionnel)
├── train_seq2seq.py        # Entraînement Seq2Seq
├── train.py                # Entraînement GPT (legacy)
└── scripts/
    ├── prepare_seq2seq_data.py    # Tokenise les 936 paires
    ├── prepare_conversion_data.py # Prépare train.txt/val.txt (legacy)
    ├── convert_mql4_seq2seq.py    # Inférence Seq2Seq (CLI)
    ├── convert_mql4.py            # Inférence GPT (CLI)
    ├── debug_seq2seq_data.py      # Inspecte tokens/alignment du dataset Seq2Seq
    ├── inspect_seq2seq_vocab.py   # Rapport vocabulaire depuis le .pkl
    └── mql4_generate.py / mql4_convert.py  # Génération de données synthétiques (expérimental)
data/
├── raw/                    # Paires JSONL + fichiers .mq4/.mq5 d'exemple
└── processed/              # seq2seq_dataset.pkl, train.txt, val.txt
checkpoints/                # seq2seq_best.pt, seq2seq_epoch_*.pt (gitignoré)
configs/
├── seq2seq.yaml            # Configuration du pipeline Seq2Seq
├── gpt.yaml                # Configuration du pipeline GPT (legacy)
└── default.yaml            # Config T5 obsolète (non utilisée)
docs/scripts/               # Documentation par script (CLI, options, exemples)
```

> Note : aucune suite de tests n'est encore présente (pytest est configuré en dépendance dev, mais pas de dossier `tests/`).

---

## Configuration

Tous les scripts de préparation et d'entraînement lisent leurs hyperparamètres depuis un fichier YAML via le flag `--config` (chemin relatif à la racine du projet).

| Fichier | Pipeline | Scripts concernés |
|---|---|---|
| `configs/seq2seq.yaml` | Seq2Seq (recommandé) | `prepare_seq2seq_data.py`, `train_seq2seq.py`, `convert_mql4_seq2seq.py` |
| `configs/gpt.yaml` | GPT char-level (legacy) | `prepare_conversion_data.py`, `train.py`, `convert_mql4.py` |

Exemple d'utilisation avec une config personnalisée :

```bash
python src/train_seq2seq.py --config configs/seq2seq.yaml
```

Les sections du YAML :

- `seed` — seed unique pour `torch` et `random`
- `device` — `"auto"` (détection CUDA), `"cpu"` ou `"cuda"`
- `data.*` — chemins des données et des checkpoints
- `preprocessing.*` — split train/val, délimiteurs
- `model.*` — architecture (n_embd, n_head, n_layer, dropout, block_size)
- `training.*` — hyperparamètres d'entraînement (batch_size, lr, etc.)

Tous les chemins dans le YAML sont relatifs à la racine du projet et résolus automatiquement.

---

## Notes & limites

- Le split train/validation est aléatoire (seed 42) ; idéalement il faudrait un split par famille de code pour éviter les fuites.
- Les checkpoints `seq2seq_epoch_*.pt` (~78 Mo chacun) sont volumineux ; le dépôt utilise Git LFS.
- `configs/default.yaml` (config T5) est obsolète : utiliser `configs/seq2seq.yaml` ou `configs/gpt.yaml` à la place.
- Tous les scripts d'entraînement et de préparation acceptent `--config <chemin>` pour utiliser une configuration personnalisée.
- Documentation détaillée de chaque script (CLI, options, exemples) dans `docs/scripts/*.md`.
- Le suivi d'apprentissage (loss de train/val, checkpoints) est décrit dans `AGENTS.md`.

## Licence

GPL-3.0. Vérifier les licences des scripts collectés sur GitHub avant redistribution.
