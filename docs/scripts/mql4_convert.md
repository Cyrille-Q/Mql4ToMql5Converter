# `src/scripts/mql4_convert.py` — Synthetic Data Post-Processing

## Description

Convertit le fichier JSONL généré par `mql4_generate.py` (`mql4_dataset.jsonl`,
contenant des retours à la ligne échappés `\n`) en un fichier texte brut
`mql4_dataset.txt` en déséchappant les séquences `\n`. Simple convertisseur
JSONL → texte.

## Arguments

**Aucun argument.** Chemins d'entrée et sortie codés en dur :
- Entrée : `mql4_dataset.jsonl` (dans le répertoire courant)
- Sortie : `mql4_dataset.txt` (dans le répertoire courant)

## Exemples

```bash
python src/scripts/mql4_convert.py
```

## Dépendances

- **Entrée :** `mql4_dataset.jsonl` (généré par `src/scripts/mql4_generate.py`)
- **Sortie :** `mql4_dataset.txt`
- **Aucune config nécessaire**

## Pipeline

```
mql4_generate.py → mql4_convert.py
```