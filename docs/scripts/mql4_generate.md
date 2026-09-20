# `src/scripts/mql4_generate.py` — Synthetic MQL4 Data Generator

## Description

Génère 5000 snippets MQL4 synthétiques avec des indicateurs randomisés
(iMA, iRSI, iATR, iCCI), des conditions et des appels OrderSend. Les données sont
écrites dans `mql4_dataset.jsonl` dans le répertoire courant, au format
`{"text":"..."}` par ligne.

## Arguments

**Aucun argument.** Le script est autonome et ne prend pas de paramètres en ligne
de commande.

## Exemples

```bash
# Génération des données synthétiques
python src/scripts/mql4_generate.py
```

## Dépendances

- **Sortie :** `mql4_dataset.jsonl` (dans le répertoire courant)
- **Aucune config nécessaire**

## Pipeline

```
mql4_generate.py → mql4_convert.py
```