# `data/raw/collect_mql_dataset.py` — GitHub Dataset Collection

## Description

Collecte des paires de fichiers MQL4/MQL5 depuis la recherche de code GitHub.
Recherche des fichiers `.mq4`/`.mq5`, apparie les fichiers par nom de base
(matching exact + flou via `SequenceMatcher`), les télécharge dans `collected_pairs/`,
et écrit un fichier `mql_dataset_collected.jsonl` avec les paires collectées.

**Nécessite une variable d'environnement `GITHUB_TOKEN`** pour l'authentification
à l'API GitHub.

## Arguments

| Argument     | Type       | Requis | Défaut | Description                                    |
|--------------|------------|--------|--------|------------------------------------------------|
| `--output-dir` | `str`    | Non    | `.`    | Répertoire de destination racine               |
| `--query`    | `str` (append) | Non | `None` | Requête GitHub code-search supplémentaire (répétable) |
| `--sleep`    | `float`    | Non    | `1.2`  | Secondes entre chaque appel API                |
| `--dry-run`  | `flag`     | Non    | `False`| Liste les paires sans télécharger              |

## Exemples

```bash
# Collecte dans le répertoire courant
python data/raw/collect_mql_dataset.py

# Collecte dans un répertoire spécifique
python data/raw/collect_mql_dataset.py --output-dir data/raw

# Simulation sans téléchargement
python data/raw/collect_mql_dataset.py --output-dir data/raw --dry-run

# Avec des requêtes supplémentaires
python data/raw/collect_mql_dataset.py --query "language:mql4" --query "language:mql5"
```

## Dépendances

- **Environnement :** variable `GITHUB_TOKEN` définie
- **Connexion :** accès Internet à l'API GitHub
- **Sorties :** `mql_dataset_collected.jsonl` + dossiers `collected_pairs/`