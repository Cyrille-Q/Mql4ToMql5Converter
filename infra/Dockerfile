FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

# HOME obligatoire pour la compatibilité AI Training (OVHcloud)
WORKDIR /workspace
ENV HOME=/workspace
ENV PYTHONUNBUFFERED=1

# Code de l'application + config
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src

# Données brutes (jsonl) — la préparation régénère le pkl + tokenizer
COPY data/raw/mql_all_936.jsonl ./data/raw/mql_all_936.jsonl

# Dépendances manquantes dans l'image pytorch/pytorch de base
RUN pip install --break-system-packages "sentencepiece>=0.2" "safetensors>=0.4"

# Dossier de checkpoints accessible par l'utilisateur OVHcloud (UID 42420)
RUN mkdir -p /workspace/checkpoints && chown -R 42420:42420 /workspace

# Job : prépare les données (jsonl → pkl) puis entraîne le T5
CMD ["sh", "-c", "python src/scripts/prepare_t5_data.py --config configs/t5.yaml && python src/train_t5.py --config configs/t5.yaml"]
