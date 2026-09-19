import json
import os
import sys
import random
import argparse

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

parser = argparse.ArgumentParser(description='Prepare GPT conversion data')
parser.add_argument('--config', default='configs/gpt.yaml',
                    help='Path to YAML config file (default: configs/gpt.yaml)')
args = parser.parse_args()

cfg = load_config(args.config)

random.seed(cfg.seed)

INPUT_JSONL = cfg.data.input_jsonl
OUTPUT_DIR = cfg.data.processed_dir
TRAIN_FILE = os.path.join(OUTPUT_DIR, os.path.basename(cfg.data.train_file))
VAL_FILE = os.path.join(OUTPUT_DIR, os.path.basename(cfg.data.val_file))

END_TOKEN = cfg.preprocessing.end_token

os.makedirs(OUTPUT_DIR, exist_ok=True)

pairs = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        mql4 = data['mql4'].strip()
        mql5 = data['mql5'].strip()
        formatted = f"MQL4: {mql4}\n\nMQL5: {mql5}{END_TOKEN}"
        pairs.append(formatted)

random.shuffle(pairs)

split_idx = int(cfg.preprocessing.train_split * len(pairs))
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for p in train_pairs:
        f.write(p + "\n")

with open(VAL_FILE, 'w', encoding='utf-8') as f:
    for p in val_pairs:
        f.write(p + "\n")

print(f"Total examples: {len(pairs)}")
print(f"Train: {len(train_pairs)} -> {TRAIN_FILE}")
print(f"Val: {len(val_pairs)} -> {VAL_FILE}")
print(f"End token: '{END_TOKEN}'")
print(f"Vocab will include: {set(''.join(pairs))}")