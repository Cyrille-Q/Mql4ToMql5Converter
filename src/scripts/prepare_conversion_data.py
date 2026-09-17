import json
import os
import random

random.seed(42)

INPUT_JSONL = "data/raw/mql_dataset_manual.jsonl"
OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.txt")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.txt")

END_TOKEN = "###"

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

split_idx = int(0.9 * len(pairs))
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