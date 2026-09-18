import json
import os
import sys
import random
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.tokenizers.mql_tokenizer import MQLTokenizer, SOS_IDX, EOS_IDX

random.seed(42)

INPUT_JSONL = os.path.join(PROJECT_ROOT, "data/raw/mql_dataset_manual.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/seq2seq_dataset.pkl")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load pairs
pairs = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        pairs.append((data['mql4'].strip(), data['mql5'].strip()))

print(f"Loaded {len(pairs)} pairs")

# Fit tokenizer on all MQL4 and MQL5 texts
all_texts = []
for mql4, mql5 in pairs:
    all_texts.append(mql4)
    all_texts.append(mql5)

tokenizer = MQLTokenizer()
tokenizer.fit(all_texts)
print(f"Vocab size: {tokenizer.vocab_size}")

# Tokenize
enc_inputs = []
dec_inputs = []
labels = []

for mql4, mql5 in pairs:
    enc_ids = tokenizer.encode(mql4)
    dec_ids = tokenizer.encode(mql5)
    enc_inputs.append(enc_ids)
    dec_inputs.append([SOS_IDX] + dec_ids)
    labels.append(dec_ids + [EOS_IDX])

# Train/val split (same seed 42, 90/10)
indices = list(range(len(pairs)))
random.shuffle(indices)
split_idx = int(0.9 * len(pairs))
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Check sequence lengths
enc_lens = [len(e) for e in enc_inputs]
dec_lens = [len(d) for d in dec_inputs]
print(f"Encoder input:  min={min(enc_lens)}, max={max(enc_lens)}, avg={sum(enc_lens)/len(enc_lens):.0f}")
print(f"Decoder input:  min={min(dec_lens)}, max={max(dec_lens)}, avg={sum(dec_lens)/len(dec_lens):.0f}")

# Save
dataset = {
    'tokenizer': tokenizer,
    'enc_inputs': enc_inputs,
    'dec_inputs': dec_inputs,
    'labels': labels,
    'train_idx': train_idx,
    'val_idx': val_idx,
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(dataset, f)

print(f"\nSaved to {OUTPUT_FILE}")