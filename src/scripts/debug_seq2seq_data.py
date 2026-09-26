import os
import sys
import json
import pickle
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config
from src.tokenizers.mql_tokenizer import SPECIAL_TOKENS

parser = argparse.ArgumentParser(description='Debug Seq2Seq dataset: inspect tokenized pairs')
parser.add_argument('--config', default='configs/seq2seq.yaml',
                    help='Path to YAML config file (default: configs/seq2seq.yaml)')
parser.add_argument('--index', type=int, default=None,
                    help='Show a single pair by index')
parser.add_argument('--range', type=int, nargs=2, default=None, metavar=('START', 'END'),
                    help='Show pairs from START to END (inclusive)')
parser.add_argument('--max', type=int, default=3,
                    help='Maximum number of pairs to show (default: 3)')
parser.add_argument('--all', action='store_true',
                    help='Show all pairs (overrides --max)')
parser.add_argument('--mode', choices=['enc', 'dec', 'labels', 'all'], default='all',
                    help='Which sequences to display (default: all)')
parser.add_argument('--show-code', action='store_true',
                    help='Show raw MQL4/MQL5 code strings')
parser.add_argument('--show-meta', action='store_true',
                    help='Show category and complexity metadata')
parser.add_argument('--raw-ids', action='store_true',
                    help='Show raw ID lists in addition to detailed table')
parser.add_argument('--alignment', action='store_true',
                    help='Show teacher-forcing alignment (dec_inputs vs labels side-by-side)')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
sys.path.insert(0, PROJECT_ROOT)

PKL_PATH = os.path.join(cfg.data.processed_dir, cfg.data.output_pkl)
JSONL_PATH = os.path.join(cfg.data.raw_dir, cfg.data.input_jsonl)

SPECIAL_NAMES = {
    '<PAD>': 'PAD',
    '<SOS>': 'SOS',
    '<EOS>': 'EOS',
    '<UNK>': 'UNK',
}

def load_metadata(jsonl_path):
    metadata = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            metadata.append({
                'category': data.get('category', '?'),
                'complexity': data.get('complexity', '?'),
                'mql4': data['mql4'].strip(),
                'mql5': data['mql5'].strip(),
            })
    return metadata

def is_whitespace(tok):
    return tok in (' ', '\n', '\t', '\r') or (tok and all(c in ' \n\t\r' for c in tok))

def format_token_cell(tok):
    if tok in SPECIAL_TOKENS:
        name = SPECIAL_NAMES.get(tok, tok)
        return f"<{name}>"
    if is_whitespace(tok):
        return repr(tok)
    return tok

def print_separator(char='=', width=60):
    print(char * width)

def print_header(label, pair_idx, count, is_train=None, meta=None):
    status = ''
    if is_train is not None:
        status = ' [train]' if is_train else ' [val]'
    print()
    print_separator('=')
    label_line = f" Pair #{pair_idx}{status} "
    if meta:
        label_line += f"(cat: {meta['category']}, complexity: {meta['complexity']})"
    print(label_line)
    print_separator('=')
    print(f" {label} — {count} tokens")
    print()

def print_code(label, code):
    print(f"-- Raw {label} --")
    print(code)
    print()

def print_token_table(tokenizer, ids, show_raw_ids=False):
    print(f" {'idx':>4} | {'token':<22} | repr")
    print(f" {'-'*4}-+-{'-'*22}-+-{'-'*20}")

    for i, tid in enumerate(ids):
        tok = tokenizer.itos.get(tid, '<UNK>')
        cell = format_token_cell(tok)

        is_special = tok in SPECIAL_TOKENS
        suffix = '  ** SPECIAL' if is_special else f"  {repr(tok)}"
        print(f" {tid:>4} | {cell:<22} | {suffix}")

    print(f" ---- count: {len(ids)} tokens ----")

    if show_raw_ids:
        print(f" IDs: {ids}")
    print()

def format_short(tok, max_len=20):
    cell = format_token_cell(tok)
    if len(cell) > max_len:
        cell = cell[:max_len-3] + '...'
    return cell

def print_alignment(tokenizer, dec_ids, lab_ids):
    print(" Alignment (teacher forcing) : chaque position du decodeur predit la position suivante")
    print(f" {'Pos':>4} | {'dec_input (entre)':<28} | {'labels (cible)':<28} | note")
    print(f" {'-'*4}-+-{'-'*28}-+-{'-'*28}-+-{'-'*20}")

    max_rows = max(len(dec_ids), len(lab_ids))
    for pos in range(max_rows):
        dec_tok = tokenizer.itos.get(dec_ids[pos], '<UNK>') if pos < len(dec_ids) else ''
        dec_cell = format_short(dec_tok)
        dec_suffix = ' ** SPECIAL' if dec_tok in SPECIAL_TOKENS else ''

        lab_tok = tokenizer.itos.get(lab_ids[pos], '<UNK>') if pos < len(lab_ids) else ''
        lab_cell = format_short(lab_tok)
        lab_suffix = ' ** SPECIAL' if lab_tok in SPECIAL_TOKENS else ''

        note = ''
        if pos == 0:
            note = '<-- amorce'
        elif pos == len(lab_ids) - 1 and lab_tok in SPECIAL_TOKENS:
            note = '<-- fin'

        print(f" {pos:>4} | {dec_cell:<28} | {lab_cell:<28} | {note}")

    print()
    print(" Schema :")
    print("   dec_input = [<SOS>,       mql5_ids              ]")
    print("   labels    = [           mql5_ids,       <EOS>   ]")
    print("                dec_input[pos] predit labels[pos] a chaque position")
    print()

def print_legend():
    print(" Legende des sequences :")
    print("   enc_inputs  = [tokens MQL4 bruts]                <- pas de <SOS>/<EOS>")
    print("   dec_inputs  = [<SOS>, tokens MQL5]              <- entree du decodeur")
    print("   labels      = [tokens MQL5, <EOS>]              <- cible decalee (+1)")
    print("   Les tokens <PAD> (id=0) sont ajoutes au moment du batching uniquement")
    print()

print(f"Loading dataset from: {PKL_PATH}")
with open(PKL_PATH, 'rb') as f:
    dataset = pickle.load(f)

tokenizer = dataset['tokenizer']
enc_inputs = dataset['enc_inputs']
dec_inputs = dataset['dec_inputs']
labels = dataset['labels']
train_idx = set(dataset['train_idx'])

print(f"Loaded {len(enc_inputs)} pairs, vocab_size={tokenizer.vocab_size}")
print(f"Train: {len(train_idx)} / Val: {len(dataset['val_idx'])}")
print()

if args.alignment:
    print_legend()

metadata = load_metadata(JSONL_PATH) if args.show_meta else None

# Determine which indices to show
if args.index is not None:
    indices = [args.index]
elif args.range is not None:
    start, end = args.range
    indices = list(range(start, end + 1))
elif args.all:
    indices = list(range(len(enc_inputs)))
else:
    indices = list(range(min(args.max, len(enc_inputs))))

# Validate indices
for i in indices:
    if i < 0 or i >= len(enc_inputs):
        print(f"Warning: index {i} is out of range (0-{len(enc_inputs)-1}), skipping")
        continue

    is_train = i in train_idx
    meta = metadata[i] if metadata else None

    if args.show_code:
        print_header(f"Encoder Input (MQL4)", i, len(enc_inputs[i]), is_train, meta)
        print_code("MQL4", meta['mql4'] if meta else '')

    if args.mode in ('enc', 'all'):
        print_header(f"Encoder Input (MQL4)", i, len(enc_inputs[i]), is_train, meta)
        print_token_table(tokenizer, enc_inputs[i], args.raw_ids)

    if args.show_code:
        print()
        print_separator('-')
        print(f" Raw MQL5 code")
        print_separator('-')
        print_code("MQL5", meta['mql5'] if meta else '')

    if args.mode in ('dec', 'all'):
        print_header(f"Decoder Input (SOS + MQL5)", i, len(dec_inputs[i]), is_train, meta)
        print_token_table(tokenizer, dec_inputs[i], args.raw_ids)

    if args.mode in ('labels', 'all'):
        print_header(f"Labels (MQL5 + EOS)", i, len(labels[i]), is_train, meta)
        print_token_table(tokenizer, labels[i], args.raw_ids)

    if args.alignment and args.mode in ('dec', 'labels', 'all'):
        print_separator('-')
        print(f" Decoder alignment (pair #{i})")
        print_separator('-')
        print_alignment(tokenizer, dec_inputs[i], labels[i])