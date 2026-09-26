import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.tokenizers.mql_tokenizer import SPECIAL_TOKENS

_reconfigure = getattr(sys.stdout, 'reconfigure', None)
if _reconfigure is not None:
    _reconfigure(encoding='utf-8')

DEFAULT_PKL = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'seq2seq_dataset.pkl')


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Seq2Seq vocabulary from .pkl dataset'
    )
    parser.add_argument(
        '--pkl',
        default=DEFAULT_PKL,
        help=f'Path to .pkl dataset (default: {DEFAULT_PKL})',
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pkl):
        print(f"Error: file not found — {args.pkl}", file=sys.stderr)
        sys.exit(1)

    with open(args.pkl, 'rb') as f:
        dataset = pickle.load(f)

    tokenizer = dataset['tokenizer']
    enc_inputs = dataset['enc_inputs']
    train_idx = dataset['train_idx']
    val_idx = dataset['val_idx']

    print('=' * 60)
    print('  Seq2Seq Dataset — Vocabulary Report')
    print('=' * 60)
    print(f'  Dataset       : {args.pkl}')
    print(f'  Vocab size    : {tokenizer.vocab_size}')
    print(f'  Total pairs   : {len(enc_inputs)}')
    print(f'  Train pairs   : {len(train_idx)}')
    print(f'  Val pairs     : {len(val_idx)}')
    print('=' * 60)
    print()
    print(f'{"Index":<8} {"Token":<40}')
    print(f'{"-" * 8} {"-" * 40}')

    for idx in range(tokenizer.vocab_size):
        tok = tokenizer.itos[idx]
        marker = ' (special)' if tok in SPECIAL_TOKENS else ''
        print(f'{idx:<8} {repr(tok):<40}{marker}')

    print()
    print('=' * 60)
    print(f'  Done — {tokenizer.vocab_size} tokens listed.')
    print('=' * 60)


if __name__ == '__main__':
    main()