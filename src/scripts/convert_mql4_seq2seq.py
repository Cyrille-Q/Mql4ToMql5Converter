import argparse
import os
import pickle
import sys

import torch

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

parser = argparse.ArgumentParser(description='Convert MQL4 to MQL5 using T5-small')
parser.add_argument('checkpoint_path', help='Path to model checkpoint (.pt)')
parser.add_argument('mql4_input', nargs='?', default=None,
                    help='MQL4 code string or path to .mq4 file')
parser.add_argument('--config', default='configs/seq2seq.yaml',
                    help='Path to YAML config file (default: configs/seq2seq.yaml)')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
sys.path.insert(0, PROJECT_ROOT)

from src.models.t5_small import T5Small
from src.tokenizers.mql_sp_tokenizer import EOS_IDX, PAD_IDX, SOS_IDX

DATASET_FILE = cfg.data.output_pkl


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = T5Small(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        d_ff=checkpoint['d_ff'],
        num_heads=checkpoint['num_heads'],
        d_kv=checkpoint['d_kv'],
        n_layer=checkpoint['n_layer'],
        dropout=checkpoint.get('dropout', 0.1),
        pad_idx=PAD_IDX,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset['tokenizer']


def convert_mql4(model, tokenizer, mql4_code, device, max_new_tokens=500):
    enc_ids = tokenizer.encode(mql4_code.strip())
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        gen = model.generate(
            enc_tensor,
            max_new_tokens=max_new_tokens,
            eos_token=EOS_IDX,
            sos_token=SOS_IDX,
        )

    output = tokenizer.decode(gen[0].tolist())
    return output.strip()


if __name__ == '__main__':
    checkpoint_path = args.checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer(DATASET_FILE)

    model = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded: vocab={tokenizer.vocab_size}, device={device}")

    if args.mql4_input:
        input_arg = args.mql4_input
        if os.path.isfile(input_arg):
            with open(input_arg, 'r', encoding='utf-8') as f:
                mql4_code = f.read()
        else:
            mql4_code = input_arg
    else:
        mql4_code = """#property strict
extern int Period=14;
void OnTick(){
    double ma = iMA(Symbol(),0,Period,0,MODE_SMA,PRICE_CLOSE,0);
    if(Close[0] > ma) Print("above MA");
}"""

    print("\n=== Input MQL4 ===")
    print(mql4_code)

    mql5_code = convert_mql4(model, tokenizer, mql4_code, device)

    print("\n=== Generated MQL5 ===")
    print(mql5_code)
