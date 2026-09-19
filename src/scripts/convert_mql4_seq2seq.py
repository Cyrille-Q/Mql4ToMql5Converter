import os
import sys
import pickle
import argparse
import torch

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

parser = argparse.ArgumentParser(description='Convert MQL4 to MQL5 using Seq2Seq')
parser.add_argument('checkpoint_path', help='Path to model checkpoint (.pt)')
parser.add_argument('mql4_input', nargs='?', default=None,
                    help='MQL4 code string or path to .mq4 file')
parser.add_argument('--config', default='configs/seq2seq.yaml',
                    help='Path to YAML config file (default: configs/seq2seq.yaml)')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
sys.path.insert(0, PROJECT_ROOT)

from src.models.seq2seq import Seq2SeqTransformer
from src.tokenizers.mql_tokenizer import MQLTokenizer, EOS_IDX, SOS_IDX, PAD_IDX


DATASET_FILE = cfg.data.output_pkl


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Seq2SeqTransformer(
        vocab_size=checkpoint['vocab_size'],
        n_embd=checkpoint['n_embd'],
        block_size=checkpoint['block_size'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        dropout=checkpoint.get('dropout', 0.2),
        pad_idx=PAD_IDX,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint['block_size']


def load_tokenizer(dataset_path):
    import pickle
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset['tokenizer']


def convert_mql4(model, tokenizer, mql4_code, device, max_new_tokens=500):
    enc_ids = tokenizer.encode(mql4_code.strip())
    enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        gen_ids = model.generate(
            enc_tensor,
            max_new_tokens=max_new_tokens,
            eos_token=EOS_IDX,
            sos_token=SOS_IDX,
        )

    output = tokenizer.decode(gen_ids)
    return output.strip()


if __name__ == '__main__':
    checkpoint_path = args.checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer(DATASET_FILE)

    model, block_size = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded: vocab={tokenizer.vocab_size}, block_size={block_size}, device={device}")

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

    print(f"\n=== Input MQL4 ===")
    print(mql4_code)

    mql5_code = convert_mql4(model, tokenizer, mql4_code, device)

    print(f"\n=== Generated MQL5 ===")
    print(mql5_code)