import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.seq2seq import Seq2SeqTransformer
from src.tokenizers.mql_tokenizer import MQLTokenizer, EOS_IDX, SOS_IDX, PAD_IDX


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
    if len(sys.argv) < 2:
        print("Usage: python convert_mql4_seq2seq.py <checkpoint_path> [mql4_code_or_file]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATASET_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "seq2seq_dataset.pkl")
    tokenizer = load_tokenizer(DATASET_FILE)

    model, block_size = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded: vocab={tokenizer.vocab_size}, block_size={block_size}, device={device}")

    if len(sys.argv) > 2:
        input_arg = sys.argv[2]
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