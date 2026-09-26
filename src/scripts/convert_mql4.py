import os
import sys
import argparse
import torch

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

parser = argparse.ArgumentParser(description='Convert MQL4 to MQL5 using GPT')
parser.add_argument('checkpoint_path', help='Path to model checkpoint (.pt)')
parser.add_argument('mql4_input', nargs='?', default=None,
                    help='MQL4 code string or path to .mq4 file')
parser.add_argument('--config', default='configs/gpt.yaml',
                    help='Path to YAML config file (default: configs/gpt.yaml)')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from models import gpt

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    n_embd = checkpoint['n_embd']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    block_size = checkpoint['block_size']
    dropout = checkpoint.get('dropout', 0.2)
    
    model = gpt.GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Reconstruct vocab from checkpoint (need to reload from data)
    return model, block_size

def load_vocab(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, len(chars)

def convert_mql4(model, encode, decode, mql4_code, block_size, device, max_new_tokens=500, end_token=cfg.preprocessing.end_token):
    prompt = f"MQL4: {mql4_code.strip()}\n\nMQL5:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    # Crop if too long
    if context.size(1) > block_size:
        context = context[:, -block_size:]
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    
    output = decode(generated)
    
    # Extract only the MQL5 part after the prompt
    if "MQL5:" in output:
        output = output.split("MQL5:", 1)[1]
    
    # Stop at end token
    if end_token in output:
        output = output.split(end_token)[0] + end_token
    
    return output.strip()

if __name__ == '__main__':
    checkpoint_path = args.checkpoint_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load vocab from training data
    train_file = cfg.data.train_file
    encode, decode, vocab_size = load_vocab(train_file)

    # Load model
    model, block_size = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded: vocab={vocab_size}, block_size={block_size}, device={device}")

    # Get input MQL4 code
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

    mql5_code = convert_mql4(model, encode, decode, mql4_code, block_size, device)

    print(f"\n=== Generated MQL5 ===")
    print(mql5_code)