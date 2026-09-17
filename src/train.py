# let's now encode the entire text dataset and store it into a torch.Tensor
import os
#import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from GPT import gpt

torch.manual_seed(1337) # Random number
#sys.path.append(os.path.dirname(__file__))
def load_and_encode(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode


def get_batch(split, batch_size, block_size):
    """Generate a batch of training data"""

    #batch_size = 4 # how many independent sequences will we process in parallel?
    #A larger batch size means more samples are processed simultaneously, which can help with faster training and provide more stable gradient estimates, but requires more memory.

    #block_size = 8 # what is the maximum context length for predictions?
    #The model will consider only the most recent block_size tokens as the context for generating the next token prediction.
    #For example, if block_size = 8, each sequence in the batch will have a context window of 8 tokens, allowing the model to “see” up to 8 tokens at a time. This limits the amount of text the model considers, which can be useful for controlling memory use and focusing the model on recent context when predicting the next token.    

    data = train_data if split == 'train' else val_data

    # Pick random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack sequences into batches
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


if __name__ == '__main__':

    # Configuration des chemins
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "train.txt")
    VAL_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "val.txt")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Hyperparameters
    batch_size = 2
    block_size = 512
    max_iters = 5000
    eval_interval = 50
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 10
    n_embd = 128
    n_head = 4
    n_layer = 3
    dropout = 0.2

    END_TOKEN = "###"

    train_data, vocab_size, encode, decode = load_and_encode(TRAIN_FILE)
    val_data, _, _, _ = load_and_encode(VAL_FILE)
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab: {''.join(sorted(set(decode(list(range(vocab_size))))))}")

    # Get a batch
    xb, yb = get_batch('train', batch_size, block_size)

    print(f"Input batch shape: {xb.shape}")  # (batch_size, block_size)
    print(f"Target batch shape: {yb.shape}")
    print(f"\nFirst sequence in batch:")
    print(f"Input: {decode(xb[0].tolist())!r}")
    print(f"Target: {decode(yb[0].tolist())!r}")

    # Create model
    model = gpt.GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split, batch_size, block_size)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Sauvegarde checkpoint régulier
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_iter_{iter:06d}.pt"
            )
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': losses['train'].item() if hasattr(losses['train'], 'item') else losses['train'],
                'val_loss': losses['val'].item() if hasattr(losses['val'], 'item') else losses['val'],
                'vocab_size': vocab_size,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'block_size': block_size,
                'batch_size': batch_size,
                'dropout': dropout,
            }, checkpoint_path)
            print(f"    -> Checkpoint saved: {checkpoint_path}")

            # Sauvegarde du meilleur modèle si val_loss s'améliore
            current_val_loss = losses['val'].item() if hasattr(losses['val'], 'item') else losses['val']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                    'vocab_size': vocab_size,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                }, best_model_path)
                print(f"    -> New best model saved! (val_loss: {best_val_loss:.4f})")

        # Get batch
        xb, yb = get_batch('train', batch_size, block_size)

        # Forward & backward
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Test generation with a prompt
    test_prompt = "MQL4: #property strict\nextern int Period=14;\nvoid OnTick(){}\n\nMQL5:"
    context = torch.tensor([encode(test_prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    output = decode(generated)
    # Stop at end token
    if END_TOKEN in output:
        output = output.split(END_TOKEN)[0] + END_TOKEN
    print("\n=== Generation test ===")
    print(output)    