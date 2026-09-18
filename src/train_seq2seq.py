import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.seq2seq import Seq2SeqTransformer
from src.tokenizers.mql_tokenizer import PAD_IDX, SOS_IDX, EOS_IDX

torch.manual_seed(1337)

# Paths
DATASET_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "seq2seq_dataset.pkl")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
batch_size = 2
block_size = 512
max_epochs = 500
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.2

# Load dataset
with open(DATASET_FILE, 'rb') as f:
    dataset = pickle.load(f)

tokenizer = dataset['tokenizer']
enc_inputs = dataset['enc_inputs']
dec_inputs = dataset['dec_inputs']
labels = dataset['labels']
train_idx = dataset['train_idx']
val_idx = dataset['val_idx']

vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")
print(f"Train examples: {len(train_idx)}, Val examples: {len(val_idx)}")

def collate_batch(indices):
    enc = [torch.tensor(enc_inputs[i], dtype=torch.long) for i in indices]
    dec = [torch.tensor(dec_inputs[i], dtype=torch.long) for i in indices]
    lbl = [torch.tensor(labels[i], dtype=torch.long) for i in indices]
    enc_pad = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    dec_pad = pad_sequence(dec, batch_first=True, padding_value=PAD_IDX)
    lbl_pad = pad_sequence(lbl, batch_first=True, padding_value=PAD_IDX)
    return enc_pad, dec_pad, lbl_pad

# Model
model = Seq2SeqTransformer(vocab_size, n_embd, block_size, n_head, n_layer, dropout, pad_idx=PAD_IDX)
model = model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss(split, eval_iters=10):
    model.eval()
    indices = train_idx if split == 'train' else val_idx
    losses = []
    for _ in range(eval_iters):
        batch_indices = [indices[i] for i in torch.randint(len(indices), (batch_size,))]
        enc_input, dec_input, lbl = collate_batch(batch_indices)
        enc_input, dec_input, lbl = enc_input.to(device), dec_input.to(device), lbl.to(device)
        _, loss = model(enc_input, dec_input, lbl)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
step = 0

for epoch in range(max_epochs):
    # Shuffle training indices
    epoch_indices = torch.randperm(len(train_idx)).tolist()
    epoch_indices = [train_idx[i] for i in epoch_indices]

    model.train()
    epoch_loss = 0
    num_batches = 0

    for i in range(0, len(epoch_indices), batch_size):
        batch_indices = epoch_indices[i:i + batch_size]
        enc_input, dec_input, lbl = collate_batch(batch_indices)
        enc_input, dec_input, lbl = enc_input.to(device), dec_input.to(device), lbl.to(device)

        logits, loss = model(enc_input, dec_input, lbl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        step += 1

    avg_train_loss = epoch_loss / num_batches

    # Eval
    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        avg_val_loss = estimate_loss('val')
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"epoch {epoch:3d} | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | step {step}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"seq2seq_epoch_{epoch:04d}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout,
        }, checkpoint_path)

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "seq2seq_best.pt")
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'vocab_size': vocab_size,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'block_size': block_size,
                'dropout': dropout,
            }, best_path)
            print(f"  -> New best model! (val_loss: {best_val_loss:.4f})")

    # Test generation on val example every 50 epochs
    if epoch % 50 == 0 and len(val_idx) > 0:
        model.eval()
        test_idx = val_idx[0]
        enc_tensor = torch.tensor([enc_inputs[test_idx]], dtype=torch.long, device=device)
        gen_ids = model.generate(enc_tensor, max_new_tokens=300, eos_token=EOS_IDX, sos_token=SOS_IDX)
        generated = tokenizer.decode(gen_ids)
        expected = tokenizer.decode(dec_inputs[test_idx][1:] + [EOS_IDX])
        print(f"\n  === Generation test (epoch {epoch}) ===")
        print(f"  Generated (first 100 chars): {generated[:100]}")
        print(f"  Expected  (first 100 chars): {expected[:100]}")
        print()

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")