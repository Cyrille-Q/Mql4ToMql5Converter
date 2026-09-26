import argparse
import os
import pickle
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config, resolve_device
from src.utils import metrics

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

parser = argparse.ArgumentParser(description='Train Seq2Seq model')
parser.add_argument('--config', default='configs/seq2seq.yaml',
                    help='Path to YAML config file (default: configs/seq2seq.yaml)')
parser.add_argument('--resume', default=None,
                    help='Path to a .pt checkpoint to resume training from')
parser.add_argument('--verbose', action='store_true',
                    help='Afficher le test de génération à chaque checkpoint'
                         ' (plus de détails à l\'écran)')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
sys.path.insert(0, PROJECT_ROOT)

from src.models.seq2seq import Seq2SeqTransformer
from src.tokenizers.mql_tokenizer import PAD_IDX, SOS_IDX, EOS_IDX

torch.manual_seed(cfg.seed)

# Paths
DATASET_FILE = cfg.data.output_pkl
CHECKPOINT_DIR = cfg.data.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
batch_size = cfg.training.batch_size
block_size = cfg.model.block_size
max_epochs = cfg.training.max_epochs
eval_interval = cfg.training.eval_interval
learning_rate = cfg.training.learning_rate
device = resolve_device(cfg)
n_embd = cfg.model.n_embd
n_head = cfg.model.n_head
n_layer = cfg.model.n_layer
dropout = cfg.model.dropout
eval_iters = cfg.training.eval_iters
checkpoint_interval = cfg.training.checkpoint_interval
NB_TOKENS = 80  # tokens générés vs attendus affichés lors du test verbose

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

# Training state (possibly overridden below via --resume)
start_epoch = 0
step = 0
best_val_loss = float('inf')

# Model + optimizer: if resuming, hyperparameters from the checkpoint take
# precedence over the config so the constructed model matches the checkpoint.
if args.resume:
    print(f"Loading checkpoint from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    n_embd = ckpt['n_embd']
    n_head = ckpt['n_head']
    n_layer = ckpt['n_layer']
    block_size = ckpt['block_size']
    dropout = ckpt['dropout']
    if ckpt['vocab_size'] != vocab_size:
        print(f"Warning: checkpoint vocab_size ({ckpt['vocab_size']}) "
              f"!= dataset vocab_size ({vocab_size})")
    model = Seq2SeqTransformer(vocab_size, n_embd, block_size, n_head, n_layer,
                               dropout, pad_idx=PAD_IDX).to(device)
    ckpt_pos = ckpt['model_state_dict']['position_embedding.weight'].shape[0]
    if ckpt_pos != model.position_embedding.num_embeddings:
        new_emb = nn.Embedding(ckpt_pos, model.position_embedding.embedding_dim,
                               device=model.position_embedding.weight.device)
        n_old = min(model.position_embedding.num_embeddings, ckpt_pos)
        new_emb.weight.data[:n_old] = model.position_embedding.weight.data[:n_old]
        model.position_embedding = new_emb
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        print("Warning: checkpoint has no optimizer_state_dict -> resuming with a fresh AdamW")
    start_epoch = ckpt.get('epoch', -1) + 1
    step = ckpt.get('step', 0)
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f"Resuming from epoch {start_epoch} (step {step}), "
          f"previous best val_loss {best_val_loss:.4f}")
else:
    model = Seq2SeqTransformer(vocab_size, n_embd, block_size, n_head, n_layer,
                               dropout, pad_idx=PAD_IDX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

@torch.no_grad()
def estimate_loss(split, eval_iters=eval_iters):
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
history = []

for epoch in range(start_epoch, max_epochs):
    # Shuffle training indices
    epoch_indices = torch.randperm(len(train_idx)).tolist()
    epoch_indices = [train_idx[i] for i in epoch_indices]

    model.train()
    epoch_loss = 0
    num_batches = 0

    batches: Any = range(0, len(epoch_indices), batch_size)
    if tqdm is not None:
        batches = tqdm(batches, desc=f"epoch {epoch:3d}", leave=False)

    for i in batches:
        batch_indices = epoch_indices[i:i + batch_size]
        enc_input, dec_input, lbl = collate_batch(batch_indices)
        enc_input, dec_input, lbl = enc_input.to(device), dec_input.to(device), lbl.to(device)

        _, loss = model(enc_input, dec_input, lbl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        step += 1
        if tqdm is not None:
            batches.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / num_batches

    # Summary train loss each epoch + eval (val loss) every eval_interval epochs
    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        avg_val_loss = estimate_loss('val')
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        history.append({'epoch': epoch, 'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss, 'step': step})
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

        # Test generation on a random training example (only in verbose mode)
        if args.verbose and len(train_idx) > 0:
            model.eval()
            test_idx = train_idx[torch.randint(len(train_idx), (1,)).item()]
            enc_tensor = torch.tensor([enc_inputs[test_idx]], dtype=torch.long, device=device)
            nb_exp = len(dec_inputs[test_idx]) - 1  # tokens attendus (hors SOS)
            max_start = max(nb_exp - NB_TOKENS, 0)
            start = int(torch.randint(max_start + 1, (1,)).item())
            gen_ids = model.generate(enc_tensor, max_new_tokens=start + NB_TOKENS,
                                     eos_token=EOS_IDX, sos_token=SOS_IDX)
            gen_tokens = gen_ids[1 + start:1 + start + NB_TOKENS]
            exp_tokens = dec_inputs[test_idx][1 + start:1 + start + NB_TOKENS]
            print(f"\n  === Generation test (epoch {epoch}, train example {test_idx}, "
                  f"start {start}) ===")
            print(f"  Generated: {tokenizer.decode(gen_tokens)}")
            print(f"  Expected : {tokenizer.decode(exp_tokens)}")
            print()
    else:
        print(f"epoch {epoch:3d} | train loss {avg_train_loss:.4f} | step {step}")

# Export metrics (loss history)
history_csv = os.path.join(CHECKPOINT_DIR, "seq2seq_loss_history.csv")
metrics.write_history_csv(history_csv, history, ['epoch', 'train_loss', 'val_loss', 'step'])
print(f"Loss history saved to {history_csv}")
history_json = os.path.join(CHECKPOINT_DIR, "seq2seq_loss_history.json")
metrics.write_history_json(history_json, history)
print(f"Loss history saved to {history_json}")
curve_path = os.path.join(CHECKPOINT_DIR, "seq2seq_loss_curve.png")
if metrics.plot_history(curve_path, history):
    print(f"Loss curve saved to {curve_path}")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")