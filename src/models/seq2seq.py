import torch
import torch.nn as nn
from torch.nn import functional as F


class EncoderHead(nn.Module):
    """Single head of self-attention (bidirectional, no causal mask)"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class DecoderHead(nn.Module):
    """Single head of causal self-attention (with tril mask)"""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        if T > self.tril.shape[0]:
            tril = torch.tril(torch.ones(T, T, device=x.device))
        else:
            tril = self.tril[:T, :T]
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class CrossHead(nn.Module):
    """Single head of cross-attention (query=decoder, key/value=encoder)"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, enc_mask=None):
        B, T_dec, C = x.shape
        B_enc, T_enc, C_enc = enc_out.shape
        k = self.key(enc_out)
        q = self.query(x)
        v = self.value(enc_out)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        if enc_mask is not None:
            wei = wei.masked_fill(enc_mask[:, None, :] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads in parallel"""

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(self.proj(x))
        return out


class EncoderMultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__(num_heads, head_size, n_embd, dropout)
        self.heads = nn.ModuleList([EncoderHead(n_embd, head_size, dropout) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return super().forward(out)


class DecoderMultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__(num_heads, head_size, n_embd, dropout)
        self.heads = nn.ModuleList([DecoderHead(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return super().forward(out)


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__(num_heads, head_size, n_embd, dropout)
        self.heads = nn.ModuleList([CrossHead(n_embd, head_size, dropout) for _ in range(num_heads)])

    def forward(self, x, enc_out, enc_mask=None):
        out = torch.cat([h(x, enc_out, enc_mask) for h in self.heads], dim=-1)
        return super().forward(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Transformer encoder block: bidirectional self-attention + FFWD"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = EncoderMultiHeadSelfAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block: causal self-attn + cross-attn + FFWD"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = DecoderMultiHeadSelfAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ca = MultiHeadCrossAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc_out, enc_mask=None):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), enc_out, enc_mask)
        x = x + self.ffwd(self.ln3(x))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, pad_idx=0):
        super().__init__()
        self.block_size = block_size
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.encoder = nn.Sequential(*[
            EncoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.decoder = nn.Sequential(*[
            DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _embed(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        if T > self.position_embedding.num_embeddings:
            new_emb = nn.Embedding(T, self.position_embedding.embedding_dim, device=x.device)
            n_old = self.position_embedding.num_embeddings
            new_emb.weight.data[:n_old] = self.position_embedding.weight.data
            self.position_embedding = new_emb
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        return tok_emb + pos_emb

    def encode(self, x):
        B, T = x.shape
        x = self._embed(x)
        x = self.encoder(x)
        return x

    def decode(self, x, enc_out, enc_mask=None):
        x = self._embed(x)
        for block in self.decoder:
            x = block(x, enc_out, enc_mask)
        x = self.ln_f(x)
        return x

    def forward(self, enc_input, dec_input, labels=None):
        enc_mask = (enc_input != self.pad_idx)
        enc_out = self.encode(enc_input)
        dec_out = self.decode(dec_input, enc_out, enc_mask)
        logits = self.lm_head(dec_out)

        if labels is None:
            return logits, None

        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        labels_flat = labels.view(B * T)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.pad_idx)
        return logits, loss

    @torch.no_grad()
    def generate(self, enc_input, max_new_tokens=300, eos_token=2, sos_token=1):
        self.eval()
        enc_mask = (enc_input != self.pad_idx)
        enc_out = self.encode(enc_input)
        B = enc_input.shape[0]
        dec_input = torch.full((B, 1), sos_token, dtype=torch.long, device=enc_input.device)

        for _ in range(max_new_tokens):
            logits = self.lm_head(self.decode(dec_input, enc_out, enc_mask))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_token], dim=-1)
            if next_token.item() == eos_token:
                break

        return dec_input[0].tolist()