"""T5-small encoder-decoder conforming to HuggingFace T5 conventions.

The module is structured so that ``state_dict()`` keys exactly match the
HuggingFace ``google/t5-small`` naming (``encoder.block.{i}.layer.0.
SelfAttention.{q,k,v,o}``, ``decoder.block.{i}.layer.1.EncDecAttention.*``,
``shared.weight``, ``lm_head.weight`` ...).  This makes it straightforward to
export the trained checkpoint to GGUF for ``llama.cpp`` reusing the standard
HuggingFace ``convert_hf_to_gguf.py`` script.

Architecture (T5-small):
    d_model=512, d_ff=2048, num_heads=8, d_kv=64, 6 layers encoder + decoder,
    ReLU feed-forward, LayerNorm eps=1e-6, tied embeddings + lm_head, absolute
    positions replaced by T5 *relative position bias* (32 buckets).

Special tokens are passed explicitly (PAD/SOS/EOS come from the tokenizer).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Map a relative position matrix to bucket ids (identical to HF T5)."""
    num_buckets = num_buckets // 2 if bidirectional else num_buckets
    ret = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        ret[relative_position > 0] = num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)

    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class T5Attention(nn.Module):
    """Multi-head attention with optional (HF-style) relative position bias."""

    def __init__(self, d_model, num_heads, d_kv, dropout, has_relative_attention_bias):
        super().__init__()
        self.inner_dim = num_heads * d_kv
        self.n_heads = num_heads
        self.d_kv = d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.dropout = dropout

        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(32, self.n_heads)

    def _rel_embedding(self, relative_position, dtype, device):
        buckets = _relative_position_bucket(relative_position, bidirectional=True)
        bias = self.relative_attention_bias(buckets.to(device))  # (q_len, k_len, heads)
        bias = bias.permute([2, 0, 1]).unsqueeze(0).to(dtype)  # (1, heads, q_len, k_len)
        return bias

    def compute_bias(self, query_length, key_length, dtype, device):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        return self._rel_embedding(relative_position, dtype, device)

    def _shape(self, tensor, bsz, seq_len):
        return tensor.view(bsz, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None):
        bsz, q_len, _ = hidden_states.shape

        kv_states = hidden_states if key_value_states is None else key_value_states
        _, kv_len, _ = kv_states.shape

        q = self._shape(self.q(hidden_states), bsz, q_len)
        k = self._shape(self.k(kv_states), bsz, kv_len)
        v = self._shape(self.v(kv_states), bsz, kv_len)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(q_len, kv_len, q.dtype, q.device)
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads, q_len, kv_len),
                    device=q.device,
                    dtype=q.dtype,
                )

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_kv)
        scores = scores + position_bias

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores.float(), dim=-1).type_as(v)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        out = self.o(out)
        return out, position_bias


class T5LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(
            variance + self.variance_epsilon
        )
        return self.weight * x


class DenseReluDense(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        h = F.relu(self.wi(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.wo(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.DenseReluDense = DenseReluDense(d_model, d_ff, dropout)
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, x):
        return x + self.DenseReluDense(self.layer_norm(x))


class T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, dropout, has_relative_attention_bias):
        super().__init__()
        self.SelfAttention = T5Attention(
            d_model, num_heads, d_kv, dropout, has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, x, mask=None, position_bias=None):
        attn, position_bias = self.SelfAttention(x, mask=mask, position_bias=position_bias)
        return x + attn, position_bias


class T5LayerCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, dropout):
        super().__init__()
        self.EncDecAttention = T5Attention(
            d_model, num_heads, d_kv, dropout, has_relative_attention_bias=False
        )
        self.layer_norm = T5LayerNorm(d_model)

    def forward(self, x, kv, mask=None, position_bias=None):
        attn, position_bias = self.EncDecAttention(
            x, mask=mask, key_value_states=kv, position_bias=position_bias
        )
        return x + attn, position_bias


class T5Block(nn.Module):
    """A single encoder or decoder block (module-level container)."""

    def __init__(self, d_model, d_ff, num_heads, d_kv, dropout, has_relative_attention_bias,
                 is_decoder):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                d_model, num_heads, d_kv, dropout, has_relative_attention_bias
            )
        )
        if is_decoder:
            self.layer.append(T5LayerCrossAttention(d_model, num_heads, d_kv, dropout))
        self.layer.append(T5LayerFF(d_model, d_ff, dropout))


class T5Stack(nn.Module):
    """Encoder or decoder stack; ``embed_tokens`` is shared with the other stack."""

    def __init__(self, vocab_size, d_model, d_ff, num_heads, d_kv, n_layer, dropout,
                 is_decoder, pad_idx):
        super().__init__()
        self.is_decoder = is_decoder
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.block = nn.ModuleList([
            T5Block(
                d_model, d_ff, num_heads, d_kv, dropout,
                has_relative_attention_bias=(i == 0),
                is_decoder=is_decoder,
            )
            for i in range(n_layer)
        ])
        self.final_layer_norm = T5LayerNorm(d_model)


class T5Small(nn.Module):
    """T5-small encoder-decoder with HuggingFace-compatible state dict keys.

    Args:
        vocab_size: number of tokens.
        d_model: hidden size (512 for T5-small).
        d_ff: feed-forward hidden size (2048 for T5-small).
        num_heads: attention heads (8).
        d_kv: head dimension (64).
        n_layer: number of encoder AND decoder layers (6).
        dropout: dropout rate (0.1 by default).
        pad_idx: padding token id (used to build masks and ignore index).
    """

    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_heads=8, d_kv=64,
                 n_layer=6, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self._d_model = d_model
        self._d_ff = d_ff
        self._heads = num_heads
        self._d_kv = d_kv
        self._n_layer = n_layer
        self._dropout = dropout

        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5Stack(
            vocab_size, d_model, d_ff, num_heads, d_kv, n_layer, dropout,
            is_decoder=False, pad_idx=pad_idx,
        )
        self.decoder = T5Stack(
            vocab_size, d_model, d_ff, num_heads, d_kv, n_layer, dropout,
            is_decoder=True, pad_idx=pad_idx,
        )
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight

    # ── core forward ──────────────────────────────────────────────
    def forward(self, enc_input, dec_input, labels=None):
        encoder_outputs = self._encode(enc_input)
        logits = self._run_decoder(dec_input, encoder_outputs, enc_input)

        if labels is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.pad_idx,
        )
        return logits, loss

    def _encode(self, enc_input):
        encoder_hidden = self.shared(enc_input)
        encoder_hidden = F.dropout(encoder_hidden, p=self._dropout, training=self.training)

        position_bias = None
        for block in self.encoder.block:
            hidden_states, position_bias = block.layer[0](
                encoder_hidden, mask=None, position_bias=position_bias
            )
            hidden_states = block.layer[1](hidden_states)
            encoder_hidden = hidden_states

        encoder_hidden = self.encoder.final_layer_norm(encoder_hidden)
        return encoder_hidden

    def _run_decoder(self, dec_input, encoder_outputs, enc_input):
        decoder_hidden = self.shared(dec_input)
        seq_len = decoder_hidden.shape[1]
        causal = _make_causal_mask(seq_len, decoder_hidden.device, decoder_hidden.dtype)
        enc_pad_mask = _expand_mask(enc_input != self.pad_idx, decoder_hidden.dtype)

        position_bias = None
        for block in self.decoder.block:
            decoder_hidden, position_bias = block.layer[0](
                decoder_hidden, mask=causal, position_bias=position_bias
            )
            decoder_hidden, _ = block.layer[1](
                decoder_hidden, kv=encoder_outputs, mask=enc_pad_mask, position_bias=None
            )
            decoder_hidden = block.layer[2](decoder_hidden)

        decoder_hidden = self.decoder.final_layer_norm(decoder_hidden)
        return self.lm_head(decoder_hidden)

    # ── generation (greedy auto-regression) ───────────────────────
    @torch.no_grad()
    def generate(self, enc_input, max_new_tokens=300, eos_token=2, sos_token=1):
        """Greedy generation from a batch of encoder inputs (returns token ids)."""
        self.eval()
        encoder_outputs = self._encode(enc_input)
        bsz = enc_input.shape[0]
        dec_input = torch.full((bsz, 1), sos_token, dtype=torch.long, device=enc_input.device)

        for _ in range(max_new_tokens):
            logits = self._run_decoder(dec_input, encoder_outputs, enc_input)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_token], dim=-1)
            if next_token.squeeze(-1).tolist() == [eos_token]:
                break

        return dec_input


def _expand_mask(mask, dtype):
    """Boolean keep mask (B, src_len) -> additive mask (B, 1, 1, src_len) in logits space.

    Broadcasts against the attention scores of shape (B, heads, tgt_len, src_len).
    """
    expanded = mask.unsqueeze(1).unsqueeze(1).to(dtype)
    return (1.0 - expanded) * torch.finfo(dtype).min


def _make_causal_mask(seq_len, device, dtype):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return (1.0 - mask) * torch.finfo(dtype).min
