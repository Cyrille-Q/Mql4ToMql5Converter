"""SentencePiece tokenizer for MQL, following T5 id conventions (PAD=0).

SentencePiece is required for a clean export to GGUF/llama.cpp (which expects
a standard ``.model`` + the T5 special-token layout).  The piece indices for
<pad> / <unk> / <s> / </s> are fixed as 0/1/2/3 to match HF T5 expectations.

The object is pickled inside ``data/processed/t5_dataset.pkl`` alongside
the encoded sequences, so both the trainer and the inference CLI can reuse it.
"""

import json
import os
from typing import Optional

import sentencepiece as spm

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class MQLSPTokenizer:
    """Thin wrapper around a trained SentencePiece BPE model (T5 layout)."""

    def __init__(self, model_path=None, vocab_size=8000, model_type='bpe',
                 character_coverage=0.9995):
        self.model_path = str(model_path) if model_path else None
        self._vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self._sp: Optional[spm.SentencePieceProcessor] = None
        if model_path:
            self._sp = spm.SentencePieceProcessor(model_file=self.model_path)

    # -- training --------------------------------------------------
    def fit(self, texts, model_dir):
        """Train a BPE SentencePiece model on the given texts and load it."""
        os.makedirs(model_dir, exist_ok=True)
        corpus = os.path.join(model_dir, 'corpus.txt')
        with open(corpus, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')

        prefix = os.path.join(model_dir, 'sp')
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=prefix,
            model_type=self.model_type,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            pad_id=PAD_IDX, pad_piece='<pad>',
            unk_id=UNK_IDX, unk_piece='<unk>',
            bos_id=SOS_IDX, bos_piece='<s>',
            eos_id=EOS_IDX, eos_piece='</s>',
            train_extremely_large_corpus=True,
        )
        os.remove(corpus)
        self.model_path = prefix + '.model'
        self._sp = spm.SentencePieceProcessor(model_file=self.model_path)
        return self

    # -- interface (compatible with MQLTokenizer usage) ------------
    def encode(self, text):
        assert self._sp is not None, "tokenizer not fitted/loaded"
        return self._sp.encode(text, out_type=int)

    def decode(self, ids, skip_special=True):
        assert self._sp is not None, "tokenizer not fitted/loaded"
        ids = list(ids)
        if skip_special:
            ids = [i for i in ids if i not in (PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX)]
        return self._sp.decode(ids)

    def get_piece_size(self):
        return self._sp.get_piece_size() if self._sp else 0

    @property
    def vocab_size(self):
        return self._sp.get_piece_size() if self._sp else self._vocab_size

    def save(self, path):
        """Save a small JSON descriptor (load() method is symmetric)."""
        data = {
            'model_path': self.model_path,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'character_coverage': self.character_coverage,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path, model_base=None):
        """Load a tokenizer from a JSON descriptor written by save()."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_path = data['model_path']
        if model_base and not os.path.isabs(model_path):
            model_path = os.path.join(model_base, model_path)
        return cls(model_path=model_path,
                   vocab_size=data['vocab_size'],
                   model_type=data['model_type'],
                   character_coverage=data.get('character_coverage', 0.9995))


def find_special_ids(model_path):
    """Return (pad, unk, bos, eos) piece ids from a trained .model file."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp.piece_to_id('<pad>'), sp.piece_to_id('<unk>'), \
        sp.piece_to_id('<s>'), sp.piece_to_id('</s>')
