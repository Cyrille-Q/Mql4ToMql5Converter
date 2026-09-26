import re
import json
from pathlib import Path

SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

TOKEN_PATTERN = re.compile(
    r'[a-zA-Z_]\w*'       # keywords & identifiers
    r'|0x[0-9a-fA-F]+'    # hex literals (0xFF)
    r'|\d'                # individual digits (digit-level numbers)
    r'|"[^"]*"'            # double-quoted strings
    r"|'[^']*'"            # single-quoted strings
    r'|//'                # single-line comment marker
    r'|/\*'               # block comment open
    r'|\*/'               # block comment close
    r'|\s+'               # whitespace (preserves formatting)
    r'|\S'                # any other non-whitespace char
)


class MQLTokenizer:
    def __init__(self):
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        self._fitted = False

    def fit(self, texts):
        tokens = set()
        for text in texts:
            for match in TOKEN_PATTERN.finditer(text):
                tokens.add(match.group())
        for tok in sorted(tokens):
            if tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok
        self._fitted = True

    def encode(self, text):
        ids = []
        for match in TOKEN_PATTERN.finditer(text):
            tok = match.group()
            ids.append(self.stoi.get(tok, UNK_IDX))
        return ids

    def decode(self, ids, skip_special=True):
        tokens = []
        for i in ids:
            tok = self.itos.get(i, '<UNK>')
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return ''.join(tokens)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def save(self, path):
        data = {
            'stoi': self.stoi,
            'itos': {int(k): v for k, v in self.itos.items()},
            '_fitted': self._fitted,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data['stoi']
        tok.itos = {int(k): v for k, v in data['itos'].items()}
        tok._fitted = data['_fitted']
        return tok