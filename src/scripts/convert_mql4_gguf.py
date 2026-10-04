"""Inference of a converted T5 token-level MQL4->MQL5 model via llama.cpp.

Requires a GGUF model previously produced with ``export_t5_gguf.py`` (or the
equivalent manual ``convert_hf_to_gguf.py`` call), plus a locally built
llama.cpp binary supporting the T5 encoder-decoder architecture.

Two driver modes are supported:

* ``--llama-bin``: shell out to a llama.cpp CLI (e.g. ``llama-encoder-decoder``
  or ``llama-cli``).  The exact interface of these binaries evolves quickly;
  the ``--llama-args`` option lets you tune the generated command line.
* (default) ``--server``: drive a running ``llama-server`` instance over HTTP
  (``/completion`` endpoint).  Faster and more stable for interactive use.

Tokenization is done locally with the same SentencePiece model used at
training time, so the ids fed to the GGUF model are identical to PyTorch.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.tokenizers.mql_sp_tokenizer import MQLSPTokenizer


def load_spm(path):
    return MQLSPTokenizer(model_path=path)


def run_http(host, port, prompt, max_tokens):
    url = f"http://{host}:{port}/completion"
    payload = json.dumps({"prompt": prompt, "n_predict": max_tokens}).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))['content']


def run_cli(binary, args, prompt):
    cmd = [binary] + args + [prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp exited {result.returncode}:\n{result.stderr}")
    # llama-cli prints the continuation on stdout after the prompt.
    out = result.stdout.strip()
    if out.startswith(prompt):
        out = out[len(prompt):].strip()
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Convert MQL4 to MQL5 via a GGUF T5 model in llama.cpp'
    )
    parser.add_argument('--spm', required=True, help='Path to the trained tokenizer.model')
    parser.add_argument('mql4_input', nargs='?', default=None,
                        help='MQL4 code string or path to a .mq4 file')
    parser.add_argument('--max-tokens', type=int, default=500)
    parser.add_argument('--server', action='store_true',
                        help='Use llama-server HTTP mode (host:port)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--llama-bin', default='llama-encoder-decoder',
                        help='llama.cpp CLI binary (shell mode)')
    parser.add_argument('--llama-args', default='-m', help='Prefix args before the prompt '
                        '(e.g. "-m model.gguf"); the prompt is passed as the last arg.')
    args = parser.parse_args()

    sp = load_spm(args.spm)
    print(f"Tokenizer loaded: vocab size {sp.get_piece_size()}")

    if args.mql4_input:
        if os.path.isfile(args.mql4_input):
            with open(args.mql4_input, 'r', encoding='utf-8') as f:
                mql4 = f.read()
        else:
            mql4 = args.mql4_input
    else:
        mql4 = ("#property strict\nextern int Period=14;\nvoid OnTick(){\n"
                "double ma = iMA(Symbol(),0,Period,0,MODE_SMA,PRICE_CLOSE,0);\n"
                "if(Close[0] > ma) Print(\"above MA\");\n}")

    print(f"\n=== Input MQL4 ===\n{mql4}\n")

    # The SentencePiece model is also used by llama.cpp for T5; we pass the raw
    # source as the prompt and tokenization happens inside llama.cpp.
    if args.server:
        output = run_http(args.host, args.port, mql4, args.max_tokens)
    else:
        llm_args = (args.llama_args or '-m').split()
        output = run_cli(args.llama_bin, llm_args, mql4)

    print(f"\n=== Generated MQL5 ===\n{output.strip()}")


if __name__ == '__main__':
    main()
