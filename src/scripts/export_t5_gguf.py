"""Export the trained T5-small checkpoint to a HuggingFace-style directory,
then (optionally) convert it to GGUF for ``llama.cpp`` using the official
``convert_hf_to_gguf.py`` script (from the llama.cpp repository).

The ``.pt`` checkpoint saved by ``src/train_seq2seq.py`` contains a state dict
whose keys already match HuggingFace T5 naming, so exporting is mostly a matter
of re-serializing tensors as ``model.safetensors`` plus writing ``config.json``
and the SentencePiece tokenizer files.

Usage (project root):
    python src/scripts/export_t5_gguf.py checkpoints/seq2seq_best.pt \\
        --config configs/seq2seq.yaml --output-dir exports/t5-mql

    # with automated GGUF conversion (llama.cpp repo checked out somewhere)
    python src/scripts/export_t5_gguf.py checkpoints/seq2seq_best.pt \\
        --gguf-converter "C:/llama.cpp/convert_hf_to_gguf.py" \\
        --quant f16 --output-dir exports/t5-mql
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

# Ajouter la racine du projet au path avant les imports locaux
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

parser = argparse.ArgumentParser(
    description='Export T5-small checkpoint to HuggingFace + GGUF (llama.cpp)'
)
parser.add_argument('checkpoint_path', help='Path to a .pt checkpoint from src/train_seq2seq.py')
parser.add_argument('--config', default='configs/seq2seq.yaml',
                    help='Path to YAML config file (default: configs/seq2seq.yaml)')
parser.add_argument('--output-dir', default='exports/t5-mql',
                    help='Directory where the HuggingFace-style model will be written')
parser.add_argument('--gguf-converter', default=None,
                    help='Absolute path to llama.cpp convert_hf_to_gguf.py. If set, GGUF '
                         'conversion is executed right after the HF export.')
parser.add_argument('--quant', default='f16',
                    help='GGUF quantization type (f16, q8_0, ...) passed to the converter.')
args = parser.parse_args()

cfg = load_config(args.config)
PROJECT_ROOT = cfg._project_root
sys.path.insert(0, PROJECT_ROOT)

from src.models.t5_small import T5Small
from src.tokenizers.mql_sp_tokenizer import PAD_IDX


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = T5Small(
        vocab_size=ckpt['vocab_size'],
        d_model=ckpt['d_model'],
        d_ff=ckpt['d_ff'],
        num_heads=ckpt['num_heads'],
        d_kv=ckpt['d_kv'],
        n_layer=ckpt['n_layer'],
        dropout=ckpt.get('dropout', 0.1),
        pad_idx=PAD_IDX,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt


def write_config(ckpt, out_dir):
    cfg_data = {
        'architectures': ['T5ForConditionalGeneration'],
        'model_type': 't5',
        'd_model': ckpt['d_model'],
        'd_ff': ckpt['d_ff'],
        'num_layers': ckpt['n_layer'],
        'num_heads': ckpt['num_heads'],
        'd_kv': ckpt['d_kv'],
        'vocab_size': ckpt['vocab_size'],
        'dropout_rate': ckpt.get('dropout', 0.1),
        'layer_norm_epsilon': 1e-06,
        'initializer_factor': 1.0,
        'relative_attention_num_buckets': 32,
        'relative_attention_max_distance': 128,
        'feed_forward_proj': 'relu',
        'is_encoder_decoder': True,
        'pad_token_id': PAD_IDX,
        'eos_token_id': 3,
        'decoder_start_token_id': 2,
        'tie_word_embeddings': True,
        'use_cache': False,
        'gradient_checkpointing': False,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_data, f, indent=2)


def write_tokenizer(out_dir):
    """Copy the SentencePiece model and write minimal HF tokenizer metadata."""
    sp_model = cfg.data.tokenizer_dir  # actually the dir; model is <dir>/sp.model
    sp_model_file = os.path.join(sp_model, 'sp.model')
    if not os.path.exists(sp_model_file):
        # tokenizer_dir may point directly at the model file in some configs
        sp_model_file = sp_model

    shutil.copy(sp_model_file, os.path.join(out_dir, 'tokenizer.model'))

    tokenizer_config = {
        'model_max_length': cfg.model.block_size,
        'pad_token': '<pad>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'bos_token': '<s>',
        'extra_special_tokens': {},
        'sp_model_kwargs': {},
        'tokenizer_class': 'T5Tokenizer',
    }
    with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        'pad_token': '<pad>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'bos_token': '<s>',
    }
    with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, indent=2)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(args.checkpoint_path, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Serialize weights as safetensors (keys already follow HF T5 naming).
    # Tied embeddings share memory across shared.weight / embed_tokens / lm_head;
    # clone them so safetensors sees disjoint tensors and llama.cpp can load
    # each HF key independently.
    from safetensors.torch import save_file

    tensors = {k: v.float() for k, v in model.state_dict().items()}
    shared = tensors.get('shared.weight')
    if shared is not None:
        for alias in ('encoder.embed_tokens.weight',
                      'decoder.embed_tokens.weight',
                      'lm_head.weight'):
            if alias in tensors:
                tensors[alias] = shared.clone()
    save_file(tensors, os.path.join(args.output_dir, 'model.safetensors'))

    write_config(ckpt, args.output_dir)
    write_tokenizer(args.output_dir)

    print(f"HF-style model written to {args.output_dir}")
    print("  - model.safetensors")
    print("  - config.json")
    print("  - tokenizer.model / tokenizer_config.json / special_tokens_map.json")

    if not args.gguf_converter:
        print("\nGGUF conversion skipped. Re-run with --gguf-converter <convert_hf_to_gguf.py> "
              "or run it manually:")
        print(f"  python {args.gguf_converter or '<llama.cpp>/convert_hf_to_gguf.py'} \\\n"
              f"      {args.output_dir} --outtype {args.quant} --outfile {args.output_dir}.gguf")
        return

    print(f"\nConverting to GGUF ({args.quant}) ...")
    result = subprocess.run(
        [sys.executable, args.gguf_converter, args.output_dir,
         '--outtype', args.quant, '--outfile', f"{args.output_dir}.gguf"],
        check=False,
    )
    if result.returncode != 0:
        print(f"convert_hf_to_gguf.py failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"GGUF model written to {args.output_dir}.gguf")


if __name__ == '__main__':
    main()
