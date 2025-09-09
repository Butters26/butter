#!/usr/bin/env python3
"""
Export a HuggingFace GPT-2 style causal LM to Monday binary format.

Supports models with GPT-2 block structure:
 - transformer.wte (token embeddings)
 - transformer.wpe (positional embeddings)
 - transformer.h.{i}.attn.c_attn (combined QKV)
 - transformer.h.{i}.attn.c_proj (output projection)
 - transformer.h.{i}.ln_1, ln_2 (layer norms)
 - transformer.h.{i}.mlp.c_fc (FFN up), mlp.c_proj (FFN down)
 - transformer.ln_f (final norm)

Usage:
  python3 export_hf_to_monday.py --model gpt2 --out /workspace/monday_from_hf.bin
  python3 export_hf_to_monday.py --path /path/to/hf/ckpt --out /workspace/monday_from_hf.bin

Note: You need transformers and torch installed to run this exporter.
"""
import argparse
import json
import math
import os
import struct
import sys
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
except Exception as e:
    print("This exporter requires transformers and torch. Install them first.")
    raise


def write_tensor(f, name: str, tensor: np.ndarray, meta: Optional[Dict] = None):
    checksum = __import__('hashlib').sha256(tensor.tobytes()).hexdigest()
    metadata = {
        'name': name,
        'shape': tensor.shape,
        'dtype': str(tensor.dtype),
        'checksum': checksum
    }
    if meta is not None:
        metadata.update(meta)
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    f.write(struct.pack('<I', len(metadata_bytes)))
    f.write(metadata_bytes)
    f.write(tensor.tobytes())


def gpt2_to_monday(state: Dict[str, torch.Tensor], cfg: AutoConfig, out_path: str):
    hidden_size = int(cfg.n_embd)
    num_layers = int(cfg.n_layer)
    num_heads = int(cfg.n_head)
    max_seq_len = int(getattr(cfg, 'n_positions', 1024))
    vocab_size = int(cfg.vocab_size)
    head_size = hidden_size // num_heads

    # Header tensor count: 10 + 13*L
    num_tensors = 10 + 13 * num_layers

    traits = {
        'creativity': 1.4,
        'empathy': 1.5,
        'humor': 1.3,
        'honesty': 1.4,
        'safety': 0.7,
        'engagement': 1.3
    }
    notus = {
        'memory': 1.5,
        'context': 1.4,
        'learning': 1.3,
        'analytics': 1.4,
        'safety': 0.7
    }

    with open(out_path, 'wb') as f:
        header = {
            'magic': 'MOND',
            'version': 2,
            'num_tensors': num_tensors,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'traits': traits,
            'notus': notus
        }
        header_bytes = json.dumps(header).encode('utf-8')
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)

        # Embeddings
        wte = state['transformer.wte.weight'].detach().cpu().numpy().astype(np.float32)  # (V,C)
        write_tensor(f, 'tok_embedding', wte)
        write_tensor(f, 'output.weight', wte.T.copy())

        meta = {
            'vocab_size': int(vocab_size),
            'd_model': int(hidden_size),
            'num_layers': int(num_layers),
            'num_heads': int(num_heads),
            'head_size': int(head_size)
        }
        write_tensor(f, '_meta', np.array([0], dtype=np.float32), meta=meta)

        # Positional embeddings
        if 'transformer.wpe.weight' in state:
            wpe = state['transformer.wpe.weight'].detach().cpu().numpy().astype(np.float32)
            # Ensure shape is (max_seq_len, hidden_size)
            if wpe.shape[0] < max_seq_len:
                # pad if needed
                pad = np.zeros((max_seq_len - wpe.shape[0], hidden_size), dtype=np.float32)
                wpe = np.concatenate([wpe, pad], axis=0)
            elif wpe.shape[0] > max_seq_len:
                wpe = wpe[:max_seq_len]
            write_tensor(f, 'position_embeddings', wpe)
        else:
            # fallback sinusoidal if absent
            pos = np.zeros((max_seq_len, hidden_size), dtype=np.float32)
            for pos_idx in range(max_seq_len):
                for i in range(0, hidden_size, 2):
                    pos[pos_idx, i] = math.sin(pos_idx / (10000 ** (i / hidden_size)))
                    if i + 1 < hidden_size:
                        pos[pos_idx, i + 1] = math.cos(pos_idx / (10000 ** (i / hidden_size)))
            write_tensor(f, 'position_embeddings', pos)

        # Attention mask placeholder (causal mask for max_seq_len)
        mask = np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32))
        write_tensor(f, 'attention_mask', mask)

        # Embedding layer norm (GPT-2 has none; use identity)
        write_tensor(f, 'embedding_ln.weight', np.ones((hidden_size,), dtype=np.float32))
        write_tensor(f, 'embedding_ln.bias', np.zeros((hidden_size,), dtype=np.float32))

        # Layers
        for i in range(num_layers):
            prefix = f'transformer.h.{i}.'
            # Norms
            ln1_w = state[prefix + 'ln_1.weight'].detach().cpu().numpy().astype(np.float32)
            ln1_b = state[prefix + 'ln_1.bias'].detach().cpu().numpy().astype(np.float32)
            ln2_w = state[prefix + 'ln_2.weight'].detach().cpu().numpy().astype(np.float32)
            ln2_b = state[prefix + 'ln_2.bias'].detach().cpu().numpy().astype(np.float32)

            write_tensor(f, f'layer_{i}.attn_ln.gamma', ln1_w)
            write_tensor(f, f'layer_{i}.attn_ln.beta', ln1_b)
            write_tensor(f, f'layer_{i}.ffn_ln.gamma', ln2_w)
            write_tensor(f, f'layer_{i}.ffn_ln.beta', ln2_b)

            # Attention projections
            c_attn_w = state[prefix + 'attn.c_attn.weight'].detach().cpu().numpy().astype(np.float32)  # (C, 3C)
            # Split Q,K,V along last dim
            q_w, k_w, v_w = np.split(c_attn_w, 3, axis=1)
            write_tensor(f, f'layer_{i}.attn.query.weight', q_w)
            write_tensor(f, f'layer_{i}.attn.key.weight', k_w)
            write_tensor(f, f'layer_{i}.attn.value.weight', v_w)

            c_proj_w = state[prefix + 'attn.c_proj.weight'].detach().cpu().numpy().astype(np.float32)  # (C,C)
            write_tensor(f, f'layer_{i}.attn.output.weight', c_proj_w)

            # Bias for attn output (GPT-2 attn.c_proj has bias)
            if prefix + 'attn.c_proj.bias' in state:
                attn_b = state[prefix + 'attn.c_proj.bias'].detach().cpu().numpy().astype(np.float32)
            else:
                attn_b = np.zeros((hidden_size,), dtype=np.float32)
            write_tensor(f, f'layer_{i}.attn.output.bias', attn_b)

            # FFN
            up_w = state[prefix + 'mlp.c_fc.weight'].detach().cpu().numpy().astype(np.float32)  # (C,FF)
            up_b = state[prefix + 'mlp.c_fc.bias'].detach().cpu().numpy().astype(np.float32)    # (FF,)
            down_w = state[prefix + 'mlp.c_proj.weight'].detach().cpu().numpy().astype(np.float32)  # (FF,C)
            down_b = state[prefix + 'mlp.c_proj.bias'].detach().cpu().numpy().astype(np.float32)    # (C,)
            write_tensor(f, f'layer_{i}.ffn.up.weight', up_w)
            write_tensor(f, f'layer_{i}.ffn.up.bias', up_b)
            write_tensor(f, f'layer_{i}.ffn.down.weight', down_w)
            write_tensor(f, f'layer_{i}.ffn.down.bias', down_b)

        # Output norm
        ln_f_w = state['transformer.ln_f.weight'].detach().cpu().numpy().astype(np.float32)
        ln_f_b = state['transformer.ln_f.bias'].detach().cpu().numpy().astype(np.float32)
        write_tensor(f, 'output_ln.weight', ln_f_w)
        write_tensor(f, 'output_ln.bias', ln_f_b)

        # Output bias (if available)
        if 'lm_head.bias' in state:
            out_bias = state['lm_head.bias'].detach().cpu().numpy().astype(np.float32)
        else:
            out_bias = np.zeros((vocab_size,), dtype=np.float32)
        write_tensor(f, 'output.bias', out_bias)

    print(f"Export complete: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--model', type=str, help='HF model id (e.g., gpt2)')
    g.add_argument('--path', type=str, help='Local HF model directory')
    ap.add_argument('--out', type=str, required=True, help='Output Monday .bin path')
    args = ap.parse_args()

    model_id_or_path = args.model or args.path
    cfg = AutoConfig.from_pretrained(model_id_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
    state = hf_model.state_dict()
    gpt2_to_monday(state, cfg, args.out)


if __name__ == '__main__':
    main()

