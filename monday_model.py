#!/usr/bin/env python3
import numpy as np
import struct
import os
import gc
import math
from typing import Dict, List, Tuple, Optional, Union
import json
import hashlib
from dataclasses import dataclass
import logging

@dataclass
class ModelResponse:
    """Standardized response format for model outputs"""
    text: str
    tokens: List[int]
    logprobs: List[float]
    finish_reason: str

@dataclass
class ModelConfig:
    """Configuration for model generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 100
    stop_tokens: List[str] = None
    
class LayerNorm:
    def __init__(self, size: int, eps: float = 1e-5):
        self.size = size
        self.eps = eps
        self.gamma = np.ones(size, dtype=np.float32)
        self.beta = np.zeros(size, dtype=np.float32)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization with numerical stability"""
        x = x.astype(np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Clip variance to prevent division by zero
        var = np.maximum(var, self.eps)
        
        # Normalize and scale
        x_norm = (x - mean) / np.sqrt(var)
        return (self.gamma * x_norm + self.beta).astype(np.float32)

class MondayModel:
    def __init__(self):
        # Setup logging early
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MondayModel")

        # Default model architecture
        self.hidden_size = 1024
        self.num_layers = 12
        self.num_heads = 16
        self.vocab_size = 32000
        self.max_seq_len = 2048
        self.eps = 1e-6
        self.dropout_rate = 0.1
        self.ffn_hidden_size = 4 * self.hidden_size

        # Optional tiny mode for quick testing (enabled by default)
        tiny_flag = os.getenv('MONDAY_TINY', '1')
        if str(tiny_flag).lower() in ('1', 'true', 'yes', 'on'):
            self.hidden_size = 128
            self.num_layers = 2
            self.num_heads = 4
            self.vocab_size = 2048
            self.max_seq_len = 128
            self.ffn_hidden_size = 4 * self.hidden_size
            self.logger.info("MONDAY_TINY enabled: hidden_size=%d, layers=%d, heads=%d, vocab=%d, max_seq=%d",
                             self.hidden_size, self.num_layers, self.num_heads, self.vocab_size, self.max_seq_len)
        
        self.head_size = self.hidden_size // self.num_heads
        
        # Initialize weights dict
        self.weights = {}
        self.loaded = False
        self.header_info: Dict[str, Union[int, str, Dict]] = {}
        self.num_tensors_declared: int = 0
        self.num_tensors_loaded: int = 0
        
        # Monday's personality traits (now used for output biasing)
        self.traits = {
            'creativity': 1.4,
            'empathy': 1.5,
            'humor': 1.3,
            'honesty': 1.4,
            'safety': 0.7,
            'engagement': 1.3
        }
        
        # Notus framework components (now used for attention biasing)
        self.notus = {
            'memory': 1.5,
            'context': 1.4,
            'learning': 1.3,
            'analytics': 1.4,
            'safety': 0.7
        }

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax with numerical stability"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def init_weight(self, shape: Tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        """Initialize weights using scaled normal distribution with stability measures"""
        # Use Xavier/Glorot initialization for better gradient flow
        fan_in = shape[0] if len(shape) >= 1 else 1
        fan_out = shape[1] if len(shape) >= 2 else 1
        std = np.sqrt(2.0 / (fan_in + fan_out)) * scale
        
        # Initialize with bounded values
        weight = np.random.normal(0, std, shape).astype(np.float32)
        
        # Apply stability measures
        weight = np.clip(weight, -3*std, 3*std)  # Clip outliers
        weight = weight / (np.std(weight) + self.eps)  # Normalize
        weight = weight * std  # Rescale to desired range
        
        return weight.astype(np.float32)

    def init_attention_weights(self, scale: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize attention weights with proper scaling"""
        q = self.init_weight((self.hidden_size, self.hidden_size), scale)
        k = self.init_weight((self.hidden_size, self.hidden_size), scale)
        v = self.init_weight((self.hidden_size, self.hidden_size), scale)
        o = self.init_weight((self.hidden_size, self.hidden_size), scale)
        return q, k, v, o

    def write_tensor(self, f, name: str, tensor: np.ndarray, meta: Optional[Dict] = None):
        """Write tensor with validation and optional metadata"""
        checksum = hashlib.sha256(tensor.tobytes()).hexdigest()
        
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
        
        chunk_size = 1024 * 1024  # 1MB chunks
        data = tensor.tobytes()
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            f.write(chunk)
            f.flush()
            
        del data
        gc.collect()
        
    def read_tensor(self, f) -> Tuple[str, np.ndarray, Dict]:
        """Read tensor with validation and stability measures"""
        metadata_size = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(metadata_size))
        
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        size = np.prod(shape) * dtype.itemsize
        
        # Read data in chunks for large tensors
        chunk_size = 1024 * 1024  # 1MB chunks
        data = bytearray()
        remaining = size
        
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                raise EOFError("Unexpected end of file")
            data.extend(chunk)
            remaining -= len(chunk)
        
        # Convert to tensor with stability measures
        tensor = np.frombuffer(data, dtype=dtype).reshape(shape)
        
        # Validate checksum
        checksum = hashlib.sha256(tensor.tobytes()).hexdigest()
        if checksum != metadata['checksum']:
            raise ValueError(f"Checksum mismatch for tensor {metadata['name']}")
        
        # Apply stability measures
        tensor = tensor.astype(np.float64)  # Higher precision for processing
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return metadata['name'], tensor, metadata

    def create_attention_mask(self, size: int) -> np.ndarray:
        """Create causal attention mask"""
        mask = np.tril(np.ones((size, size)))
        return mask.astype(np.float32)
        
    def create_layer(self, f, layer_id: int):
        """Create transformer layer with proper attention and FFN"""
        print(f"  Creating layer {layer_id} weights...")
        
        # Initialize attention weights with stability measures
        q, k, v, o = self.init_attention_weights()
        
        # Scale attention weights for better initial stability
        scale = np.sqrt(1.0 / self.hidden_size)
        q = q * scale
        k = k * scale
        v = v * scale
        o = o * scale
        
        # Normalize weights
        q = self._normalize_tensor(q)
        k = self._normalize_tensor(k)
        v = self._normalize_tensor(v)
        o = self._normalize_tensor(o)
        
        # Initialize layer norms
        attn_ln = LayerNorm(self.hidden_size)
        ffn_ln = LayerNorm(self.hidden_size)
        
        # Write attention components
        prefix = f'layer_{layer_id}'
        self.write_tensor(f, f'{prefix}.attn.query.weight', q)
        self.write_tensor(f, f'{prefix}.attn.key.weight', k)
        self.write_tensor(f, f'{prefix}.attn.value.weight', v)
        self.write_tensor(f, f'{prefix}.attn.output.weight', o)
        
        # Write layer norms
        self.write_tensor(f, f'{prefix}.attn_ln.gamma', attn_ln.gamma)
        self.write_tensor(f, f'{prefix}.attn_ln.beta', attn_ln.beta)
        
        # Initialize and scale FFN weights
        ffn_scale = np.sqrt(2.0 / (self.hidden_size + self.ffn_hidden_size))
        ffn_up = self.init_weight((self.hidden_size, self.ffn_hidden_size)) * ffn_scale
        ffn_down = self.init_weight((self.ffn_hidden_size, self.hidden_size)) * ffn_scale
        
        # Normalize FFN weights
        ffn_up = self._normalize_tensor(ffn_up)
        ffn_down = self._normalize_tensor(ffn_down)
        
        # Write FFN components
        self.write_tensor(f, f'{prefix}.ffn.up.weight', ffn_up)
        self.write_tensor(f, f'{prefix}.ffn.down.weight', ffn_down)
        
        self.write_tensor(f, f'{prefix}.ffn_ln.gamma', ffn_ln.gamma)
        self.write_tensor(f, f'{prefix}.ffn_ln.beta', ffn_ln.beta)
        
        # Initialize biases near zero with normalization
        attn_bias = np.random.normal(0, 0.001, self.hidden_size).astype(np.float32)
        ffn_bias1 = np.random.normal(0, 0.001, self.ffn_hidden_size).astype(np.float32)
        ffn_bias2 = np.random.normal(0, 0.001, self.hidden_size).astype(np.float32)
        
        # Normalize biases
        attn_bias = self._normalize_tensor(attn_bias)
        ffn_bias1 = self._normalize_tensor(ffn_bias1)
        ffn_bias2 = self._normalize_tensor(ffn_bias2)
        
        # Write biases
        self.write_tensor(f, f'{prefix}.attn.output.bias', attn_bias)
        self.write_tensor(f, f'{prefix}.ffn.up.bias', ffn_bias1)
        self.write_tensor(f, f'{prefix}.ffn.down.bias', ffn_bias2)
        
        del q, k, v, o, ffn_up, ffn_down
        gc.collect()
        
    def build_model(self, filepath: str):
        """Build complete Monday model with proper architecture"""
        print("Building Monday model...")
        
        tensors_per_layer = 13
        # Correct tensor count:
        #  - Pre-layer tensors: tok_embedding, output.weight, _meta, position_embeddings,
        #    attention_mask, embedding_ln.weight, embedding_ln.bias => 7
        #  - Per-layer tensors: 13 each
        #  - Post-layer tensors: output_ln.weight, output_ln.bias, output.bias => 3
        #  Total = 7 + L*13 + 3 = 10 + L*13
        num_tensors = 10 + self.num_layers * tensors_per_layer
        
        with open(filepath, 'wb') as f:
            header = {
                'magic': 'MOND',
                'version': 2,
                'num_tensors': num_tensors,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len,
                'traits': self.traits,
                'notus': self.notus
            }
            header_bytes = json.dumps(header).encode('utf-8')
            f.write(struct.pack('<I', len(header_bytes)))
            f.write(header_bytes)
            
            print("\nCreating embeddings...")
            # Create token embeddings with proper initialization
            print("\nCreating token embeddings...")
            E_tok = np.random.randn(self.vocab_size, self.hidden_size).astype(np.float32) * (1.0 / np.sqrt(self.hidden_size))
            # Initialize special tokens with slightly larger values
            E_tok[:4] = np.random.randn(4, self.hidden_size).astype(np.float32) * (2.0 / np.sqrt(self.hidden_size))
            # Normalize embeddings
            E_tok = self._normalize_tensor(E_tok)
            self.write_tensor(f, 'tok_embedding', E_tok)
            
            # Tie output projection to embeddings, with the ONLY canonical name
            W_out = E_tok.T.astype(np.float32)  # (hidden_size, vocab_size)
            self.write_tensor(f, 'output.weight', W_out)
            
            # Store metadata
            meta = {
                'vocab_size': int(self.vocab_size),
                'd_model': int(self.hidden_size),
                'num_layers': int(self.num_layers),
                'num_heads': int(self.num_heads),
                'head_size': int(self.head_size)
            }
            self.write_tensor(f, '_meta', np.array([0], dtype=np.float32), meta=meta)
            
            del E_tok, W_out
            
            # Create position embeddings with sinusoidal initialization
            print("Creating position embeddings...")
            pos = np.zeros((self.max_seq_len, self.hidden_size), dtype=np.float32)
            for pos_idx in range(self.max_seq_len):
                for i in range(0, self.hidden_size, 2):
                    pos[pos_idx, i] = np.sin(pos_idx / (10000 ** (i / self.hidden_size)))
                    if i + 1 < self.hidden_size:
                        pos[pos_idx, i + 1] = np.cos(pos_idx / (10000 ** (i / self.hidden_size)))
            pos = self._normalize_tensor(pos)
            self.write_tensor(f, 'position_embeddings', pos)
            del pos
            
            # Create attention mask
            print("Creating attention mask...")
            mask = self.create_attention_mask(self.max_seq_len)
            self.write_tensor(f, 'attention_mask', mask)
            del mask
            
            # Create embedding layer norm
            print("Creating embedding normalization...")
            emb_ln = LayerNorm(self.hidden_size)
            self.write_tensor(f, 'embedding_ln.weight', emb_ln.gamma)
            self.write_tensor(f, 'embedding_ln.bias', emb_ln.beta)
            del emb_ln
            
            for i in range(self.num_layers):
                print(f"\nLayer {i+1}/{self.num_layers}")
                self.create_layer(f, i)
                gc.collect()
                
            print("\nCreating output components...")
            # Output layer normalization
            out_ln = LayerNorm(self.hidden_size)
            self.write_tensor(f, 'output_ln.weight', out_ln.gamma)
            self.write_tensor(f, 'output_ln.bias', out_ln.beta)
            
            # Add output bias with small initialization
            out_bias = np.random.normal(0, 0.001, self.vocab_size).astype(np.float32)
            out_bias = self._normalize_tensor(out_bias)
            self.write_tensor(f, 'output.bias', out_bias)
            
            del out_bias, out_ln
            
            f.flush()
        
        file_size = os.path.getsize(filepath)
        print(f"\nModel saved to: {filepath}")
        print(f"File size: {file_size:,} bytes")
        
        checksum = hashlib.sha256(open(filepath, 'rb').read()).hexdigest()
        checksum_file = filepath + '.sha256'
        with open(checksum_file, 'w') as f:
            f.write(checksum)
        print(f"Checksum saved to: {checksum_file}")

    def load_model(self, filepath: str):
        """Load model weights from file"""
        self.logger.info(f"Loading model from {filepath}")
        
        with open(filepath, 'rb') as f:
            # Read and validate header
            header_size = struct.unpack('<I', f.read(4))[0]
            header = json.loads(f.read(header_size))
            
            if header['magic'] != 'MOND':
                raise ValueError("Invalid model file format")
            self.header_info = header
            self.num_tensors_declared = int(header.get('num_tensors', 0))
            
            # Load all tensors
            loaded_count = 0
            for _ in range(header['num_tensors']):
                name, tensor, meta = self.read_tensor(f)
                if name == '_meta':
                    # Update model dimensions from metadata
                    self.vocab_size = int(meta['vocab_size'])
                    self.hidden_size = int(meta['d_model'])
                    self.num_layers = int(meta['num_layers'])
                    self.num_heads = int(meta['num_heads'])
                    self.head_size = int(meta['head_size'])
                    loaded_count += 1
                    continue
                self.weights[name] = tensor
                loaded_count += 1
            
            # Validate critical tensors
            E_tok = self.weights['tok_embedding']  # expect (V, C)
            W_out = self.weights['output.weight']  # expect (C, V)
            
            assert E_tok.shape == (self.vocab_size, self.hidden_size), \
                f"tok_embedding shape {E_tok.shape} != {(self.vocab_size, self.hidden_size)}"
            assert W_out.shape == (self.hidden_size, self.vocab_size), \
                f"output.weight shape {W_out.shape} != {(self.hidden_size, self.vocab_size)}"
            
            # Print model dimensions
            print(f"vocab_size,d_model = {self.vocab_size}, {self.hidden_size}")
            print(f"E_tok: {E_tok.shape}, W_out: {W_out.shape}")
                
        self.loaded = True
        self.num_tensors_loaded = loaded_count
        self.logger.info("Model loaded successfully")

    def validate(self) -> List[str]:
        """Validate loaded tensors for presence, shapes, and finiteness."""
        issues: List[str] = []
        if not self.loaded:
            issues.append("Model not loaded")
            return issues
        
        # Helper to check tensor
        def check(name: str, expected_shape: Optional[Tuple[int, ...]] = None):
            if name not in self.weights:
                issues.append(f"Missing tensor: {name}")
                return
            t = self.weights[name]
            if expected_shape is not None and tuple(t.shape) != tuple(expected_shape):
                issues.append(f"Shape mismatch for {name}: {t.shape} != {expected_shape}")
            if t.size == 0:
                issues.append(f"Empty tensor: {name}")
            if not np.isfinite(t).all():
                issues.append(f"Non-finite values in: {name}")
        
        V = int(self.vocab_size)
        C = int(self.hidden_size)
        L = int(self.num_layers)
        H = int(self.num_heads)
        HS = int(self.head_size)
        FF = int(self.ffn_hidden_size)
        T = int(self.max_seq_len)
        
        # Global tensors
        check('tok_embedding', (V, C))
        check('output.weight', (C, V))
        check('position_embeddings', (T, C))
        check('attention_mask', (T, T))
        check('embedding_ln.weight', (C,))
        check('embedding_ln.bias', (C,))
        check('output_ln.weight', (C,))
        check('output_ln.bias', (C,))
        check('output.bias', (V,))
        
        # Per-layer tensors
        for i in range(L):
            p = f'layer_{i}'
            check(f'{p}.attn.query.weight', (C, C))
            check(f'{p}.attn.key.weight', (C, C))
            check(f'{p}.attn.value.weight', (C, C))
            check(f'{p}.attn.output.weight', (C, C))
            check(f'{p}.attn.output.bias', (C,))
            check(f'{p}.attn_ln.gamma', (C,))
            check(f'{p}.attn_ln.beta', (C,))
            check(f'{p}.ffn.up.weight', (C, FF))
            check(f'{p}.ffn.up.bias', (FF,))
            check(f'{p}.ffn.down.weight', (FF, C))
            check(f'{p}.ffn.down.bias', (C,))
            check(f'{p}.ffn_ln.gamma', (C,))
            check(f'{p}.ffn_ln.beta', (C,))
        
        # Count check
        declared = int(self.num_tensors_declared)
        loaded = int(self.num_tensors_loaded)
        if declared != 0 and loaded != declared:
            issues.append(f"Tensor count mismatch: loaded {loaded} != declared {declared}")
        
        # Structural sanity checks
        if C % max(1, H) != 0:
            issues.append(f"hidden_size {C} not divisible by num_heads {H}")
        if HS * H != C:
            issues.append(f"head_size*num_heads {HS}*{H} != hidden_size {C}")
        
        return issues

    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute scaled dot-product attention with numerical stability"""
        # Convert to float64 for intermediate calculations
        q = q.astype(np.float64)
        k = k.astype(np.float64)
        v = v.astype(np.float64)
        
        # Add small noise to prevent exact zeros
        noise_scale = 1e-8
        q = q + np.random.normal(0, noise_scale, q.shape)
        k = k + np.random.normal(0, noise_scale, k.shape)
        v = v + np.random.normal(0, noise_scale, v.shape)
        
        # Normalize each head
        q = self._normalize_tensor(q)
        k = self._normalize_tensor(k)
        v = self._normalize_tensor(v)
        
        # Scaled dot product with careful normalization
        scale = 1.0 / np.sqrt(float(self.head_size))
        
        # Compute attention scores with stability measures
        # Compute in chunks to avoid overflow
        batch_size, num_heads, seq_len, head_dim = q.shape
        scores = np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=np.float64)
        
        chunk_size = 32  # Process attention in chunks
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                
                # Compute chunk scores
                q_chunk = q[:, :, i:end_i, :]
                k_chunk = k[:, :, j:end_j, :].transpose(0, 1, 3, 2)
                chunk_scores = np.matmul(q_chunk, k_chunk)
                chunk_scores = chunk_scores * scale
                
                # Normalize chunk
                chunk_scores = self._normalize_tensor(chunk_scores)
                
                # Store chunk
                scores[:, :, i:end_i, j:end_j] = chunk_scores
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (1 - mask) * -1e2  # Less extreme masking value
            
        # Compute attention probabilities with stable softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_shifted = scores - scores_max
        
        # Compute exp in chunks to avoid overflow
        exp_scores = np.zeros_like(scores)
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            chunk = scores_shifted[:, :, :, i:end_i]
            exp_scores[:, :, :, i:end_i] = np.exp(chunk)
        
        # Compute attention weights with stability
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True) + self.eps
        attn_weights = exp_scores / sum_exp
        
        # Normalize attention weights
        attn_weights = self._normalize_tensor(attn_weights)
        
        # Apply attention to values in chunks
        out = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float64)
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            chunk_weights = attn_weights[:, :, :, i:end_i]
            chunk_values = v[:, :, i:end_i, :]
            chunk_out = np.matmul(chunk_weights, chunk_values)
            chunk_out = self._normalize_tensor(chunk_out)
            out = out + chunk_out
            
        # Final normalization
        out = self._normalize_tensor(out)
        
        return out.astype(np.float32)

    def _normalize_tensor(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Normalize tensor for numerical stability"""
        # Convert to float64 for better precision
        x = x.astype(np.float64)
        
        # Handle NaN and Inf
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Add small noise to prevent exact zeros
        noise_scale = eps * 0.01  # Reduced noise scale
        x = x + np.random.normal(0, noise_scale, x.shape)
        
        # Normalize based on tensor rank and shape
        if len(x.shape) > 2:  # For attention tensors (batch, heads, seq, dim)
            # Process in chunks to avoid overflow
            chunk_size = 32
            for b in range(x.shape[0]):
                for h in range(x.shape[1]):
                    for i in range(0, x.shape[2], chunk_size):
                        end_i = min(i + chunk_size, x.shape[2])
                        chunk = x[b, h, i:end_i]
                        
                        # Normalize chunk
                        mean = np.mean(chunk)
                        std = np.std(chunk)
                        chunk = (chunk - mean) / (std + eps)
                        
                        # Clip and scale chunk
                        chunk = np.clip(chunk, -3, 3)
                        chunk = chunk * 0.1
                        
                        x[b, h, i:end_i] = chunk
                        
        elif len(x.shape) == 2:  # For weight matrices
            # Process in chunks
            row_chunk = 64
            col_chunk = 64
            for i in range(0, x.shape[0], row_chunk):
                end_i = min(i + row_chunk, x.shape[0])
                for j in range(0, x.shape[1], col_chunk):
                    end_j = min(j + col_chunk, x.shape[1])
                    
                    # Get chunk
                    chunk = x[i:end_i, j:end_j]
                    
                    # Normalize chunk
                    mean = np.mean(chunk)
                    std = np.std(chunk)
                    chunk = (chunk - mean) / (std + eps)
                    
                    # Clip and scale chunk
                    chunk = np.clip(chunk, -3, 3)
                    chunk = chunk * 0.1
                    
                    x[i:end_i, j:end_j] = chunk
                    
        else:  # For bias vectors
            # Process in chunks
            chunk_size = 1024
            for i in range(0, x.shape[0], chunk_size):
                end_i = min(i + chunk_size, x.shape[0])
                chunk = x[i:end_i]
                
                # Normalize chunk
                mean = np.mean(chunk)
                std = np.std(chunk)
                chunk = (chunk - mean) / (std + eps)
                
                # Clip and scale chunk
                chunk = np.clip(chunk, -3, 3)
                chunk = chunk * 0.1
                
                x[i:end_i] = chunk
        
        # Add tiny offset to prevent exact zeros
        x = x + np.sign(x) * eps
        
        # Final stability measures
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(x, -10, 10)
        
        return x.astype(np.float32)
        
    def safe_matmul(self, a_2d: np.ndarray, b_2d: np.ndarray) -> np.ndarray:
        """Safe matrix multiplication with shape and value checks"""
        # expect (N,K) @ (K,M) -> (N,M)
        assert a_2d.ndim == 2 and b_2d.ndim == 2
        assert a_2d.shape[1] == b_2d.shape[0], f"matmul shape mismatch: {a_2d.shape} @ {b_2d.shape}"
        
        # Check for NaN/Inf
        if not np.isfinite(a_2d).all():
            raise FloatingPointError("NaN/Inf in left operand")
        if not np.isfinite(b_2d).all():
            raise FloatingPointError("NaN/Inf in right operand")
            
        # Convert to float32
        a_2d = a_2d.astype(np.float32)
        b_2d = b_2d.astype(np.float32)
        
        # Compute in chunks
        chunk_size = 32
        N, K = a_2d.shape
        M = b_2d.shape[1]
        result = np.zeros((N, M), dtype=np.float32)
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            a_chunk = a_2d[i:end_i]
            
            for j in range(0, M, chunk_size):
                end_j = min(j + chunk_size, M)
                b_chunk = b_2d[:, j:end_j]
                
                # Compute chunk
                chunk = np.matmul(a_chunk, b_chunk)
                result[i:end_i, j:end_j] = chunk
                
                # Check for non-finite values
                if not np.isfinite(chunk).all():
                    raise FloatingPointError("Non-finite values in matmul result")
        
        return result
        
    def _matmul_in_chunks(self, a: np.ndarray, b: np.ndarray, chunk_size: int = 32) -> np.ndarray:
        """Perform matrix multiplication in chunks to avoid overflow"""
        # Convert to float64 for intermediate calculations
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        
        # Get output shape
        out_shape = (a.shape[0], b.shape[1]) if len(a.shape) == 2 else a.shape[:-1] + (b.shape[-1],)
        result = np.zeros(out_shape, dtype=np.float64)
        
        # Reshape inputs if needed
        if len(a.shape) > 2:
            a_2d = a.reshape(-1, a.shape[-1])
            b_2d = b
        else:
            a_2d = a
            b_2d = b
            
        # Process in chunks
        for i in range(0, a_2d.shape[0], chunk_size):
            end_i = min(i + chunk_size, a_2d.shape[0])
            a_chunk = a_2d[i:end_i]
            
            # Compute chunk result using safe_matmul
            chunk_result = self.safe_matmul(a_chunk, b_2d)
            
            # Store result
            if len(a.shape) > 2:
                result.reshape(-1, b.shape[-1])[i:end_i] = chunk_result
            else:
                result[i:end_i] = chunk_result
        
        # Final normalization
        result = self._normalize_tensor(result)
        
        return result

    def _forward_layer(self, x: np.ndarray, layer_id: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through a single transformer layer with stability measures"""
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        prefix = f'layer_{layer_id}'
        
        # Layer normalization before attention
        ln_weight = self.weights[f'{prefix}.attn_ln.gamma']
        ln_bias = self.weights[f'{prefix}.attn_ln.beta']
        ln_out = x * ln_weight + ln_bias
        ln_out = self._normalize_tensor(ln_out)
        
        # Multi-head attention with shape checking and normalization
        # Project to Q, K, V with stability measures
        ln_flat = ln_out.reshape(-1, self.hidden_size)
        ln_flat = self._normalize_tensor(ln_flat)
        
        # Compute Q, K, V projections in chunks
        q = self._matmul_in_chunks(ln_flat, self.weights[f'{prefix}.attn.query.weight'])
        q = self._normalize_tensor(q).reshape(batch_size, seq_len, self.hidden_size)
        
        k = self._matmul_in_chunks(ln_flat, self.weights[f'{prefix}.attn.key.weight'])
        k = self._normalize_tensor(k).reshape(batch_size, seq_len, self.hidden_size)
        
        v = self._matmul_in_chunks(ln_flat, self.weights[f'{prefix}.attn.value.weight'])
        v = self._normalize_tensor(v).reshape(batch_size, seq_len, self.hidden_size)
        
        # Reshape for multi-head attention with normalization
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Normalize each head
        for b in range(batch_size):
            for h in range(self.num_heads):
                q[b, :, h, :] = self._normalize_tensor(q[b, :, h, :])
                k[b, :, h, :] = self._normalize_tensor(k[b, :, h, :])
                v[b, :, h, :] = self._normalize_tensor(v[b, :, h, :])
        
        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Compute attention with stability measures
        attn_out = self._attention(q, k, v, mask)
        attn_out = attn_out.transpose(0, 2, 1, 3)
        
        # Normalize attention output
        for b in range(batch_size):
            for h in range(self.num_heads):
                attn_out[b, :, h, :] = self._normalize_tensor(attn_out[b, :, h, :])
        
        # Reshape and project output
        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_size)
        attn_out = self._normalize_tensor(attn_out)
        
        # Output projection with normalization
        attn_flat = attn_out.reshape(-1, self.hidden_size)
        attn_flat = self._normalize_tensor(attn_flat)
        
        # Compute output projection in chunks
        out_proj = self._matmul_in_chunks(attn_flat, self.weights[f'{prefix}.attn.output.weight'])
        out_proj = self._normalize_tensor(out_proj).reshape(batch_size, seq_len, self.hidden_size)
        out_proj = out_proj + self.weights[f'{prefix}.attn.output.bias']
        attn_out = self._normalize_tensor(out_proj)
        
        # First residual connection
        x = x + attn_out
        
        # Layer normalization before FFN with stability measures
        ln_weight = self.weights[f'{prefix}.ffn_ln.gamma']
        ln_bias = self.weights[f'{prefix}.ffn_ln.beta']
        ln_out = x * ln_weight + ln_bias
        ln_out = self._normalize_tensor(ln_out)
        
        # Flatten for matrix multiplication
        ln_flat = ln_out.reshape(-1, self.hidden_size)
        ln_flat = self._normalize_tensor(ln_flat)
        
        # First FFN projection with normalization and chunking
        ffn_up = self._matmul_in_chunks(ln_flat, self.weights[f'{prefix}.ffn.up.weight'])
        ffn_up = self._normalize_tensor(ffn_up)
        ffn_up = ffn_up + self.weights[f'{prefix}.ffn.up.bias']
        
        # Apply activation with stability measures
        ffn_act = self.gelu(ffn_up)
        ffn_act = self._normalize_tensor(ffn_act)
        
        # Second FFN projection with normalization and chunking
        ffn_down = self._matmul_in_chunks(ffn_act, self.weights[f'{prefix}.ffn.down.weight'])
        ffn_down = self._normalize_tensor(ffn_down)
        ffn_down = ffn_down + self.weights[f'{prefix}.ffn.down.bias']
        
        # Reshape and normalize
        ffn_out = ffn_down.reshape(batch_size, seq_len, self.hidden_size)
        ffn_out = self._normalize_tensor(ffn_out)
        
        # Second residual connection
        x = x + ffn_out
        
        return x

    def _sample_top_p(self, logits: np.ndarray, top_p: float) -> int:
        """Sample from the distribution using top-p (nucleus) sampling"""
        # Verify logits shape
        assert logits.ndim == 1 and logits.shape[0] == self.vocab_size, \
            f"logits must be (vocab_size,), got {logits.shape}"
        
        # Stabilize softmax
        logits = logits.astype(np.float32)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        s = probs.sum()
        if not np.isfinite(s) or s == 0.0:
            # fall back to uniform over vocab to avoid crash
            probs = np.ones_like(probs, dtype=np.float32)
            s = probs.sum()
        probs /= s
        
        # sort by probability, accumulate until top_p
        sorted_idx = np.argsort(-probs)  # descending
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p, side='right') + 1
        keep_idx = sorted_idx[:cutoff]
        
        # renormalize kept mass
        kept = probs[keep_idx]
        kept /= kept.sum()
        
        # sample
        choice = np.random.choice(keep_idx, p=kept)
        return int(choice)

    def generate(self, 
                prompt: Union[str, List[int]], 
                config: Optional[ModelConfig] = None) -> ModelResponse:
        """Generate text from prompt"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        config = config or ModelConfig()
        
        # Convert prompt to token ids
        if isinstance(prompt, str):
            # TODO: Add tokenizer integration
            token_ids = [0]  # Placeholder
        elif isinstance(prompt, (list, tuple)):
            token_ids = list(prompt)
        elif isinstance(prompt, np.ndarray):
            token_ids = prompt.astype(np.int64).tolist()
        else:
            raise TypeError("prompt must be str, list[int], tuple[int], or np.ndarray of ints")
            
        # Ensure prompt isn't too long
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            
        # Initialize generation
        tokens = list(token_ids)
        logprobs = []
        
        # Create attention mask
        mask = self.create_attention_mask(len(tokens))
        
        # Get initial embeddings with normalization
        x = self.weights['tok_embedding'][tokens]
        x = self._normalize_tensor(x)
        
        pos = self.weights['position_embeddings'][:len(tokens)]
        pos = self._normalize_tensor(pos)
        
        # Combine token and position embeddings
        x = x + pos
        
        # Apply embedding layer norm
        ln_weight = self.weights['embedding_ln.weight']
        ln_bias = self.weights['embedding_ln.bias']
        x = x * ln_weight + ln_bias
        x = self._normalize_tensor(x)
        
        # Add batch and sequence dimensions
        x = x.reshape(1, len(tokens), self.hidden_size)
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
        
        # Forward pass through layers with normalization
        for i in range(self.num_layers):
            x = self._forward_layer(x, i, mask)
            x = self._normalize_tensor(x)
            assert np.isfinite(x).all(), f"non-finite activations after layer {i}"
            
        # x: (1, T, C) → take last token of first (only) batch
        h_last = x[0, -1, :].astype(np.float32)  # (C,)
        
        # Project to vocabulary size with stability measures
        W_out = self.weights['output.weight'].astype(np.float32)  # (C, V)
        assert W_out.shape == (self.hidden_size, self.vocab_size)
        logits = h_last @ W_out  # (V,)
        
        # Add bias with normalization
        bias = self.weights['output.bias'].astype(np.float32)  # (V,)
        assert bias.shape == (self.vocab_size,)
        logits = logits + bias
        
        # No final layer norm - just normalize
        logits = self._normalize_tensor(logits)
        
        # Final stability measures
        logits = np.clip(logits, -10, 10)  # Less extreme clipping
        logits = logits - np.max(logits)  # Shift for numerical stability
        logits = logits * 0.1  # Scale down for better stability
        
        # Apply temperature
        logits = logits / config.temperature
        
        # Verify logits shape before sampling
        assert logits.ndim == 1 and logits.shape[0] == self.vocab_size, \
            f"logits must be (vocab_size,), got {logits.shape}"
        
        # Sample next token
        if config.top_p < 1.0:
            next_token = self._sample_top_p(logits, config.top_p)
        else:
            next_token = np.random.choice(len(logits), p=self.softmax(logits))
            
        tokens.append(next_token)
        logprobs.append(float(logits[next_token]))
        
        # TODO: Add tokenizer integration for text conversion
        return ModelResponse(
            text="",  # Placeholder until tokenizer added
            tokens=tokens,
            logprobs=logprobs,
            finish_reason="length" if len(tokens) >= config.max_tokens else "stop"
        )

    def __call__(self, 
                 prompt: Union[str, List[int]], 
                 config: Optional[ModelConfig] = None) -> ModelResponse:
        """Convenience method for generate()"""
        return self.generate(prompt, config)
        
def main():
    model = MondayModel()
    model_path = '/workspace/monday_v2.bin'
    model.build_model(model_path)
    
    # Test loading and basic inference
    try:
        print("\nTesting model loading and inference...")
        model.load_model(model_path)
        issues = model.validate()
        if issues:
            print("\nValidation issues found:")
            for msg in issues:
                print(f" - {msg}")
        else:
            print("\nValidation passed: no issues found.")
        
        # Create a simple test sequence
        test_tokens = np.array([1, 2, 3], dtype=np.int32)  # Simple test sequence
        print(f"Input tokens: {test_tokens}")
        
        # Test generation with controlled parameters
        config = ModelConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=5
        )
        
        response = model.generate(test_tokens, config)
        print("\nGeneration test results:")
        print(f"Output tokens: {response.tokens}")
        print(f"Log probabilities: {response.logprobs}")
        print(f"Finish reason: {response.finish_reason}")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == '__main__':
    main()

