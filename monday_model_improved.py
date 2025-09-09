#!/usr/bin/env python3
"""
Improved MondayModel - A more efficient and stable transformer implementation
"""
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
    """Efficient layer normalization implementation"""
    def __init__(self, size: int, eps: float = 1e-5):
        self.size = size
        self.eps = eps
        self.gamma = np.ones(size, dtype=np.float32)
        self.beta = np.zeros(size, dtype=np.float32)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class SimpleTokenizer:
    """Basic tokenizer for demonstration - replace with proper tokenizer"""
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Simple word-based tokenization
        self.word_to_id = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.id_to_word = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}
        
    def encode(self, text: str) -> List[int]:
        """Simple encoding - split by spaces and map to IDs"""
        words = text.split()
        token_ids = []
        for word in words:
            if word not in self.word_to_id:
                # Simple hash-based mapping for unknown words
                word_id = hash(word) % (self.vocab_size - 4) + 4
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
            token_ids.append(self.word_to_id[word])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Simple decoding"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append("<unk>")
        return " ".join(words)

class MondayModel:
    def __init__(self):
        # Model architecture
        self.hidden_size = 1024
        self.num_layers = 12
        self.num_heads = 16
        self.head_size = self.hidden_size // self.num_heads
        self.vocab_size = 32000
        self.max_seq_len = 2048
        self.eps = 1e-6
        self.ffn_hidden_size = 4 * self.hidden_size
        
        # Initialize weights dict
        self.weights = {}
        self.loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MondayModel")
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(self.vocab_size)
        
        # Monday's personality traits
        self.traits = {
            'creativity': 1.4,
            'empathy': 1.5,
            'humor': 1.3,
            'honesty': 1.4,
            'safety': 0.7,
            'engagement': 1.3
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
        """Initialize weights using Xavier initialization"""
        fan_in = shape[0] if len(shape) >= 1 else 1
        fan_out = shape[1] if len(shape) >= 2 else 1
        std = np.sqrt(2.0 / (fan_in + fan_out)) * scale
        return np.random.normal(0, std, shape).astype(np.float32)

    def write_tensor(self, f, name: str, tensor: np.ndarray, meta: Optional[Dict] = None):
        """Write tensor with validation and metadata"""
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
        f.write(tensor.tobytes())
        
    def read_tensor(self, f) -> Tuple[str, np.ndarray, Dict]:
        """Read tensor with validation"""
        metadata_size = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(metadata_size))
        
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        size = np.prod(shape) * dtype.itemsize
        
        data = f.read(size)
        if len(data) != size:
            raise EOFError("Unexpected end of file")
        
        tensor = np.frombuffer(data, dtype=dtype).reshape(shape)
        
        # Validate checksum
        checksum = hashlib.sha256(tensor.tobytes()).hexdigest()
        if checksum != metadata['checksum']:
            raise ValueError(f"Checksum mismatch for tensor {metadata['name']}")
            
        return metadata['name'], tensor, metadata

    def create_attention_mask(self, size: int) -> np.ndarray:
        """Create causal attention mask"""
        return np.tril(np.ones((size, size), dtype=np.float32))
        
    def create_layer(self, f, layer_id: int):
        """Create transformer layer"""
        print(f"  Creating layer {layer_id} weights...")
        
        # Initialize attention weights
        q = self.init_weight((self.hidden_size, self.hidden_size))
        k = self.init_weight((self.hidden_size, self.hidden_size))
        v = self.init_weight((self.hidden_size, self.hidden_size))
        o = self.init_weight((self.hidden_size, self.hidden_size))
        
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
        
        # Initialize FFN weights
        ffn_up = self.init_weight((self.hidden_size, self.ffn_hidden_size))
        ffn_down = self.init_weight((self.ffn_hidden_size, self.hidden_size))
        
        # Write FFN components
        self.write_tensor(f, f'{prefix}.ffn.up.weight', ffn_up)
        self.write_tensor(f, f'{prefix}.ffn.down.weight', ffn_down)
        
        self.write_tensor(f, f'{prefix}.ffn_ln.gamma', ffn_ln.gamma)
        self.write_tensor(f, f'{prefix}.ffn_ln.beta', ffn_ln.beta)
        
        # Initialize biases
        attn_bias = np.zeros(self.hidden_size, dtype=np.float32)
        ffn_bias1 = np.zeros(self.ffn_hidden_size, dtype=np.float32)
        ffn_bias2 = np.zeros(self.hidden_size, dtype=np.float32)
        
        # Write biases
        self.write_tensor(f, f'{prefix}.attn.output.bias', attn_bias)
        self.write_tensor(f, f'{prefix}.ffn.up.bias', ffn_bias1)
        self.write_tensor(f, f'{prefix}.ffn.down.bias', ffn_bias2)
        
    def build_model(self, filepath: str):
        """Build complete Monday model"""
        print("Building Monday model...")
        
        tensors_per_layer = 13
        num_tensors = (
            2 +  # Embeddings + positions
            self.num_layers * tensors_per_layer +  # Layer components
            3    # Output layer + norm
        )
        
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
                'traits': self.traits
            }
            header_bytes = json.dumps(header).encode('utf-8')
            f.write(struct.pack('<I', len(header_bytes)))
            f.write(header_bytes)
            
            print("\nCreating embeddings...")
            # Create token embeddings
            E_tok = self.init_weight((self.vocab_size, self.hidden_size))
            self.write_tensor(f, 'tok_embedding', E_tok)
            
            # Tie output projection to embeddings
            W_out = E_tok.T.astype(np.float32)
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
            
            # Create position embeddings
            print("Creating position embeddings...")
            pos = np.zeros((self.max_seq_len, self.hidden_size), dtype=np.float32)
            for pos_idx in range(self.max_seq_len):
                for i in range(0, self.hidden_size, 2):
                    pos[pos_idx, i] = np.sin(pos_idx / (10000 ** (i / self.hidden_size)))
                    if i + 1 < self.hidden_size:
                        pos[pos_idx, i + 1] = np.cos(pos_idx / (10000 ** (i / self.hidden_size)))
            self.write_tensor(f, 'position_embeddings', pos)
            
            # Create embedding layer norm
            print("Creating embedding normalization...")
            emb_ln = LayerNorm(self.hidden_size)
            self.write_tensor(f, 'embedding_ln.weight', emb_ln.gamma)
            self.write_tensor(f, 'embedding_ln.bias', emb_ln.beta)
            
            for i in range(self.num_layers):
                print(f"\nLayer {i+1}/{self.num_layers}")
                self.create_layer(f, i)
                
            print("\nCreating output components...")
            # Output layer normalization
            out_ln = LayerNorm(self.hidden_size)
            self.write_tensor(f, 'output_ln.weight', out_ln.gamma)
            self.write_tensor(f, 'output_ln.bias', out_ln.beta)
            
            # Add output bias
            out_bias = np.zeros(self.vocab_size, dtype=np.float32)
            self.write_tensor(f, 'output.bias', out_bias)
            
            f.flush()
        
        file_size = os.path.getsize(filepath)
        print(f"\nModel saved to: {filepath}")
        print(f"File size: {file_size:,} bytes")

    def load_model(self, filepath: str):
        """Load model weights from file"""
        self.logger.info(f"Loading model from {filepath}")
        
        with open(filepath, 'rb') as f:
            # Read and validate header
            header_size = struct.unpack('<I', f.read(4))[0]
            header = json.loads(f.read(header_size))
            
            if header['magic'] != 'MOND':
                raise ValueError("Invalid model file format")
            
            # Load all tensors
            for _ in range(header['num_tensors']):
                name, tensor, meta = self.read_tensor(f)
                if name == '_meta':
                    # Update model dimensions from metadata
                    self.vocab_size = int(meta['vocab_size'])
                    self.hidden_size = int(meta['d_model'])
                    self.num_layers = int(meta['num_layers'])
                    self.num_heads = int(meta['num_heads'])
                    self.head_size = int(meta['head_size'])
                    continue
                self.weights[name] = tensor
            
            # Validate critical tensors
            E_tok = self.weights['tok_embedding']
            W_out = self.weights['output.weight']
            
            assert E_tok.shape == (self.vocab_size, self.hidden_size)
            assert W_out.shape == (self.hidden_size, self.vocab_size)
                
        self.loaded = True
        self.logger.info("Model loaded successfully")

    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute scaled dot-product attention"""
        # Scaled dot product
        scale = 1.0 / np.sqrt(self.head_size)
        scores = np.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (1 - mask) * -1e9
            
        # Compute attention probabilities
        attn_weights = self.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = np.matmul(attn_weights, v)
        
        return out

    def _forward_layer(self, x: np.ndarray, layer_id: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through a single transformer layer"""
        batch_size, seq_len, _ = x.shape
        prefix = f'layer_{layer_id}'
        
        # Layer normalization before attention
        ln_weight = self.weights[f'{prefix}.attn_ln.gamma']
        ln_bias = self.weights[f'{prefix}.attn_ln.beta']
        ln_out = x * ln_weight + ln_bias
        
        # Multi-head attention
        # Project to Q, K, V
        q = np.matmul(ln_out, self.weights[f'{prefix}.attn.query.weight'])
        k = np.matmul(ln_out, self.weights[f'{prefix}.attn.key.weight'])
        v = np.matmul(ln_out, self.weights[f'{prefix}.attn.value.weight'])
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Compute attention
        attn_out = self._attention(q, k, v, mask)
        attn_out = attn_out.transpose(0, 2, 1, 3)
        
        # Reshape and project output
        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_size)
        attn_out = np.matmul(attn_out, self.weights[f'{prefix}.attn.output.weight'])
        attn_out = attn_out + self.weights[f'{prefix}.attn.output.bias']
        
        # First residual connection
        x = x + attn_out
        
        # Layer normalization before FFN
        ln_weight = self.weights[f'{prefix}.ffn_ln.gamma']
        ln_bias = self.weights[f'{prefix}.ffn_ln.beta']
        ln_out = x * ln_weight + ln_bias
        
        # FFN
        ffn_up = np.matmul(ln_out, self.weights[f'{prefix}.ffn.up.weight'])
        ffn_up = ffn_up + self.weights[f'{prefix}.ffn.up.bias']
        ffn_act = self.gelu(ffn_up)
        
        ffn_down = np.matmul(ffn_act, self.weights[f'{prefix}.ffn.down.weight'])
        ffn_down = ffn_down + self.weights[f'{prefix}.ffn.down.bias']
        
        # Second residual connection
        x = x + ffn_down
        
        return x

    def _sample_top_p(self, logits: np.ndarray, top_p: float) -> int:
        """Sample from the distribution using top-p (nucleus) sampling"""
        # Compute probabilities
        probs = self.softmax(logits)
        
        # Sort by probability
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff = np.searchsorted(cumsum, top_p, side='right') + 1
        keep_idx = sorted_idx[:cutoff]
        
        # Renormalize
        kept = probs[keep_idx]
        kept /= kept.sum()
        
        # Sample
        choice = np.random.choice(keep_idx, p=kept)
        return int(choice)

    def generate(self, 
                prompt: Union[str, List[int]], 
                config: Optional[ModelConfig] = None) -> ModelResponse:
        """Generate text from prompt"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        config = config or ModelConfig()
        
        # Convert prompt to token ids if string
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
            
        # Ensure prompt isn't too long
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            
        # Initialize generation
        tokens = token_ids.copy()
        logprobs = []
        
        # Create attention mask
        mask = self.create_attention_mask(len(tokens))
        
        # Get initial embeddings
        x = self.weights['tok_embedding'][tokens]
        pos = self.weights['position_embeddings'][:len(tokens)]
        x = x + pos
        
        # Apply embedding layer norm
        ln_weight = self.weights['embedding_ln.weight']
        ln_bias = self.weights['embedding_ln.bias']
        x = x * ln_weight + ln_bias
        
        # Add batch dimension
        x = x.reshape(1, len(tokens), self.hidden_size)
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
        
        # Forward pass through layers
        for i in range(self.num_layers):
            x = self._forward_layer(x, i, mask)
            
        # Get last token representation
        h_last = x[0, -1, :]
        
        # Project to vocabulary
        logits = h_last @ self.weights['output.weight']
        logits = logits + self.weights['output.bias']
        
        # Apply temperature
        logits = logits / config.temperature
        
        # Sample next token
        if config.top_p < 1.0:
            next_token = self._sample_top_p(logits, config.top_p)
        else:
            probs = self.softmax(logits)
            next_token = np.random.choice(len(logits), p=probs)
            
        tokens.append(next_token)
        logprobs.append(float(logits[next_token]))
        
        # Convert to text
        text = self.tokenizer.decode(tokens)
        
        return ModelResponse(
            text=text,
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
    model.build_model('monday_v2_improved.bin')
    
    # Test loading and basic inference
    try:
        print("\nTesting model loading and inference...")
        model.load_model('monday_v2_improved.bin')
        
        # Test with text prompt
        test_prompt = "Hello world"
        print(f"Input prompt: '{test_prompt}'")
        
        config = ModelConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=5
        )
        
        response = model.generate(test_prompt, config)
        print("\nGeneration test results:")
        print(f"Output text: '{response.text}'")
        print(f"Output tokens: {response.tokens}")
        print(f"Log probabilities: {response.logprobs}")
        print(f"Finish reason: {response.finish_reason}")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == '__main__':
    main()