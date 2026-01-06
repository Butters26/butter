# Monday Model

A simple, CPU-only Transformer (pre-LN) scaffold with a custom binary format for quick iteration and weight loading.

## Quick start (tiny mode)

```bash
MONDAY_TINY=1 python3 /workspace/monday_model.py
```
- Tiny mode uses small dims for fast smoke tests.
- Outputs `monday_tiny.bin` (unless overridden).

## Full size scaffold

```bash
MONDAY_TINY=0 python3 /workspace/monday_model.py
```
- Uses larger default dims and saves/loads `monday_v2.bin`.

## Don’t overwrite weights (default)
- The script will only build if the file is missing, or if you force it.
- To force a rebuild:
```bash
MONDAY_FORCE_REBUILD=1 python3 /workspace/monday_model.py
```

## Load a specific file
```bash
MONDAY_MODEL_PATH=/absolute/path/to/your_weights.bin python3 /workspace/monday_model.py
```
This bypasses the default filenames.

## Validate
The script prints validation results after load (shape checks, finiteness, tensor counts).

## HuggingFace → Monday exporter
We include an exporter for GPT-2-style checkpoints:

```bash
# Example using a public HF model id
python3 /workspace/export_hf_to_monday.py --model gpt2 --out /workspace/monday_from_hf.bin

# Or from a local HF directory
python3 /workspace/export_hf_to_monday.py --path /path/to/hf/ckpt --out /workspace/monday_from_hf.bin
```

Notes:
- Requires `transformers` and `torch` to be installed.
- Maps GPT-2 blocks to Monday tensor names/shapes.
- Token embeddings are tied to output projection as in Monday.

## Personality and purpose
The code preserves Monday’s personality fields (traits/notus metadata). These are not altered by exporter or runtime changes.

## License
MIT