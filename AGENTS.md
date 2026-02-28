# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Monday Model is a pure-Python, CPU-only Transformer inference scaffold with a custom binary weight format (`.bin` files with `MOND` magic header). The only runtime dependency is `numpy`.

### Running the application

The primary entry point is `monday_model.py`. See `README.md` for all run modes (`MONDAY_TINY`, `MONDAY_FORCE_REBUILD`, `MONDAY_MODEL_PATH`).

Quick smoke test:
```bash
MONDAY_TINY=1 python3 /workspace/monday_model.py
```

### Caveats

- **No linting, test framework, or build system.** The project has no `requirements.txt`, `pyproject.toml`, `Makefile`, or CI config. Linting/testing means running the scripts directly and checking for errors.
- **`monday_model_improved.py` has a pre-existing bug:** its `_attention` method uses `k.transpose(-2, -1)` which is PyTorch syntax, not valid numpy. Inference will fail with `ValueError: axes don't match array`. The model builds and loads fine; only `generate()` is broken.
- **`export_hf_to_monday.py` requires `torch` and `transformers`**, which are not installed by default. These are optional and only needed for HuggingFace model conversion.
- Pre-built weight files (`monday_tiny.bin`, `monday_v2.bin`) are committed to the repo. The script skips rebuilding if the file exists; use `MONDAY_FORCE_REBUILD=1` to regenerate.
