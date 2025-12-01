# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational GPT-2 style Transformer implementation in Rust. Designed to teach fundamental concepts behind language models like ChatGPT.

## Build and Run Commands

```bash
# Build
cargo build --release

# Run with Japanese training data (default)
cargo run

# Run with English training data
cargo run -- en

# Run tests
cargo test

# Check code
cargo check

# Format code
cargo fmt

# Lint
cargo clippy
```

## Architecture

### Core Components (src/working_transformer.rs)

**WorkingTransformer** - Main model structure containing:
- Token and position embeddings (`token_embeddings`, `position_embeddings`)
- Transformer layers (`layers: Vec<TransformerLayer>`)
- Final layer norm and output projection

**TransformerLayer** - Single transformer block with:
- Multi-head attention weights (W_Q, W_K, W_V, W_O) with proper head splitting
- `num_heads` and `head_dim` configuration
- Pre-LN layer normalization (ln1, ln2)
- Feed-forward network (ff_w1, ff_b1, ff_w2, ff_b2) with GELU activation

**TrainableTransformer** - Training wrapper that combines:
- The underlying `WorkingTransformer` model
- `SimpleTokenizer` for text processing
- Learning rate and simplified gradient descent

**SimpleTokenizer** - Whitespace-based tokenizer with:
- Special tokens: `<bos>` (1), `<eos>` (0), `<unk>` (2)
- Vocabulary built from training data

### Data Flow

1. Text → `SimpleTokenizer.encode()` → token IDs
2. Token IDs → embedding lookup + position encoding
3. Through transformer layers (attention → FFN with residuals)
4. Final layer norm → output projection → logits
5. Softmax with temperature → sampling → next token

## Known Limitations (Educational Simplifications)

- **Simplified backpropagation**: Only embeddings and output layer are updated; attention/FFN/LayerNorm weights remain fixed (by design for educational clarity)
- **Basic tokenization**: Whitespace-based only
- **No GPU support**: CPU-only implementation

## Recent Fixes

- **Multi-head attention**: Properly splits Q/K/V into heads, computes attention per head
- **Sequence-level gradients**: Training now considers full sequence context via attention
- **Position embedding bounds**: Safely clips indices when `seq_len > max_seq_len`
- **Sparse gradient updates**: Efficient updates targeting only relevant tokens
- **Independent weights**: Output projection is independently initialized (no weight tying)

## Training Data Format

One sentence per line in `data.txt` (Japanese) or `data_en.txt` (English):

```text
Hello world
The quick brown fox
```

## Dependencies

- `ndarray` - N-dimensional arrays for matrix operations
- `rand`, `rand_distr` - Random number generation for initialization and sampling
- `anyhow` - Error handling
