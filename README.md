# Simple LLM - Educational GPT-2 Implementation in Rust

An educational implementation of a GPT-2 style Transformer model written in Rust. This project is designed to help understand the fundamental concepts behind language models like ChatGPT.

## Overview

This project implements a simplified version of the Transformer architecture, featuring:

- Multi-Head Attention mechanism
- Layer Normalization (Pre-LN configuration)
- Position-wise Feed-Forward Networks
- Text generation with temperature sampling

## Features

- ✅ Working Transformer implementation with attention layers
- ✅ Simple tokenizer for text processing
- ✅ **Real training with backpropagation** - the model actually learns patterns!
- ✅ Interactive text generation interface
- ✅ Support for Japanese and English text
- ✅ Educational focus with clear, understandable code
- ✅ Gradient descent optimization for embeddings and output layer

## Prerequisites

- Rust 1.70 or later
- Basic understanding of neural networks (helpful but not required)

## Installation

```bash
git clone https://github.com/yourusername/simple-llm.git
cd simple-llm
cargo build --release
```

## Usage

### Prepare Training Data

Create training data files with one sentence per line:

**For Japanese (data.txt):**
```text
こんにちは 世界
今日は いい 天気です
プログラミングは 楽しい
```

**For English (data_en.txt):**
```text
Hello world
The quick brown fox jumps over the lazy dog
Rust is a systems programming language
```

### Run Training and Generation

```bash
# For Japanese training data
cargo run

# For English training data
cargo run -- en
```

The program will:
1. Load and tokenize the training data
2. Train the model for 100 epochs (watch the loss decrease!)
3. Enter interactive mode where you can input prompts for text generation

### Interactive Mode

After training, you can interact with the model:

```
Enter a prompt (or 'quit' to exit): Hello
Generated text: world
```

### Training Progress Example

```
Epoch 1: 平均損失 = 4.7132
Epoch 50: 平均損失 = 3.5241
Epoch 100: 平均損失 = 2.9444
```

The model actually learns! The loss decreases over time, and it successfully learns patterns like "Hello" → "world" from the training data.

## Project Structure

```
simple-llm/
├── src/
│   ├── lib.rs                    # Library entry point
│   ├── main.rs                   # Main executable
│   └── working_transformer.rs    # Transformer implementation
├── docs/                         # Documentation
├── data.txt                      # Training data
├── Cargo.toml                    # Project configuration
├── Cargo.lock                    # Dependency lock file
└── README.md                     # This file
```

## Documentation

For detailed documentation, see the [docs/](docs/) directory:
- [PROCESSING_FLOW.md](docs/PROCESSING_FLOW.md) - Detailed explanation of how the model works
- [IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md) - Implementation status and features

## Limitations

This is an educational implementation with several limitations:

- **Simplified backpropagation** - Only embeddings and output layer are updated (attention weights remain fixed)
- **Basic gradient computation** - Manual implementation without automatic differentiation
- No GPU support
- Basic tokenization (whitespace-based)
- Limited model size for educational purposes
- No advanced optimizers (uses simple gradient descent)

**Note**: While the model can now learn simple patterns (like "Hello" → "world"), it's still a simplified implementation designed for educational purposes. For complex language understanding, a full implementation with complete backpropagation through all layers would be needed.

## Learning Resources

If you're new to Transformers and language models:

1. Start by reading [PROCESSING_FLOW.md](docs/PROCESSING_FLOW.md) to understand the overall architecture
2. Run the code with a small dataset to see how training works
3. Experiment with different prompts in interactive mode
4. Modify the code to deepen your understanding

## Contributing

This is an educational project. Contributions that improve clarity, add explanations, or fix bugs are welcome!

## License

This project is dual-licensed under MIT and Apache 2.0. See LICENSE files for details.

## Acknowledgments

This implementation is inspired by the original Transformer paper "Attention Is All You Need" and the GPT-2 architecture by OpenAI.

---

**Note**: Remember to update the author information and repository URL in `Cargo.toml` before publishing!