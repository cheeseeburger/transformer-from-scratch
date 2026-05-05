# Transformer From Scratch 

A complete implementation of the Transformer architecture from scratch using PyTorch.
No HuggingFace. No pre-built transformer libraries. Every component built manually.


## Why I Built This

Most people use pre-built libraries without understanding what's happening underneath.
This project is about understanding the core mechanics of attention and sequence modeling
that power GPT, BERT, and every modern LLM.



## Architecture Overview

### 1. Multi-Head Attention (`attention.py`)
The core of the transformer. Instead of one attention function, we run
multiple attention heads in parallel — each learning different relationships
between tokens.

Formula: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

- Q (Query) — what am I looking for?
- K (Key) — what do I have?
- V (Value) — what do I return?
- Scaling by sqrt(d_k) prevents vanishing gradients with large values

### 2. Positional Encoding (`encoder.py`)
Transformers have no inherent sense of order — unlike RNNs.
Positional encoding injects position information using sin/cos waves
at different frequencies so the model knows where each token sits
in the sequence.

### 3. Encoder (`encoder.py`)
Each encoder block has:
- Multi-head self attention
- Feed forward network (2 linear layers + ReLU)
- Residual connections (Add & Norm) after each sublayer
- Layer Normalization for training stability

### 4. Decoder (`decoder.py`)
Each decoder block has THREE sublayers (unlike encoder's two):
- Masked self attention — decoder looks at itself but can't see future tokens
- Cross attention — Q from decoder, K and V from encoder output
  (this is how decoder connects to and uses the encoder's understanding)
- Feed forward network

### 5. Causal Masking
During training, the decoder must not cheat by looking at future tokens.
A triangular mask blocks future positions so the model predicts
each token using only past context.



## Task: Sequence Reversal

The model is trained to reverse integer sequences.

Input:  [3, 7, 1, 9, 4, 2, 8, 6, 5, 10]
Output: [10, 5, 6, 8, 2, 4, 9, 1, 7, 3]

Simple enough to verify correctness, meaningful enough to
prove the encoder-decoder pipeline works end to end.



## Training Results

| Epoch | Loss   |
|-------|--------|
| 100   | 2.5809 |
| 200   | 2.0367 |
| 300   | 1.1414 |
| 400   | 0.4761 |
| 500   | 0.3047 |

Loss dropped from **2.58 → 0.30** over 500 epochs — 
proving the model is genuinely learning the reversal pattern.



## Project Structure
---

## How To Run

```bash
# Install dependencies
pip install torch numpy matplotlib

# Verify transformer works
python transformer.py

# Train the model
python train.py
```

---

## Key Concepts Implemented
- Scaled dot-product attention
- Multi-head attention with head splitting and concatenation
- Sinusoidal positional encoding
- Residual connections
- Layer normalization
- Teacher forcing during training
- Padding and causal masking
- Cross attention between encoder and decoder

---

## Built With
- Python 3.9
- PyTorch

---

*Built from scratch to understand the internals of modern LLMs.*
