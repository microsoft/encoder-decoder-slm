# Efficient Small Language Models with Encoder-Decoder Architecture

This repository contains the implementation of "Return of the Encoder: Maximizing Parameter Efficiency for SLMs", showcasing efficient encoder-decoder architectures for small language models (≤1B parameters).

## Overview

Recent trends in language modeling have favored large decoder-only architectures. However, for small language models (SLMs) with 1 billion parameters or fewer, our work demonstrates that encoder-decoder architectures offer significant efficiency advantages:

- 47% lower first-token latency
- 4.7x higher throughput on edge devices
- Superior performance on asymmetric sequence tasks
- Efficient handling of vision-language tasks

## Key Features

### 1. Efficient Encoder-Decoder Architecture
- Optimized 2/3-1/3 encoder-decoder split
- Integration with modern advances:
  - Rotary Positional Embeddings (RoPE)
  - Vision Transformer (ViT) for multimodal tasks
- Efficient sequence length handling

### 2. Cross-Architecture Knowledge Distillation
- Novel framework for distilling from decoder-only teachers
- Sequence alignment strategies for cross-architecture transfer
- Temperature-based optimization

### 3. Vision-Language Integration
- High-resolution image processing pipeline
- Efficient token compression strategy
- Vision-text alignment through projection layers

## Repository Structure
```
efficient-slm/
├── core/
│   ├── modeling/               # Core architecture implementation
│   ├── distillation/           # Knowledge distillation framework
│   └── vision/                 # Vision-language integration
├── configs/                    # Model and training configurations
├── training/                   # Training scripts and utilities
├── evaluation/                # Evaluation scripts
└── examples/                  # Usage examples and notebooks
```

## Performance

Our architecture demonstrates consistent improvements across various tasks:

- SQuAD 2.0: 0.69/0.94 (RougeLsum/Ragas-GPT)
- IELTS: 0.32/0.46
- CodeXGLUE: 0.93/0.74
- XSum: 0.27/0.20

Hardware efficiency (330M parameter model):
- GPU: 86ms first-token latency, 37.4 tokens/s throughput
- CPU: 1591ms first-token latency, 15.3 tokens/s throughput
- NPU: 189ms first-token latency, 123.8 tokens/s throughput

## Installation and Usage

We will be adding the code soon... Stay tuned

## Citation

That too will be added soon :D
