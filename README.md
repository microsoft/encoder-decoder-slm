# Return of the Encoder: Efficient Small Language Models

## Overview
While large language models continue to grow in size, smaller models (≤1B parameters) require thoughtful architectural decisions. Our work demonstrates that encoder-decoder models inherently outperform decoder-only architectures before any optimizations:

- Base encoder-decoder achieves +2-4% performance improvement across tasks
- After knowledge distillation, performance gains increase to +6-8%
- Significantly more efficient than decoder-only counterparts:
 - 📉 47% lower first-token latency
 - 🚀 4.7x higher throughput on edge devices
 - 💾 11-16% less memory usage
 - ⚡ 22% fewer FLOPs for sequence generation

We note that our work focuses on architectural comparisons rather than competing with recent SLM developments (e.g., SmolLM, MobileLLM). Our analysis isolates the fundamental advantages of encoder-decoder versus decoder-only designs in sub-1B parameter regimes, with particular emphasis on deployment efficiency.

![Architectural Comparison](IntroFigure.png)
*Architectural Efficiency in SLMs. Left: Comparison of architectures where encoder-decoder creates a fixed input representation with KV cache only for output, while decoder-only requires growing KV caches for both input and output. Top right: Inference time scaling with input length, showing encoder-decoder's efficient fixed-representation approach versus decoder-only's steeper computational growth. Bottom right: Performance across tasks showing encoder-decoder's advantages at fixed compute budget, further enhanced by KD.*

## Technical Highlights
- **Efficient Base Architecture**: 2/3-1/3 encoder-decoder split consistently outperforms decoder-only
- **Enhanced Performance**: Knowledge distillation from larger teachers while maintaining architectural benefits
- **Hardware Efficiency**: Superior across GPU (86ms), CPU (1591ms), and NPU (189ms) platforms

## Performance
Our 330M parameter model outperforms decoder-only baselines (given same training data & FLOPs):
- SQuAD 2.0: 0.69/0.94 vs 0.57/0.90
- IELTS: 0.32/0.46 vs 0.31/0.40
- CodeXGLUE: 0.93/0.74 vs 0.93/0.63
- XSum: 0.27/0.20 vs 0.24/0.19
We also show that results continue as we scale the models up to 1B parameters.

## Usage
### Package Installation
Before running our code, please create a virtual conda environment using python==3.10, and install necessary packages.
```bash
cd encoder-decoder-slm
conda create -n slm_env python=3.10 -y
conda activate slm_env
pip install --upgrade pip
pip install -e .
```

### Text2text Inference
We provide example inference code for a text2text model. Feel free to modify the `question` and `context` values in `src/mu/generate_text2text.py` if you want to try other examples.
```bash
cd encoder-decoder-slm
python -m mu.generate_text2text
```

### Text+image2text Inference
We provide example inference code for a text+image2text model. Several images are included under `artifacts/images` for you to try. Feel free to modify the `image_file` and `question` values in `src/mu/generate_text+image2text.py` if you want to try other examples.
```bash
cd encoder-decoder-slm
python -m mu.generate_text+image2text
```

### Training
Run KD training using the following command
```bash
cd encoder-decoder-slm
torchrun --nproc_per_node=${GPU_COUNT} -m mu.train_text2text_by_kd
```
Note that the KD training code references a 'teacher.pt' (which should be placed at `artifacts/models/teacher.pt`) which is a LoRA fine-tuned version of Phi-3-mini available on Hugging Face.

⭐ Star this repository to get notified when we release the rest of the codes and models!

