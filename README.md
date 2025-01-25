# Return of the Encoder: Efficient Small Language Models

📢 **Coming Soon**: Code and pretrained models will be available shortly!

## Overview
While large language models continue to grow, we show that smaller models (≤1B parameters) can be remarkably efficient with the right architecture. Our encoder-decoder approach achieves:

- 📉 47% lower first-token latency
- 🚀 4.7x higher throughput on edge devices  
- 💪 Superior performance on asymmetric tasks
- 🖼️ Efficient vision-language capabilities

## Key Innovations

- **Efficient Architecture**: Optimized 2/3-1/3 encoder-decoder split with RoPE and ViT integration
- **Novel Distillation**: Cross-architecture knowledge transfer from larger decoder-only teachers
- **Hardware Efficiency**: Demonstrated across GPU (86ms), CPU (1591ms), and NPU (189ms) platforms

## Results Highlights
Our 330M parameter model achieves strong performance across:
- SQuAD 2.0: 0.69/0.94
- IELTS: 0.32/0.46  
- CodeXGLUE: 0.93/0.74
- XSum: 0.27/0.20

⭐ Star this repository to get notified when we release the code and models!
