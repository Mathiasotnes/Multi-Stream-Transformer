# Multi-Stream Transformer

---

### 1. Introduction

This project implements a novel transformer architecture that processes text through multiple parallel streams, each trained on different objectives. The core idea is to enhance the model's understanding and generation capabilities by combining different types of pattern recognition. The parallel streams will be fused into a common residual stream optimized for next-token prediction, and the goal is to enhance the model's underlying comprehension.

---

### 2. Setup

The required libraries can be downloaded throught the requirements file:
```
pip install -r requirements.txt
```

Run this afterwards for installing pytorch version compatible with CUDA 11.6:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

---

### 3. Repo activity
![Alt](https://repobeats.axiom.co/api/embed/63f6b8809ab51b7382c8edf0af1a101375e5a4dd.svg "Repobeats analytics image")
