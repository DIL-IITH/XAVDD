# XAVDD: Audio-Visual Dataset Distillation

This repository provides implementation for our NeurIPS 2026 paper `X-AVDD: Cross-Attentive Audio-Visual Dataset Distillation'. It proposes an audio-visual dataset distillation pipeline using bidirectional cross-attention (BiXAttn). The pipeline trains teacher networks, extracts class-wise audio and visual representations, initializes and optimizes synthetic audio-visual samples, and evaluates student networks trained on the distilled dataset.

## 1. Installation

Requirements: Python 3, Bash, and a CUDA-capable GPU with a compatible PyTorch installation. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install numpy pillow tqdm prettytable torchmetrics wandb h5py
```

## 2. Dataset preparation

The code expects **preprocessed, paired audio and video tensors**, similar to AVDD. The directory structure taking **VGG10K** as example:

```text
data/
└── VGG10K/
    ├── train_data/
    │   ├── audio/
    │   │   ├── class0/
    │   │   │   ├── tensor0.pt
    │   │   │   └── tensor1.pt
    │   │   └── ... (class1 through class9)
    │   └── video/
    │       ├── class0/
    │       │   ├── tensor0.pt
    │       │   └── tensor1.pt
    │       └── ... (class1 through class9)
    └── test_data/
        ├── audio/
        └── video/
```


## 3. Run the complete pipeline

Make the shell scripts executable:

```bash
chmod +x launch_program.sh launch_synthesis_vgg10k.sh recover.sh evaluate_syn_data_vgg10k.sh
```

The launcher accepts three positional arguments:

```bash
bash launch_program.sh <GPU_ID> <DATASET> <MODEL>
```

For example, for VGG10K using ConvNet on GPU 0:

```bash
bash launch_program.sh 0 vgg10k convnet
```
