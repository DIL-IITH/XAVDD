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

## 4. Pipeline stages

The following commands reproduce the stages in `launch_program.sh` for VGG10K/ConvNet using GPU 0.
### Step 1: Train the BiXAttn teacher network
Train the teacher used for dataset synthesis on the original training dataset. The model uses modality-specific encoders and bidirectional cross-attention blocks.

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_teacher_network.py \
    --dataset vgg10k --device cuda:0 --model convnet \
    --batch_size 128 --lr 0.001 --epochs 100 \
    --fusion ensemble --with_cross_attn
```

Output: `pretrained/vgg10k/convnet/teacher_model_bixattn.pt`

### Step 2: Train the teacher network for soft labels
Train a separate audio-visual teacher without cross-attention for student supervision during evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_teacher_network.py \
    --dataset vgg10k --device cuda:0 --model convnet \
    --batch_size 128 --lr 0.001 --epochs 100 --fusion ensemble
```

Output: `pretrained/vgg10k/convnet/teacher_model_wo_attn.pt`

### Step 3: Extract class-wise embeddings
Use the frozen BiXAttn teacher to extract patch embeddings and cross-attended embeddings from the real training dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python3 obtain_embeddings.py \
    --dataset vgg10k --model convnet --device cuda:0 --batch_size 128 \
    --arch_path pretrained/vgg10k/convnet/teacher_model_bixattn.pt
```

Outputs:

```text
patch_embeddings/vgg10k/convnet/{audio,video}/class<ID>.pt
attended_embeddings/vgg10k/convnet/{audio,video}/class<ID>.pt
```

### Step 4: Obtain initialization data

Sample paired real audio-visual examples per class to initialize the synthetic data.

```bash
CUDA_VISIBLE_DEVICES=0 python3 obtain_init_data.py --dataset vgg10k
```

Outputs:

```text
init_data/vgg10k/audio/tensor_<ID>.pt
init_data/vgg10k/frame/tensor_<ID>.pt
```

### Step 5: Generate the distilled dataset

Optimize synthetic audio and visual inputs using the frozen BiXAttn teacher. 

```bash
bash launch_synthesis_vgg10k.sh 0
```

Synthetic data is saved as paired PNG visual samples and PyTorch audio tensors:

```text
syn_data/vgg10k/convnet/syn_av_init_attn/
├── new000/
│   ├── class000_id000.png
│   ├── class000_id000.pt
│   └── ...
├── new001/
└── ... (through new009)
```

### Step 6: Evaluate the distilled dataset

Train student networks without cross-attention on the synthetic dataset and evaluate them on the original test split. The teacher without cross-attention provides soft targets during training.

```bash
bash evaluate_syn_data_vgg10k.sh 0 \
    pretrained/vgg10k/convnet/teacher_model_wo_attn.pt convnet 0.0
```

For additional CLI options, run `python3 <script>.py --help` after installing dependencies.


## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{
mohanty2026xavdd,
title={X-{AVDD}: Cross-Attentive Audio-Visual Dataset Distillation},
author={Saumyaranjan Mohanty and Aravind Reddy and Konda Reddy Mopuri},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=MzMWvUkNZw}
}
```

## 🙏 Acknowledgments

This codebase uses dataset released by [AVDD][https://github.com/sakshamsingh1/AVDD]. We thank the authors for making their work publicly available.