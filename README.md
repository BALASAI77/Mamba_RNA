# Mamba RNA Foundation Model - A100 Training Guide

This repository contains the complete codebase to pretrain and evaluate a Mamba-based RNA foundation model. It is optimized for high-performance GPUs (like NVIDIA A100).

## 1. Setup Environment

On your A100 machine, ensure you have CUDA drivers installed (typically CUDA 11.8 or 12.x).

```bash
# Create a fresh environment
conda create -n mamba_arch python=3.10
conda activate mamba_arch

# Install dependencies
pip install -r requirements.txt
```

> **Note**: `mamba-ssm` and `causal-conv1d` require CUDA compilation. If pip installation fails, ensure `nvcc` is in your path or install pre-built wheels matching your CUDA version.

## 2. Prepare Data

Transfer your large dataset (the `.tar.gz` or `.csv` file) to the machine. You do NOT need to unzip it manually.

```bash
# Example: Prepare the 20 million sequence dataset
# This script uses parallel processing to tokenize data and save it to disk for instant loading.
python prepare_large_dataset.py --csv_file /path/to/clustered_10k_dataset.csv.tar.gz --output_dir processed_20m_data
```

## 3. Pretrain on A100

Run the training script. It will automatically use the GPU.

```bash
# Run training (using the processed data from step 2)
# Since prepare_large_dataset saves to a folder, point data_path to it.
python train.py --data_path processed_20m_data --output_dir mamba_rna_checkpoints --epochs 15 --batch_size 128
```

> **Tips for A100:**
> - You can likely increase `--batch_size` to 128, 256 or more depending on sequence length (A100 has 40GB/80GB VRAM).
> - Enable `bf16` if supported (Ampere architecture supports it well) by modifying `train.py` arguments if needed, though `fp16` (default) works fine.

**Checkpoints:**
- Saved to `mamba_rna_checkpoints/`
- `best_checkpoint/`: The best model based on validation loss.
- `epoch5_checkpoints/`, `epoch10_checkpoints/`: Periodic snapshots.

## 4. Evaluate Downstream Tasks

After pretraining, you can verify performance on downstream tasks (Splice Site Prediction, Secondary Structure, etc.).

**Prepare or Simulate Data:**
```bash
# Generate dummy data for testing the pipeline if real data is not ready
python create_dummy_data.py
```

**Run Evaluation:**
```bash
# Classification Task
python evaluate_downstream.py \
    --task_type classification \
    --train_file dummy_data/splice_train.csv \
    --val_file dummy_data/splice_val.csv \
    --model_path mamba_rna_checkpoints/best_checkpoint

# Secondary Structure Prediction
python evaluate_downstream.py \
    --task_type ssp \
    --train_file dummy_data/ssp_train.csv \
    --val_file dummy_data/ssp_val.csv \
    --model_path mamba_rna_checkpoints/best_checkpoint
```



## Troubleshooting Installation

### Error: `CUDA version mismatch` (e.g., PyTorch 12.8 vs System 13.1)
The error `detected CUDA version 13.1 mismatches...` means you installed the latest CUDA compiler (13.1) but your PyTorch was built for 12.8. They must match.

**Solution:** Downgrade the compiler to match PyTorch.

```bash
# 1. Remove the wrong version (13.x)
conda remove -y cuda-nvcc

# 2. Install the CORRECT version (12.8)
conda install -c nvidia cuda-nvcc=12.8

# 3. Now install Mamba
MAX_JOBS=4 pip install mamba-ssm causal-conv1d
```

## 5. Inference


To run the model on a single sequence:

```bash
python predict_inference.py --sequence "AUGGC..." --cls_model_path mamba_rna_checkpoints/best_checkpoint
```
