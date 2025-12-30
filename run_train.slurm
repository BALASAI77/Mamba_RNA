#!/bin/bash
#SBATCH --job-name=mamba_train
#SBATCH --account=lfovia
#SBATCH --qos=lfovia_qos
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --time=24:00:00

# 1. Load Modules
module load miniconda/3

# 2. Activate Environment
source activate mamba_arch

# 3. Clean any existing temp vars to avoid "No space left"
mkdir -p ~/tmp_build
export TMPDIR=~/tmp_build

# 4. Run Training
# Note: Using 1 GPU (as per user limit)
# Batch size reduced to 32/64 to fit in 32GB RAM / 1 GPU VRAM
python train.py \
    --data_path processed_dataset \
    --output_dir mamba_rna_checkpoints \
    --epochs 15 \
    --batch_size 64
