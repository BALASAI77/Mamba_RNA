# Master Guide: Mamba RNA Pretraining on A100 Cluster

This guide covers every step from logging in to running your training job. Follow these exact commands.

## 1. Login to Cluster
Open your terminal (on your local machine) and SSH into the cluster.
*(IP and default password taken from your usage guide)*
```bash
ssh user@103.42.201.29
# Password: user123 (or your changed password)
```

## 2. Clone the Repository
Once logged in:
```bash
git clone https://github.com/BALASAI77/Mamba_RNA.git
cd Mamba_RNA
```

## 3. Set Up Environment (CRITICAL STEP)
This specific setup handles the "CUDA Mismatch" and "No Space" errors we found.

```bash
# 1. Load Conda Module
module load miniconda/3

# 2. Create Environment
conda create -n mamba_arch python=3.10 -y
source activate mamba_arch

# 3. Install PyTorch with MATCHING CUDA Compiler (12.4)
# This prevents the "CUDA 13.1 mismatch" error.
# We explicitly install cuda-nvcc to ensure the compiler is present for Mamba.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 cuda-nvcc -c pytorch -c nvidia -y

# 4. Install Helper Packages
pip install packaging

# 5. Install Mamba (Compilation Step)
# We set MAX_JOBS to prevent crashing, and TMPDIR to prevent "No space left"y
mkdir -p ~/tmp_build
export TMPDIR=~/tmp_build
# Ensure nvcc is found in path
export CUDA_HOME=$CONDA_PREFIX
MAX_JOBS=4 pip install mamba-ssm causal-conv1d

# 6. Install remaining requirements
pip install -r requirements.txt
```

## 4. Transfer & Prepare Data
You need to move your dataset from your **local machine** to the **cluster**.

**A. Upload Data (Run on LOCAL Terminal):**
```bash
# Replace /local/path/to/... with where your file actually is
scp /local/path/to/clustered_10k_dataset.csv.tar.gz user@103.42.201.29:~/Mamba_RNA/
```

**B. Prepare Data (Run on CLUSTER Terminal):**
```bash
# This extracts the tar.gz and creates a 'processed_dataset' folder
python prepare_large_dataset.py --csv_file clustered_10k_dataset.csv.tar.gz
```

## 5. Submit Training Job
We use **SLURM** to submit the job to the background. 
I have already created `run_train.slurm` tailored to your account limits (1 GPU, 6 CPUs, 32GB RAM).

```bash
# Submit the job
sbatch run_train.slurm
```

## 6. Monitor Your Job
After submitting, checks its status:

```bash
# Check queue status
squeue --me

# Watch the training logs live (replace JOBID with the number from squeue, e.g., 345)
tail -f training_345.out
```

**If the job crashes:** check the error file:
```bash
cat training_345.err
```

## 7. Results
When finished:
*   **Checkpoints** are in: `mamba_rna_checkpoints/`
*   **Logs** are in the `.out` and `.err` files.
