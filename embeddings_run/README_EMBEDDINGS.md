# Training Mamba RNA on Embeddings

This folder contains the complete setup for training the Mamba RNA model using pre-computed compressed embeddings (`.pt` files). This assumes you have already generated `processed_dataset.pt` or `processed_dataset.pt.xz`.

## 1. Setup

### A. Environment
Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

### B. Data
Place your `processed_dataset.pt` file in this directory.
If you have the compressed version (`processed_dataset.pt.xz`), decompress it first:
```bash
xz -d -k processed_dataset.pt.xz
```

## 2. Running Training

Since the costly data processing is already done, you can launch training immediately. The `dataset.py` in this folder is optimized to execute `int8`/`int16` compressed tensors directly.

**Command:**
```bash
python train.py \
    --data_path processed_dataset.pt \
    --epochs 50 \
    --output_dir checkpoints_embeddings \
    --batch_size 32 \
    --gradient_accumulation_steps 1
```

### Key Flags:
*   `--data_path`: Path to your `.pt` file.
*   `--batch_size`: Adjust based on your GPU memory (e.g., 32 for A100, 8 for smaller GPUs).
*   `--fp16`: (Optional) Use `--fp16` if your GPU supports mixed precision.

## 3. Verify Training
To check if training is actually happening:
1.  Look for the progress bar (tqdm) in the console.
2.  Check the Loss value—it should decrease over steps.
3.  Check `checkpoints_embeddings/` for saved models (defaults to every epoch).

## 4. Troubleshooting
*   **OOM (Out Of Memory)**: Reduce `--batch_size`.
*   **"File not found"**: Ensure `processed_dataset.pt` is in the current folder.
