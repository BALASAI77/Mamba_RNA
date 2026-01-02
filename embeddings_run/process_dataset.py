import pandas as pd
import torch
import json
import os
import argparse
from tqdm import tqdm
from dataset import BNATokenizer

def process_csv_to_pt(csv_path, output_path, max_length=512):
    print(f"Counting lines in {csv_path}...")
    # Pass 1: Get total size
    total_samples = 0
    with open(csv_path, 'r') as f:
        total_samples = sum(1 for line in f) - 1 # Minus header
    
    print(f"Total samples: {total_samples}")
    
    # Pre-allocate tensors (Int8/Int16) to save memory
    print(f"Pre-allocating tensors (~{total_samples * max_length * 3 / 1024**3:.2f} GB)...")
    final_input_ids = torch.empty((total_samples, max_length), dtype=torch.int8)
    final_structure_ids = torch.empty((total_samples, max_length), dtype=torch.int16)
    
    chunksize = 100000
    tokenizer = BNATokenizer()
    
    current_idx = 0
    
    # Pass 2: Fill
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if 'seq' not in chunk.columns:
            print("Error: 'seq' column not found.")
            return

        # Prepare batch lists
        batch_input_ids = []
        batch_structure_ids = []

        for index, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing chunk (start {current_idx})"):
            seq = str(row['seq'])
            
            # --- STRUCTURE PARSING ---
            structure_data = [0] * max_length
            if 'connects' in row and pd.notna(row['connects']):
                try:
                    raw_struct = json.loads(row['connects'])
                    current_len = min(len(raw_struct), max_length)
                    for i in range(current_len):
                        val = raw_struct[i]
                        if i + 1 < max_length and val > 0:
                            structure_data[i+1] = int(val)
                except:
                    pass
            
            # --- TOKENIZATION ---
            tokens = tokenizer.encode(seq, max_length=max_length)
            if len(tokens) < max_length:
                padding = torch.full((max_length - len(tokens),), tokenizer.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            
            batch_input_ids.append(tokens)
            batch_structure_ids.append(torch.tensor(structure_data, dtype=torch.long))
        
        # Convert batch to tensor and assign
        batch_len = len(batch_input_ids)
        
        # Optimize batch cast
        batch_inputs = torch.stack(batch_input_ids).to(torch.int8)
        batch_structs = torch.stack(batch_structure_ids).to(torch.int16)
        
        final_input_ids[current_idx : current_idx + batch_len] = batch_inputs
        final_structure_ids[current_idx : current_idx + batch_len] = batch_structs
        
        current_idx += batch_len
        
        # Free memory
        del batch_input_ids, batch_structure_ids, batch_inputs, batch_structs, chunk
        
    print(f"Saving to {output_path}...")
    torch.save({
        "input_ids": final_input_ids,
        "structure_ids": final_structure_ids
    }, output_path)
    
    print(f"Done. Saved shape: {final_input_ids.shape}")
    print(f"Dtypes: input_ids={final_input_ids.dtype}, structure_ids={final_structure_ids.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output_pt", type=str, default="processed_dataset.pt", help="Output .pt file")
    args = parser.parse_args()
    
    process_csv_to_pt(args.input_csv, args.output_pt)
