import argparse
import os
import tarfile
from datasets import load_dataset
from dataset import BNATokenizer
import multiprocessing

def extract_tar_file(file_path):
    """
    Extracts a .tar.gz file and returns the path to the extracted CSV.
    Assumes the tarball contains a single CSV or we take the first one found.
    """
    print(f"Extracting {file_path}...")
    extracted_files = []
    
    # Extract to the directory where the tar file is
    extract_dir = os.path.dirname(os.path.abspath(file_path))
    
    with tarfile.open(file_path, "r:gz") as tar:
        # Check for dangerous paths (simple safeguard)
        for member in tar.getmembers():
            if os.path.isabs(member.name) or ".." in member.name:
                print(f"Skipping suspicious extraction path: {member.name}")
                continue
            
            tar.extract(member, path=extract_dir)
            full_path = os.path.join(extract_dir, member.name)
            if full_path.endswith(".csv"):
                 extracted_files.append(full_path)
    
    if not extracted_files:
        raise ValueError(f"No CSV files found in {file_path}")
        
    print(f"Extracted: {extracted_files[0]}")
    return extracted_files[0]

def process_batch(batch):
    tokenizer = BNATokenizer()
    # batch['seq'] is a list of strings
    
    encoded_inputs = []
    
    # Simple iteration - for 20M optimization we might want to vectorize more but this is safe
    for seq in batch['seq']:
        # Ensure it's a string
        if seq is None:
            seq = ""
        seq = str(seq)
        
        # We handle tokenization. 
        # Note: The original tokenizer returns a tensor, but for 'datasets' mapping
        # it is often better to return python lists, and let the collator convert to tensors.
        # But we can reuse the logic.
        
        # To avoid overhead of re-initializing tokenizer or slow loops, 
        # we minimally reimplement parts here or just use it.
        # Ideally we'd modify BNATokenizer to work on batches of strings.
        
        # Let's use the object logic but Convert tensor to list for Arrow storage
        token_tensor = tokenizer.encode(seq, max_length=512, truncation=True)
        encoded_inputs.append(token_tensor.tolist())

    return {"input_ids": encoded_inputs}

def main():
    parser = argparse.ArgumentParser(description="Prepare large RNA dataset")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the large CSV file or .tar.gz archive")
    parser.add_argument("--output_dir", type=str, default="processed_dataset", help="Directory to save the processed HF dataset")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for parallel processing")
    
    args = parser.parse_args()

    input_path = args.csv_file
    
    # Auto-extract if tar.gz
    if input_path.endswith(".tar.gz"):
        input_path = extract_tar_file(input_path)

    print(f"Loading dataset from {input_path}...")
    # 'load_dataset' with 'csv' format is memory efficient (streaming/arrow)
    # It doesn't load everything to RAM at once.
    dataset = load_dataset("csv", data_files=input_path, split="train")

    print(f"Dataset loaded. Number of rows: {len(dataset)}")
    print("Example row:", dataset[0])

    # Check for 'seq' column
    if 'seq' not in dataset.column_names:
        # Try to guess or fallback
        # If the user has no header, 'load_dataset' might assign 'column_1', 'column_2' etc.
        # We assume the user ensures a header or we rename.
        print(f"Columns found: {dataset.column_names}")
        if 'column_1' in dataset.column_names and 'seq' not in dataset.column_names:
             print("Renaming 'column_1' to 'seq'...")
             dataset = dataset.rename_column('column_1', 'seq')
        elif 'sequence' in dataset.column_names:
             dataset = dataset.rename_column('sequence', 'seq')

    print("Tokenizing dataset...")
    # Process in parallel
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        desc="Tokenizing",
        remove_columns=[c for c in dataset.column_names if c not in ['structure', 'id']] # Keep structure/id if needed, remove raw seq to save space?
        # Actually, let's keep everything or decide based on disk space. 
        # Use remove_columns=['seq'] if we want to save space since we have input_ids.
    )

    print(f"Saving processed dataset to {args.output_dir}...")
    processed_dataset.save_to_disk(args.output_dir)
    print("Done! You can now load this dataset in train.py using 'load_from_disk'")

if __name__ == "__main__":
    main()
