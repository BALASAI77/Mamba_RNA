import pandas as pd
import numpy as np
import os
import random

def create_classification_data(path, n_samples=50):
    """
    Creates dummy data for Splice Site Prediction (Binary Classification)
    """
    data = []
    bases = ['A', 'U', 'G', 'C']
    for _ in range(n_samples):
        seq_len = random.randint(50, 100)
        seq = "".join(random.choices(bases, k=seq_len))
        label = random.randint(0, 1) # 0 or 1
        data.append({"seq": seq, "label": label})
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Created classification dummy data at {path}")

def create_ssp_data(path, n_samples=50):
    """
    Creates dummy data for Secondary Structure Prediction (Contact Map / DotBracket)
    For simplicitly we use dot-bracket string as label for TokenClassification
    """
    data = []
    bases = ['A', 'U', 'G', 'C']
    structs = ['.', '(', ')']
    for _ in range(n_samples):
        seq_len = random.randint(50, 100)
        seq = "".join(random.choices(bases, k=seq_len))
        
        # Create random balanced structure? No, just random chars for dummy test
        # Real structure is harder to mock perfectly without logic, but model doesn't care about valid physics for compilation.
        structure = "".join(random.choices(structs, k=seq_len))
        
        data.append({"seq": seq, "structure": structure})
        
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Created SSP dummy data at {path}")

if __name__ == "__main__":
    os.makedirs("dummy_data", exist_ok=True)
    create_classification_data("dummy_data/splice_train.csv")
    create_classification_data("dummy_data/splice_val.csv")
    create_ssp_data("dummy_data/ssp_train.csv")
    create_ssp_data("dummy_data/ssp_val.csv")
