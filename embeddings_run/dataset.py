import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import json
import numpy as np


class BNATokenizer:
    """Simple RNA Tokenizer for A, U, G, C, N and specials."""
    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "A": 5,
            "U": 6,
            "G": 7,
            "C": 8,
            "N": 9
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.unk_token_id = 1
        
    def __len__(self):
        return len(self.vocab)
        
    def encode(self, text, max_length=512, truncation=True, padding="max_length"):
        # text is a string of RNA bases
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text.upper()]
        
        # Add CLS/SEP
        tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token_id]
            
        if padding == "max_length" and len(tokens) < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            
        return torch.tensor(tokens, dtype=torch.long)

class RNADataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer if tokenizer else BNATokenizer()
        self.max_length = max_length
        self.is_preprocessed = False
        
        if file_path.endswith('.pt'):
            print(f"Loading pre-processed dataset from {file_path}...")
            data = torch.load(file_path)
            self.input_ids = data['input_ids']
            self.structure_ids = data['structure_ids']
            self.is_preprocessed = True
            self.length = len(self.input_ids)
            print(f"Loaded {self.length} samples.")
            return

        # Legacy CSV handling
        try:
            # Read first line to check for header
            df_head = pd.read_csv(file_path, nrows=5)
             # ... (rest of old logic fallback) ...
            if 'seq' in df_head.columns or 'name' in df_head.columns:
                 self.data = pd.read_csv(file_path)
            else:
                 self.data = pd.read_csv(file_path, header=None, names=['id', 'seq', 'structure'])
        except Exception:
             self.data = pd.read_csv(file_path)
             
        self.length = len(self.data)

             
        self.tokenizer = tokenizer if tokenizer else BNATokenizer()
        self.max_length = max_length
        
        # Normalize columns
        if 'seq' not in self.data.columns and 'id' in self.data.columns:
             pass
        if 'seq' not in self.data.columns and len(self.data.columns) >= 2:
             self.data.rename(columns={self.data.columns[1]: 'seq'}, inplace=True)
        if 'structure' not in self.data.columns and len(self.data.columns) >= 3:
             self.data.rename(columns={self.data.columns[2]: 'structure'}, inplace=True)
             
    def __len__(self):
        if self.is_preprocessed:
            return self.length
        return len(self.data)
        
    def __getitem__(self, idx):
        if self.is_preprocessed:
            return {
                "input_ids": self.input_ids[idx].long(),
                "structure_ids": self.structure_ids[idx].long()
            }

        row = self.data.iloc[idx]
        seq = str(row['seq'])
        
        # Parse structure
        structure_ids = []
        if 'structure' in row:
            try:
                s_str = row['structure']
                if isinstance(s_str, str):
                    structure_ids = json.loads(s_str)
                elif isinstance(s_str, list):
                    structure_ids = s_str
            except:
                structure_ids = [0] * len(seq)
        
        # Handle length mismatch
        if len(structure_ids) != len(seq):
            if len(structure_ids) < len(seq):
                structure_ids += [0] * (len(seq) - len(structure_ids))
            else:
                structure_ids = structure_ids[:len(seq)]

        input_ids = self.tokenizer.encode(seq, max_length=self.max_length)
        
        # Pad structure_ids to match input_ids length
        # [CLS] -> 0
        padded_structure = [0] * len(input_ids)
        
        n_bases = min(len(seq), self.max_length - 2)
        
        for i in range(n_bases):
            # i is 0-based sequence index
            # Structure value val is 1-based index (if > 0)
            # We want to map this to TOKEN index.
            # Token index for sequence pos i is i+1.
            # If sequence pos i pairs with pos k (k is 0-based), structure val is k+1.
            # Token index for pos k is k+1 = structure val.
            # So we just take structure val as the paired token index directly.
            
            val = structure_ids[i]
            if val > 0:
                # Ensure val points to a valid token index
                if val <= n_bases: # val is 1-based index limit
                     padded_structure[i+1] = int(val) # Points to token index directly
        
        return {"input_ids": input_ids, "structure_ids": torch.tensor(padded_structure, dtype=torch.long)}

def data_collator_mlm(features, tokenizer, mlm_probability=0.15):
    """
    Data collator for Masked Language Modeling.
    """
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = input_ids.clone()
    
    # Probability matrix
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        # 1 if special token, 0 otherwise
        [1 if (token in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]) else 0 for token in val]
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% convert to [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    # 10% random (we deal with this simply by doing nothing for the other 20%, or valid random, but for simplicity just Masking most of time)
    # The default BERT masking logic is a bit more complex (80% mask, 10% random, 10% original).
    # For this implementation, let's stick to simple masking for now or add the rest if critical.
    # Let's add full logic for completeness.
    
    # 10% random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    # The remaining 10% (masked_indices & ~indices_replaced & ~indices_random) stay original
    
    return {"input_ids": input_ids, "labels": labels}

def data_collator_structural_mlm(features, tokenizer, mlm_probability=0.15):
    """
    Structure-Guided Masked Language Modeling collator.
    Masks paired bases together.
    """
    input_ids = torch.stack([f["input_ids"] for f in features])
    structure_ids = torch.stack([f["structure_ids"] for f in features])
    labels = input_ids.clone()
    
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    special_tokens_mask = [
        [1 if (token in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]) else 0 for token in val]
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Enforce structure masking: if i is masked and paired with j, mask j
    batch_size, seq_len = input_ids.shape
    for b in range(batch_size):
        # Get indices currently masked
        current_masked = torch.where(masked_indices[b])[0]
        for idx in current_masked:
            # Check structure
            partner = structure_ids[b, idx].item()
            if partner > 0 and partner < seq_len:
                # Mask the partner too
                masked_indices[b, partner] = True
    
    labels[~masked_indices] = -100

    # 80% convert to [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    return {"input_ids": input_ids, "labels": labels}
