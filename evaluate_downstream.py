import argparse
import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from model import MambaRNAConfig
from model_downstream import MambaRNAForSequenceClassification, MambaRNAForTokenClassification
from dataset import BNATokenizer

def compute_metrics_cls(eval_pred):
    logits, labels = eval_pred
    # logits might be tuple
    if isinstance(logits, tuple): logits = logits[0]
    
    predictions = torch.tensor(logits).argmax(dim=-1)
    labels = torch.tensor(labels)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

def tokenize_cls(examples, tokenizer, max_length):
    # examples['seq'] -> list of strings
    # examples['label'] -> list of ints
    tokenized = [tokenizer.encode(seq, max_length=max_length) for seq in examples['seq']]
    # Pad to max_length for batching (simple manual padding or use collator)
    # The simple BNATokenizer encode returns tensor. We convert to list for HF Dataset.
    input_ids = [t.tolist() for t in tokenized]
    return {"input_ids": input_ids, "labels": examples['label']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "ssp"], help="Task to evaluate")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="mamba_rna_checkpoints/best_checkpoint")
    parser.add_argument("--output_dir", type=str, default="downstream_results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    tokenizer = BNATokenizer()
    
    print(f"Loading datasets from {args.train_file} and {args.val_file}...")
    dataset = load_dataset("csv", data_files={"train": args.train_file, "val": args.val_file})
    
    # Load config and model
    print(f"Loading model from {args.model_path}...")
    # We load config from the checkpoint to ensure dimensions match
    config = MambaRNAConfig.from_pretrained(args.model_path)
    
    if args.task_type == "classification":
        # Check num labels
        labels_set = set(dataset['train']['label'])
        num_labels = len(labels_set)
        print(f"Detected {num_labels} classes for classification.")
        config.num_labels = num_labels
        
        model = MambaRNAForSequenceClassification.from_pretrained(args.model_path, config=config)
        
        # Tokenize
        def preprocess_function(examples):
            return tokenize_cls(examples, tokenizer, 512)
            
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        compute_metrics = compute_metrics_cls
        
    elif args.task_type == "ssp":
        print("Secondary Structure Prediction (Token Classification)")
        # For dummy SSP, we treat it as 3-class classification: . ( ) -> 0 1 2
        # Real task might be more complex.
        
        vocab_map = {'.': 0, '(': 1, ')': 2}
        config.num_labels = 3
        model = MambaRNAForTokenClassification.from_pretrained(args.model_path, config=config)
        
        def preprocess_ssp(examples):
            input_ids_batch = []
            labels_batch = []
            for seq, struc in zip(examples['seq'], examples['structure']):
                # Encode seq
                tokens = tokenizer.encode(seq, max_length=512)
                input_ids_batch.append(tokens.tolist())
                
                # Encode structure
                # Align with tokens (CLS ... SEP)
                # CLS -> -100
                # SEP -> -100
                # Pad -> -100
                lbls = [-100] # CLS
                struc_trunc = struc[:512-2] # truncate
                for char in struc_trunc:
                    lbls.append(vocab_map.get(char, 0)) # Default to 0 (.)
                lbls.append(-100) # SEP
                
                # Padding
                if len(lbls) < 512:
                    lbls += [-100] * (512 - len(lbls))
                
                labels_batch.append(lbls)
                
            return {"input_ids": input_ids_batch, "labels": labels_batch}

        tokenized_datasets = dataset.map(preprocess_ssp, batched=True)
        compute_metrics = None # TODO: Add token accuracy

    # Data Collator
    # We need a simple collator that pads input_ids
    from transformers import DataCollatorWithPadding
    # Our tokenizer isn't standard HF, so DataCollatorWithPadding might verify tokenizer type.
    # We'll use a simple lambda.
    
    def collate_fn(features):
        batch = {}
        # Pad input_ids
        max_len = max(len(f['input_ids']) for f in features)
        
        padded_ids = []
        padded_labels = []
        
        for f in features:
            # Pad inputs
            ids = f['input_ids'] + [0] * (max_len - len(f['input_ids']))
            padded_ids.append(ids)
            
            # Pad labels if present (simple stacking)
            if 'labels' in f:
                if isinstance(f['labels'], int): # for CLS task
                    padded_labels.append(f['labels'])
                else: # for Token task
                     # Pad with -100
                     l = f['labels'] + [-100] * (max_len - len(f['labels']))
                     padded_labels.append(l)
        
        batch['input_ids'] = torch.tensor(padded_ids)
        batch['labels'] = torch.tensor(padded_labels)
        return batch

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    print("Starting Training/Evaluation...")
    trainer.train()
    
    print("Evaluation Results:")
    eval_res = trainer.evaluate()
    print(eval_res)

if __name__ == "__main__":
    main()
