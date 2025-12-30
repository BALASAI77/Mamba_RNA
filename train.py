import os
import argparse
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from model import MambaRNAConfig, MambaRNAForMaskedLM
from dataset import RNADataset, BNATokenizer, data_collator_structural_mlm

class CustomCheckpointCallback(TrainerCallback):
    """
    Callback to save checkpoints every 5 epochs with a specific naming format.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch is often a float (e.g. 1.0), so we round it
        epoch = int(round(state.epoch))
        if epoch > 0 and epoch % 5 == 0:
            output_dir = os.path.join(args.output_dir, f"epoch{epoch}_checkpoints")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save model with unwrapping if necessary
            if 'model' in kwargs:
                model_to_save = kwargs['model']
                if hasattr(model_to_save, 'module'):
                    model_to_save = model_to_save.module
                model_to_save.save_pretrained(output_dir)
            
            print(f"Saved custom checkpoint for epoch {epoch} to {output_dir}")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Clustered_10k_dataset.csv")

    parser.add_argument("--output_dir", type=str, default="mamba_rna_checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print("Initializing Model...")
    # Config similar to StructRFM base but with Mamba
    config = MambaRNAConfig(
        vocab_size=10, 
        d_model=768, 
        n_layer=12,
        dropout=0.1
    )
    model = MambaRNAForMaskedLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print(f"Loading Data from {args.data_path}...")
    tokenizer = BNATokenizer()
    dataset = RNADataset(args.data_path, tokenizer=tokenizer, max_length=512)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        dataloader_num_workers=4,
        remove_unused_columns=False, # Important for custom datasets
        gradient_checkpointing=True,
        save_total_limit=1,           # Internal temporary checkpoints
        load_best_model_at_end=True,  # Load best model at end
        metric_for_best_model="eval_loss", 
        greater_is_better=False,      
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: data_collator_structural_mlm(x, tokenizer),
        callbacks=[CustomCheckpointCallback()]
    )

    print("Starting Training...")
    trainer.train()

    # Save the absolute best model (loaded automatically at end) to "best_checkpoint"
    best_path = os.path.join(args.output_dir, "best_checkpoint")
    print(f"Saving best model to {best_path}...")
    trainer.save_model(best_path)

if __name__ == "__main__":
    train()
