import torch
import argparse
from model_downstream import MambaRNAForSequenceClassification, MambaRNAForTokenClassification
from dataset import BNATokenizer
from model import MambaRNAConfig
import torch.nn.functional as F

def predict(args):
    tokenizer = BNATokenizer()
    seq = args.sequence.upper()
    print(f"\nScanning Sequence (Len {len(seq)}):\n{seq}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    input_ids = tokenizer.encode(seq, max_length=512)
    input_ids = input_ids.unsqueeze(0).to(device) # Batch dim & Move to device
    
    # -----------------------------------------------
    # 1. Classification (Splice Site / ncRNA)
    # -----------------------------------------------
    print("\n[1] Running Classification Model...")
    try:
        # Load from the specified checkpoint (e.g. results_cls/checkpoint-21)
        cls_path = args.cls_model_path
        config_cls = MambaRNAConfig.from_pretrained(cls_path)
        model_cls = MambaRNAForSequenceClassification.from_pretrained(cls_path, config=config_cls)
        model_cls.to(device)
        model_cls.eval()
        
        with torch.no_grad():
            outputs = model_cls(input_ids)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            
        print(f"   -> Prediction: Class {pred_class}")
        print(f"   -> Probabilities: {probs[0].tolist()}")
    except Exception as e:
        print(f"   -> Failed to run CLS: {e}")

    # -----------------------------------------------
    # 2. Secondary Structure Prediction (SSP)
    # -----------------------------------------------
    print("\n[2] Running Secondary Structure Prediction Model...")
    try:
        ssp_path = args.ssp_model_path
        config_ssp = MambaRNAConfig.from_pretrained(ssp_path)
        model_ssp = MambaRNAForTokenClassification.from_pretrained(ssp_path, config=config_ssp)
        model_ssp.to(device)
        model_ssp.eval()
        
        vocab_map_inv = {0: '.', 1: '(', 2: ')'}
        
        with torch.no_grad():
            outputs = model_ssp(input_ids)
            logits = outputs['logits'] # [B, L, NumLabels]
            preds = torch.argmax(logits, dim=-1)[0] # [L]
            
        # Decode
        # tokens indices: [CLS] ... [SEP] [PAD]
        # input_ids[0][0] is CLS.
        # Sequence starts at index 1.
        
        structure_str = ""
        # iterate over sequence length
        # input_ids includes cls/sep. 
        # len(seq) is actual length. 
        # indices 1 to len(seq) are the bases.
        
        for i in range(1, len(seq) + 1):
            if i < len(preds):
                idx = preds[i].item()
                structure_str += vocab_map_inv.get(idx, '?')
            else:
                structure_str += "?"
                
        print(f"   -> Predicted Structure:\n{structure_str}")
        
    except Exception as e:
        print(f"   -> Failed to run SSP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="AUGGCUACUACGUAGCUAGCUAGCUACGAUCGAUCGAUCGAU")
    parser.add_argument("--cls_model_path", type=str, default="results_cls/checkpoint-21")
    parser.add_argument("--ssp_model_path", type=str, default="results_ssp/checkpoint-21")
    args = parser.parse_args()
    
    predict(args)
