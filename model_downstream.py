import torch
import torch.nn as nn
from model import MambaRNAConfig, MambaRNAModel, MambaRNAForMaskedLM

class MambaRNAForSequenceClassification(MambaRNAForMaskedLM):
    def __init__(self, config):
        # We initialize with same base config
        # We can reuse MambaRNAForMaskedLM init or just MambaRNAModel
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 2
        
        # Override the head
        self.lm_head = None 
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        
        # Initialize weights
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        # MambaRNAModel forward
        features = self.backbone(input_ids, **kwargs) # [B, L, D]
        
        # Pooler: Use CLS token at index 0
        cls_output = features[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output) # [B, NumLabels]
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
            
        return output

class MambaRNAForTokenClassification(MambaRNAForMaskedLM):
    """
    For tasks like predicting 3-state secondary structure (., (, )) per base
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 3
        
        self.lm_head = None
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
        
        self.post_init()
        
    def forward(self, input_ids, labels=None, **kwargs):
        features = self.backbone(input_ids, **kwargs) # [B, L, D]
        
        sequence_output = self.dropout(features)
        logits = self.classifier(sequence_output) # [B, L, NumLabels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on active parts (flatten)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
            
        return output

class MambaRNAForContactMap(MambaRNAForMaskedLM):
    """
    For predicting pairwise interactions (Contact Map).
    Output: [B, L, L] probability map.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = None
        
        # Bilinear projection or simple Outer Product
        self.proj_left = nn.Linear(config.d_model, 128)
        self.proj_right = nn.Linear(config.d_model, 128)
        self.classifier_pair = nn.Linear(128, 1) # inner product -> scalar
        
        self.post_init()
        
    def forward(self, input_ids, labels=None, **kwargs):
        features = self.backbone(input_ids, **kwargs) # [B, L, D]
        
        # Low rank projection for memory efficiency
        left = self.proj_left(features) # [B, L, 128]
        right = self.proj_right(features) # [B, L, 128]
        
        # Outer product via broadcasting
        # [B, L, 1, 128] * [B, 1, L, 128] -> elementwise? No.
        # We want [B, L, L].
        # (L, 128) @ (128, L) = (L, L)
        
        # Batch matmul
        logits = torch.matmul(left, right.transpose(1, 2)) # [B, L, L]
        # Divide by sqrt(dim)
        logits = logits / (128 ** 0.5)
        
        # No sigmoid applied here, loss will take BCEWithLogits
        
        loss = None
        if labels is not None:
            # Labels: [B, L, L] float 0/1
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
            
        return output
