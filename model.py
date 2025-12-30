import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba not installed, using mock for development checks if needed.")
    Mamba = None

class MambaRNAConfig(PretrainedConfig):
    model_type = "mamba_rna"
    def __init__(
        self,
        vocab_size=10,
        d_model=768,
        n_layer=12,
        dropout=0.1,
        ssm_cfg=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = dropout
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {}

class BiMambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        if Mamba is None:
            raise ImportError("mamba_ssm package is not installed.")
            
        # Forward Mamba
        self.forward_mixer = Mamba(
            d_model=config.d_model,
            **config.ssm_cfg
        )
        # Backward Mamba
        self.backward_mixer = Mamba(
            d_model=config.d_model,
            **config.ssm_cfg
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        self.linear_fuse = nn.Linear(2 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, dim]
        
        # Forward pass
        out_fwd = self.forward_mixer(hidden_states)
        
        # Backward pass - flip sequence, run mamba, flip back
        out_rev = self.backward_mixer(hidden_states.flip(dims=[1])).flip(dims=[1])
        
        # Combine
        combined = torch.cat([out_fwd, out_rev], dim=-1)
        out = self.linear_fuse(combined)
        out = self.dropout(out)
        
        # Request residual connection handling here if Mamba doesn't handle it internally in a block context
        # Standard Mamba block usually does Norm -> Mixer -> Add Residual.
        # Here we are inside the mixer part essentially.
        
        return out

# Wrapper to make it a full block with Residual + Norm
class MambaEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = BiMambaBlock(config)
        
    def forward(self, hidden_states):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        if self.training and getattr(self, "gradient_checkpointing", False):
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mixer),
                hidden_states,
                use_reentrant=False
            )
            return residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.mixer(hidden_states)
            return residual + hidden_states

class MambaRNAModel(PreTrainedModel):
    config_class = MambaRNAConfig
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MambaEncoderLayer):
            module.gradient_checkpointing = value
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaEncoderLayer(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return x

class MambaRNAForMaskedLM(PreTrainedModel):
    config_class = MambaRNAConfig
    supports_gradient_checkpointing = True
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.backbone.gradient_checkpointing = value

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaRNAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
    def forward(self, input_ids, labels=None, **kwargs):
        features = self.backbone(input_ids, **kwargs)
        logits = self.lm_head(features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
            
        return output
