"""
Complete PDE model: Encoder + Llama Transformer + Decoder
For causal autoregressive prediction of PDE dynamics.
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
from encoder import PDE1DEncoder, PDE2DEncoder, PDE1DDecoder, PDE2DDecoder



class PDECausalModel(nn.Module):
    """
    Complete model for PDE causal prediction.
    
    Architecture:
        Input [B, T, *spatial, 6]
          ↓
        Encoder → Tokens [B, seq_len, hidden_dim]
          ↓
        Llama Transformer [B, seq_len, hidden_dim]
          ↓
        Decoder → Output [B, T, *spatial, 6]
    
    Args:
        config (dict): Model configuration from yaml
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.in_channels = config['model']['in_channels']
        self.hidden_dim = config['model']['encoder_hidden_dim']
        
        # Create encoders and decoders for different dimensions
        self.encoder_1d = PDE1DEncoder(self.in_channels, self.hidden_dim)
        self.encoder_2d = PDE2DEncoder(self.in_channels, self.hidden_dim)
        
        self.decoder_1d = PDE1DDecoder(self.in_channels, self.hidden_dim)
        self.decoder_2d = PDE2DDecoder(self.in_channels, self.hidden_dim)
        
        # Create Llama Transformer from scratch
        llama_config = LlamaConfig(
            hidden_size=config['model']['transformer']['hidden_size'],
            num_hidden_layers=config['model']['transformer']['num_hidden_layers'],
            num_attention_heads=config['model']['transformer']['num_attention_heads'],
            num_key_value_heads=config['model']['transformer']['num_key_value_heads'],
            intermediate_size=config['model']['transformer']['intermediate_size'],
            hidden_act=config['model']['transformer']['hidden_act'],
            max_position_embeddings=config['model']['transformer']['max_position_embeddings'],
            rms_norm_eps=config['model']['transformer']['rms_norm_eps'],
            rope_theta=config['model']['transformer']['rope_theta'],
            attention_dropout=config['model']['transformer']['attention_dropout'],
            use_cache=config['model']['transformer']['use_cache'],
        )
        
        # Initialize Llama model from scratch (no pretrained weights)
        self.transformer = LlamaModel(llama_config)
        
        # Remove embedding layer (we don't need it)
        self.transformer.embed_tokens = None
        
        print(f"Initialized Llama Transformer with {llama_config.num_hidden_layers} layers")
        print(f"Hidden dim: {llama_config.hidden_size}, Heads: {llama_config.num_attention_heads}, KV heads: {llama_config.num_key_value_heads}")
        
        # Count parameters
        self._log_parameters()
    
    def _log_parameters(self):
        """Log model parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder_1d.parameters()) + \
                        sum(p.numel() for p in self.encoder_2d.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_1d.parameters()) + \
                        sum(p.numel() for p in self.decoder_2d.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params = encoder_params + decoder_params + transformer_params
        
        print(f"\nModel Parameter Counts:")
        print(f"  Encoders: {encoder_params:,}")
        print(f"  Transformer: {transformer_params:,}")
        print(f"  Decoders: {decoder_params:,}")
        print(f"  Total: {total_params:,}")
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with automatic dimension detection.
        
        Args:
            x: Input tensor
                - 1D: [B, T, H, 6]
                - 2D: [B, T, H, W, 6]
            attention_mask: Optional attention mask for causal masking
                            [B, seq_len] where 1 = attend, 0 = ignore
        
        Returns:
            Output tensor with same shape as input
        """
        ndim = x.ndim - 3  # Exclude B, T, C
        
        if ndim == 1:  # 1D
            # Encode
            tokens = self.encoder_1d(x)  # [B, 4096, hidden_dim]
            
            # Transformer (with inputs_embeds, skipping embedding layer)
            transformer_output = self.transformer(
                inputs_embeds=tokens,
                attention_mask=attention_mask,
                use_cache=False
            )
            hidden = transformer_output.last_hidden_state  # [B, 4096, hidden_dim]
            
            # Decode
            output = self.decoder_1d(hidden)  # [B, T, H, 6]
            
        elif ndim == 2:  # 2D
            # Encode
            tokens = self.encoder_2d(x)  # [B, 4096, hidden_dim]
            
            # Transformer
            transformer_output = self.transformer(
                inputs_embeds=tokens,
                attention_mask=attention_mask,
                use_cache=False
            )
            hidden = transformer_output.last_hidden_state  # [B, 4096, hidden_dim]
            
            # Decode
            output = self.decoder_2d(hidden)  # [B, T, H, W, 6]
            
        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")
        
        return output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive prediction.
    
    Position i can only attend to positions <= i.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        mask: [seq_len, seq_len] where 1 = can attend, 0 = cannot attend
              For Transformers library, this will be converted to additive mask
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


