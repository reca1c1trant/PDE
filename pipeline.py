"""
PDE Pipeline: Encoder + Llama Transformer + Decoder
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

        # Causal mask constants
        self.num_timesteps = 16
        self.tokens_per_timestep = 256  # 4096 / 16
        self._causal_mask = None

        # Create encoders and decoders
        self.encoder_1d = PDE1DEncoder(self.in_channels, self.hidden_dim)
        self.encoder_2d = PDE2DEncoder(self.in_channels, self.hidden_dim)
        self.decoder_1d = PDE1DDecoder(self.in_channels, self.hidden_dim)
        self.decoder_2d = PDE2DDecoder(self.in_channels, self.hidden_dim)

        # Llama config with FlashAttention-2 support
        use_flash_attn = config['model'].get('use_flash_attention', True)
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"

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
            use_cache=False,
            attn_implementation=attn_impl,
        )

        # Initialize Llama from scratch
        self.transformer = LlamaModel(llama_config)
        self.transformer.embed_tokens = None

        # Gradient checkpointing
        if config['model'].get('gradient_checkpointing', True):
            self.transformer.gradient_checkpointing_enable()

        self._log_info(llama_config, use_flash_attn)

    def _log_info(self, llama_config, use_flash_attn):
        """Log model info."""
        encoder_params = sum(p.numel() for p in self.encoder_1d.parameters()) + \
                        sum(p.numel() for p in self.encoder_2d.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_1d.parameters()) + \
                        sum(p.numel() for p in self.decoder_2d.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params = encoder_params + decoder_params + transformer_params

        print(f"\n{'='*60}")
        print(f"PDECausalModel Initialized")
        print(f"{'='*60}")
        print(f"Transformer: {llama_config.num_hidden_layers} layers, "
              f"{llama_config.hidden_size} hidden, "
              f"{llama_config.num_attention_heads} heads")
        print(f"FlashAttention-2: {'Enabled' if use_flash_attn else 'Disabled'}")
        print(f"Gradient Checkpointing: {self.transformer.is_gradient_checkpointing}")
        print(f"\nParameters:")
        print(f"  Encoders:    {encoder_params:,}")
        print(f"  Transformer: {transformer_params:,}")
        print(f"  Decoders:    {decoder_params:,}")
        print(f"  Total:       {total_params:,}")
        print(f"{'='*60}\n")

    def _get_causal_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create block causal mask (cached)."""
        if self._causal_mask is None or self._causal_mask.device != device:
            self._causal_mask = create_block_causal_mask(
                self.num_timesteps,
                self.tokens_per_timestep,
                device,
                dtype
            )
        return self._causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with block causal mask.

        Args:
            x: [B, T, H, 6] (1D) or [B, T, H, W, 6] (2D)

        Returns:
            Output tensor with same shape as input
        """
        ndim = x.ndim - 3
        B = x.shape[0]

        causal_mask = self._get_causal_mask(x.device, x.dtype)
        attention_mask = causal_mask.expand(B, -1, -1, -1)

        if ndim == 1:
            tokens = self.encoder_1d(x)
            hidden = self.transformer(inputs_embeds=tokens, attention_mask=attention_mask).last_hidden_state
            output = self.decoder_1d(hidden)
        elif ndim == 2:
            tokens = self.encoder_2d(x)
            hidden = self.transformer(inputs_embeds=tokens, attention_mask=attention_mask).last_hidden_state
            output = self.decoder_2d(hidden)
        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")

        return output


def create_block_causal_mask(
    num_timesteps: int,
    tokens_per_timestep: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create Timestep-Level Block Causal Mask.

    Args:
        num_timesteps: Number of timesteps (16)
        tokens_per_timestep: Tokens per timestep (256)
        device: Device
        dtype: Data type

    Returns:
        mask: [1, 1, seq_len, seq_len], 0.0=attend, -inf=mask
    """
    block_mask = torch.tril(torch.ones(num_timesteps, num_timesteps, device=device))
    mask = block_mask.repeat_interleave(tokens_per_timestep, dim=0)
    mask = mask.repeat_interleave(tokens_per_timestep, dim=1)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.to(dtype=dtype)


def compute_masked_loss(pred, target, channel_mask):
    """
    Compute MSE loss with channel masking.

    Args:
        pred: [B, T, *spatial, 6]
        target: [B, T, *spatial, 6]
        channel_mask: [B, 6]

    Returns:
        loss: scalar
    """
    squared_error = (pred - target) ** 2
    ndim = pred.ndim
    mask_shape = [pred.shape[0]] + [1] * (ndim - 2) + [pred.shape[-1]]
    mask = channel_mask.view(*mask_shape)
    masked_error = squared_error * mask
    num_valid = mask.sum()
    return masked_error.sum() / num_valid if num_valid > 0 else masked_error.sum()
