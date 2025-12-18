"""
PDE Pipeline: Encoder + Llama Transformer + Decoder
For causal autoregressive prediction of PDE dynamics.
"""

import os
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
from encoder import (
    PDE1DEncoder, PDE2DEncoder, PDE1DDecoder, PDE2DDecoder,
    PDE2DEncoderV2, PDE2DDecoderV2, create_encoder_v2, create_decoder_v2,
    FFTEncoder2D, FFTDecoder2D, create_encoder_v3, create_decoder_v3
)


def _is_main_process():
    """Check if current process is main (rank 0)."""
    return os.environ.get('LOCAL_RANK', '0') == '0'


class PDECausalModel(nn.Module):
    """
    Complete model for PDE causal prediction.

    Architecture:
        Input [B, T, *spatial, 6]
          ↓
        Normalize (per-sample, spatial-wise)
          ↓
        Noise Injection (training only)
          ↓
        Encoder → Tokens [B, seq_len, hidden_dim]
          ↓
        Llama Transformer [B, seq_len, hidden_dim]
          ↓
        Decoder → Output [B, T, *spatial, 6]
          ↓
        Denormalize
          ↓
        Output

    Args:
        config (dict): Model configuration from yaml
    """

    def __init__(self, config: dict):
        super().__init__()

        self.in_channels = config['model']['in_channels']
        # hidden_dim syncs with transformer.hidden_size automatically
        self.hidden_dim = config['model']['transformer']['hidden_size']

        # Normalization and noise config
        self.noise_level = config['model'].get('noise_level', 0.0)
        self.norm_eps = 1e-6

        # Causal mask constants
        self.num_timesteps = 16
        self.tokens_per_timestep = 256  # 4096 / 16
        self._causal_mask = None

        # Create encoders and decoders
        # 检查encoder版本 (通过 config.model.encoder.version 判断)
        encoder_config = config['model'].get('encoder', {})
        encoder_version = encoder_config.get('version', 'v1')

        if encoder_version == 'v3':
            # V3: FFT-based encoder
            self.encoder_2d = create_encoder_v3(config)
            self.decoder_2d = create_decoder_v3(config)
        elif encoder_version == 'v2':
            # V2: CNN with ResBlocks
            self.encoder_2d = create_encoder_v2(config)
            self.decoder_2d = create_decoder_v2(config)
        else:
            # V1: 原始版本 (向后兼容)
            self.encoder_2d = PDE2DEncoder(self.in_channels, self.hidden_dim)
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
            torch_dtype=torch.bfloat16,  # Fix FlashAttention dtype warning
        )

        # Initialize Llama from scratch
        self.transformer = LlamaModel(llama_config)
        self.transformer.embed_tokens = None

        # Gradient checkpointing (use_reentrant=False for DDP compatibility)
        if config['model'].get('gradient_checkpointing', True):
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Convert entire model to bf16 for FSDP compatibility
        self.to(torch.bfloat16)

        self._log_info(llama_config, use_flash_attn, encoder_version, encoder_config)

    def _log_info(self, llama_config, use_flash_attn, encoder_version='v1', encoder_config=None):
        """Log model info (only on main process)."""
        if not _is_main_process():
            return

        encoder_params = sum(p.numel() for p in self.encoder_2d.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_2d.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params = encoder_params + decoder_params + transformer_params

        print(f"\n{'='*60}")
        print(f"PDECausalModel Initialized")
        print(f"{'='*60}")
        print(f"Transformer: {llama_config.num_hidden_layers} layers, "
              f"{llama_config.hidden_size} hidden, "
              f"{llama_config.num_attention_heads} heads")
        if encoder_version == 'v3' and encoder_config:
            hidden_ch = encoder_config.get('hidden_channels', 64)
            modes = encoder_config.get('modes', 64)
            n_blocks = encoder_config.get('n_blocks', 4)
            print(f"Encoder: V3 FFT (hidden_channels={hidden_ch}, modes={modes}, n_blocks={n_blocks})")
        elif encoder_version == 'v2' and encoder_config:
            mid_ch = encoder_config.get('mid_channels', 256)
            use_res = encoder_config.get('use_resblock', True)
            print(f"Encoder: V2 (mid_channels={mid_ch}, resblock={use_res})")
        else:
            print(f"Encoder: V1 (original)")
        print(f"FlashAttention-2: {'Enabled' if use_flash_attn else 'Disabled'}")
        print(f"Gradient Checkpointing: {self.transformer.is_gradient_checkpointing}")
        print(f"Dtype: {llama_config.torch_dtype}")
        print(f"Normalization: Enabled (spatial-wise, eps={self.norm_eps})")
        print(f"Noise Injection: {self.noise_level if self.noise_level > 0 else 'Disabled'}")
        print(f"\nParameters:")
        print(f"  Encoders:    {encoder_params:,}")
        print(f"  Transformer: {transformer_params:,}")
        print(f"  Decoders:    {decoder_params:,}")
        print(f"  Total:       {total_params:,}")
        print(f"{'='*60}\n")

    def _get_causal_mask(self, num_timesteps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get or create block causal mask (cached for training, dynamic for inference).

        Args:
            num_timesteps: Actual number of timesteps in input
            device: Device
            dtype: Data type
        """
        # For training (fixed 16 timesteps), use cached mask
        if num_timesteps == self.num_timesteps:
            if self._causal_mask is None or self._causal_mask.device != device:
                self._causal_mask = create_block_causal_mask(
                    self.num_timesteps,
                    self.tokens_per_timestep,
                    device,
                    dtype
                )
            return self._causal_mask

        # For inference with different timesteps, create dynamically
        return create_block_causal_mask(
            num_timesteps,
            self.tokens_per_timestep,
            device,
            dtype
        )

    def _normalize(self, x: torch.Tensor):
        """
        Normalize input tensor over spatial dimensions.

        Args:
            x: [B, T, H, C] (1D) or [B, T, H, W, C] (2D)

        Returns:
            x_norm: Normalized tensor
            mean: [B, T, 1, C] or [B, T, 1, 1, C]
            std: [B, T, 1, C] or [B, T, 1, 1, C]
        """
        ndim = x.ndim - 3  # 1 for 1D, 2 for 2D

        if ndim == 1:
            # [B, T, H, C] -> reduce over H (dim=2)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + self.norm_eps
        else:
            # [B, T, H, W, C] -> reduce over H, W (dim=2,3)
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + self.norm_eps

        x_norm = (x - mean) / std
        return x_norm, mean, std

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        return x * std + mean

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise during training."""
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with normalization, noise injection, and block causal mask.

        Args:
            x: [B, T, H, 6] (1D) or [B, T, H, W, 6] (2D)

        Returns:
            Output tensor with same shape as input
        """
        ndim = x.ndim - 3
        B = x.shape[0]
        T = x.shape[1]  # Actual number of timesteps

        # Normalize
        x_norm, mean, std = self._normalize(x)

        # Noise injection (training only)
        x_norm = self._add_noise(x_norm)

        # Causal mask (dynamic based on actual timesteps)
        causal_mask = self._get_causal_mask(T, x.device, x.dtype)
        attention_mask = causal_mask.expand(B, -1, -1, -1)

        if ndim == 1:
            pass
            # tokens = self.encoder_1d(x_norm)
            # hidden = self.transformer(inputs_embeds=tokens, attention_mask=attention_mask).last_hidden_state
            # output = self.decoder_1d(hidden)

        elif ndim == 2:
            tokens = self.encoder_2d(x_norm)
            hidden = self.transformer(inputs_embeds=tokens, attention_mask=attention_mask).last_hidden_state
            output = self.decoder_2d(hidden)

        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")

        # Denormalize
        output = self._denormalize(output, mean, std)

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


def compute_masked_loss(pred, target, channel_mask, alpha=0.0, sigma=None):
    """
    Compute combined MSE + RMSE/nRMSE loss with channel masking.

    Loss = alpha * RMSE + (1 - alpha) * MSE           (if sigma is None)
    Loss = alpha * nRMSE + (1 - alpha) * MSE          (if sigma is provided)

    nRMSE formula (per channel c):
        L_c = sqrt(mean(((pred_c - target_c) / sigma_c)^2))
        nRMSE = mean(L_c for all valid channels)

    Args:
        pred: [B, T, *spatial, C]
        target: [B, T, *spatial, C]
        channel_mask: [B, C]
        alpha: Weight for RMSE/nRMSE (0=pure MSE, 1=pure RMSE/nRMSE)
        sigma: [C] global std per channel, or None for plain RMSE

    Returns:
        loss: scalar
    """
    # Get valid channels from first sample (assume consistent across batch)
    valid_mask = channel_mask[0].bool()  # [C]

    # Filter to valid channels only
    pred_valid = pred[..., valid_mask]  # [B, T, *spatial, C_valid]
    target_valid = target[..., valid_mask]  # [B, T, *spatial, C_valid]

    # MSE (global)
    mse = ((pred_valid - target_valid) ** 2).mean()

    # Combined loss
    if alpha > 0:
        if sigma is not None:
            # nRMSE with global sigma
            sigma_valid = sigma[valid_mask]  # [C_valid]
            # Reshape sigma for broadcasting: [1, 1, ..., C_valid]
            sigma_shape = [1] * (pred_valid.ndim - 1) + [sigma_valid.shape[0]]
            sigma_broadcast = sigma_valid.view(*sigma_shape)

            # Per-channel normalized squared error
            normalized_error = (pred_valid - target_valid) / (sigma_broadcast + 1e-8)
            # Per-channel RMSE: sqrt(mean over B,T,spatial)
            per_channel_mse = (normalized_error ** 2).mean(dim=tuple(range(pred_valid.ndim - 1)))  # [C_valid]
            per_channel_rmse = torch.sqrt(per_channel_mse + 1e-8)
            # nRMSE = mean over channels
            nrmse = per_channel_rmse.mean()
            loss = alpha * nrmse + (1 - alpha) * mse
        else:
            # Plain RMSE (no normalization)
            rmse = torch.sqrt(mse + 1e-8)
            loss = alpha * rmse + (1 - alpha) * mse
    else:
        loss = mse

    return loss
