"""
PDE Pipeline V2: Encoder + Llama Transformer + Decoder + Temporal Residual

Key difference from V1:
- Added Zero-Init Residual Projection: output = input + delta
- Model learns the temporal difference (Δu) instead of full field (u)
- More suitable for PDE tasks where consecutive timesteps are similar

Physical insight:
- For dt ≈ 0.001, |Δu| ≈ 0.001 while |u| ≈ 1.0
- Learning Δu is 1000x easier than learning u
- Residual connection provides strong inductive bias for temporal evolution
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


class ResidualProjection(nn.Module):
    """
    Zero-initialized projection for residual learning.

    Initially outputs zero, so output = input (identity mapping).
    Gradually learns to predict Δ = target - input.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # Zero initialization: initial output is zero
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] tensor
        Returns:
            delta: [B, T, H, W, C] tensor (initially all zeros)
        """
        B, T, H, W, C = x.shape
        # Reshape for Conv2d: [B*T, C, H, W]
        x_perm = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        delta = self.proj(x_perm)
        # Reshape back: [B, T, H, W, C]
        delta = delta.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)
        return delta


class PDECausalModelV2(nn.Module):
    """
    PDE Causal Model V2 with Temporal Residual Connection.

    Key difference from V1:
        output = input + delta  (where delta is predicted by the model)

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
        Decoder → Features [B, T, *spatial, 6]
          ↓
        Zero-Init Residual Projection → Delta [B, T, *spatial, 6]
          ↓
        Output = Input + Delta

    Args:
        config (dict): Model configuration from yaml
    """

    def __init__(self, config: dict):
        super().__init__()

        self.in_channels = config['model']['in_channels']
        self.hidden_dim = config['model']['transformer']['hidden_size']

        # Normalization and noise config
        self.noise_level = config['model'].get('noise_level', 0.0)
        self.norm_eps = 1e-6

        # Causal mask constants
        self.num_timesteps = 16
        self.tokens_per_timestep = 256  # 4096 / 16
        self._causal_mask = None

        # Create encoders and decoders
        encoder_config = config['model'].get('encoder', {})
        encoder_version = encoder_config.get('version', 'v1')

        if encoder_version == 'v3':
            self.encoder_2d = create_encoder_v3(config)
            self.decoder_2d = create_decoder_v3(config)
        elif encoder_version == 'v2':
            self.encoder_2d = create_encoder_v2(config)
            self.decoder_2d = create_decoder_v2(config)
        else:
            self.encoder_2d = PDE2DEncoder(self.in_channels, self.hidden_dim)
            self.decoder_2d = PDE2DDecoder(self.in_channels, self.hidden_dim)

        # Zero-Init Residual Projection (V2 key feature)
        self.residual_proj = ResidualProjection(self.in_channels)

        # Llama config
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
            torch_dtype=torch.bfloat16,
        )

        # Initialize Llama from scratch
        self.transformer = LlamaModel(llama_config)
        self.transformer.embed_tokens = None

        # Gradient checkpointing
        if config['model'].get('gradient_checkpointing', True):
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Convert to bf16
        self.to(torch.bfloat16)

        self._log_info(llama_config, use_flash_attn, encoder_version, encoder_config)

    def _log_info(self, llama_config, use_flash_attn, encoder_version='v1', encoder_config=None):
        """Log model info (only on main process)."""
        if not _is_main_process():
            return

        encoder_params = sum(p.numel() for p in self.encoder_2d.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_2d.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        residual_params = sum(p.numel() for p in self.residual_proj.parameters())
        total_params = encoder_params + decoder_params + transformer_params + residual_params

        print(f"\n{'='*60}")
        print(f"PDECausalModel V2 (Temporal Residual)")
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
            channels = encoder_config.get('channels', [64, 128, 256])
            use_res = encoder_config.get('use_resblock', True)
            print(f"Encoder: V2 (channels={channels}, resblock={use_res})")
        else:
            print(f"Encoder: V1 (original)")
        print(f"Residual: Zero-Init Projection (output = input + delta)")
        print(f"FlashAttention-2: {'Enabled' if use_flash_attn else 'Disabled'}")
        print(f"Gradient Checkpointing: {self.transformer.is_gradient_checkpointing}")
        print(f"Dtype: {llama_config.torch_dtype}")
        print(f"Normalization: Enabled (spatial-wise, eps={self.norm_eps})")
        print(f"Noise Injection: {self.noise_level if self.noise_level > 0 else 'Disabled'}")
        print(f"\nParameters:")
        print(f"  Encoders:      {encoder_params:,}")
        print(f"  Transformer:   {transformer_params:,}")
        print(f"  Decoders:      {decoder_params:,}")
        print(f"  Residual Proj: {residual_params:,}")
        print(f"  Total:         {total_params:,}")
        print(f"{'='*60}\n")

    def _get_causal_mask(self, num_timesteps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create block causal mask."""
        if num_timesteps == self.num_timesteps:
            if self._causal_mask is None or self._causal_mask.device != device:
                self._causal_mask = create_block_causal_mask(
                    self.num_timesteps,
                    self.tokens_per_timestep,
                    device,
                    dtype
                )
            return self._causal_mask

        return create_block_causal_mask(
            num_timesteps,
            self.tokens_per_timestep,
            device,
            dtype
        )

    def _normalize(self, x: torch.Tensor):
        """Normalize input tensor over spatial dimensions."""
        ndim = x.ndim - 3

        if ndim == 1:
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + self.norm_eps
        else:
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
        Forward pass with temporal residual connection.

        output = input + delta

        Where delta is predicted by: Encoder -> Transformer -> Decoder -> ResidualProj

        Args:
            x: [B, T, H, 6] (1D) or [B, T, H, W, 6] (2D)

        Returns:
            Output tensor with same shape as input
        """
        ndim = x.ndim - 3
        B = x.shape[0]
        T = x.shape[1]

        # Store original input for residual
        x_input = x

        # Normalize
        x_norm, mean, std = self._normalize(x)

        # Noise injection (training only)
        x_norm = self._add_noise(x_norm)

        # Causal mask
        causal_mask = self._get_causal_mask(T, x.device, x.dtype)
        attention_mask = causal_mask.expand(B, -1, -1, -1)

        if ndim == 1:
            raise NotImplementedError("1D not implemented for V2")

        elif ndim == 2:
            tokens = self.encoder_2d(x_norm)
            hidden = self.transformer(inputs_embeds=tokens, attention_mask=attention_mask).last_hidden_state
            features = self.decoder_2d(hidden)

        else:
            raise ValueError(f"Unsupported input dimension: {x.shape}")

        # Denormalize decoder output
        features = self._denormalize(features, mean, std)

        # Residual projection: compute delta from features
        # Initially delta ≈ 0, so output ≈ input (identity)
        delta = self.residual_proj(features)

        # Final output = input + delta
        output = x_input + delta

        return output


def create_block_causal_mask(
    num_timesteps: int,
    tokens_per_timestep: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create Timestep-Level Block Causal Mask."""
    block_mask = torch.tril(torch.ones(num_timesteps, num_timesteps, device=device))
    mask = block_mask.repeat_interleave(tokens_per_timestep, dim=0)
    mask = mask.repeat_interleave(tokens_per_timestep, dim=1)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.to(dtype=dtype)


def compute_nrmse_loss(pred: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor):
    """
    Compute nRMSE loss (per-sample, per-channel).

    Same as train_e2e.py for consistency.
    """
    eps = 1e-8

    if channel_mask.dim() == 2:
        valid_mask = channel_mask[0].bool()
    else:
        valid_mask = channel_mask.bool()

    pred_valid = pred[..., valid_mask]
    target_valid = target[..., valid_mask]

    B, T, H, W, C = pred_valid.shape

    pred_flat = pred_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)
    target_flat = target_valid.permute(0, 4, 1, 2, 3).reshape(B, C, -1)

    mse_per_bc = ((pred_flat - target_flat) ** 2).mean(dim=2)
    rmse_per_bc = torch.sqrt(mse_per_bc + eps)

    rms_per_bc = torch.sqrt((target_flat ** 2).mean(dim=2) + eps)

    nrmse_per_bc = rmse_per_bc / rms_per_bc

    nrmse = nrmse_per_bc.mean()
    rmse = rmse_per_bc.mean()

    return nrmse, rmse
