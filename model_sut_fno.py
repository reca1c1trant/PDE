"""
Spectral U-Net Transformer + FNO (SUT-FNO) Model.

Architecture:
    Input [B, T, 128, 128, C]
        ↓
    Spectral Encoder (with FNO layers)
        ↓ + Spectral Skip Connections
    Llama Transformer
        ↓
    Spectral Decoder (with Spectral Fusion + FNO)
        ↓
    FNO Refinement Head
        ↓
    Output [B, T-1, 128, 128, C]

Key features:
1. Spectral Skip: FFT coefficients bypass transformer bottleneck
2. Multi-scale FNO: Frequency domain processing at each scale
3. FNO Refinement: Final polish for PDE accuracy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import LlamaConfig, LlamaModel

from encoder_spectral import SpectralEncoder
from decoder_spectral import SpectralDecoder


class SUTFNOModel(nn.Module):
    """
    Spectral U-Net Transformer + FNO.

    For PDE prediction: given timesteps t[0:T-1], predict t[1:T].
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Extract configs
        model_cfg = config.get('model', {})
        encoder_cfg = model_cfg.get('encoder', {})
        decoder_cfg = model_cfg.get('decoder', {})
        transformer_cfg = model_cfg.get('transformer', {})
        fno_head_cfg = model_cfg.get('fno_head', {})

        in_channels = model_cfg.get('in_channels', 6)
        hidden_dim = transformer_cfg.get('hidden_size', 768)

        # Spectral Encoder
        self.encoder = SpectralEncoder(
            in_channels=in_channels,
            channels=encoder_cfg.get('channels', [64, 128, 256]),
            fno_modes=encoder_cfg.get('fno_modes', [32, 16, 8]),
            spectral_skip_modes=encoder_cfg.get('spectral_skip_modes', [32, 16, 8]),
            hidden_dim=hidden_dim,
        )

        # Llama Transformer
        llama_config = LlamaConfig(
            hidden_size=transformer_cfg.get('hidden_size', 768),
            num_hidden_layers=transformer_cfg.get('num_hidden_layers', 10),
            num_attention_heads=transformer_cfg.get('num_attention_heads', 12),
            num_key_value_heads=transformer_cfg.get('num_key_value_heads', 4),
            intermediate_size=transformer_cfg.get('intermediate_size', 3072),
            hidden_act=transformer_cfg.get('hidden_act', 'silu'),
            max_position_embeddings=transformer_cfg.get('max_position_embeddings', 4096),
            rms_norm_eps=transformer_cfg.get('rms_norm_eps', 1e-5),
            rope_theta=transformer_cfg.get('rope_theta', 500000.0),
            attention_dropout=transformer_cfg.get('attention_dropout', 0.0),
            vocab_size=1,  # Not used, we use inputs_embeds
        )
        self.transformer = LlamaModel(llama_config)

        # Spectral Decoder
        self.decoder = SpectralDecoder(
            out_channels=in_channels,
            channels=decoder_cfg.get('channels', [256, 128, 64]),
            skip_channels=encoder_cfg.get('channels', [64, 128, 256])[::-1],  # Reverse order
            fno_modes=decoder_cfg.get('fno_modes', [16, 24, 32]),
            hidden_dim=hidden_dim,
            fno_head_modes=fno_head_cfg.get('modes', 32),
            fno_head_width=fno_head_cfg.get('width', 64),
            fno_head_layers=fno_head_cfg.get('n_layers', 3),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, H, W, C] input tensor (T frames)

        Returns:
            output: [B, T-1, H, W, C] predicted next frames
        """
        B, T, H, W, C = x.shape

        # Use T-1 frames as input to predict T-1 frames (shifted by 1)
        x_input = x[:, :-1]  # [B, T-1, H, W, C]
        T_input = T - 1

        # Encode
        latent, spectral_skips = self.encoder(x_input)
        # latent: [B, (T-1)*256, hidden_dim]
        # spectral_skips: list of [B*(T-1), C, modes, modes//2+1]

        # Transform
        transformer_output = self.transformer(
            inputs_embeds=latent,
            use_cache=False,
        )
        latent_out = transformer_output.last_hidden_state  # [B, (T-1)*256, hidden_dim]

        # Decode
        output = self.decoder(latent_out, spectral_skips, temporal_length=T_input)
        # output: [B, T-1, H, W, C]

        return output

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + transformer_params + decoder_params

        return {
            'encoder': encoder_params,
            'transformer': transformer_params,
            'decoder': decoder_params,
            'total': total,
        }

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Count trainable parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total = encoder_params + transformer_params + decoder_params

        return {
            'encoder': encoder_params,
            'transformer': transformer_params,
            'decoder': decoder_params,
            'total': total,
        }


def create_sut_fno_model(config: dict) -> SUTFNOModel:
    """Factory function to create SUT-FNO model."""
    model = SUTFNOModel(config)

    # Enable gradient checkpointing if configured
    if config.get('model', {}).get('gradient_checkpointing', False):
        model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SUT-FNO Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test config
    config = {
        'model': {
            'in_channels': 6,
            'gradient_checkpointing': True,
            'encoder': {
                'channels': [64, 128, 256],
                'fno_modes': [32, 16, 8],
                'spectral_skip_modes': [32, 16, 8],
            },
            'decoder': {
                'channels': [256, 128, 64],
                'fno_modes': [16, 24, 32],
            },
            'transformer': {
                'hidden_size': 768,
                'num_hidden_layers': 10,
                'num_attention_heads': 12,
                'num_key_value_heads': 4,
                'intermediate_size': 3072,
                'hidden_act': 'silu',
                'max_position_embeddings': 4096,
                'rms_norm_eps': 1e-5,
                'rope_theta': 500000.0,
                'attention_dropout': 0.0,
            },
            'fno_head': {
                'modes': 32,
                'width': 64,
                'n_layers': 3,
            },
        },
    }

    # Create model
    model = create_sut_fno_model(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Encoder:     {params['encoder']:>12,}")
    print(f"  Transformer: {params['transformer']:>12,}")
    print(f"  Decoder:     {params['decoder']:>12,}")
    print(f"  Total:       {params['total']:>12,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 17, 128, 128, 6).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test backward pass
    print(f"\nTesting backward pass...")
    loss = output.float().sum()
    loss.backward()
    print("Backward pass successful!")

    # Memory usage
    if device.type == 'cuda':
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print("\n" + "=" * 60)
    print("SUT-FNO Model test passed!")
    print("=" * 60)
