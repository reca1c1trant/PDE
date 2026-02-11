"""
LoRA wrapper for PDEModelV2 (Neighborhood Attention).

Applies LoRA to:
- NA Attention: qkv, proj
- FFN (SwiGLU): gate_proj, up_proj, down_proj

Encoder and Decoder can be optionally unfrozen for better adaptation.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model
from pretrain.model_v2 import PDEModelV2


def _is_main_process() -> bool:
    return os.environ.get('LOCAL_RANK', '0') == '0'


class PDELoRAModelV2(nn.Module):
    """
    LoRA wrapper for PDEModelV2.

    Applies LoRA to transformer layers while optionally keeping
    encoder/decoder trainable for domain adaptation.
    """

    def __init__(
        self,
        config: Dict,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        """
        Args:
            config: Model configuration dict
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: If True, freeze encoder weights
            freeze_decoder: If True, freeze decoder weights
        """
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        # Create base model
        self.model = PDEModelV2(config)

        # Load pretrained weights
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Apply LoRA to transformer
        self._apply_lora(config)

        # Freeze encoder/decoder if specified
        if freeze_encoder:
            self._freeze_module(self.model.encoder, "Encoder")
        if freeze_decoder:
            self._freeze_module(self.model.decoder, "Decoder")

        # Log parameter summary
        if _is_main_process():
            self._log_param_summary()

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights."""
        if _is_main_process():
            print(f"\nLoading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if _is_main_process():
            if missing:
                print(f"  Missing keys: {len(missing)}")
                if len(missing) <= 5:
                    for k in missing:
                        print(f"    - {k}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            print(f"  Loaded pretrained weights successfully")

    def _apply_lora(self, config: Dict):
        """Apply LoRA to transformer layers."""
        lora_cfg = config.get('model', {}).get('lora', {})

        # Default LoRA config for V2
        lora_config = LoraConfig(
            r=lora_cfg.get('r', 16),
            lora_alpha=lora_cfg.get('alpha', 32),
            lora_dropout=lora_cfg.get('dropout', 0.05),
            target_modules=lora_cfg.get('target_modules', [
                "qkv", "proj",  # NA Attention
                "gate_proj", "up_proj", "down_proj",  # FFN
            ]),
            bias="none",
            task_type=None,
        )

        # Apply LoRA to transformer
        self.model.transformer = get_peft_model(self.model.transformer, lora_config)

        if _is_main_process():
            print(f"\nLoRA config:")
            print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}")
            print(f"  target_modules={lora_config.target_modules}")

    def _freeze_module(self, module: nn.Module, name: str):
        """Freeze all parameters in a module."""
        count = 0
        for param in module.parameters():
            param.requires_grad = False
            count += param.numel()
        if _is_main_process():
            print(f"  Froze {name}: {count:,} parameters")

    def _log_param_summary(self):
        """Log trainable parameter summary."""
        # Count by component
        enc_total = sum(p.numel() for p in self.model.encoder.parameters())
        enc_train = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)

        dec_total = sum(p.numel() for p in self.model.decoder.parameters())
        dec_train = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)

        trans_total = sum(p.numel() for p in self.model.transformer.parameters())
        trans_train = sum(p.numel() for p in self.model.transformer.parameters() if p.requires_grad)

        total = enc_total + dec_total + trans_total
        trainable = enc_train + dec_train + trans_train

        print(f"\n{'='*60}")
        print(f"PDELoRAModelV2 Parameter Summary")
        print(f"{'='*60}")
        print(f"Encoder:     {enc_total:>12,} total, {enc_train:>10,} trainable")
        print(f"Decoder:     {dec_total:>12,} total, {dec_train:>10,} trainable")
        print(f"Transformer: {trans_total:>12,} total, {trans_train:>10,} trainable (LoRA)")
        print(f"{'-'*60}")
        print(f"Total:       {total:>12,} total, {trainable:>10,} trainable")
        print(f"Trainable ratio: {100 * trainable / total:.2f}%")
        print(f"{'='*60}\n")

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.model(x, channel_mask, return_normalized)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_lora_checkpoint(
    model: PDELoRAModelV2,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    metrics: Dict,
    save_path: str,
    config: Dict,
    patience_counter: int = 0,
    best_val_loss: float = float('inf'),
):
    """
    Save checkpoint with all trainable weights.

    Includes: LoRA weights + Encoder + Decoder (if unfrozen).
    """
    trainable_state_dict = {}
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.data.clone()

    checkpoint = {
        'global_step': global_step,
        'trainable_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'patience_counter': patience_counter,
        'best_val_loss': best_val_loss,
    }

    torch.save(checkpoint, save_path)


def load_lora_checkpoint(
    model: PDELoRAModelV2,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load checkpoint for resuming training."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load trainable state dict
    if 'trainable_state_dict' in checkpoint:
        trainable_state_dict = checkpoint['trainable_state_dict']
        model_state = model.model.state_dict()
        model_state.update(trainable_state_dict)
        model.model.load_state_dict(model_state)

        if _is_main_process():
            print(f"Loaded {len(trainable_state_dict)} trainable parameters")

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load optimizer state: {e}")

    # Load scheduler state
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load scheduler state: {e}")

    return checkpoint


if __name__ == "__main__":
    """Test LoRA model."""
    print("Testing PDELoRAModelV2...")

    config = {
        'model': {
            'in_channels': 18,
            'hidden_dim': 256,
            'patch_size': 16,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.0,
            'encoder': {'stem_hidden': 64, 'stem_out': 128},
            'intra_patch': {'num_layers': 1, 'temporal_window': 3, 'num_heads': 4},
            'na': {'base_kernel': 5},
            'decoder': {'stem_channels': 128, 'hidden_channels': 64},
            'lora': {
                'r': 8,
                'alpha': 16,
                'dropout': 0.05,
                'target_modules': ["qkv", "proj", "gate_proj", "up_proj", "down_proj"],
            },
        }
    }

    # Test with encoder/decoder unfrozen (default)
    print("\n[Test 1] Encoder/Decoder UNFROZEN")
    model = PDELoRAModelV2(config, pretrained_path=None)

    # Test forward
    x = torch.randn(2, 8, 128, 128, 18)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test with encoder/decoder frozen
    print("\n[Test 2] Encoder/Decoder FROZEN")
    model_frozen = PDELoRAModelV2(
        config,
        pretrained_path=None,
        freeze_encoder=True,
        freeze_decoder=True,
    )

    print("\nPDELoRAModelV2 test passed!")
