"""
LoRA wrapper for PDEModelV3 (Unified Shared FFN Transformer).

Applies LoRA to:
- NA Attention: qkv, proj (all active NA variants)
- FFN (SwiGLU): gate_proj, up_proj, down_proj

Encoder and Decoder can be optionally unfrozen for better adaptation.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model
from pretrain.model_v3 import PDEModelV3


def _is_main_process() -> bool:
    return os.environ.get('LOCAL_RANK', '0') == '0'


class PDELoRAModelV3(nn.Module):
    """
    LoRA wrapper for PDEModelV3.

    Applies LoRA to shared transformer layers while optionally keeping
    encoder/decoder trainable for domain adaptation.
    """

    def __init__(
        self,
        config: Dict,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        # Create base model
        self.model = PDEModelV3(config)

        # Load pretrained weights (before LoRA so names match)
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Apply LoRA to transformer
        self._apply_lora(config)

        # Freeze encoder/decoder if specified
        if freeze_encoder:
            self._freeze_encoders()
        if freeze_decoder:
            self._freeze_decoders()

        if _is_main_process():
            self._log_param_summary()

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights, handling DDP prefixes and dimension mismatches."""
        if _is_main_process():
            print(f"\nLoading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip DDP/compile prefixes
        cleaned = {}
        for k, v in state_dict.items():
            k = k.removeprefix('module.').removeprefix('_orig_mod.')
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        if _is_main_process():
            if missing:
                print(f"  Missing keys (randomly init): {len(missing)}")
                for k in missing[:10]:
                    print(f"    - {k}")
                if len(missing) > 10:
                    print(f"    ... and {len(missing) - 10} more")
            if unexpected:
                print(f"  Unexpected keys (ignored): {len(unexpected)}")
                for k in unexpected[:10]:
                    print(f"    - {k}")
                if len(unexpected) > 10:
                    print(f"    ... and {len(unexpected) - 10} more")
            loaded = len(cleaned) - len(unexpected)
            print(f"  Loaded {loaded} keys successfully")

    def _apply_lora(self, config: Dict):
        """Apply LoRA to transformer layers."""
        lora_cfg = config.get('model', {}).get('lora', {})

        lora_config = LoraConfig(
            r=lora_cfg.get('r', 16),
            lora_alpha=lora_cfg.get('alpha', 32),
            lora_dropout=lora_cfg.get('dropout', 0.05),
            target_modules=lora_cfg.get('target_modules', [
                "qkv", "proj",       # NA Attention
                "gate_proj", "up_proj", "down_proj",  # FFN (SwiGLU)
            ]),
            bias="none",
            task_type=None,
        )

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

    def _freeze_encoders(self):
        """Freeze all encoder components that exist."""
        self._freeze_module(self.model.encoder, "Encoder (2D)")
        if hasattr(self.model, 'encoder_1d'):
            self._freeze_module(self.model.encoder_1d, "Encoder (1D)")
        if hasattr(self.model, 'encoder_3d'):
            self._freeze_module(self.model.encoder_3d, "Encoder (3D)")

    def _freeze_decoders(self):
        """Freeze all decoder components that exist."""
        self._freeze_module(self.model.decoder, "Decoder (2D)")
        if hasattr(self.model, 'decoder_1d'):
            self._freeze_module(self.model.decoder_1d, "Decoder (1D)")
        if hasattr(self.model, 'decoder_3d'):
            self._freeze_module(self.model.decoder_3d, "Decoder (3D)")

    def _log_param_summary(self):
        """Log trainable parameter summary."""
        components = {}

        # 2D encoder/decoder (always present)
        components['Encoder (2D)'] = self.model.encoder
        components['Decoder (2D)'] = self.model.decoder

        # Optional 1D/3D components
        for dim_name in ['1d', '3d']:
            for comp_type, label in [('encoder', 'Encoder'), ('decoder', 'Decoder')]:
                attr = f'{comp_type}_{dim_name}'
                if hasattr(self.model, attr):
                    components[f'{label} ({dim_name.upper()})'] = getattr(self.model, attr)

        components['Transformer'] = self.model.transformer

        total_all = 0
        trainable_all = 0

        print(f"\n{'='*60}")
        print(f"PDELoRAModelV3 Parameter Summary")
        print(f"{'='*60}")

        for name, module in components.items():
            t = sum(p.numel() for p in module.parameters())
            tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
            suffix = " (LoRA)" if name == 'Transformer' else ""
            print(f"{name + suffix + ':':<20} {t:>12,} total, {tr:>10,} trainable")
            total_all += t
            trainable_all += tr

        print(f"{'-'*60}")
        print(f"{'Total:':<20} {total_all:>12,} total, {trainable_all:>10,} trainable")
        print(f"Trainable ratio: {100 * trainable_all / total_all:.2f}%")
        print(f"{'='*60}\n")

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        """Forward pass (same interface as PDEModelV3)."""
        return self.model(x, channel_mask, return_normalized)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_lora_checkpoint(
    model: PDELoRAModelV3,
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
    model: PDELoRAModelV3,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load checkpoint for resuming training."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'trainable_state_dict' in checkpoint:
        trainable_state_dict = checkpoint['trainable_state_dict']
        model_state = model.model.state_dict()
        model_state.update(trainable_state_dict)
        model.model.load_state_dict(model_state)

        if _is_main_process():
            print(f"Loaded {len(trainable_state_dict)} trainable parameters")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load optimizer state: {e}")

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load scheduler state: {e}")

    return checkpoint
