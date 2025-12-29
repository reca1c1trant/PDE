"""
LoRA wrapper for PDECausalModel.

Loads pretrained weights and applies LoRA to the Transformer layers.
Freezes Encoder, Decoder, and Transformer base weights.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model, PeftModel
from pipeline import PDECausalModel


def _is_main_process() -> bool:
    return os.environ.get('LOCAL_RANK', '0') == '0'


def create_pde_model_with_lora(
    config: Dict,
    pretrained_path: Optional[str] = None,
    lora_config: Optional[Dict] = None,
) -> nn.Module:
    """
    Create PDECausalModel with LoRA applied to Transformer.

    Args:
        config: Model configuration dict
        pretrained_path: Path to pretrained checkpoint
        lora_config: LoRA configuration dict with keys:
            - r: LoRA rank
            - alpha: LoRA alpha (scaling)
            - dropout: LoRA dropout
            - target_modules: List of module names to apply LoRA

    Returns:
        model: PDECausalModel with LoRA, frozen base weights
    """
    # Create base model
    model = PDECausalModel(config)

    # Load pretrained weights
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if _is_main_process():
            print(f"\n{'='*60}")
            print(f"Loaded pretrained weights from: {pretrained_path}")
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            print(f"{'='*60}\n")

    # Default LoRA config
    if lora_config is None:
        lora_config = {
            'r': 16,
            'alpha': 32,
            'dropout': 0.05,
            'target_modules': [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        }

    # Apply LoRA to transformer
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type=None,  # Custom model, not a standard task
    )

    # Disable gradient checkpointing before wrapping with PEFT
    # (PEFT's prepare_model_for_gradient_checkpointing fails because
    # LlamaModel has no embedding layer when we use inputs_embeds)
    if hasattr(model.transformer, 'gradient_checkpointing_disable'):
        model.transformer.gradient_checkpointing_disable()

    # Wrap transformer with LoRA
    model.transformer = get_peft_model(model.transformer, peft_config)

    # Re-enable gradient checkpointing on the base model if needed
    # Use use_reentrant=False for proper gradient flow with LoRA
    if config.get('model', {}).get('gradient_checkpointing', False):
        model.transformer.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Freeze encoder and decoder
    for param in model.encoder_2d.parameters():
        param.requires_grad = False

    for param in model.decoder_2d.parameters():
        param.requires_grad = False

    # Log trainable parameters
    if _is_main_process():
        _log_trainable_params(model)

    return model


def _log_trainable_params(model: nn.Module):
    """Log trainable vs frozen parameters."""
    total_params = 0
    trainable_params = 0

    # Encoder
    enc_total = sum(p.numel() for p in model.encoder_2d.parameters())
    enc_train = sum(p.numel() for p in model.encoder_2d.parameters() if p.requires_grad)

    # Decoder
    dec_total = sum(p.numel() for p in model.decoder_2d.parameters())
    dec_train = sum(p.numel() for p in model.decoder_2d.parameters() if p.requires_grad)

    # Transformer (with LoRA)
    trans_total = sum(p.numel() for p in model.transformer.parameters())
    trans_train = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)

    total_params = enc_total + dec_total + trans_total
    trainable_params = enc_train + dec_train + trans_train

    print(f"\n{'='*60}")
    print(f"LoRA Model Parameter Summary")
    print(f"{'='*60}")
    print(f"Encoder:     {enc_total:>12,} total, {enc_train:>10,} trainable")
    print(f"Decoder:     {dec_total:>12,} total, {dec_train:>10,} trainable")
    print(f"Transformer: {trans_total:>12,} total, {trans_train:>10,} trainable")
    print(f"{'-'*60}")
    print(f"Total:       {total_params:>12,} total, {trainable_params:>10,} trainable")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*60}\n")


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)
    return lora_params


def save_lora_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    metrics: Dict,
    save_path: str,
    config: Dict,
):
    """Save LoRA checkpoint (only trainable params)."""
    # Get LoRA state dict from transformer
    lora_state_dict = {}
    for name, param in model.transformer.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.clone()

    checkpoint = {
        'global_step': global_step,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }

    torch.save(checkpoint, save_path)


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load LoRA checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load LoRA weights
    lora_state_dict = checkpoint['lora_state_dict']
    model_state = model.transformer.state_dict()
    model_state.update(lora_state_dict)
    model.transformer.load_state_dict(model_state)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


class PDELoRAModel(nn.Module):
    """
    Wrapper class for PDECausalModel with LoRA.

    This wrapper handles:
    - Loading pretrained weights
    - Applying LoRA to transformer
    - Freezing encoder/decoder
    - Forward pass with proper dtype handling
    """

    def __init__(self, config: Dict, pretrained_path: Optional[str] = None):
        super().__init__()

        lora_config = config.get('model', {}).get('lora', None)

        self.model = create_pde_model_with_lora(
            config=config,
            pretrained_path=pretrained_path,
            lora_config=lora_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return get_lora_params(self.model)


if __name__ == "__main__":
    """Test LoRA model creation."""
    import yaml

    # Minimal config for testing
    config = {
        'model': {
            'in_channels': 6,
            'noise_level': 0.0,
            'use_flash_attention': False,
            'gradient_checkpointing': False,
            'encoder': {
                'version': 'v2',
                'channels': [64, 128, 256],
                'use_resblock': True,
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
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05,
                'target_modules': [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
            },
        },
    }

    print("Creating PDELoRAModel (without pretrained weights)...")
    model = PDELoRAModel(config, pretrained_path=None)

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 16, 128, 128, 6, dtype=torch.bfloat16)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check trainable params
    trainable = model.get_trainable_params()
    print(f"\nTrainable parameters: {len(trainable)}")

    print("\nLoRA model test passed!")
