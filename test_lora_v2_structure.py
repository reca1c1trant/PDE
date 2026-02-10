"""
Test script to verify LoRA target modules for PDEModelV2.

This script prints all Linear layer names in the model to identify
correct target_modules for PEFT LoRA.
"""

import torch
import torch.nn as nn
from model_v2 import PDEModelV2


def find_linear_layers(model: nn.Module, prefix: str = "") -> list:
    """Recursively find all Linear layers and their full names."""
    linear_layers = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            linear_layers.append((full_name, module.in_features, module.out_features))
        else:
            linear_layers.extend(find_linear_layers(module, full_name))
    return linear_layers


def main():
    # Create small test model
    config = {
        'model': {
            'in_channels': 18,
            'hidden_dim': 256,
            'patch_size': 16,
            'num_layers': 2,  # Small for testing
            'num_heads': 4,
            'dropout': 0.0,
            'encoder': {
                'stem_hidden': 64,
                'stem_out': 128,
            },
            'intra_patch': {
                'num_layers': 1,
                'temporal_window': 3,
                'num_heads': 4,
            },
            'na': {
                'base_kernel': 5,
            },
            'decoder': {
                'stem_channels': 128,
                'hidden_channels': 64,
            },
        }
    }

    print("Creating PDEModelV2...")
    model = PDEModelV2(config)

    print("\n" + "=" * 70)
    print("All Linear layers in PDEModelV2:")
    print("=" * 70)

    linear_layers = find_linear_layers(model)

    # Group by component
    encoder_layers = [l for l in linear_layers if l[0].startswith('encoder')]
    transformer_layers = [l for l in linear_layers if l[0].startswith('transformer')]
    decoder_layers = [l for l in linear_layers if l[0].startswith('decoder')]

    print("\n[ENCODER]")
    for name, in_f, out_f in encoder_layers:
        print(f"  {name}: Linear({in_f}, {out_f})")

    print("\n[TRANSFORMER - NA Attention]")
    na_layers = [l for l in transformer_layers if 'na.na' in l[0]]
    for name, in_f, out_f in na_layers[:10]:  # Show first 10
        print(f"  {name}: Linear({in_f}, {out_f})")
    if len(na_layers) > 10:
        print(f"  ... and {len(na_layers) - 10} more")

    print("\n[TRANSFORMER - FFN]")
    ffn_layers = [l for l in transformer_layers if 'ffn' in l[0]]
    for name, in_f, out_f in ffn_layers[:10]:
        print(f"  {name}: Linear({in_f}, {out_f})")
    if len(ffn_layers) > 10:
        print(f"  ... and {len(ffn_layers) - 10} more")

    print("\n[DECODER]")
    for name, in_f, out_f in decoder_layers:
        print(f"  {name}: Linear({in_f}, {out_f})")

    # Suggest target_modules
    print("\n" + "=" * 70)
    print("Suggested LoRA target_modules:")
    print("=" * 70)

    # Extract unique module name patterns
    patterns = set()
    for name, _, _ in transformer_layers:
        # Get the last part of the name (the actual layer name)
        parts = name.split('.')
        if 'qkv' in parts or 'proj' in parts[-1] == 'proj':
            patterns.add(parts[-1])
        if any(p in parts for p in ['gate_proj', 'up_proj', 'down_proj']):
            patterns.add(parts[-1])

    # NA attention layers
    print("\nFor NA Attention (in natten):")
    print('  target_modules = ["qkv", "proj"]')

    print("\nFor FFN (SwiGLU):")
    print('  target_modules = ["gate_proj", "up_proj", "down_proj"]')

    print("\nCombined (recommended):")
    print('  target_modules = ["qkv", "proj", "gate_proj", "up_proj", "down_proj"]')

    # Test PEFT compatibility
    print("\n" + "=" * 70)
    print("Testing PEFT LoRA application...")
    print("=" * 70)

    try:
        from peft import LoraConfig, get_peft_model

        # Try applying LoRA to transformer only
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["qkv", "proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        # Apply to transformer
        model.transformer = get_peft_model(model.transformer, lora_config)

        print("\n✓ PEFT LoRA applied successfully!")

        # Count trainable params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nParameter summary:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable (LoRA): {trainable_params:,}")
        print(f"  Ratio: {100 * trainable_params / total_params:.2f}%")

        # List trainable params
        print("\nTrainable parameter names (first 20):")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in trainable_names[:20]:
            print(f"  {name}")
        if len(trainable_names) > 20:
            print(f"  ... and {len(trainable_names) - 20} more")

    except ImportError:
        print("PEFT not installed, skipping LoRA test")
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
