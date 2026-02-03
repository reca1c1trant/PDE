"""
Inspect checkpoint structure and print all parameter shapes.

Usage:
    python inspect_checkpoint.py --ckpt /path/to/checkpoint.pt
"""

import argparse
import torch
from collections import defaultdict


def inspect_checkpoint(ckpt_path: str):
    """Load and inspect checkpoint structure."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"\n{'='*60}")
    print("Checkpoint keys:")
    print(f"{'='*60}")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"  {key}: dict with {len(ckpt[key])} items")
        elif isinstance(ckpt[key], torch.Tensor):
            print(f"  {key}: tensor {ckpt[key].shape}")
        else:
            print(f"  {key}: {type(ckpt[key]).__name__}")

    # Find state_dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f"\nUsing 'model_state_dict'")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"\nUsing 'state_dict'")
    else:
        # Assume the checkpoint itself is the state dict
        state_dict = ckpt
        print(f"\nUsing checkpoint directly as state_dict")

    print(f"\n{'='*60}")
    print("Model State Dict Structure")
    print(f"{'='*60}")

    # Group by component
    components = defaultdict(list)
    for key, value in state_dict.items():
        # Extract component name (first part before '.')
        parts = key.split('.')
        if len(parts) >= 2:
            component = parts[0]
            if parts[0] in ['encoder_2d', 'decoder_2d']:
                component = f"{parts[0]}.{parts[1]}"
        else:
            component = 'other'
        components[component].append((key, value.shape if isinstance(value, torch.Tensor) else type(value)))

    # Print grouped
    for component in sorted(components.keys()):
        print(f"\n[{component}]")
        for key, shape in components[component]:
            print(f"  {key}: {shape}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    total_params = 0
    component_params = defaultdict(int)
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            n = value.numel()
            total_params += n

            parts = key.split('.')
            if 'encoder' in key:
                component_params['encoder'] += n
            elif 'decoder' in key:
                component_params['decoder'] += n
            elif 'transformer' in key:
                component_params['transformer'] += n
            else:
                component_params['other'] += n

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    for comp, n in sorted(component_params.items()):
        print(f"  {comp}: {n / 1e6:.2f}M ({100*n/total_params:.1f}%)")

    # Specifically check encoder first layer (the one we need to modify)
    print(f"\n{'='*60}")
    print("Encoder First Layer (need modification)")
    print(f"{'='*60}")

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Look for first conv layers
            if 'encoder' in key and ('vector_conv.0' in key or 'scalar_conv.0' in key):
                print(f"  {key}: {value.shape}")

    print(f"\n{'='*60}")
    print("Decoder Last Layer (need modification)")
    print(f"{'='*60}")

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Look for last conv layers in decoder
            if 'decoder' in key and 'conv' in key:
                parts = key.split('.')
                # Find the layer index
                for i, p in enumerate(parts):
                    if p.isdigit() and int(p) >= 6:  # Last layers
                        print(f"  {key}: {value.shape}")
                        break


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint structure")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()

    inspect_checkpoint(args.ckpt)


if __name__ == "__main__":
    main()
