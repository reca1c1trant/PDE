"""
Convert old checkpoint (6 channels) to new format (18 channels).

Key changes:
1. encoder_2d.scalar_conv.0: [64, 3, 3, 3] → [64, 15, 3, 3]
2. decoder_2d.scalar_conv.7: [64, 3, 4, 4] → [64, 15, 4, 4]
3. Move u,v from old vector channels to new scalar channels 2,3

Usage:
    python convert_checkpoint.py --input old.pt --output new.pt
"""

import argparse
import torch
import copy


def convert_checkpoint(input_path: str, output_path: str):
    """Convert old 6-channel checkpoint to new 18-channel format."""
    print(f"Loading: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')

    old_state = ckpt['model_state_dict']
    new_state = {}

    # Track what we modified
    modified_keys = []
    copied_keys = []

    for key, value in old_state.items():
        # ============================================================
        # Encoder first layer: scalar_conv.0
        # Old: [64, 3, 3, 3] processes [0, 0, 0] (empty scalar)
        # New: [64, 15, 3, 3] processes [0, 0, u, v, 0, ...]
        # ============================================================
        if key == 'encoder_2d.scalar_conv.0.weight':
            # Old shape: [64, 3, 3, 3]
            # New shape: [64, 15, 3, 3]
            old_weight = value  # [64, 3, 3, 3]
            new_weight = torch.zeros(64, 15, 3, 3)

            # Copy old vector_conv weights (u, v channels) to scalar positions 2, 3
            old_vector_weight = old_state['encoder_2d.vector_conv.0.weight']  # [64, 3, 3, 3]
            new_weight[:, 2, :, :] = old_vector_weight[:, 0, :, :]  # u → scalar ch 2
            new_weight[:, 3, :, :] = old_vector_weight[:, 1, :, :]  # v → scalar ch 3

            new_state[key] = new_weight
            modified_keys.append(f"{key}: {old_weight.shape} → {new_weight.shape}")
            continue

        # ============================================================
        # Encoder first layer: vector_conv.0
        # Old: [64, 3, 3, 3] processes [u, v, 0] (mistakenly)
        # New: [64, 3, 3, 3] processes [vx, vy, vz] (true velocity)
        # For diffusion-reaction, velocity is all zeros, so reinitialize
        # ============================================================
        if key == 'encoder_2d.vector_conv.0.weight':
            # Keep the old weights but they won't be meaningful for new velocity
            # Could also reinitialize, but keeping for potential transfer
            new_state[key] = value
            copied_keys.append(f"{key}: {value.shape} (kept, may need retraining)")
            continue

        # ============================================================
        # Decoder last layer: scalar_conv.7.weight
        # Old: [64, 3, 4, 4] (ConvTranspose2d: [in_ch, out_ch, H, W])
        # New: [64, 15, 4, 4]
        # ============================================================
        if key == 'decoder_2d.scalar_conv.7.weight':
            old_weight = value  # [64, 3, 4, 4]
            new_weight = torch.zeros(64, 15, 4, 4)

            # Copy old vector_conv output weights (u, v) to scalar positions 2, 3
            old_vector_weight = old_state['decoder_2d.vector_conv.7.weight']  # [64, 3, 4, 4]
            new_weight[:, 2, :, :] = old_vector_weight[:, 0, :, :]  # u → scalar ch 2
            new_weight[:, 3, :, :] = old_vector_weight[:, 1, :, :]  # v → scalar ch 3

            new_state[key] = new_weight
            modified_keys.append(f"{key}: {old_weight.shape} → {new_weight.shape}")
            continue

        # ============================================================
        # Decoder last layer: scalar_conv.7.bias
        # Old: [3]
        # New: [15]
        # ============================================================
        if key == 'decoder_2d.scalar_conv.7.bias':
            old_bias = value  # [3]
            new_bias = torch.zeros(15)

            # Copy old vector_conv bias (u, v) to scalar positions 2, 3
            old_vector_bias = old_state['decoder_2d.vector_conv.7.bias']  # [3]
            new_bias[2] = old_vector_bias[0]  # u → scalar ch 2
            new_bias[3] = old_vector_bias[1]  # v → scalar ch 3

            new_state[key] = new_bias
            modified_keys.append(f"{key}: {old_bias.shape} → {new_bias.shape}")
            continue

        # ============================================================
        # Decoder last layer: vector_conv.7 (keep as is)
        # These weights output [vx, vy, vz], which are all zeros for
        # diffusion-reaction. May need retraining for other datasets.
        # ============================================================
        if key == 'decoder_2d.vector_conv.7.weight':
            new_state[key] = value
            copied_keys.append(f"{key}: {value.shape} (kept, may need retraining)")
            continue

        if key == 'decoder_2d.vector_conv.7.bias':
            new_state[key] = value
            copied_keys.append(f"{key}: {value.shape} (kept, may need retraining)")
            continue

        # ============================================================
        # All other layers: copy directly
        # ============================================================
        new_state[key] = value
        copied_keys.append(f"{key}: {value.shape}")

    # Create new checkpoint
    new_ckpt = {
        'model_state_dict': new_state,
        'converted_from': input_path,
        'conversion_note': 'Converted from 6-channel to 18-channel format. '
                          'u,v moved from vector[0:2] to scalar[2:4].',
    }

    # Copy other metadata if exists
    for k in ['global_step', 'config']:
        if k in ckpt:
            new_ckpt[k] = ckpt[k]

    # Save
    print(f"\nSaving: {output_path}")
    torch.save(new_ckpt, output_path)

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")

    print(f"\nModified layers ({len(modified_keys)}):")
    for k in modified_keys:
        print(f"  {k}")

    print(f"\nCopied layers: {len(copied_keys)} (including {len([k for k in copied_keys if 'may need' in k])} that may need retraining)")

    # Verify shapes
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    print(f"encoder_2d.scalar_conv.0.weight: {new_state['encoder_2d.scalar_conv.0.weight'].shape}")
    print(f"decoder_2d.scalar_conv.7.weight: {new_state['decoder_2d.scalar_conv.7.weight'].shape}")
    print(f"decoder_2d.scalar_conv.7.bias: {new_state['decoder_2d.scalar_conv.7.bias'].shape}")

    print(f"\nDone! Converted checkpoint saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to new format")
    parser.add_argument('--input', type=str, required=True, help='Input checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
