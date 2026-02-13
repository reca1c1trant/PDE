"""
Convert PDEModelV2 checkpoint to PDEModelV3 (SharedNATransformer) format.

Weight mapping:
    encoder.* → encoder.*  (2D encoder, unchanged)
    decoder.* → decoder.*  (2D decoder, unchanged)

    transformer.layers_1x1.{i}.norm1 → transformer.layers.{i}.norm1  (shared)
    transformer.layers_1x1.{i}.norm2 → transformer.layers.{i}.norm2  (shared)
    transformer.layers_1x1.{i}.ffn   → transformer.layers.{i}.ffn    (shared)
    transformer.layers_1x1.{i}.na    → transformer.layers.{i}.na_2d_1x1
    transformer.layers_1x2.{i}.na    → transformer.layers.{i}.na_2d_1x2
    transformer.layers_1x4.{i}.na    → transformer.layers.{i}.na_2d_1x4
    transformer.final_norm           → transformer.final_norm

    Discarded: norm1/norm2/ffn from layers_1x2 and layers_1x4 (use shared from 1x1)

Usage:
    python scripts/convert_v2_to_v3.py --input checkpoints_v3_s/best_tf.pt --output checkpoints_v3_s/best_tf_v3.pt
"""

import argparse
import torch
from pathlib import Path


def convert_v2_to_v3(input_path: str, output_path: str):
    print(f"Loading: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        old_sd = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        old_sd = ckpt['state_dict']
    else:
        old_sd = ckpt

    # Strip DDP/compile prefixes
    cleaned_sd = {}
    for k, v in old_sd.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        cleaned_sd[k] = v

    new_sd = {}
    mapped = 0
    discarded = 0

    for key, value in cleaned_sd.items():
        if key.startswith('encoder.') or key.startswith('decoder.'):
            # Encoder/decoder: keep as-is
            new_sd[key] = value
            mapped += 1

        elif key.startswith('transformer.final_norm.'):
            new_sd[key] = value
            mapped += 1

        elif key.startswith('transformer.layers_1x1.'):
            # layers_1x1.{i}.{component}.{param}
            rest = key[len('transformer.layers_1x1.'):]
            parts = rest.split('.', 1)
            layer_idx = parts[0]
            component_param = parts[1]  # e.g., "norm1.weight", "na.qkv.weight", "ffn.gate_proj.weight"

            component = component_param.split('.')[0]

            if component == 'na':
                # V2: na.na.qkv.weight (wrapper -> NattenNA3D)
                # V3: na_2d_1x1.qkv.weight (direct NattenNA3D)
                # Strip "na." prefix, then strip inner "na." from wrapper
                inner = component_param[3:]  # "na.qkv.weight"
                if inner.startswith('na.'):
                    inner = inner[3:]  # "qkv.weight"
                new_key = f'transformer.layers.{layer_idx}.na_2d_1x1.{inner}'
            else:
                # norm1, norm2, ffn, dropout → shared
                new_key = f'transformer.layers.{layer_idx}.{component_param}'

            new_sd[new_key] = value
            mapped += 1

        elif key.startswith('transformer.layers_1x2.'):
            rest = key[len('transformer.layers_1x2.'):]
            parts = rest.split('.', 1)
            layer_idx = parts[0]
            component_param = parts[1]

            component = component_param.split('.')[0]

            if component == 'na':
                inner = component_param[3:]
                if inner.startswith('na.'):
                    inner = inner[3:]
                new_key = f'transformer.layers.{layer_idx}.na_2d_1x2.{inner}'
                new_sd[new_key] = value
                mapped += 1
            else:
                discarded += 1

        elif key.startswith('transformer.layers_1x4.'):
            rest = key[len('transformer.layers_1x4.'):]
            parts = rest.split('.', 1)
            layer_idx = parts[0]
            component_param = parts[1]

            component = component_param.split('.')[0]

            if component == 'na':
                inner = component_param[3:]
                if inner.startswith('na.'):
                    inner = inner[3:]
                new_key = f'transformer.layers.{layer_idx}.na_2d_1x4.{inner}'
                new_sd[new_key] = value
                mapped += 1
            else:
                discarded += 1

        else:
            # Unknown key, keep as-is
            new_sd[key] = value
            mapped += 1

    print(f"Mapped: {mapped} keys")
    print(f"Discarded: {discarded} keys (redundant norm/ffn from 1x2/1x4)")
    print(f"New state dict: {len(new_sd)} keys")

    # Save
    if 'model_state_dict' in ckpt:
        ckpt['model_state_dict'] = new_sd
    elif 'state_dict' in ckpt:
        ckpt['state_dict'] = new_sd
    else:
        ckpt = new_sd

    torch.save(ckpt, output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert V2 checkpoint to V3 format")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    convert_v2_to_v3(args.input, args.output)


if __name__ == '__main__':
    main()
