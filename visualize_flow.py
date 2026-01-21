"""
Visualize Flow Mixing predictions vs ground truth.

Select 5 samples from validation set, show the last output timestep (t=16)
compared to ground truth.

Style: Similar to 1.png (side-by-side comparison with colorbar)
Coordinates: x, y in [0, 1]
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model_fno_baseline import create_fno_baseline
from model_mlp_baseline import create_mlp_baseline
from dataset_flow import FlowMixingDataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model_type = config.get('model', {}).get('type', 'fno')
    if model_type == 'mlp':
        model = create_mlp_baseline(config)
    else:
        model = create_fno_baseline(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def visualize_samples(
    model,
    dataset: FlowMixingDataset,
    device: torch.device,
    num_samples: int = 5,
    save_path: str = "flow_visualization.png",
    seed: int = 42,
):
    """
    Visualize model predictions vs ground truth.

    Args:
        model: Trained model
        dataset: Validation dataset
        device: torch device
        num_samples: Number of samples to visualize
        save_path: Output image path
        seed: Random seed for sample selection
    """
    # Select random samples
    np.random.seed(seed)
    total_clips = len(dataset)
    sample_indices = np.random.choice(total_clips, min(num_samples, total_clips), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Coordinate range [0, 1]
    extent = [0, 1, 0, 1]

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get data
            sample = dataset[idx]
            data = sample['data'].unsqueeze(0).to(device)  # [1, T, H, W, C]
            vtmax = sample['vtmax'].item()

            # Model prediction
            output = model(data)  # [1, T-1, H, W, C]

            # Get last timestep (t=16, index 15 in output)
            # input: t=0,1,...,16 (17 frames)
            # output: t=1,2,...,16 (16 frames)
            # last output = output[:, -1] = prediction for t=16

            pred_last = output[0, -1, :, :, 0].cpu().numpy()  # [H, W]
            gt_last = data[0, -1, :, :, 0].cpu().numpy()      # [H, W], t=16

            # Compute error
            error = np.abs(pred_last - gt_last)

            # Determine color range
            vmin = min(gt_last.min(), pred_last.min())
            vmax = max(gt_last.max(), pred_last.max())

            # Plot Ground Truth
            im0 = axes[i, 0].imshow(
                gt_last, origin='lower', extent=extent,
                cmap='jet', vmin=vmin, vmax=vmax
            )
            axes[i, 0].set_title(f'Ground Truth (t=16)\nvtmax={vtmax:.2f}', fontsize=10)
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('y')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

            # Plot Prediction
            im1 = axes[i, 1].imshow(
                pred_last, origin='lower', extent=extent,
                cmap='jet', vmin=vmin, vmax=vmax
            )
            axes[i, 1].set_title(f'Prediction (t=16)', fontsize=10)
            axes[i, 1].set_xlabel('x')
            axes[i, 1].set_ylabel('y')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Plot Error
            im2 = axes[i, 2].imshow(
                error, origin='lower', extent=extent,
                cmap='hot'
            )
            rmse = np.sqrt(np.mean(error**2))
            axes[i, 2].set_title(f'|Error| (RMSE={rmse:.4f})', fontsize=10)
            axes[i, 2].set_xlabel('x')
            axes[i, 2].set_ylabel('y')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--output', type=str, default='flow_visualization.png', help='Output image path')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)

    # Create validation dataset
    val_dataset = FlowMixingDataset(
        data_path=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed'],
        clips_per_sample=None,
    )
    print(f"Validation dataset: {len(val_dataset)} clips")

    # Visualize
    visualize_samples(
        model=model,
        dataset=val_dataset,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
