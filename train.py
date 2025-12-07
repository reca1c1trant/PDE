"""
Training script for PDE Causal Autoregressive Model.
Uses Accelerate for distributed training and WandB for logging.

Usage:
    # First time setup (choose options interactively)
    accelerate config

    # Launch training
    accelerate launch train.py

    # Or single GPU
    python train.py
"""

import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import logging

from dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from pipeline import PDECausalModel, compute_masked_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    train_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='train',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed']
    )

    val_dataset = PDEDataset(
        data_dir=config['dataset']['path'],
        temporal_length=config['dataset']['temporal_length'],
        split='val',
        train_ratio=config['dataset']['train_ratio'],
        seed=config['dataset']['seed']
    )

    batch_size = config['dataloader']['batch_size']

    train_sampler = DimensionGroupedSampler(train_dataset, batch_size, shuffle=True)
    val_sampler = DimensionGroupedSampler(val_dataset, batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, accelerator, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)

    for batch_idx, batch in enumerate(pbar):
        data = batch['data']  # [B, 17, *spatial, 6]
        channel_mask = batch['channel_mask']  # [B, 6]

        # Causal AR split
        input_data = data[:, :-1]   # [B, 16, ...]
        target_data = data[:, 1:]   # [B, 16, ...]

        # Forward
        with accelerator.accumulate(model):
            output = model(input_data)
            loss = compute_masked_loss(output, target_data, channel_mask)

            accelerator.backward(loss)

            # Gradient clipping
            if config['training'].get('grad_clip'):
                accelerator.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Log to WandB
        if batch_idx % config['logging']['log_interval'] == 0:
            accelerator.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/lr': optimizer.param_groups[0]['lr'],
                'gpu/memory_gb': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            })

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def validate(model, val_loader, accelerator):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
        data = batch['data']
        channel_mask = batch['channel_mask']

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        output = model(input_data)
        loss = compute_masked_loss(output, target_data, channel_mask)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    # Load config
    config = load_config()
    set_seed(config['dataset']['seed'])

    # Initialize Accelerator
    mixed_precision = config['training'].get('mixed_precision', 'bf16')
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=grad_accum,
        log_with="wandb"
    )

    # Logging
    if accelerator.is_main_process:
        logger.info(f"Mixed Precision: {mixed_precision}")
        logger.info(f"Gradient Accumulation: {grad_accum}")
        logger.info(f"Num Processes: {accelerator.num_processes}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    if accelerator.is_main_process:
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    model = PDECausalModel(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training'].get('betas', (0.9, 0.999))
    )

    # Scheduler
    scheduler = None
    if config['training'].get('use_scheduler', True):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    if scheduler:
        scheduler = accelerator.prepare(scheduler)

    # Init WandB tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config['logging']['project'],
            config=config,
            init_kwargs={"wandb": {"name": f"pde-{config['dataset']['seed']}"}}
        )

    # Checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{config['training']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, accelerator, epoch, config)
        val_loss = validate(model, val_loader, accelerator)

        if scheduler:
            scheduler.step()

        # Log epoch metrics
        accelerator.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss
        })

        if accelerator.is_main_process:
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save checkpoints
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }

            # Latest
            torch.save(checkpoint, save_dir / 'latest_checkpoint.pt')

            # Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, save_dir / 'best_model.pt')
                logger.info(f"Saved best model (val_loss: {val_loss:.6f})")

            # Periodic
            if epoch % config['logging'].get('save_interval', 10) == 0:
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')

    accelerator.end_training()

    if accelerator.is_main_process:
        logger.info(f"\nTraining completed! Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
