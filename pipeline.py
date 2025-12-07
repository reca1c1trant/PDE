"""
Training pipeline for PDE Causal Autoregressive Model.
Predicts future timesteps given past timesteps.
"""

import yaml
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from pde_dataset import PDEDataset, DimensionGroupedSampler, collate_fn
from model import PDECausalModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    # Create datasets (temporal_length=17 for causal AR)
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
    
    # Create samplers
    batch_size = config['dataloader']['batch_size']
    
    train_sampler = DimensionGroupedSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=config['dataloader']['shuffle']
    )
    
    val_sampler = DimensionGroupedSampler(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create dataloaders
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


def compute_masked_loss(pred, target, channel_mask):
    """
    Compute MSE loss with channel masking.
    Only compute loss on real channels (not padded ones).
    
    Args:
        pred: [B, T, *spatial, 6]
        target: [B, T, *spatial, 6]
        channel_mask: [B, 6] - 1 for real channels, 0 for padded
    
    Returns:
        loss: scalar
    """
    # Compute per-element squared error
    squared_error = (pred - target) ** 2  # [B, T, *spatial, 6]
    
    # Expand mask to match spatial dimensions
    # [B, 6] → [B, 1, ..., 1, 6]
    ndim = pred.ndim
    mask_shape = [pred.shape[0]] + [1] * (ndim - 2) + [pred.shape[-1]]
    mask = channel_mask.view(*mask_shape)  # [B, 1, ..., 1, 6]
    
    # Apply mask
    masked_error = squared_error * mask  # [B, T, *spatial, 6]
    
    # Compute mean over valid elements
    num_valid = mask.sum()
    if num_valid > 0:
        loss = masked_error.sum() / num_valid
    else:
        loss = masked_error.sum()  # Fallback (shouldn't happen)
    
    return loss


def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """Train for one epoch with causal AR."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        data = batch['data'].to(device)  # [B, 17, *spatial, 6]
        channel_mask = batch['channel_mask'].to(device)  # [B, 6]
        
        # Causal AR: input = t[0:16], target = t[1:17]
        input_data = data[:, :-1]    # [B, 16, *spatial, 6]
        target_data = data[:, 1:]    # [B, 16, *spatial, 6]
        
        # Forward pass
        # Note: Causal masking is handled inside Transformer by default
        # when using autoregressive position encodings
        output = model(input_data)  # [B, 16, *spatial, 6]
        
        # Compute loss
        loss = compute_masked_loss(output, target_data, channel_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for stability)
        if config['training'].get('grad_clip', None):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Log to WandB
        if batch_idx % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/batch': batch_idx,
                'train/lr': optimizer.param_groups[0]['lr']
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        data = batch['data'].to(device)
        channel_mask = batch['channel_mask'].to(device)
        
        # Causal AR: input = t[0:16], target = t[1:17]
        input_data = data[:, :-1]
        target_data = data[:, 1:]
        
        # Forward pass
        output = model(input_data)
        
        # Compute loss
        loss = compute_masked_loss(output, target_data, channel_mask)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    """Main training function."""
    # Load configuration
    config = load_config('config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize WandB
    wandb.init(
        project=config['logging']['project'],
        entity=config['logging'].get('entity', None),
        config=config,
        name=f"causal-ar-{config['dataset']['seed']}"
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = PDECausalModel(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params:,}")
    wandb.config.update({'num_parameters': num_params})
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training'].get('betas', (0.9, 0.999))
    )
    
    # Setup learning rate scheduler
    if config['training'].get('use_scheduler', False):
        if config['training'].get('scheduler_type') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'] - config['training'].get('warmup_epochs', 0),
                eta_min=config['training'].get('min_lr', 1e-6)
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training'].get('min_lr', 1e-6)
            )
    else:
        scheduler = None
    
    # Watch model with WandB
    wandb.watch(model, log='all', log_freq=100)
    
    # Create checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss
        })
        
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        # Save latest
        latest_path = save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoints
        if epoch % config['logging'].get('save_interval', 10) == 0:
            periodic_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)
            logger.info(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    # Finish WandB run
    wandb.finish()
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()