"""
Main training script for PDE solver.
Demonstrates usage of PDEDataset with WandB logging.
"""

import yaml
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
import logging

from pde_dataset import PDEDataset, DimensionGroupedSampler, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    # Create datasets
    train_dataset = PDEDataset(
        data_dir=config['dataset'],
        temporal_length=config['temporal_length'],
        split='train',
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )
    
    val_dataset = PDEDataset(
        data_dir=config['dataset'],
        temporal_length=config['temporal_length'],
        split='val',
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )
    
    # Create samplers with batch_size
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
    
    # Create dataloaders using batch_sampler
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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        # TODO: Implement your forward pass
        # For now, just compute a dummy loss
        loss = torch.mean(data)  # Placeholder
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log to WandB
        if batch_idx % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/batch': batch_idx
            })
            
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            
            # TODO: Implement your forward pass
            loss = torch.mean(data)  # Placeholder
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    """Main training function."""
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize WandB
    wandb.init(
        project=config['logging']['project'],
        entity=config['logging']['entity'],
        config=config,
        name=f"pde-solver-{config['seed']}"
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # TODO: Initialize your model
    # model = YourModel(config['model']).to(device)
    # For now, create a dummy model
    model = torch.nn.Linear(10, 10).to(device)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    
    # Watch model with WandB
    wandb.watch(model, log='all', log_freq=100)
    
    # Create checkpoint directory
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss
        })
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    # Finish WandB run
    wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()