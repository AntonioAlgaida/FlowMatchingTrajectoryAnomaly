# scripts/run_train.py
import os
import sys
import hydra
import logging
from pathlib import Path

from omegaconf import DictConfig

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.waymo_dataset import WaymoDataset
from src.models.deep_flow import DeepFlow
from src.engine.trainer import Trainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    # 1. Load Data & Model
    train_set = WaymoDataset(cfg, split='training', in_memory=cfg.training.in_memory)
    val_set = WaymoDataset(cfg, split='validation', in_memory=cfg.training.in_memory)
    model = DeepFlow(cfg)

    # 2. Initialize Trainer
    trainer = Trainer(cfg, model, train_set, val_set)
    
    # 3. Checkpoint Resume Logic
    checkpoint_path = "checkpoints/latest.pth" # Try resuming from latest first
    if os.path.exists(checkpoint_path):
        print("\n" + "="*50)
        choice = input(f"🔍 Found existing checkpoint. Resume training? [y/N]: ").lower()
        if choice == 'y':
            trainer.load_checkpoint(checkpoint_path)
        print("="*50 + "\n")

    # 4. Training Loop
    print(f"Starting Training on {trainer.device}...")
    for epoch in range(cfg.training.epochs):
        train_loss, train_loss_flow, train_loss_coord = trainer.train_epoch(epoch)
        val_loss, val_loss_flow, val_loss_coord = trainer.validate()
        
        # --- THE MISSING CALL: SAVE EVERY EPOCH ---
        trainer.save_checkpoint(val_loss, epoch)
        # ------------------------------------------
        
        trainer.scheduler.step()
        
        # Log summary
        lr = trainer.optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch} | TRAIN: {train_loss:.4f} (Flow: {train_loss_flow:.4f}, Coord: {train_loss_coord:.4f}) | VAL: {val_loss:.4f} (Flow: {val_loss_flow:.4f}, Coord: {val_loss_coord:.4f}) | LR: {lr:.6e}")
        logger.info(f"Epoch {epoch} | TRAIN: {train_loss:.4f} (Flow: {train_loss_flow:.4f}, Coord: {train_loss_coord:.4f}) | VAL: {val_loss:.4f} (Flow: {val_loss_flow:.4f}, Coord: {val_loss_coord:.4f}) | LR: {lr:.6e}")
        
if __name__ == "__main__":
    main()