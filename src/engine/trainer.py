import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.engine.losses import CFMLoss

import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, model, train_dataset, val_dataset):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # 1. DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, 
            shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, 
            shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor
        )

        # 2. Optimization
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs)
        self.criterion = CFMLoss(cfg).to(self.device)
        
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss_flow = 0
        total_loss_coord = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move all tensors in dictionary to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward + Loss
            loss, loss_flow, loss_coord = self.criterion(self.model, batch)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_flow += loss_flow.item()
            total_loss_coord += loss_coord.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "flow": f"{loss_flow.item():.4f}", "coord": f"{loss_coord.item():.4f}"})
            
        return total_loss / len(self.train_loader), total_loss_flow / len(self.train_loader), total_loss_coord / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_loss_flow = 0
        total_loss_coord = 0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, loss_flow, loss_coord = self.criterion(self.model, batch)
            total_loss += loss.item()
            total_loss_flow += loss_flow.item()
            total_loss_coord += loss_coord.item()
        
        return total_loss / len(self.val_loader), total_loss_flow / len(self.val_loader), total_loss_coord / len(self.val_loader)
    
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
            
        print(f"✅ State loaded from {path}")
        logger.info(f"✅ State loaded from {path}")
        return True

    def save_checkpoint(self, val_loss, epoch):
        """
        Saves the complete state to allow for perfect resumption.
        """
        os.makedirs("checkpoints", exist_ok=True)
        
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'epoch': epoch
        }
        
        # Always save the latest to resume after a crash
        torch.save(state, "checkpoints/latest.pth")
        
        # Save the best model for evaluation
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, "checkpoints/best_model.pth")
            # print(f" ⭐ New Best Model! (Loss: {val_loss:.4f})")
            logger.info(f" ⭐ New Best Model! (Loss: {val_loss:.4f})")
