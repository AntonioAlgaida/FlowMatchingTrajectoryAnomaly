# src/engine/losses.py
import torch
import torch.nn as nn
import json
import os

class CFMLoss(nn.Module):
    def __init__(self, cfg, sigma_min=1e-4, lambda_coord=0.1):
        super().__init__()
        self.sigma_min = sigma_min
        self.lambda_coord = lambda_coord # Weight for the coordinate loss
        
        # Load PCA basis for differentiable reconstruction
        pca_path = os.path.join(os.path.dirname(cfg.data.processed_dir), "pca_basis.json")
        with open(pca_path, 'r') as f:
            pca_data = json.load(f)
        
        # Register as buffers so they move to the correct GPU automatically
        self.register_buffer("pca_comp", torch.tensor(pca_data['components']).float()) # [12, 160]
        self.register_buffer("pca_mean", torch.tensor(pca_data['mean']).float())       # [160]
        self.scale_pos = 50.0

    def forward(self, model, batch):
        x1 = batch['target_action'] # [B, 12] PCA coefficients
        weights = batch['sample_weight'] # [B]
        
        # --- NEW: NORMALIZE WEIGHTS ---
        # This ensures the total gradient magnitude stays consistent
        weights = weights / weights.mean() 
        # ------------------------------

        B, device = x1.shape[0], x1.device
        
        t = torch.rand(B, device=device)
        t_exp = t.view(-1, 1)
        
        # Noise in manifold space
        x0 = torch.randn_like(x1)

        # 1. Standard Flow Matching Math
        xt = (1 - (1 - self.sigma_min) * t_exp) * x0 + t_exp * x1
        ut = x1 - (1 - self.sigma_min) * x0

        # Predict velocity in PCA space
        vt = model(batch, xt, t)

        # --- Loss A: Weighted Flow Matching Loss (Manifold Space) ---
        # 1. Compute MSE per sample: [B, 12] -> [B]
        loss_flow_per_sample = (vt - ut).pow(2).mean(dim=1)
        # 2. Apply weights and take batch mean
        loss_flow = (loss_flow_per_sample * weights).mean()
        
        # --- Loss B: Coordinate Reconstruction Loss (Physical Space) ---
        # We find the predicted x1 by taking the current velocity to its logical conclusion:
        # Since u_t is constant in OT-CFM, x1_pred = (vt / 1.0) + x0 (approx)
        # More rigorously: we want the predicted velocity to point at the real x1
        x1_pred = vt + (1 - self.sigma_min) * x0
        
        # Transform BOTH to meters
        # Predicted traj in meters:
        traj_pred_m = (torch.matmul(x1_pred, self.pca_comp) + self.pca_mean) * self.scale_pos
        # Ground truth traj in meters:
        traj_gt_m = (torch.matmul(x1, self.pca_comp) + self.pca_mean) * self.scale_pos
        
        # 1. Compute MSE in meters per sample
        mse_coord_per_sample = (traj_pred_m - traj_gt_m).pow(2).mean(dim=1)

        # 2. Use RMSE instead of MSE for coordinates to keep scales sane
        # This changes the loss from "meters squared" to just "meters"
        rmse_coord_per_sample = torch.sqrt(mse_coord_per_sample + 1e-6)

        # 3. Apply normalized weights
        loss_coord = (rmse_coord_per_sample * weights).mean()

        # 4. Use a smaller lambda
        # Try lambda_coord = 1.0 or 0.1
        return loss_flow + (self.lambda_coord * loss_coord), loss_flow, loss_coord