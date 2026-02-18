# src/models/flow_head.py
import torch
import torch.nn as nn
from src.models.layers import SinusoidalPosEmb, ResidualBlock

class FlowHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = 12
        self.time_dim = 64
        self.context_dim = 256
        
        # --- CRITICAL CHANGE ---
        # Input = Latent(12) + Time(64) + Context(256) + GOAL_SKIP(256)
        self.input_dim = 12 + 64 + 256 + 256 # = 588
        # -----------------------
        
        self.hidden_dim = 512
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            # Make the network slightly deeper/wider to handle the strong constraint
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim), # Added one more block
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

    def forward(self, x_t, t, context, goal_emb): # <--- Accept Goal Here
        t_embed = self.time_mlp(t)
        
        # Concatenate EVERYTHING
        feat = torch.cat([x_t, t_embed, context, goal_emb], dim=-1)
        
        v_t = self.net(feat)
        return v_t