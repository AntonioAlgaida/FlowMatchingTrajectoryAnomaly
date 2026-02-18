# src/models/deep_flow.py
import torch
import torch.nn as nn
from src.models.encoder import SceneEncoder
from src.models.flow_head import FlowHead

class DeepFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SceneEncoder(cfg)
        self.flow_head = FlowHead(cfg)
        self.sigma_min = 1e-4

    def forward(self, batch, xt, t):
        """
        batch: dictionary from WaymoDataset
        xt: noisy PCA coefficients [B, 6]
        t: time steps [B]
        """
        # Encode scene conditioned on the explicit goal
        context, goal_emb = self.encoder(
            batch['agent_context'], 
            batch['agent_mask'], 
            batch['map_context'], 
            batch['map_mask'],
            batch['goal_pos'],
            batch['goal_lane'] # Lane-aware goal
        )
        
        # 2. Pass both to the Head
        v_t = self.flow_head(xt, t, context, goal_emb)
        return v_t