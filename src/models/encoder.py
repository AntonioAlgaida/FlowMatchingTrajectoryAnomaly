# src/models/encoder.py
import torch
import torch.nn as nn
from src.models.layers import MLP, ResidualBlock

class SceneEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = 256
        
        # --- Stage A: Local Encoders ---
        # 1. Agent Encoder
        # Input: 11 steps * 10 features = 110
        # Output: 256 latent dim
        self.agent_local = MLP(in_dim=110, hidden_dim=256, out_dim=256)
        
        # 2. Map Encoder (Mini-PointNet)
        # Input: 7 features (x, y, z, type, r, y, g) per point
        # We process points independently then MaxPool
        self.map_local = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            ResidualBlock(256)
        )
        
        # 3. Goal Encoder
        # Input: Goal Point (2) + Goal Lane Polyline (20*2 = 40) = 42
        self.goal_local = MLP(in_dim=42, hidden_dim=256, out_dim=256)
        
        # --- Stage B: Global Interaction (Transformer) ---
        # Fuses Agents + Map + Goal into a coherent scene representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True, 
            activation='gelu',
            norm_first=True # Pre-Norm for better stability
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # --- Stage C: Ego-Centric Filtering ---
        # The Ego Agent (Query) attends to the whole scene (Keys/Values)
        self.ego_cross_attn = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            batch_first=True
        )

    def forward(self, agent_ctx, agent_mask, map_ctx, map_mask, goal_pos, goal_lane):
        """
        Args:
            agent_ctx: [B, 32, 11, 10] - Agents history
            agent_mask: [B, 32, 11] - True if valid
            map_ctx: [B, 256, 20, 7] - Map polylines
            map_mask: [B, 256] - True if valid
            goal_pos: [B, 2] - Target destination
            goal_lane: [B, 20, 2] - Lane-aware goal representation
            
        Returns:
            context_vector: [B, 256] - Global context for flow
            goal_emb: [B, 256] - Direct goal embedding for skip-connection
        """
        B = agent_ctx.shape[0]
        device = agent_ctx.device

        # 0. Sanitize Inputs (Crucial for Transformer stability)
        # Replaces NaN/Inf with 0.0 to prevent gradient explosions
        agent_ctx = torch.nan_to_num(agent_ctx)
        map_ctx = torch.nan_to_num(map_ctx)
        goal_pos = torch.nan_to_num(goal_pos)
        goal_lane = torch.nan_to_num(goal_lane)

        # --- 1. Local Encoding ---
        
        # A. Agents: Flatten time and feature dims -> [B, 32, 110]
        agent_in = agent_ctx.reshape(B, 32, -1)
        agent_tokens = self.agent_local(agent_in) # [B, 32, 256]
        
        # B. Map: Process points then pool -> [B, 256, 256]
        map_in = map_ctx.reshape(-1, 20, 7) # [B*256, 20, 7]
        map_pt_embeddings = self.map_local(map_in) # [B*256, 20, 256]
        # Max-pool over the 20 points dimension to get permutation invariance
        map_tokens = torch.max(map_pt_embeddings, dim=1)[0] 
        map_tokens = map_tokens.reshape(B, 256, 256)
        
        # C. Goal: Project coordinates -> [B, 256]
        # Concatenate Point [B, 2] and Lane [B, 40] -> [B, 42]
        goal_input = torch.cat([goal_pos, goal_lane.reshape(B, -1)], dim=-1)
        goal_emb = self.goal_local(goal_input) # [B, 256]
        goal_token = goal_emb.unsqueeze(1) # [B, 1, 256]

        # --- 2. Global Fusion ---
        
        # Concatenate all tokens: Agents + Map + Goal
        # Sequence Length = 32 + 256 + 1 = 289
        all_tokens = torch.cat([agent_tokens, map_tokens, goal_token], dim=1)
        
        # Create Padding Mask (PyTorch expects True = PAD/IGNORE)
        # 1. Collapse agent mask: valid if ANY step is valid
        a_valid = agent_mask.any(dim=-1) # [B, 32]
        # 2. Invert valid masks to get padding masks
        a_pad = ~a_valid
        m_pad = ~map_mask
        # 3. Goal is always valid (False = Do not pad)
        g_pad = torch.zeros(B, 1, device=device, dtype=torch.bool)
        
        global_pad_mask = torch.cat([a_pad, m_pad, g_pad], dim=1) # [B, 289]

        # Pass through Transformer
        global_tokens = self.global_transformer(all_tokens, src_key_padding_mask=global_pad_mask)

        # --- 3. Ego-Centric Filtering ---
        
        # The Ego is always at Index 0 of the Agent block
        ego_query = global_tokens[:, 0:1, :] # [B, 1, 256]
        
        # Cross-Attention: Ego asks "What in this scene matters to ME?"
        context_vector, _ = self.ego_cross_attn(
            query=ego_query, 
            key=global_tokens, 
            value=global_tokens, 
            key_padding_mask=global_pad_mask
        )
        
        context_vector = context_vector.squeeze(1) # [B, 256]

        # Return BOTH the fused context and the raw goal embedding
        return context_vector, goal_emb