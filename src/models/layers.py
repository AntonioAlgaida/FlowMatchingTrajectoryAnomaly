# src/models/layers.py
import torch
import torch.nn as nn
import math
class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [B] (the time values t)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb # [B, dim]
    
class FourierGoalEncoder(nn.Module):
    def __init__(self, input_dim=2, out_dim=256, num_freqs=32):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = out_dim
        
        # Fourier frequencies
        self.register_buffer("freqs", torch.pow(2, torch.arange(num_freqs)) * torch.pi)
        
        # 2 coords * (sin + cos) * num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * num_freqs, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x: [B, 2]
        B = x.shape[0]
        # x_expand: [B, 2, 1] * freqs: [1, 1, num_freqs] -> [B, 2, num_freqs]
        x_proj = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        # Concatenate sin and cos: [B, 2, 2 * num_freqs]
        feat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.mlp(feat.view(B, -1)) # [B, out_dim]