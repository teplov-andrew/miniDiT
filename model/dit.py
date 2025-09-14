import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import extract_patches, reconstruct_image


class ConditionalNorm2d(nn.Module):
    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, features):
        h = self.norm(x)                                 
        gamma, beta = self.to_gamma_beta(features).chunk(2, dim=-1)
        return (1 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        
        factor = -math.log(10000.0) / max(half - 1, 1)
        
        expo = torch.arange(half, device=device, dtype=x.dtype) * factor
        if x.ndim == 1:
            x = x.unsqueeze(1)
        
        x = x.to(dtype=expo.dtype)
        emb = x * expo.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, num_features=128):
        super(TransformerBlock, self).__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.multihead_att = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.cond_norm = ConditionalNorm2d(hidden_size, num_features)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, features):
        norm_x = self.norm(x)
        x = self.multihead_att(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.cond_norm(x, features)
        x = self.mlp(norm_x) + x
        return x
    
    
class DiT(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=4, 
                 hidden_size=128, num_features=128, num_layers=3, num_heads=4):
        super(DiT, self).__init__()    
        
        self.patch_size = patch_size
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU()
        )
        
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
    
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
    
        self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, num_features=num_features) for _ in range(num_layers)
        ])
            
        self.fc_out = nn.Linear(hidden_size, channels_in * patch_size * patch_size)

    def forward(self, image_in, index):
        index_features = self.time_mlp(index)

        patch_seq = extract_patches(image_in, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        embs = patch_emb + self.pos_embedding

        for block in self.blocks:
            embs = block(embs, index_features)

        image_out = self.fc_out(embs)

        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size) 