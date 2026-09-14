"""
MODEL_PCA.PY -- Continuous Latent Color Shift Predictor (ResMLP_256) for SD3.

Predicts (m1, m2, m3) PCA subspace shift coordinates from:
  1. Base Color CIELAB (L*, a*, b*)
  2. Target Color CIELAB (L*, a*, b*)
  3. Directional & Perceptual Features (Delta Lab, Chroma, Periodic Hue Embeddings)
"""

import math
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(self.fc1(x))
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual


class ResMLP_256(nn.Module):
    """
    Standard ResMLP architecture:
      Input (15D) -> Linear(15->256) -> 2 ResBlocks (256D) -> LayerNorm -> Linear(256->3)
    """
    def __init__(self, in_dim: int = 15, hidden: int = 256, n_blocks: int = 2, out_dim: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.out_proj(h)


class LinearBaseline(nn.Module):
    def __init__(self, in_dim=15, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class ShallowMLP(nn.Module):
    def __init__(self, in_dim=15, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MediumMLP(nn.Module):
    def __init__(self, in_dim=15, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, in_dim=15, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "Linear": lambda in_d, out_d: LinearBaseline(in_d, out_d),
    "Shallow_128": lambda in_d, out_d: ShallowMLP(in_d, 128, out_d),
    "Medium_256": lambda in_d, out_d: MediumMLP(in_d, 256, out_d),
    "Deep_256": lambda in_d, out_d: DeepMLP(in_d, 256, out_d),
    "ResMLP_256": lambda in_d, out_d: ResMLP_256(in_d, 256, 2, out_d),
}


class MLPShiftPCA:
    """
    Inference wrapper: featurizes (base_lab, target_lab) -> (m1, m2, m3).
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_name = ckpt.get("model_name", "ResMLP_256")
        
        if model_name in MODEL_REGISTRY:
            self.model = MODEL_REGISTRY[model_name](15, 3).to(self.device)
        else:
            self.model = ResMLP_256(in_dim=15, hidden=256, n_blocks=2, out_dim=3).to(self.device)
            
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def featurize(base_lab: Tuple[float, float, float],
                  target_lab: Tuple[float, float, float]) -> torch.Tensor:
        L_b, a_b, b_b = float(base_lab[0]), float(base_lab[1]), float(base_lab[2])
        L_t, a_t, b_t = float(target_lab[0]), float(target_lab[1]), float(target_lab[2])

        dL = L_t - L_b
        da = a_t - a_b
        db = b_t - b_b

        c_base = math.sqrt(a_b**2 + b_b**2)
        c_target = math.sqrt(a_t**2 + b_t**2)

        h_base = math.atan2(b_b, a_b)
        h_target = math.atan2(b_t, a_t)

        sin_hb, cos_hb = math.sin(h_base), math.cos(h_base)
        sin_ht, cos_ht = math.sin(h_target), math.cos(h_target)

        feat = [
            L_b, a_b, b_b,
            L_t, a_t, b_t,
            dL, da, db,
            c_base, c_target,
            sin_hb, cos_hb,
            sin_ht, cos_ht
        ]
        return torch.tensor([feat], dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, base_lab: Tuple[float, float, float],
                 target_lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
        inp = self.featurize(base_lab, target_lab).to(self.device)
        m = self.model(inp).squeeze(0).cpu().numpy()
        return float(m[0]), float(m[1]), float(m[2])


def load_mlp_pca(ckpt_path: str, device: str = "cuda") -> MLPShiftPCA:
    return MLPShiftPCA(ckpt_path, device=device)

