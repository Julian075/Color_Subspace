"""
MODEL_PCA.PY -- Model definition and inference wrapper for PCA Shift MLP in FLUX.2.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Architecture Definitions
# =============================================================================
class LinearBaseline(nn.Module):
    def __init__(self, in_dim=15, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLP_Shallow(nn.Module):
    def __init__(self, in_dim=15, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLP_Medium(nn.Module):
    def __init__(self, in_dim=15, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLP_Deep(nn.Module):
    def __init__(self, in_dim=15, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResMLP_256(nn.Module):
    def __init__(self, in_dim=15, hidden=256, n_blocks=2, out_dim=3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_head(h)


class ResMLP_512(nn.Module):
    def __init__(self, in_dim=15, hidden=512, n_blocks=3, out_dim=3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_head(h)


# =============================================================================
# Inference Wrapper
# =============================================================================
ARCH_MAP = {
    "Linear_Baseline": lambda in_dim: LinearBaseline(in_dim, 3),
    "MLP_Shallow_64":  lambda in_dim: MLP_Shallow(in_dim, 64, 3),
    "MLP_Medium_128":  lambda in_dim: MLP_Medium(in_dim, 128, 3),
    "MLP_Deep_256":    lambda in_dim: MLP_Deep(in_dim, 256, 3),
    "ResMLP_256":      lambda in_dim: ResMLP_256(in_dim, 256, 2, 3),
    "ResMLP_512":      lambda in_dim: ResMLP_512(in_dim, 512, 3, 3),
}


class MLPShiftPCA:
    """End-to-end wrapper for predicting PCA shift vectors from color pairs."""
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MLP checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_name = checkpoint.get("model_name", "ResMLP_256")
        in_dim = checkpoint.get("in_dim", 15)

        if model_name in ARCH_MAP:
            self.model = ARCH_MAP[model_name](in_dim).to(self.device)
        else:
            self.model = ResMLP_256(in_dim=in_dim, hidden=256, n_blocks=2, out_dim=3).to(self.device)

        sd = checkpoint.get("state_dict", checkpoint.get("model_state", checkpoint))
        self.model.load_state_dict(sd)
        self.model.eval()

        self.x_mean = np.array(checkpoint["x_mean"], dtype=np.float32)
        self.x_std = np.array(checkpoint["x_std"], dtype=np.float32)

    def __call__(self, init_lab, target_lab):
        return self.predict_m(init_lab, target_lab)

    def predict_m(self, init_lab, target_lab) -> np.ndarray:
        """
        Predicts continuous (m1, m2, m3) PCA shift vector.
        init_lab: (L, a, b) tuple or array
        target_lab: (L, a, b) tuple or array
        """
        base_L, base_a, base_b = float(init_lab[0]), float(init_lab[1]), float(init_lab[2])
        mod_L, mod_a, mod_b = float(target_lab[0]), float(target_lab[1]), float(target_lab[2])

        delta_L = mod_L - base_L
        delta_a = mod_a - base_a
        delta_b = mod_b - base_b

        c_base = math.hypot(base_a, base_b)
        c_mod = math.hypot(mod_a, mod_b)
        delta_c = c_mod - c_base

        h_base_rad = math.atan2(base_b, base_a)
        h_mod_rad = math.atan2(mod_b, mod_a)

        sin_h_base, cos_h_base = math.sin(h_base_rad), math.cos(h_base_rad)
        sin_dh, cos_dh = math.sin(h_mod_rad - h_base_rad), math.cos(h_mod_rad - h_base_rad)

        feat = np.array([[
            base_L, base_a, base_b,
            delta_L, delta_a, delta_b,
            mod_L, mod_a, mod_b,
            c_base, delta_c,
            sin_h_base, cos_h_base,
            sin_dh, cos_dh,
        ]], dtype=np.float32)

        feat_norm = (feat - self.x_mean) / self.x_std
        with torch.no_grad():
            t = torch.from_numpy(feat_norm).to(self.device)
            m = self.model(t).cpu().numpy()[0]

        return m.astype(np.float32)


def load_mlp_pca(ckpt_path: str, device: str = "cuda") -> MLPShiftPCA:
    return MLPShiftPCA(ckpt_path, device)
