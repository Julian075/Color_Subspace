"""
TRAIN_AND_SEARCH_MLP_PCA.PY -- Architecture Search & Training for PCA Latent Shift MLP.

Evaluates multiple MLP architectures (shallow to deep residual) across:
  - R² score on (m1, m2, m3)
  - MAE on (m1, m2, m3)
  - Chroma Error: |ΔC* - ΔC*_pred|
  - Hue Angle Error: |Δh° - Δh°_pred|
  - Angular Directional Error (degrees) in PCA latent space
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = str(SCRIPT_DIR / "coleccion_datos_mlp_pca_out" / "dataset_mlp_pca_25k.csv")
OUT_DIR = str(SCRIPT_DIR / "mlp_training_out")


# =========================== FEATURE ENGINEERING ===========================
def compute_chroma_and_hue(a, b):
    chroma = np.sqrt(a**2 + b**2)
    hue_rad = np.arctan2(b, a)
    hue_deg = np.degrees(hue_rad) % 360.0
    return chroma, hue_rad, hue_deg


def angular_diff_deg(h1_deg, h2_deg):
    diff = np.abs(h1_deg - h2_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)


def prepare_features(df):
    """Extracts rich color-space features for the MLP."""
    base_L = df["base_L"].values.astype(np.float32)
    base_a = df["base_a"].values.astype(np.float32)
    base_b = df["base_b"].values.astype(np.float32)

    mod_L = df["mod_L"].values.astype(np.float32)
    mod_a = df["mod_a"].values.astype(np.float32)
    mod_b = df["mod_b"].values.astype(np.float32)

    delta_L = df["delta_L"].values.astype(np.float32)
    delta_a = df["delta_a"].values.astype(np.float32)
    delta_b = df["delta_b"].values.astype(np.float32)

    # Chroma and Hue features
    c_base, h_base_rad, h_base_deg = compute_chroma_and_hue(base_a, base_b)
    c_mod, h_mod_rad, h_mod_deg = compute_chroma_and_hue(mod_a, mod_b)
    delta_c = c_mod - c_base
    delta_h_deg = angular_diff_deg(h_mod_deg, h_base_deg)

    # Sine/Cosine representations of angles (smooth continuous boundary)
    sin_h_base, cos_h_base = np.sin(h_base_rad), np.cos(h_base_rad)
    sin_h_mod, cos_h_mod = np.sin(h_mod_rad), np.cos(h_mod_rad)
    sin_dh, cos_dh = np.sin(h_mod_rad - h_base_rad), np.cos(h_mod_rad - h_base_rad)

    # 15D Comprehensive Feature Vector
    X = np.stack([
        base_L, base_a, base_b,            # 0..2: Base Lab
        delta_L, delta_a, delta_b,         # 3..5: Delta Lab target
        mod_L, mod_a, mod_b,               # 6..8: Target Lab
        c_base, delta_c,                   # 9..10: Base Chroma & Delta Chroma
        sin_h_base, cos_h_base,            # 11..12: Base Hue periodic
        sin_dh, cos_dh,                    # 13..14: Delta Hue periodic
    ], axis=1).astype(np.float32)

    # Targets: (m1, m2, m3)
    y = np.stack([
        df["m1"].values.astype(np.float32),
        df["m2"].values.astype(np.float32),
        df["m3"].values.astype(np.float32),
    ], axis=1)

    return X, y, df["baseline_id"].values


# =========================== DATASET & SPLITS ===========================
class ColorShiftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_by_baseline(X, y, baseline_ids, train_ratio=0.80, val_ratio=0.10):
    unique_baselines = np.unique(baseline_ids)
    np.random.shuffle(unique_baselines)

    n_total = len(unique_baselines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_b = set(unique_baselines[:n_train])
    val_b = set(unique_baselines[n_train:n_train + n_val])
    test_b = set(unique_baselines[n_train + n_val:])

    train_mask = np.isin(baseline_ids, list(train_b))
    val_mask = np.isin(baseline_ids, list(val_b))
    test_mask = np.isin(baseline_ids, list(test_b))

    return (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])


# =========================== ARCHITECTURES ===========================
class LinearBaseline(nn.Module):
    def __init__(self, in_dim, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLP_Shallow(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3):
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
    def __init__(self, in_dim, hidden=128, out_dim=3):
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
    def __init__(self, in_dim, hidden=256, out_dim=3):
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
    def __init__(self, in_dim, hidden=256, n_blocks=2, out_dim=3):
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
    def __init__(self, in_dim, hidden=512, n_blocks=3, out_dim=3):
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


# =========================== EVALUATION METRICS ===========================
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-8
    r2_per_axis = 1.0 - (ss_res / ss_tot)
    r2_overall = 1.0 - (np.sum(ss_res) / np.sum(ss_tot))
    return r2_per_axis, r2_overall


def evaluate_model(model, X_test, y_test, x_mean, x_std):
    model.eval()
    with torch.no_grad():
        X_norm = (X_test - x_mean) / x_std
        X_t = torch.from_numpy(X_norm).to(DEVICE)
        preds_t = model(X_t)
        preds = preds_t.cpu().numpy()

    # 1. R2 scores
    r2_axes, r2_total = compute_r2(y_test, preds)

    # 2. MAE per axis
    mae_axes = np.mean(np.abs(y_test - preds), axis=0)

    # 3. Directional Angular Error in PCA space (degrees)
    dot = np.sum(preds * y_test, axis=1)
    norm_p = np.linalg.norm(preds, axis=1) + 1e-8
    norm_t = np.linalg.norm(y_test, axis=1) + 1e-8
    cos_sim = np.clip(dot / (norm_p * norm_t), -1.0, 1.0)
    angular_error_deg = np.mean(np.degrees(np.arccos(cos_sim)))

    # 4. Magnitude Error (Chroma-equivalent in PCA space)
    mag_pred = np.linalg.norm(preds, axis=1)
    mag_true = np.linalg.norm(y_test, axis=1)
    mag_mae = np.mean(np.abs(mag_true - mag_pred))

    # 5. Delta Chroma & Delta Hue metrics
    # In PCA space: m2 & m3 control the chromatic a*b* plane
    hue_pred_rad = np.arctan2(preds[:, 2], preds[:, 1])
    hue_true_rad = np.arctan2(y_test[:, 2], y_test[:, 1])
    hue_error_deg = np.mean(angular_diff_deg(np.degrees(hue_pred_rad) % 360, np.degrees(hue_true_rad) % 360))

    chroma_pca_pred = np.sqrt(preds[:, 1]**2 + preds[:, 2]**2)
    chroma_pca_true = np.sqrt(y_test[:, 1]**2 + y_test[:, 2]**2)
    chroma_pca_mae = np.mean(np.abs(chroma_pca_true - chroma_pca_pred))

    return {
        "r2_overall": float(r2_total),
        "r2_m1": float(r2_axes[0]),
        "r2_m2": float(r2_axes[1]),
        "r2_m3": float(r2_axes[2]),
        "mae_m1": float(mae_axes[0]),
        "mae_m2": float(mae_axes[1]),
        "mae_m3": float(mae_axes[2]),
        "angular_error_deg": float(angular_error_deg),
        "mag_mae": float(mag_mae),
        "hue_error_deg": float(hue_error_deg),
        "chroma_pca_mae": float(chroma_pca_mae),
    }


# =========================== TRAINING LOOP ===========================
def train_architecture(arch_name, model_cls, X_train, y_train, X_val, y_val, x_mean, x_std, epochs=80, lr=1e-3):
    print(f"\n--- Training {arch_name} ---")
    in_dim = X_train.shape[1]
    model = model_cls(in_dim).to(DEVICE)

    # Normalize inputs
    X_train_norm = (X_train - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std

    train_ds = ColorShiftDataset(X_train_norm, y_train)
    val_ds = ColorShiftDataset(X_val_norm, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_cnt = 15, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx)
                val_loss += loss_fn(pred, by).item() * len(bx)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.5f})")
                break

    model.load_state_dict(best_weights)
    return model


# =========================== MAIN RUNNER ===========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print(f">>> STARTING MLP ARCHITECTURE SEARCH & TRAINING FOR PCA SPACE")
    print(f"    Dataset: {DATASET_PATH}")
    print(f"    Device:  {DEVICE}")
    print("=" * 80)

    # 1. Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} samples across {df['baseline_id'].nunique():,} baselines.")

    X, y, baseline_ids = prepare_features(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_baseline(X, y, baseline_ids)

    print(f"Dataset Split (by baseline_id):")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Compute normalization statistics strictly from Train set
    x_mean = np.mean(X_train, axis=0, keepdims=True)
    x_std = np.std(X_train, axis=0, keepdims=True) + 1e-7

    # 2. Architecture Candidates
    CANDIDATES = [
        ("Linear_Baseline", LinearBaseline),
        ("MLP_Shallow_64", MLP_Shallow),
        ("MLP_Medium_128", MLP_Medium),
        ("MLP_Deep_256", MLP_Deep),
        ("ResMLP_256", ResMLP_256),
        ("ResMLP_512", ResMLP_512),
    ]

    results = []
    trained_models = {}

    for name, model_cls in CANDIDATES:
        model = train_architecture(name, model_cls, X_train, y_train, X_val, y_val, x_mean, x_std)
        metrics = evaluate_model(model, X_test, y_test, x_mean, x_std)
        metrics["model_name"] = name
        results.append(metrics)
        trained_models[name] = model

        print(f"  [Test Results - {name}]")
        print(f"    R² Overall:       {metrics['r2_overall']:.4f} (m1: {metrics['r2_m1']:.3f}, m2: {metrics['r2_m2']:.3f}, m3: {metrics['r2_m3']:.3f})")
        print(f"    MAE (m1,m2,m3):   ({metrics['mae_m1']:.4f}, {metrics['mae_m2']:.4f}, {metrics['mae_m3']:.4f})")
        print(f"    Chroma MAE:       {metrics['chroma_pca_mae']:.4f}")
        print(f"    Hue Angle Error:  {metrics['hue_error_deg']:.2f}°")
        print(f"    Direction Error:  {metrics['angular_error_deg']:.2f}°")

    # 3. Select Winner
    # Ranked primarily by R² Overall + Lowest Direction Error + Lowest Hue Error
    results_df = pd.DataFrame(results).sort_values(by="r2_overall", ascending=False)
    winner_name = results_df.iloc[0]["model_name"]
    winner_model = trained_models[winner_name]

    print("\n" + "=" * 80)
    print(">>> ARCHITECTURE SEARCH LEADERBOARD (Ranked on Held-Out Test Set):")
    print("=" * 80)
    print(results_df[["model_name", "r2_overall", "r2_m1", "r2_m2", "r2_m3", "hue_error_deg", "chroma_pca_mae", "angular_error_deg"]].to_string(index=False))

    print(f"\n>>> WINNING MODEL: {winner_name} (R² = {results_df.iloc[0]['r2_overall']:.4f})")

    # 4. Save Final Best Model & Metadata
    model_save_path = os.path.join(OUT_DIR, "mlp_shift_pca_best.pt")
    metadata_save_path = os.path.join(OUT_DIR, "mlp_shift_pca_metadata.json")
    leaderboard_csv_path = os.path.join(OUT_DIR, "architecture_search_results.csv")

    results_df.to_csv(leaderboard_csv_path, index=False)

    torch.save({
        "model_name": winner_name,
        "state_dict": winner_model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "in_dim": X.shape[1],
        "out_dim": 3,
        "test_metrics": results_df.iloc[0].to_dict(),
    }, model_save_path)

    with open(metadata_save_path, "w") as f:
        json.dump({
            "winner_model": winner_name,
            "in_dim": int(X.shape[1]),
            "feature_names": [
                "base_L", "base_a", "base_b", "delta_L", "delta_a", "delta_b",
                "mod_L", "mod_a", "mod_b", "c_base", "delta_c",
                "sin_h_base", "cos_h_base", "sin_dh", "cos_dh"
            ],
            "metrics": results_df.iloc[0].to_dict(),
        }, f, indent=2)

    print(f"\nSaved Best Model to: {model_save_path}")
    print(f"Saved Metadata to:   {metadata_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
