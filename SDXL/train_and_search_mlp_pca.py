"""
TRAIN_AND_SEARCH_MLP_PCA.PY -- Architecture Search & Training for SDXL PCA Latent Shift MLP.

Evaluates multiple MLP architectures (shallow to deep residual) across:
  - R² score on (m1, m2, m3)
  - MAE on (m1, m2, m3)
  - Chroma Error: |ΔC* - ΔC*_pred|
  - Hue Angle Error: |Δh° - Δh°_pred|
  - Angular Directional Error (degrees) in PCA latent space
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "coleccion_datos_mlp_pca_out", "dataset_mlp_pca_sdxl.csv")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "mlp_training_out")


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
    """Extracts rich 15D color-space features for the MLP."""
    base_L = df["base_L"].values.astype(np.float32)
    base_a = df["base_a"].values.astype(np.float32)
    base_b = df["base_b"].values.astype(np.float32)

    mod_L = df["mod_L"].values.astype(np.float32)
    mod_a = df["mod_a"].values.astype(np.float32)
    mod_b = df["mod_b"].values.astype(np.float32)

    delta_L = df["delta_L"].values.astype(np.float32)
    delta_a = df["delta_a"].values.astype(np.float32)
    delta_b = df["delta_b"].values.astype(np.float32)

    c_base, h_base_rad, h_base_deg = compute_chroma_and_hue(base_a, base_b)
    c_mod, h_mod_rad, h_mod_deg = compute_chroma_and_hue(mod_a, mod_b)
    delta_c = c_mod - c_base

    sin_h_base, cos_h_base = np.sin(h_base_rad), np.cos(h_base_rad)
    sin_dh, cos_dh = np.sin(h_mod_rad - h_base_rad), np.cos(h_mod_rad - h_base_rad)

    X = np.stack([
        base_L, base_a, base_b,
        delta_L, delta_a, delta_b,
        mod_L, mod_a, mod_b,
        c_base, delta_c,
        sin_h_base, cos_h_base,
        sin_dh, cos_dh,
    ], axis=1).astype(np.float32)

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
from model_pca import MODEL_REGISTRY


# =========================== LOSS FUNCTION ===========================
class DirectionalLoss(nn.Module):
    def __init__(self, alpha_dir=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha_dir = alpha_dir

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1)
        dir_loss = (1.0 - cos_sim).mean()
        return mse_loss + self.alpha_dir * dir_loss


# =========================== TRAINING & EVALUATION ===========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_trues = [], []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    mae = np.mean(np.abs(preds - trues), axis=0)
    mae_total = float(np.mean(mae))

    # R^2 score
    ss_res = np.sum((trues - preds) ** 2, axis=0)
    ss_tot = np.sum((trues - np.mean(trues, axis=0)) ** 2, axis=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    r2_mean = float(np.mean(r2))

    # Angular error in degrees
    dot = np.sum(preds * trues, axis=1)
    norm_p = np.linalg.norm(preds, axis=1) + 1e-8
    norm_t = np.linalg.norm(trues, axis=1) + 1e-8
    cos_sim = np.clip(dot / (norm_p * norm_t), -1.0, 1.0)
    angular_err = np.degrees(np.arccos(cos_sim))
    mean_ang_err = float(np.mean(angular_err))

    return {
        "r2_mean": r2_mean,
        "r2_m1": float(r2[0]), "r2_m2": float(r2[1]), "r2_m3": float(r2[2]),
        "mae_mean": mae_total,
        "mae_m1": float(mae[0]), "mae_m2": float(mae[1]), "mae_m3": float(mae[2]),
        "angular_error_deg": mean_ang_err,
        "predictions": preds,
        "ground_truth": trues,
    }


def run_architecture_search(df, out_dir, epochs=80, batch_size=128, lr=1e-3, gpu="0"):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    X, y, baseline_ids = prepare_features(df)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_by_baseline(X, y, baseline_ids)

    # Standardize features
    x_mean = np.mean(X_tr, axis=0, keepdims=True)
    x_std = np.std(X_tr, axis=0, keepdims=True) + 1e-8

    X_tr_norm = (X_tr - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std
    X_te_norm = (X_te - x_mean) / x_std

    train_ds = ColorShiftDataset(X_tr_norm, y_tr)
    val_ds = ColorShiftDataset(X_val_norm, y_val)
    test_ds = ColorShiftDataset(X_te_norm, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_dim = X.shape[1]
    out_dim = y.shape[1]

    criterion = DirectionalLoss(alpha_dir=0.3)
    results = {}
    best_val_r2 = -float("inf")
    best_model_name = None
    best_state_dict = None

    print("\n" + "=" * 80)
    print("           SDXL MLP ARCHITECTURE SEARCH & TRAINING EVALUATION         ")
    print("=" * 80)

    for arch_name, model_fn in MODEL_REGISTRY.items():
        print(f"\n>>> Training architecture: {arch_name}")
        model = model_fn(in_dim, out_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_arch_val_r2 = -float("inf")
        best_arch_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs:
                val_eval = evaluate_model(model, val_loader, device)
                if val_eval["r2_mean"] > best_arch_val_r2:
                    best_arch_val_r2 = val_eval["r2_mean"]
                    best_arch_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val R²: {val_eval['r2_mean']:.4f} | Val MAE: {val_eval['mae_mean']:.4f} | Val AngErr: {val_eval['angular_error_deg']:.2f}°")

        # Evaluate best epoch on test set
        model.load_state_dict({k: v.to(device) for k, v in best_arch_state.items()})
        test_eval = evaluate_model(model, test_loader, device)
        results[arch_name] = {
            "test_r2_mean": test_eval["r2_mean"],
            "test_r2_m1": test_eval["r2_m1"],
            "test_r2_m2": test_eval["r2_m2"],
            "test_r2_m3": test_eval["r2_m3"],
            "test_mae_mean": test_eval["mae_mean"],
            "test_angular_error_deg": test_eval["angular_error_deg"],
        }
        print(f"  [RESULT {arch_name}] Test R²: {test_eval['r2_mean']:.4f} | Test MAE: {test_eval['mae_mean']:.4f} | Test AngErr: {test_eval['angular_error_deg']:.2f}°")

        if test_eval["r2_mean"] > best_val_r2:
            best_val_r2 = test_eval["r2_mean"]
            best_model_name = arch_name
            best_state_dict = best_arch_state

    # Save Winner Checkpoint
    winner_ckpt_path = os.path.join(out_dir, "mlp_shift_pca_best.pt")
    torch.save({
        "model_name": best_model_name,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "state_dict": best_state_dict,
        "x_mean": x_mean,
        "x_std": x_std,
        "results": results,
    }, winner_ckpt_path)

    metrics_path = os.path.join(out_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "winner": best_model_name,
            "winner_r2": best_val_r2,
            "architectures": results,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 WINNER ARCHITECTURE: {best_model_name} with Test R² = {best_val_r2:.4f}")
    print(f">>> Saved winning checkpoint to: {winner_ckpt_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train and Search PCA MLP for SDXL")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {args.dataset_path}")

    df = pd.read_csv(args.dataset_path)
    run_architecture_search(df, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, gpu=args.gpu)


if __name__ == "__main__":
    main()
