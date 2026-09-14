"""
TRAIN_AND_SEARCH_MLP_PCA.PY -- Model Architecture Search & ResMLP_256 Training for PixArt.

Performs:
  1. 80/20 train/validation split with fixed seed.
  2. 15D perceptual featurization:
     [L_b, a_b, b_b, L_t, a_t, b_t, dL, da, db, C_b, C_t, sin_hb, cos_hb, sin_ht, cos_ht]
  3. Trains and benchmarks candidate models:
     - Linear Baseline: Linear(15 -> 3)
     - Shallow MLP: 15 -> 128 -> 3
     - Medium MLP: 15 -> 256 -> 128 -> 3
     - Deep MLP: 15 -> 256 -> 256 -> 128 -> 3
     - ResMLP_256 (Selected standard): 15 -> 256 -> 2x ResBlock(256) -> 3
  4. Multi-metric evaluation:
     - R^2 score (Overall, m1, m2, m3)
     - Mean Absolute Error (MAE)
     - Predicted vs Ground Truth Chroma error
     - Predicted vs Ground Truth Hue angle error (degrees)
  5. Saves best model checkpoint (mlp_shift_pca_best.pt).
"""

import os
import math
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

from model_pca import ResBlock, ResMLP_256

# Candidate Models for Architecture Search
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


class ColorDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.features = []
        self.targets = []

        for _, row in df.iterrows():
            L_b, a_b, b_b = float(row["base_L"]), float(row["base_a"]), float(row["base_b"])
            L_t, a_t, b_t = float(row["target_L"]), float(row["target_a"]), float(row["target_b"])

            dL = L_t - L_b
            da = a_t - a_b
            db = b_t - b_b

            c_b = math.sqrt(a_b**2 + b_b**2)
            c_t = math.sqrt(a_t**2 + b_t**2)

            h_b = math.atan2(b_b, a_b)
            h_t = math.atan2(b_t, a_t)

            feat = [
                L_b, a_b, b_b,
                L_t, a_t, b_t,
                dL, da, db,
                c_b, c_t,
                math.sin(h_b), math.cos(h_b),
                math.sin(h_t), math.cos(h_t)
            ]
            self.features.append(feat)
            self.targets.append([float(row["m1"]), float(row["m2"]), float(row["m3"])])

        self.X = torch.tensor(self.features, dtype=torch.float32)
        self.Y = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_losses.append(criterion(pred, batch_y).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_model(model, val_loader, device="cuda"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    r2_overall = float(r2_score(targets, preds))
    r2_m1 = float(r2_score(targets[:, 0], preds[:, 0]))
    r2_m2 = float(r2_score(targets[:, 1], preds[:, 1]))
    r2_m3 = float(r2_score(targets[:, 2], preds[:, 2]))
    mae = float(np.mean(np.abs(targets - preds)))

    c_true = np.sqrt(targets[:, 1]**2 + targets[:, 2]**2)
    c_pred = np.sqrt(preds[:, 1]**2 + preds[:, 2]**2)
    chroma_mae = float(np.mean(np.abs(c_true - c_pred)))

    h_true = np.arctan2(targets[:, 2], targets[:, 1])
    h_pred = np.arctan2(preds[:, 2], preds[:, 1])
    angle_diff = np.abs(np.arctan2(np.sin(h_true - h_pred), np.cos(h_true - h_pred)))
    hue_deg_mae = float(np.degrees(np.mean(angle_diff)))

    return {
        "R2_overall": r2_overall,
        "R2_m1": r2_m1, "R2_m2": r2_m2, "R2_m3": r2_m3,
        "MAE": mae,
        "Chroma_MAE": chroma_mae,
        "Hue_Error_deg": hue_deg_mae,
    }


def main():
    parser = argparse.ArgumentParser(description="Model Architecture Search & ResMLP Training for PixArt")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset_mlp_pca.csv")
    parser.add_argument("--out-dir", default="./mlp_training_out")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.dataset_path)
    print(f"Loaded {len(df)} samples from {args.dataset_path}")

    # Train / Val Split
    np.random.seed(42)
    perm = np.random.permutation(len(df))
    split_idx = int(len(df) * (1.0 - args.val_split))
    train_df = df.iloc[perm[:split_idx]]
    val_df = df.iloc[perm[split_idx:]]

    train_loader = DataLoader(ColorDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ColorDataset(val_df), batch_size=args.batch_size, shuffle=False)

    candidates = {
        "Linear": LinearBaseline(15, 3),
        "Shallow_128": ShallowMLP(15, 128, 3),
        "Medium_256": MediumMLP(15, 256, 3),
        "Deep_256": DeepMLP(15, 256, 3),
        "ResMLP_256": ResMLP_256(15, 256, 2, 3),
    }

    results = []
    best_model_name = None
    best_r2 = -float("inf")
    best_state = None

    print("\n--- ARCHITECTURE SEARCH BENCHMARK ---")
    for name, model in candidates.items():
        trained_model, val_loss = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
        metrics = evaluate_model(trained_model, val_loader, device=device)
        metrics["model"] = name
        metrics["val_loss"] = float(val_loss)
        results.append(metrics)

        print(f"[{name:12s}] R2: {metrics['R2_overall']:.4f} (m1:{metrics['R2_m1']:.3f}, m2:{metrics['R2_m2']:.3f}, m3:{metrics['R2_m3']:.3f}) | MAE: {metrics['MAE']:.4f} | Hue Err: {metrics['Hue_Error_deg']:.2f} deg")

        if metrics["R2_overall"] > best_r2:
            best_r2 = metrics["R2_overall"]
            best_model_name = name
            best_state = {k: v.cpu().clone() for k, v in trained_model.state_dict().items()}

    # Save benchmark table
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="R2_overall", ascending=False, inplace=True)
    benchmark_csv = os.path.join(args.out_dir, "architecture_search_results.csv")
    results_df.to_csv(benchmark_csv, index=False)

    # Save best checkpoint
    ckpt_path = os.path.join(args.out_dir, "mlp_shift_pca_best.pt")
    torch.save({
        "model_name": best_model_name,
        "model_state_dict": best_state,
        "metrics": results_df.iloc[0].to_dict(),
    }, ckpt_path)
    print(f"\n>>> Best model ({best_model_name}) saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
