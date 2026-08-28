"""
PLOT_MLP_ARCHITECTURE_SEARCH.PY -- Generates publication-ready figures
comparing the evaluated MLP architectures for ColorSpace FLUX PCA shift.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/mlp_training_out/architecture_search_results.csv"
OUT_DIR = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/mlp_training_out"

df = pd.read_csv(CSV_PATH)
# Sort models in logical progression: Linear -> Shallow -> Medium -> Deep -> ResMLP_512 -> ResMLP_256 (Best)
model_order = ["Linear_Baseline", "MLP_Shallow_64", "MLP_Deep_256", "ResMLP_512", "MLP_Medium_128", "ResMLP_256"]
df["order"] = df["model_name"].map(lambda x: model_order.index(x) if x in model_order else 99)
df = df.sort_values("order").reset_index(drop=True)

# Set high-quality styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "font.sans-serif": "DejaVu Sans",
    "figure.autolayout": True
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

models = df["model_name"].tolist()
clean_names = [
    "Linear\nBaseline", "MLP\nShallow (64)", "MLP\nDeep (256)",
    "ResMLP\n(512)", "MLP\nMed (128)", "ResMLP\n(256) ★"
]
colors = ["#7f7f7f", "#3498db", "#9b59b6", "#e67e22", "#2ecc71", "#e74c3c"]

# -------------------------------------------------------------
# 1. Overall R2 Score & Per-Component R2
# -------------------------------------------------------------
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.22

r_overall = df["r2_overall"].values
r_m1 = df["r2_m1"].values
r_m2 = df["r2_m2"].values
r_m3 = df["r2_m3"].values

bars1 = ax1.bar(x - 1.5*width, r_overall, width, label="Overall R²", color="#2c3e50", alpha=0.95)
bars2 = ax1.bar(x - 0.5*width, r_m1, width, label="m1 R² (PC1)", color="#e74c3c", alpha=0.85)
bars3 = ax1.bar(x + 0.5*width, r_m2, width, label="m2 R² (PC2)", color="#27ae60", alpha=0.85)
bars4 = ax1.bar(x + 1.5*width, r_m3, width, label="m3 R² (PC3)", color="#2980b9", alpha=0.85)

ax1.set_title("(A) Regression Accuracy ($R^2$ Score) on Test Split", fontweight="bold")
ax1.set_ylabel("$R^2$ Score (Higher is Better)")
ax1.set_xticks(x)
ax1.set_xticklabels(clean_names)
ax1.set_ylim(0.68, 0.95)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", framealpha=0.9)

# -------------------------------------------------------------
# 2. Angular Error and Hue Error (Degrees)
# -------------------------------------------------------------
ax2 = axes[0, 1]
ang_err = df["angular_error_deg"].values
hue_err = df["hue_error_deg"].values

w2 = 0.35
ax2.bar(x - w2/2, ang_err, w2, label="Directional Error (°)", color="#d35400", alpha=0.85)
ax2.bar(x + w2/2, hue_err, w2, label="Hue Angle Error (°)", color="#8e44ad", alpha=0.85)

ax2.set_title("(B) Angular & Chromatic Direction Error", fontweight="bold")
ax2.set_ylabel("Error in Degrees (Lower is Better)")
ax2.set_xticks(x)
ax2.set_xticklabels(clean_names)
ax2.set_ylim(0, 22)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", framealpha=0.9)

# -------------------------------------------------------------
# 3. Component MAE (m1, m2, m3)
# -------------------------------------------------------------
ax3 = axes[1, 0]
mae_m1 = df["mae_m1"].values
mae_m2 = df["mae_m2"].values
mae_m3 = df["mae_m3"].values

ax3.bar(x - width, mae_m1, width, label="MAE $m_1$", color="#e74c3c", alpha=0.85)
ax3.bar(x, mae_m2, width, label="MAE $m_2$", color="#27ae60", alpha=0.85)
ax3.bar(x + width, mae_m3, width, label="MAE $m_3$", color="#2980b9", alpha=0.85)

ax3.set_title("(C) Mean Absolute Error (MAE) per PCA Latent Axis", fontweight="bold")
ax3.set_ylabel("MAE (Lower is Better)")
ax3.set_xticks(x)
ax3.set_xticklabels(clean_names)
ax3.set_ylim(0, 0.09)
ax3.grid(axis="y", linestyle="--", alpha=0.5)
ax3.legend(loc="upper right", framealpha=0.9)

# -------------------------------------------------------------
# 4. Overall Metric Summary Radar / Scatter Pareto
# -------------------------------------------------------------
ax4 = axes[1, 1]
mag_mae = df["mag_mae"].values
chroma_mae = df["chroma_pca_mae"].values

for i in range(len(models)):
    marker = "*" if "ResMLP_256" in models[i] else "o"
    size = 280 if "ResMLP_256" in models[i] else 140
    ax4.scatter(df["angular_error_deg"].iloc[i], df["r2_overall"].iloc[i], 
                color=colors[i], s=size, marker=marker, label=df["model_name"].iloc[i], zorder=5)

ax4.set_title("(D) Model Pareto Frontier: $R^2$ vs Angular Error", fontweight="bold")
ax4.set_xlabel("Directional Angular Error [°] (Lower is Better)")
ax4.set_ylabel("Overall $R^2$ Score (Higher is Better)")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend(loc="lower left", framealpha=0.9, fontsize=9)

# Add Annotation for Winner
best_row = df[df["model_name"] == "ResMLP_256"].iloc[0]
ax4.annotate("Winning Architecture\n(ResMLP_256: R²=0.895, Err=7.34°)",
             xy=(best_row["angular_error_deg"], best_row["r2_overall"]),
             xytext=(best_row["angular_error_deg"] + 1.2, best_row["r2_overall"] - 0.035),
             arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=7),
             fontweight="bold", color="#c0392b")

fig.suptitle("Architecture Search Benchmark: MLP Latent Shift Models for FLUX PCA Space", fontsize=16, fontweight="bold", y=0.99)

out_plot_path = os.path.join(OUT_DIR, "architecture_search_comparison.png")
plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
print(f"Plot successfully saved to: {out_plot_path}")
