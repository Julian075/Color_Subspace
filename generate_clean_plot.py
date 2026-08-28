"""
GENERATE_CLEAN_PLOT.PY -- Standalone visualization for Phase A PCA results.
Generates clean diagnostic plots without arbitrary axis-matching labels.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/fase_a_pca_out"
RESULTS_JSON = os.path.join(OUT_DIR, "fase_a_pca_results.json")

with open(RESULTS_JSON) as f:
    results = json.load(f)

pca_metrics = results["pca_metrics"]
evr = np.array(pca_metrics["explained_variance_ratio"]) * 100
cum_evr = np.array(pca_metrics["cumulative_variance_ratio"]) * 100
alignment = np.array(pca_metrics["alignment_matrix_rows_PC_cols_Lab"])
components = [f"PC1\n({evr[0]:.1f}%)", f"PC2\n({evr[1]:.1f}%)", f"PC3\n({evr[2]:.1f}%)"]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# Plot 1: Scree Plot (Explained Variance)
x = np.arange(len(evr))
bars = axes[0].bar(x, evr, color="#2b5c8f", width=0.45, label="Individual EVR (%)")
line = axes[0].plot(x, cum_evr, color="#d95f02", marker="o", linewidth=2.5, markersize=8, label="Cumulative (%)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(components, fontsize=11, fontweight="bold")
axes[0].set_ylabel("Explained Variance (%)", fontsize=11)
axes[0].set_ylim(0, 110)
axes[0].set_title("1. Explained Variance per Principal Component", fontsize=12, fontweight="bold")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="center right", fontsize=10)
for bar, val in zip(bars, evr):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

# Plot 2: Cosine Alignment Matrix (PC vs Canonical CIELAB)
im = axes[1].imshow(alignment, cmap="coolwarm", vmin=-1.0, vmax=1.0)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"], fontsize=10, fontweight="bold")
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(["PC1", "PC2", "PC3"], fontsize=11, fontweight="bold")
axes[1].set_title("2. Directional Alignment (Cosine Similarity)", fontsize=12, fontweight="bold")
cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label("cos(θ) alignment", fontsize=10)

for i in range(3):
    for j in range(3):
        val = alignment[i, j]
        color = "white" if abs(val) > 0.55 else "black"
        axes[1].text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

# Plot 3: 16D Latent Loadings (Linear Combination Weights of all 16 Channels)
channels = np.arange(16)
width = 0.27
pcs_dict = pca_metrics["principal_components"]
pc_keys = list(pcs_dict.keys())
c_u1 = np.array(pcs_dict[pc_keys[0]]["loading_vector_16d"])
c_u2 = np.array(pcs_dict[pc_keys[1]]["loading_vector_16d"])
c_u3 = np.array(pcs_dict[pc_keys[2]]["loading_vector_16d"])

axes[2].bar(channels - width, c_u1, width=width, label=f"PC1 ({evr[0]:.1f}%)", color="#1b9e77")
axes[2].bar(channels, c_u2, width=width, label=f"PC2 ({evr[1]:.1f}%)", color="#d95f02")
axes[2].bar(channels + width, c_u3, width=width, label=f"PC3 ({evr[2]:.1f}%)", color="#7570b3")

axes[2].set_xticks(channels)
axes[2].set_xlabel("FLUX VAE Latent Channel Index (0 – 15)", fontsize=11)
axes[2].set_ylabel("Linear Loading Weight (u_k,c)", fontsize=11)
axes[2].set_title("3. 16D Latent Loadings (Linear Combinations)", fontsize=12, fontweight="bold")
axes[2].axhline(0, color="gray", linewidth=0.8)
axes[2].grid(axis="y", linestyle="--", alpha=0.5)
axes[2].legend(loc="upper right", fontsize=9)

fig.tight_layout()
plot_path = os.path.join(OUT_DIR, "fase_a_pca_alignment_plot.png")
fig.savefig(plot_path, dpi=180)
plt.close(fig)
print(f"Generated clean plot: {plot_path}")
