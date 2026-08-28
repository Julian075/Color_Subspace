"""
DOWNLOAD_ALL_MODELS.PY -- Pre-downloads SAM3 and FLUX.1-dev into shared cache.
"""

import os
import sys
import time

os.environ["HF_HOME"] = "/leonardo_work/AIFAC_S07_004/jsantamaria/.cache/huggingface"
cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
os.makedirs(cache_dir, exist_ok=True)

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")

print("=" * 70)
print(f"Downloading models to shared cache: {cache_dir}")
print("=" * 70)

# 1. Download facebook/sam3
print("\n[1/2] Downloading facebook/sam3...")
t0 = time.time()
sam3_path = snapshot_download(
    repo_id="facebook/sam3",
    cache_dir=cache_dir,
    token=token,
    resume_download=True
)
print(f"SAM3 downloaded in {time.time() - t0:.1f}s -> {sam3_path}")

# 2. Download black-forest-labs/FLUX.1-dev
print("\n[2/2] Downloading black-forest-labs/FLUX.1-dev...")
t0 = time.time()
flux_path = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    cache_dir=cache_dir,
    token=token,
    resume_download=True
)
print(f"FLUX.1-dev downloaded in {time.time() - t0:.1f}s -> {flux_path}")

print("\n" + "=" * 70)
print("All models successfully cached on shared storage!")
print("=" * 70)
