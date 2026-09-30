# Further Color Applications: Semantic Color Transfer & Gamut Reduction

This directory provides the two core, self-contained applications for generation-time color manipulation built on top of the learned latent color subspace (PCA directions + ResMLP predictor) for **FLUX.1-dev**:

1. **[`flux_multizone_color_transfer.py`](flux_multizone_color_transfer.py)**: Multi-Zone Semantic Color Transfer from either design color palettes or photographic/artistic reference images.
2. **[`flux_spatial_gamut_reduction.py`](flux_spatial_gamut_reduction.py)**: Spatially-Adaptive Gamut Reduction / Saturation Control via pixel-wise CIELAB chroma modulation.

---

## 1. Multi-Zone Semantic Color Transfer

Steers the color distribution of the generated scene to match a reference image (design palette swatch card or photographic/artistic reference) across distinct semantic regions.

### Algorithm Summary
- **Palette & Reference Analysis**: Automatically extracts 5 to 6 principal CIELAB colors using $k$-means with CIEDE2000 deduplication, resolving dominant (background/atmosphere), focal (main subject), and accent tones.
- **Trajectory Conditioning**: Injects canonical ISCC-NBS Level 2 proxy color names into the prompt to place the initial trajectory in the target chromatic basin.
- **Semantic Masking**: Decodes $\hat{z}_0$ at gate step $s=14/28$ and segments semantic objects via SAM3 / Grounding DINO.
- **ResMLP Latent Steering**: Predicts independent latent displacements $(m_1, m_2, m_3)_k$ per region and applies a spatially composite perturbation modulated by a linearly decaying schedule.

### CLI Usage
```bash
python flux_multizone_color_transfer.py \
    --prompt "a photo of an elegant vintage coupe car parked beside an architectural glass pavilion at dusk" \
    --ref-img "path/to/reference_image.png" \
    --main-obj "car" \
    --secondary-objs "pavilion" \
    --device "cuda:1" \
    --out-dir "outputs/transfer_example"
```

---

## 2. Spatially Adaptive Gamut Reduction / Saturation Control

Continuously modulates the saturation of generated images toward a narrower target gamut without causing complementary color cancellation.

### Algorithm Summary
- **Pixel-Wise Chroma Scaling**: At gate step $s=14/28$, decodes $\hat{z}_0$ to compute the spatial CIELAB map $(L, a, b)$. For each spatial position $(y, x)$, scales chroma $C^\ast = \sqrt{(a^\ast)^2 + (b^\ast)^2}$ toward neutral gray by a reduction ratio $r \in [0, 1]$:
  $$a_{\text{target}}(y, x) = a(y, x) \cdot (1 - r), \quad b_{\text{target}}(y, x) = b(y, x) \cdot (1 - r)$$
  while preserving lightness $L^\ast(y, x)$.
- **Vectorized Latent Modulation**: Evaluates the ResMLP over the $128 \times 128$ spatial latent grid and applies the perturbation along chromatic axes $(u_1, u_2)$ modulated by the temporal schedule envelope $w(t)$.

### CLI Usage
```bash
python flux_spatial_gamut_reduction.py \
    --prompt "A colorful scarlet macaw parrot perched on a branch, vibrant plumage, jungle background" \
    --reduction 0.40 \
    --device "cuda:1" \
    --out-dir "outputs/gamut_example" \
    --prefix "macaw_red40"
```
