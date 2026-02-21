# 3D PDE Foundation Models: Literature Review & Recommendations

## 1. Paper-by-Paper Analysis

---

### 1.1 MORPH (LANL, 2025) -- **Most Directly Relevant**

**Paper**: [MORPH: PDE Foundation Models with Arbitrary Data Modality](https://arxiv.org/abs/2509.21670)

**Why it matters**: MORPH is the only paper that jointly trains 1D+2D+3D in a single model -- exactly the same goal as your architecture.

**Architecture**:
- **Unified tensor format (UPTF-7)**: Shape `(N, T, F, C, D, H, W)` where unused dims = 1
  - 1D: `(B, T, F, C, 1, 1, W)`
  - 2D: `(B, T, F, C, 1, H, W)`
  - 3D: `(B, T, F, C, D, H, W)`
- **Patch size = 8** uniformly across all dimensions
  - 1D: `1x1x8`, 2D: `1x8x8`, 3D: `8x8x8`
  - Max sequence length capped at **4096 patches**
- **4D Axial Attention** (key innovation for 3D efficiency):
  - Instead of full attention `O((TDHW)^2)`, factorizes into 4 separate 1D attentions
  - Complexity: `O(TDHW * (T + D + H + W))` -- **massive** reduction
  - Each axis attention is summed residually: `y = x + x_t + x_d + x_h + x_w`
- **Component-wise convolutions**: 3D convolutions on channel dim (8 learnable filters)
- **Inter-field cross-attention**: Fuses multiple physical fields (e.g., velocity, pressure, density)

**Normalization**: Reversible Instance Normalization (ReVIN) -- per-dataset mean/std cached and reused

**Loss**: MSE (AR(1) autoregressive, predict `x_{t+1}` from `x_t`)

**3D datasets trained on**:
- 3D-MHD: 64^3, 97 trajectories, 101 timesteps
- 3D-CFD (compressible NS): **128^3, 200 trajectories, 21 timesteps** -- same as yours!
- 3D-CFD-Turb: 64^3, 500 trajectories (fine-tuning)

**Key 3D tricks**:
- Balanced task sampling: 3D-MHD sampled at ~31% despite fewer examples
- Two-node training: 8xH100 per node (80GB each) for sufficient memory
- Patch size 8 is practical maximum for 3D (larger = too few tokens; smaller = too many)

**Model sizes**: MORPH-Ti (7M), MORPH-S (30M), MORPH-M (126M), MORPH-L (480M)

---

### 1.2 MPP (PolymathicAI, NeurIPS 2024)

**Paper**: [Multiple Physics Pretraining for Spatiotemporal Surrogate Models](https://arxiv.org/abs/2310.02994)

**Architecture**:
- **Axial ViT (AViT)**: Sequential attention over each spatial+temporal axis
  - Dense attention: `O((HWT)^2)` vs. Axial: `O(H^2 + W^2 + T^2)`
  - Spatial K, Q, V projections **shared between height and width axes**
- **Shared embedding via 1x1 convolution**: Each field (u, v, p) gets its own embedding vector, combined via inner product: `e(x,t) = u*e_u + v*e_v + p*e_p`
- **RevIN normalization**: Per-sample mean/std per channel

**Loss**: Normalized MSE (NMSE): `||M(U_t) - u_{t+1}||^2 / (||u_{t+1}||^2 + eps)` -- prevents high-magnitude systems from dominating

**3D handling**: Not trained on 3D directly. Uses "inflation" for transfer:
- 2D kernels replicated along new axis, rescaled by `1/P`
- New velocity component initialized as average of previous two projections
- Result: 11.7% improvement over random initialization on 3D finetuning

**Key insight**: **Pretrain on 2D, inflate to 3D** is a viable strategy.

---

### 1.3 DPOT (THU, ICML 2024)

**Paper**: [DPOT: Auto-Regressive Denoising Operator Transformer](https://arxiv.org/abs/2403.03542)

**Architecture**:
- **Fourier attention**: `z^(l+1)(x) = F^{-1}[W_2 * sigma(W_1 * F[z^l] + b_1) + b_2]`
  - All computation in frequency domain, linear complexity
  - Weight-sharing MLPs across all frequency modes
- **Denoising pretraining**: Inject Gaussian noise `eps ~ N(0, epsilon * ||u||)` into inputs
  - Predicts clean next timestep from noisy current step
  - Addresses distribution shift between single-step training and multi-step rollout
- **Group normalization** after each Fourier attention layer

**3D handling**: Does NOT pretrain on 3D. But transfer works:
- Replace 2D FFT with 3D FFT during finetuning
- Reduces 3D error from 41% to 22.6% via transfer from 2D pretraining

**Loss**: L2 relative error (L2RE)

**Resolution**: Fixed at 128x128 during pretraining, with interpolation for other resolutions

---

### 1.4 BCAT (UCLA, 2025)

**Paper**: [BCAT: A Block Causal Transformer for PDE Foundation Models](https://arxiv.org/abs/2501.18972)

**Architecture**:
- **Block causal attention mask**: All patches within a frame can attend to each other, but cannot see future frames. Frame-level prediction, not token-level.
- **Patch size = 8**, resolution 128x128 -> 16x16 = 256 spatial tokens per frame x 20 frames = 5120 max seq len
- **RMS norm** + query-key normalization (not LayerNorm)
- **SwiGLU activation**, 1024 hidden dim, 12 layers, 8 heads

**Key insight**: Next-frame prediction achieves **2.9x better accuracy** than next-token prediction. Block causal attention accumulates errors O(T) instead of O(T*N).

**Normalization**: Per-trajectory mean/std normalization
**Loss**: MSE between normalized predicted and ground truth
**3D**: Not supported (2D only)

---

### 1.5 PDE-Transformer (TUM, ICML 2025)

**Paper**: [PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](https://arxiv.org/abs/2505.24717)

**Architecture**:
- **Separate Channel (SC) embedding**: Each physical channel embedded independently as its own token, then interact via channel-wise axial attention
  - 2.7x-4.4x better generalization than mixed-channel embedding
- **Multi-scale U-shaped design**: PixelShuffle/PixelUnshuffle for down/upsampling with skip connections
- **Shifted window attention** (Swin-style) with log-spaced relative positions
- **RMSNorm** on Q, K in attention for stability
- **adaLN-Zero conditioning** for PDE type

**Loss**: MSE (deterministic) or flow matching (generative)
**3D**: Explicitly stated as **not supported** yet. "Currently limited to 2D regular grids."

**Key insight**: Separate-channel embedding is crucial for multi-physics generalization.

---

### 1.6 EddyFormer (2025) -- **Best 3D-specific architecture**

**Paper**: [EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](https://arxiv.org/abs/2510.24173)

**Architecture**:
- **Spectral-element method (SEM) tokenization**: Domain partitioned into `8^3 = 512` spectral elements, each with `13^3` modes
  - **Only 512 tokens** for 256^3 resolution (vs. 16.7M grid points!)
  - Each token contains rich spectral information
- **Two-stream design**:
  - SGS (subgrid-scale) stream: Local convolutions on full field
  - LES (large-eddy simulation) stream: Attention on spectrally-filtered field (`k_max=5`)
  - Only LES feeds into SGS (reflecting energy cascade physics)
- **SEMAttn**: Self-attention on local coordinates across elements
- **SEMConv**: Axial factorized spectral convolutions with compact support
- **Rotary position encoding** for domain generalization

**Loss**: Relative L2 error
**Performance**: 30% lower error than FNO/FactFormer on 3D turbulence at 256^3

**Key insights**:
- Separating large-scale (attention) from small-scale (convolution) aligns with physics
- Layer normalization only in attention blocks -- hurts performance elsewhere
- Initialization of convolution kernels at 1e-7 magnitude

---

### 1.7 PEST (2026) -- **Best 3D loss function design**

**Paper**: [PEST: Physics-Enhanced Swin Transformer for 3D Turbulence Simulation](https://arxiv.org/abs/2602.10150)

**Architecture**:
- 3D Swin Transformer with window size M^3
- Depths [2, 8, 2], hidden dim 512
- 3D convolutional patch embedding with stride P
- Temporal attention + learnable query decoder

**Loss function (multi-component)**:
1. **Frequency-adaptive spectral loss**: Partitions Fourier space into 3 bands (low/mid/high), learnable band weights via curriculum strategy
   ```
   L_data = sum_b w_b * (1/|S_b|) * sum_k |F(pred)_k - F(target)_k|^2
   ```
2. **Navier-Stokes residual constraint**: `||du/dt + (u*nabla)u + nabla_p - nu*nabla^2 u||^2`
3. **Divergence-free regularization**: `||nabla . u_hat - nabla . u||^2` (residual form, not strict zero)
4. **Uncertainty-based adaptive weighting**: `L_total = sum_i [(1/2) * exp(-s_i) * L_i + (1/2) * s_i]`
   - Automatically balances data loss, physics loss, and spectral loss
   - Warmup: first train only data+gradient loss, then add physics constraints

**Performance**: 17% lower RMSE than second-best on JHU isotropic turbulence (128^3)

**Key insights**:
- Standard L2 loss biases toward high-energy large scales
- Frequency-adaptive weighting via Parseval's theorem preserves small-scale structures
- Physics constraints provide sustained regularization during long autoregressive rollouts
- Gradient matching loss addresses window boundary artifacts

---

### 1.8 FactFormer (NeurIPS 2023)

**Paper**: [Scalable Transformer for PDE Surrogate Modeling](https://arxiv.org/abs/2305.17560)

**Architecture**:
- **Axial factorized kernel integral**: Decomposes input into 1D sub-functions via learnable projections
- Per-axis attention: Quadratic only along that axis, does not grow with number of dimensions
- Tested on 3D smoke buoyancy at **64^3** resolution

**Key insight**: Factorized attention naturally handles dimensionality curse -- per-axis cost is independent of other dimensions.

---

### 1.9 UPT (JKU, NeurIPS 2024)

**Paper**: [Universal Physics Transformers](https://arxiv.org/abs/2402.12365)

**Architecture**:
- **Hierarchical encoder**: Input points -> supernodes (message passing) -> latent tokens (perceiver pooling)
  - 50K input points -> 2048 supernodes -> 512 latent tokens (60x compression)
- **Approximator**: Standard transformer on fixed-size latent space (independent of input size)
- **Decoder**: Perceiver cross-attention to query at arbitrary points

**Loss**: Inverse encoding + inverse decoding + physics loss

**3D results**: ShapeNet-Car at 3.6K mesh points, pipe flows at 29K-59K points
- 225x faster than GINO while comparable accuracy
- Scales to 4.2M input points

**Key insight**: Compress to fixed-size latent space makes 3D tractable. Approximator never sees spatial resolution.

---

### 1.10 PDE-FM (2025)

**Paper**: [Towards a Foundation Model for PDEs Across Physics Domains](https://arxiv.org/abs/2511.21861)

**Architecture**:
- **Spatial-spectral tokenization**: Spatial tokens via PatchConv + spectral tokens via FFT (lowest m modes)
- **Mamba backbone**: Linear complexity `O(N_p * d)` vs. transformer's `O(N_p^2)`
- **FiLM conditioning**: Physics metadata (BCs, constitutive params)

**Loss**: Spatial L2 + spectral L2 with frequency-dependent weighting

**3D datasets**: Supernova (64^3), neutron star merger (192x128x66)
**Performance**: 46% VRMSE reduction over baselines on The Well benchmark

---

### 1.11 Flow Marching (2025)

**Paper**: [Flow Marching for a Generative PDE Foundation Model](https://arxiv.org/abs/2509.18611)

**Architecture**:
- **P2VAE**: Compress 128x128 -> 16x16 (12x spatial compression), 16 latent channels
- **Flow Marching Transformer (FMT)**: SiT backbone, RMSNorm, SwiGLU, FlashAttention v2
- **Latent temporal pyramids**: Downsample history (8x, 4x, 2x, 1x) -- 15x efficiency gain

**3D**: Not supported (2D only, standardized to 128x128)

**Key insight**: VAE compression + latent-space dynamics is highly efficient. History downsampling exploits Markov property.

---

### 1.12 PreLowD (2024)

**Paper**: [Pretraining a Neural Operator in Lower Dimensions](https://arxiv.org/abs/2407.17616)

**Key finding**: 1D pretrained FNO can transfer to 2D via parameter replication when using factorized architectures (same Fourier modes per axis). Up to 50% error reduction in low-data regimes.

**Limitation**: Only tested 1D->2D (not 3D). Only works for advection/diffusion-like PDEs.

---

## 2. The Well Benchmark (2024)

**Paper**: [The Well: A Large-Scale Collection of Diverse Physics Simulations](https://arxiv.org/abs/2412.00568)

3D datasets and resolutions available:
- MHD: 64^3, 256^3
- Rayleigh-Taylor instability: 128^3
- Supernova explosion: 64^3, 128^3
- Turbulence gravity cooling: 64^3
- Turbulent radiative layer 3D: 128x128x256

**Metric**: VRMSE (Variance-scaled RMSE) -- normalizes by field variance so that "predicting the mean = score of 1". Better than NRMSE for non-negative fields (pressure, density).

---

## 3. Synthesis: What Makes 3D Hard and How Papers Solve It

### 3.1 The Token Count Problem

| Method | 128^3 Resolution | Token Count | Approach |
|--------|------------------|-------------|----------|
| Your current | patch_size=8 | 8^3 * 9 = 4608 | NA4D |
| MORPH | patch_size=8 | 16^3 * T (capped 4096) | Axial attention |
| EddyFormer | 8^3 elements | 512 | SEM tokenization |
| FactFormer | - | 64^3 tokens | Axial factorized |
| UPT | perceiver | 512 fixed | Latent compression |
| Flow Marching | VAE | 16^2 * 16ch | Latent space |

**Your 4608 tokens with NA4D is reasonable**, but the kernel size matters enormously. With `base_kernel=5` in 4D (time + 3 spatial), each token attends to `5^4 = 625` neighbors. This is already heavy.

### 3.2 The Computational Bottleneck Analysis

For your architecture specifically:
- 3D encoder: `B * T * n_d * n_h * n_w` = `1 * 9 * 16 * 16 * 16 = 36864` Conv3d forward passes through CNN stem
- Each pass processes an `8^3` volume through Conv3d layers
- Then `36864` intra-patch attention operations
- Then `36864` attention pooling operations
- Then 12 transformer layers with NA4D on `9 * 16 * 16 * 16 = 36864` reshaped tokens

This is **extremely** memory and compute intensive compared to 2D.

### 3.3 Why Your 3D Loss Doesn't Converge

Based on the literature, the likely causes are:

**Problem 1: Scale mismatch between 1D/2D/3D losses**
- RMSE in normalized space averages over spatial dimensions
- 3D has 128^3 = 2M spatial points vs. 2D's ~16K
- The loss landscape is much more complex; optimizer needs more steps
- **MPP's NMSE** (dividing by target norm) explicitly prevents this

**Problem 2: Instance normalization may be too aggressive for 3D**
- You normalize over ALL spatiotemporal dimensions: `(T, D, H, W)` for 3D
- 3D compressible NS has large density/pressure gradients (shocks)
- Normalizing globally can wash out important local structure
- **MORPH uses per-dataset cached statistics** instead of per-sample
- **PEST shows** that standard normalization biases toward large-scale features

**Problem 3: Spectral bias in L2 loss**
- Standard MSE/RMSE loss is dominated by large-scale (low-frequency) features
- 3D turbulence has a Kolmogorov cascade with important small-scale structure
- The model learns to predict smooth fields (low-frequency) and ignores fine details
- **PEST's frequency-adaptive spectral loss** explicitly addresses this

**Problem 4: Patch size 8 for 3D may lose too much local information**
- 8^3 = 512 voxels collapsed to a single token via CNN + attention pool
- For compressible NS with shocks, this is aggressive compression
- **EddyFormer keeps spectral information** within each element (13^3 modes per token)

**Problem 5: Learning rate and training dynamics**
- Your config shows `lr=1e-5` with init from 2D checkpoint
- 3D parameters (encoder_3d, decoder_3d, na_3d) are randomly initialized
- They need much higher LR to catch up with pretrained 2D components
- **Poseidon uses differentiated learning rates**: `eta_new >> eta_pretrained`

---

## 4. Concrete Recommendations

### Priority 1: Loss Function Reform (High Impact, Easy)

**A. Switch from RMSE to NMSE (Normalized MSE)**
```python
# Current: RMSE in normalized space
rmse = sqrt(mean((pred - target)^2))

# Better: NMSE (from MPP)
nmse = mean((pred - target)^2) / (mean(target^2) + eps)
```
This prevents high-magnitude 3D data from dominating and provides scale-invariant optimization.

**B. Add frequency-adaptive spectral loss (from PEST)**
```python
def spectral_loss(pred, target, bands=3):
    pred_fft = torch.fft.rfftn(pred, dim=spatial_dims)
    target_fft = torch.fft.rfftn(target, dim=spatial_dims)
    # Partition into low/mid/high frequency bands
    # Learnable weights per band (or fixed: e.g., [1.0, 2.0, 4.0])
    loss = sum(w_b * mse(pred_fft[band_b], target_fft[band_b]) for b in bands)
    return loss
```
This forces the model to learn small-scale structure, not just smooth fields.

**C. Consider VRMSE from The Well**
Normalize by field variance instead of field norm -- better for non-negative fields like density/pressure.

### Priority 2: Normalization Strategy (High Impact, Medium)

**A. Switch from per-sample to per-dataset normalization (RevIN-style)**
```python
# Current: per-sample
mean = x.mean(dim=spatial_dims, keepdim=True)  # varies per sample
std = x.std(dim=spatial_dims, keepdim=True)

# Better: per-dataset cached statistics (from MORPH)
# Precompute dataset-level mean/std, cache them
# This is more stable for 3D with shocks/discontinuities
```

**B. Per-channel normalization, not global**
3D compressible NS has channels with very different scales:
- Density: O(1)
- Velocity: O(0.1-1)
- Pressure: O(100-1000)

Normalize **each channel independently**, not all together.

### Priority 3: Attention Mechanism (High Impact, Hard)

**A. Replace NA4D with Axial Attention for 3D (from MORPH)**
Your current NA4D with kernel=5 costs `O(N * 5^4) = O(625N)` per layer.

Axial attention costs `O(N * (T + D + H + W))`:
- For `T=9, D=H=W=16`: cost = `O(N * 57)` -- about 11x cheaper!

Implementation: Apply separate 1D attention along each axis, sum residually.
```python
y = x + attn_t(x) + attn_d(x) + attn_h(x) + attn_w(x)
```

**B. Consider a smaller kernel for NA4D if keeping it**
Reduce from `base_kernel=5` to `base_kernel=3`:
- `3^4 = 81` neighbors vs `5^4 = 625` -- 7.7x cheaper
- NA is already local; 3 should be sufficient for most dynamics

### Priority 4: 3D-Specific Training Strategy (Medium Impact, Easy)

**A. Differentiated learning rates**
```python
# Higher LR for new 3D components
param_groups = [
    {'params': pretrained_params, 'lr': 1e-5},     # pretrained 2D
    {'params': new_3d_params, 'lr': 1e-4},          # 3D encoder/decoder/NA
    {'params': shared_ffn_params, 'lr': 3e-5},      # shared FFN (moderate)
]
```

**B. 3D-focused training schedule**
- Phase 1: Train 3D encoder/decoder/NA with frozen 2D+FFN (10-20 epochs)
- Phase 2: Unfreeze all, lower LR

**C. Increase 3D sampling weight**
You already have `3d_cfd: clips_per_epoch: 360`, but consider:
- Increase gradient accumulation for 3D batches
- Use loss weighting (e.g., 2-5x for 3D loss)

### Priority 5: Architecture Improvements (Medium-High Impact, Hard)

**A. Larger patch size for 3D (reduce token count)**
Consider `patch_size_3d=16` instead of 8:
- 128/16 = 8 patches per axis -> `8^3 * 9 = 4608` tokens (same)
- Wait, you already get `16^3 * 9 = 36864` tokens? Let me recalculate...

Actually: with `patch_size_3d=8`, `128/8=16` patches per axis, so `16^3 * 9 = 36864` total tokens. That's extremely high for NA4D!

With `patch_size_3d=16`: `8^3 * 9 = 4608` tokens -- much more manageable.
But the CNN stem would need to handle 16^3 patches, requiring one more stage.

**B. Consider latent-space approach (from UPT/Flow Marching)**
- Compress 3D field with a lightweight 3D VAE to much smaller latent
- Run transformer in latent space
- Decode back to full resolution
- This is the most scalable approach for 3D

**C. Separate-channel embedding (from PDE-Transformer)**
Instead of mixing all channels into one token, embed each channel separately.
This naturally handles different channel counts across PDEs.

### Priority 6: Denoising Pretraining (Low-Medium Impact, Easy)

From DPOT: Add Gaussian noise to inputs during training.
```python
noise_level = 0.01 * x.norm()
x_noisy = x + noise_level * torch.randn_like(x)
output = model(x_noisy)
loss = criterion(output, target_clean)
```
This improves robustness during autoregressive rollout by exposing model to imperfect inputs.

---

## 5. Summary Comparison Table

| Feature | Your Model | MORPH | MPP | DPOT | PDE-Transformer | PEST |
|---------|-----------|-------|-----|------|-----------------|------|
| 3D support | Yes (NA4D) | Yes (axial) | Via inflation | Via FFT swap | No | Yes (Swin3D) |
| Attention | NA4D kernel=5 | Axial 4D | Axial 3D | Fourier | Shifted window | Swin 3D |
| Normalization | Per-sample inst. | Per-dataset RevIN | Per-sample RevIN | Group norm | RMSNorm + adaLN | Per-traj + adaptive |
| Loss | RMSE (normalized) | MSE | NMSE | L2RE | MSE/Flow matching | Spectral + NS + div |
| Patch size (3D) | 8 | 8 | 16 (2D) | FFT-based | 4 (2D) | Conv stride |
| Max tokens (3D) | 36864 | 4096 | N/A | N/A | N/A | 512 |
| Joint 1D/2D/3D | Yes | Yes | 2D only | 2D only | 2D only | 3D only |

---

## 6. Recommended Implementation Order

1. **Immediate** (days): Fix loss function to NMSE; reduce NA kernel to 3 for 3D; add differentiated LR
2. **Short-term** (1 week): Switch to axial attention for 3D; add spectral loss component
3. **Medium-term** (2-3 weeks): Implement per-dataset normalization (RevIN-style); add denoising training
4. **Long-term** (1 month+): Consider latent-space compression for 3D; separate-channel embedding

---

## 7. Key References

1. [MORPH](https://arxiv.org/abs/2509.21670) - Shape-agnostic 1D/2D/3D PDE foundation model
2. [MPP](https://arxiv.org/abs/2310.02994) - Multiple Physics Pretraining, axial attention
3. [DPOT](https://arxiv.org/abs/2403.03542) - Fourier attention + denoising pretraining
4. [BCAT](https://arxiv.org/abs/2501.18972) - Block causal attention, frame-level prediction
5. [PDE-Transformer](https://arxiv.org/abs/2505.24717) - Separate channel embedding, multi-scale
6. [EddyFormer](https://arxiv.org/abs/2510.24173) - SEM tokenization for 3D turbulence
7. [PEST](https://arxiv.org/abs/2602.10150) - Physics-enhanced Swin for 3D, spectral loss
8. [FactFormer](https://arxiv.org/abs/2305.17560) - Axial factorized attention for 3D
9. [UPT](https://arxiv.org/abs/2402.12365) - Latent compression for scalable 3D
10. [Poseidon](https://arxiv.org/abs/2405.19101) - Multi-scale operator transformer
11. [Flow Marching](https://arxiv.org/abs/2509.18611) - Generative PDE model with VAE compression
12. [PDE-FM](https://arxiv.org/abs/2511.21861) - Mamba backbone + spectral tokenization
13. [PreLowD](https://arxiv.org/abs/2407.17616) - Lower-dimensional pretraining transfer
14. [The Well](https://arxiv.org/abs/2412.00568) - Large-scale 3D PDE benchmark
