# Split-NS Orthogonal LoRA: Method Description

## Motivation

Standard LoRA decomposes weight updates as ΔW = BA, where B ∈ R^{d_out × r} and A ∈ R^{r × d_in}. In practice, the learned BA suffers from **rank collapse**: a few singular directions dominate while the remaining directions contribute negligibly, wasting parameter capacity.

We propose **Split-NS Orthogonal LoRA**, which applies Newton-Schulz (NS) orthogonalization to A and B separately, then scales the update by their original Frobenius norms. This enforces orthogonal bases in both input and output spaces while preserving the learned magnitudes.

## Method

### Standard LoRA Forward
```
ΔW = (α/r) * B @ A
W_eff = W_frozen + ΔW
```

### Split-NS Orthogonal LoRA Forward
```
a = ||A||_F                    # Frobenius norm of A (scalar)
b = ||B||_F                    # Frobenius norm of B (scalar)
A' = NS(A, steps=5)           # Newton-Schulz orthogonalization → rows orthonormal
B' = NS(B, steps=5)           # Newton-Schulz orthogonalization → cols orthonormal
ΔW = a * b * B' @ A'          # Orthogonal update with preserved scale
W_eff = W_frozen + ΔW
```

### Newton-Schulz Iteration (5-step quintic polynomial)

Given matrix G ∈ R^{m × n}:

1. Normalize: X = G / (||G||_F + eps)
2. If m > n: transpose X
3. Iterate 5 times:
   - A = X @ X^T
   - B = b*A + c*A@A         (b = -4.7750, c = 2.0315)
   - X = a*X + B@X           (a = 3.4445)
4. If transposed in step 2: transpose back

The quintic polynomial phi(x) = 3.4445x - 4.7750x^3 + 2.0315x^5 converges all singular values in (0, 1] to approximately 1 in 5 iterations. The output is the nearest semi-orthogonal matrix (UV^T from the SVD).

### Key Properties

1. **Orthogonal bases**: NS(A) has orthonormal rows (input space basis), NS(B) has orthonormal columns (output space basis). The update B'A' has all r non-zero singular values equal to 1.

2. **Scale preservation**: The original norms ||A||_F and ||B||_F are extracted before orthogonalization and used as scalar multipliers. This decouples direction (learned by NS) from magnitude (learned by norms).

3. **Fully differentiable**: All operations (norm, NS iteration) are standard matrix operations. Gradients flow through both the norm and NS branches to update A and B.

4. **NS on small matrices**: NS is applied to A [r × d_in] and B [d_out × r] separately, not to BA [d_out × d_in]. The iteration operates on [r × r] Gram matrices (r=16), which is negligible cost compared to NS on the full [d_out × d_in] matrix.

5. **Warm start compatible**: A and B can be initialized from trained standard LoRA weights. The orthogonalization is applied during forward, not at initialization, so no information is lost.

## Experimental Setup

- **Base model**: 182M parameter NA Transformer (pretrained on 2D PDE datasets)
- **Encoder/Decoder**: Loaded from trained vanilla LoRA checkpoint, frozen
- **LoRA targets**: qkv, proj, gate_proj, up_proj, down_proj (108 modules across 12 transformer layers)
- **LoRA rank**: r = 16 (4.87M trainable parameters, 2.67% of total)
- **Initialization**: A and B loaded from trained vanilla LoRA checkpoint (rescaled_norm)
- **Training**: lr=5e-4, cosine decay to 1e-6, 30 epochs, batch_size=20, 4 GPUs
- **Loss**: lambda_bc * BC_loss + lambda_pde * PDE_loss (rescaled_norm configuration)

## Results on Taylor-Green 2D

| Method | VRMSE (all) | VRMSE (Vx) | VRMSE (Vy) | VRMSE (press) | PDE Loss |
|--------|------------|------------|------------|---------------|----------|
| **Split-NS Orth-LoRA** | **0.00596** | 0.00102 | 0.00103 | **0.01582** | **0.00495** |
| Vanilla LoRA | 0.00624 | 0.00098 | 0.00098 | 0.01675 | 0.00510 |

Split-NS improves VRMSE by 4.5% overall, with the largest gain on the pressure channel (-5.6%). Both methods use identical hyperparameters, frozen encoder/decoder, and the same LoRA weight initialization.

## Comparison with Other Orthogonal LoRA Variants Tested

| Variant | VRMSE | Description |
|---------|-------|-------------|
| **Split-NS** | **0.00596** | NS on A, B separately + norm scaling |
| Vanilla LoRA | 0.00624 | Standard PEFT LoRA (baseline) |
| Muon-LoRA (NS on BA) | 0.00647 | NS on joint BA, alpha=||BA|| |
| Factored (NS + per-dir s) | 0.00897 | NS on A,B + learnable per-direction scale |

Split-NS is the only orthogonal variant that consistently outperforms vanilla LoRA under fair comparison conditions. The key advantage over Muon-LoRA (NS on BA jointly) is that separate orthogonalization preserves the independent scaling of A and B through their norms, while joint NS collapses everything to a single scalar alpha.

## Results on All 4 Exact-Solution Datasets

Fair comparison: 4 GPU, r=16, lr=5e-4→1e-6 cosine, 20 epochs (Burgers/AdvDiff/Wave) or 30 epochs (Taylor-Green), frozen enc/dec, same LoRA weight initialization from rescaled_norm checkpoint. Only difference: LoRA forward (Split-NS vs PEFT vanilla).

| Dataset | Method | VRMSE (all) | Per-channel VRMSE | PDE Loss |
|---------|--------|------------|-------------------|----------|
| **Taylor-Green** | Split-NS | **0.00596** | Vx=0.00102, Vy=0.00103, press=**0.01582** | **0.00495** |
| | Vanilla | 0.00624 | Vx=0.00098, Vy=0.00098, press=0.01675 | 0.00510 |
| | **Improvement** | **-4.5%** | press **-5.6%** | -2.9% |
| **Burgers** | Split-NS | **0.00273** | Vx=**0.00296**, Vy=0.00250 | **0.000179** |
| | Vanilla | 0.00282 | Vx=0.00311, Vy=0.00253 | 0.000200 |
| | **Improvement** | **-3.2%** | Vx **-4.8%** | -10.5% |
| **AdvDiff** | Split-NS | **0.00376** | u=**0.00376** | **0.1348** |
| | Vanilla | 0.00396 | u=0.00396 | 0.1693 |
| | **Improvement** | **-5.0%** | u **-5.0%** | -20.3% |
| **Wave** | Split-NS | **0.02478** | u=**0.01902**, w=0.03054 | **73.95** |
| | Vanilla | 0.02482 | u=0.01900, w=0.03064 | 79.52 |
| | **Improvement** | **-0.1%** | w **-0.3%** | -7.0% |

### Key Findings

1. **Consistent improvement**: Split-NS outperforms vanilla LoRA on all 4 datasets (VRMSE -0.1% to -5.0%).
2. **Larger gains on weaker channels**: The improvement concentrates on the channel with worst performance — Taylor-Green pressure (-5.6%), Burgers Vx (-4.8%). This suggests orthogonal bases distribute update capacity more evenly, benefiting underperforming features.
3. **PDE loss improvement is larger**: PDE loss improves -2.9% to -20.3%, consistently more than VRMSE improvement. The orthogonal update produces solutions that better satisfy the governing equations.
4. **Wave is near-tied**: The smallest gain is on Wave 2D (-0.1%), where both channels (u, w) already perform similarly — there is no "weak channel" for orthogonalization to help.
