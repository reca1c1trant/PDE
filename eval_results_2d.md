# 2D PDE Benchmark — Complete Evaluation Results

> Evaluated on validation set with 4×A40 GPUs.
> Methods:
> - **Supervised LoRA**: full-field RMSE loss, pretrain + LoRA (upper bound)
> - **Physics-only LoRA (norm)**: BC + PDE loss, eq_scales from GT residual, save by VRMSE
> - **Physics-only LoRA (no_norm)**: BC + PDE loss, eq_scales=1.0, save by 100×VRMSE + PDE
> - **Scratch (full-param)**: random init, full-param, BC + PDE loss (no pretrain)

---

## 1. Master Summary Table

### 2D Exact-Solution Datasets

| Dataset | Method | VRMSE | nRMSE | RMSE | PDE Loss |
|---------|--------|-------|-------|------|----------|
| **Taylor-Green 2D** | Supervised LoRA | **0.0014** | **0.0014** | 0.0005 | 0.0873 |
| | **Orth-LoRA** (frozen enc/dec) | **0.0060** | **0.0060** | 0.0021 | 0.0050 |
| | **rescaled_norm** (λ_pde=6.1e-7) | **0.0070** | **0.0070** | 0.0026 | 0.0056 |
| | Physics-only LoRA (norm) | 0.0294 | 0.0294 | 0.0099 | 0.0076 |
| | Physics-only LoRA (no_norm) | 0.0875 | 0.0875 | 0.0299 | 2.0116 |
| | Zero-shot (no finetune) | 0.1274 | 0.0305 | 0.0586 | 14.41 |
| | Muon-LoRA (norm) | 0.1220 | 0.1220 | 0.0420 | 2.7742 |
| | Scratch (full-param) | 0.3920 | 0.0919 | 0.1398 | — |
| **Wave 2D** | Supervised LoRA | **0.0097** | **0.0097** | 0.0247 | 73881 |
| | **Orth-LoRA** (frozen enc/dec) | **0.0218** | **0.0218** | 0.0118 | 171.4 |
| | **rescaled_norm** (λ_pde=3.2e-3) | **0.0275** | **0.0275** | 0.1156 | 125.63 |
| | Physics-only LoRA (norm) | 0.1007 | 0.1007 | 0.2706 | 142.38 |
| | Physics-only LoRA (no_norm) | 0.0478 | 0.0478 | 0.1862 | 506.55 |
| | Muon-LoRA (norm) | 0.1970 | 0.1970 | 0.9737 | 40871 |
| | Zero-shot (no finetune) | 1.0176 | 0.1374 | 3.1806 | 6.83e7 |
| | Scratch (full-param) | 0.7903 | 0.1061 | 2.9581 | 16372 |
| **AdvDiff 2D** | Supervised LoRA | **0.0023** | **0.0023** | 0.0029 | 0.4707 |
| | **Orth-LoRA** (frozen enc/dec) | **0.0038** | **0.0038** | 0.0048 | 0.0588 |
| | **rescaled_norm** (λ_pde=4.8e-4) | **0.0049** | **0.0049** | 0.0063 | 0.4325 |
| | Physics-only LoRA (norm) | 0.0133 | 0.0133 | 0.0169 | 0.3439 |
| | Physics-only LoRA (no_norm) | 0.0102 | 0.0102 | 0.0131 | 1.3274 |
| | Muon-LoRA (norm) | 0.0276 | 0.0276 | 0.0348 | 2.6490 |
| | Zero-shot (no finetune) | 1.0158 | 0.1648 | 1.2582 | 9984 |
| | Scratch (full-param) | 0.0673 | 0.0109 | 0.0833 | 9.021 |
| **Burgers 2D** | Supervised LoRA | **0.0022** | **0.0022** | 0.0002 | 0.0030 |
| | **Orth-LoRA** (frozen enc/dec) | **0.0021** | **0.0021** | 0.0002 | 0.0002 |
| | **rescaled_norm** (λ_pde=2.3e-5) | **0.0034** | **0.0034** | 0.0002 | 0.0002 |
| | Physics-only LoRA (norm) | 0.0109 | 0.0109 | 0.0005 | 0.0002 |
| | Physics-only LoRA (no_norm) | 0.1303 | 0.1303 | 0.0051 | 0.0968 |
| | Zero-shot (no finetune) | 0.0876 | 0.0280 | 0.0042 | 0.1451 |
| | Muon-LoRA (norm) | 0.0705 | 0.0705 | 0.0031 | 0.0384 |
| | Scratch (full-param) | 0.0716 | 0.0226 | 0.0042 | 0.0301 |

### The Well Datasets (postsmooth pretrain, bcw1)

| Dataset | Method | VRMSE | nRMSE |
|---------|--------|-------|-------|
| **Gray-Scott** | Supervised LoRA | **0.0754** | **0.0428** |
| | fixed_norm | 0.1110 | 0.0623 |
| | no_norm | 0.1112 | — |
| | rescaled_norm (λ_pde=8.5e-4) | 0.1117 | 0.0627 |
| | per_t_norm | 0.1246 | 0.0701 |
| **Rayleigh-Benard** | Supervised LoRA | **0.1660** | **0.1434** |
| | fixed_norm | 0.1883 | 0.1651 |
| | rescaled_norm (λ_pde=1.19) | 0.1884 | 0.1651 |
| | no_norm | 0.1940 | — |
| | per_t_norm | 0.9868 | 0.8474 |
| **Shear Flow** | Supervised LoRA | **0.0446** | **0.0446** |
| | rescaled_norm (λ_pde=1.2e-3) | 0.1151 | — |
| | no_norm | 0.1341 | — |
| | fixed_norm | 0.5007 | — |
| | per_t_norm | 0.5634 | — |
| **Active Matter** | Supervised LoRA | **0.0639** | **0.0559** |
| | fixed_norm | 0.0767 | 0.0671 |
| | rescaled_norm (λ_pde=3.3e-3) | 0.0768 | 0.0672 |
| | no_norm | 0.0769 | — |
| | per_t_norm | 0.5905 | 0.5147 |


---

## 2. Complete Experiment Table (2D Exact-Solution, VRMSE)

| Method | Taylor-Green | Wave 2D | AdvDiff 2D | Burgers 2D |
|--------|-------------|---------|-----------|-----------|
| **Supervised LoRA** | **0.0014** | **0.0097** | **0.0023** | **0.0022** |
| **rescaled_norm** (best λ_bc) | **0.0070** | **0.0243** | **0.0046** | **0.0034** |
| rescaled_norm (λ_bc=10000) | 0.0070 | 0.0275 | 0.0049 | 0.0034 |
| rescaled_norm (λ_bc=1000) | 0.0170 | 0.0489 | 0.0046 | 0.0041 |
| rescaled_norm (λ_bc=100) | 0.0232 | 0.0676 | 0.0123 | 0.0092 |
| rescaled_norm (λ_bc=100000) | 0.0098 | 0.0243 | — | 0.0043 |
| norm (λ_pde=1) | 0.0294 | 0.1007 | 0.0133 | 0.0109 |
| no_norm | 0.0875 | 0.0478 | 0.0102 | 0.1303 |
| Full FT (pretrain+full-param) | 0.0057 | 0.0252 | 0.0053 | 0.0024 |
| Muon-LoRA (norm) | 0.1220 | 0.1970 | 0.0276 | 0.0705 |
| Muon-LoRA (no_norm) | 0.1459 | 0.1694 | 0.0149 | 0.1425 |
| Zero-shot (no finetune) | 0.1274 | 1.0176 | 1.0158 | 0.0876 |
| Scratch (full-param, no pretrain) | 0.3920 | 0.7903 | 0.0673 | 0.0716 |

> Best λ_bc: TG=10000, Wave=100000, AdvDiff=1000, Burgers=10000.

### Key findings

1. **rescaled_norm is the best physics-only method** on all 4 datasets.
2. **Pretrain is critical**: Zero-shot → rescaled_norm gives 18–207× improvement; Scratch degrades 6.6–16.5× vs physics-only.
3. **Physics-only gap to supervised**: rescaled_norm reaches 49–65% of supervised accuracy using only ~3% GT (boundary).
4. **LoRA ≈ Full FT**: comparable performance, LoRA uses only 6.7% parameters.
5. **Muon-LoRA underperforms** standard LoRA by 2.7–6.5× (orthogonal constraint too restrictive).
6. **λ_bc=10000 is the strong default**, Wave benefits from 100000.

---

## 3. Orth-LoRA vs LoRA (Fair Comparison)

> Setup: frozen enc/dec, r=16, LoRA A,B initialized from trained rescaled_norm checkpoint (300 keys).
> Only difference: forward path. Orth-LoRA: ΔW = ||A|| * ||B|| * NS(B) @ NS(A). LoRA: ΔW = (α/r) * BA.
> Taylor-Green: lr=5e-4, bs=20, 4GPU, 30ep. Others: lr=1e-4, bs=4, 3-4GPU, 30ep.

| Dataset | Method | VRMSE (all) | Per-channel VRMSE | PDE Loss |
|---------|--------|------------|-------------------|----------|
| **Taylor-Green** | Orth-LoRA | **0.00596** | Vx=0.00102, Vy=0.00103, press=**0.01582** | **0.00495** |
| | LoRA | 0.00624 | Vx=0.00098, Vy=0.00098, press=0.01675 | 0.00510 |
| | Δ | **-4.5%** | press **-5.6%** | -2.9% |
| **Burgers** | Orth-LoRA | **0.00212** | Vx=**0.00232**, Vy=**0.00193** | **0.000171** |
| | LoRA | 0.00229 | Vx=0.00259, Vy=0.00200 | 0.000198 |
| | Δ | **-7.4%** | Vx **-10.4%** | -13.6% |
| **AdvDiff** | Orth-LoRA | **0.00377** | u=**0.00377** | **0.0588** |
| | LoRA | 0.00422 | u=0.00422 | 0.0625 |
| | Δ | **-10.7%** | u **-10.7%** | -5.9% |
| **Wave** | Orth-LoRA | **0.02182** | u=**0.01132**, w=**0.03232** | **171.4** |
| | LoRA | 0.02242 | u=0.01200, w=0.03284 | 196.7 |
| | Δ | **-2.7%** | u **-5.7%**, w **-1.6%** | -12.9% |

### Findings

1. **Orth-LoRA wins on all 4 datasets** (VRMSE -2.7% to -10.7%, PDE loss -2.9% to -13.6%).
2. **Largest gains on weaker channels**: TG pressure -5.6%, Burgers Vx -10.4%, Wave u -5.7%.
3. **PDE loss improvement consistently larger than VRMSE**: orthogonal update produces more physically consistent solutions.

---

## 3. Comparison to Literature Baselines

### Taylor-Green 2D (Incompressible Navier-Stokes)

| Method | Type | Metric | Value | Reference |
|--------|------|--------|-------|-----------|
| **Ours (Supervised LoRA)** | Supervised | VRMSE | **0.0014** | — |
| **Ours (Physics-only LoRA)** | Physics-only | VRMSE | **0.0294** | — |
| Ours (Scratch) | Physics-only | VRMSE | 0.3920 | — |
| FS-HNN | Supervised | Rollout MSE | 5.01e-8 | Li et al., arXiv:2603.06354 (2026) |
| FNO | Supervised | Rollout MSE | 3.43e-4 | Li et al., ICLR 2021 [1] |
| PINN | Physics-only | Rel-L2 | ~0.021 | Raissi et al., JCP 2019 [8] |

> FS-HNN uses Hamiltonian structure preservation, not directly comparable in metric.
> Our physics-only VRMSE 2.94% is competitive with PINN Rel-L2 ~2.1%, but uses no full-field labels.

### Wave 2D

| Method | Type | Metric | Value | Reference |
|--------|------|--------|-------|-----------|
| **Ours (Supervised LoRA)** | Supervised | VRMSE | **0.0097** | — |
| **Ours (Physics-only LoRA)** | Physics-only | VRMSE | **0.0478** | — |
| Ours (Scratch) | Physics-only | VRMSE | 0.7903 | — |
| CNO | Supervised | Rel-L1 | 0.0063 | Raonic et al., NeurIPS 2023 [7]; benchmarked in Poseidon [3] |
| FNO | Supervised | Rel-L1 | 0.0102 | Li et al., ICLR 2021 [1]; benchmarked in Poseidon [3] |
| UNet | Supervised | Rel-L1 | 0.0151 | benchmarked in Poseidon [3] |

> Poseidon baselines are on 64×64 (ours 256×256), different domain/IC. Metrics not directly comparable.

### AdvDiff 2D

| Method | Type | Metric | Value | Reference |
|--------|------|--------|-------|-----------|
| **Ours (Supervised LoRA)** | Supervised | VRMSE | **0.0023** | — |
| **Ours (Physics-only LoRA)** | Physics-only | VRMSE | **0.0102** | — |
| Ours (Scratch) | Physics-only | VRMSE | 0.0673 | — |
| SpecB-FNO | Supervised | NRMSE | 0.0066 | PDEBench [4] |
| FFNO | Supervised | NRMSE | 0.0072 | PDEBench [4] |
| FNO | Supervised | NRMSE | 0.0190 | PDEBench [4] |
| U-Net | Supervised | NRMSE | 0.1160 | PDEBench [4] |

> PDEBench numbers are on Diffusion-Reaction (FitzHugh-Nagumo), not pure Advection-Diffusion.
> Our physics-only nRMSE 1.02% sits between SpecB-FNO (0.66%) and FNO (1.90%), without using labels.

### Burgers 2D

| Method | Type | Metric | Value | Reference |
|--------|------|--------|-------|-----------|
| **Ours (Supervised LoRA)** | Supervised | VRMSE | **0.0022** | — |
| **Ours (Physics-only LoRA)** | Physics-only | VRMSE | **0.0109** | — |
| Ours (Scratch) | Physics-only | VRMSE | 0.0716 | — |
| ResNet | Supervised | GMean nRMSE | 0.053 | APEBench [5] |
| UNet | Supervised | GMean nRMSE | 0.162 | APEBench [5] |
| FNO | Supervised | GMean nRMSE | 0.328 | APEBench [5] |

> APEBench uses 100-step autoregressive rollout (harder setup). Different IC distribution.

### Navier-Stokes (PINO, for reference)

| Method | Type | Metric | Value | Reference |
|--------|------|--------|-------|-----------|
| PINO (fine-tuned) | Hybrid | Rel Error | 0.0184 | Li et al., arXiv:2111.03794 [2] |
| FNO | Supervised | Rel Error | 0.0304 | Li et al., ICLR 2021 [1] |
| PINN | Physics-only | Rel Error | 0.187 | Raissi et al., JCP 2019 [8] |

> PINO demonstrates supervised pretrain + physics fine-tune > physics-only from scratch — same spirit as our approach, but we achieve cross-PDE transfer.

---


## 5. LoRA vs Full Finetune (Physics-Only, rescaled_norm, best λ_bc)

| Dataset | Supervised LoRA | LoRA (rescaled_norm) | Full FT (rescaled_norm) | Scratch (no pretrain) |
|---------|----------------|---------------------|------------------------|-----------------------|
| Taylor-Green | 0.0014 | 0.0070 | **0.0057** | 0.3920 |
| Wave 2D | 0.0097 | **0.0243** | 0.0252 | 0.7903 |
| AdvDiff 2D | 0.0023 | **0.0046** | 0.0053 | 0.0673 |
| Burgers 2D | 0.0022 | 0.0034 | **0.0024** | 0.0716 |

> LoRA (6.7% params) and Full FT (100% params) perform comparably. LoRA is slightly better on AdvDiff/Wave; Full FT slightly better on TG/Burgers. The gap is small — LoRA is sufficient for physics-only finetune.

---

## 6. Normalization & lambda_bc Ablation

### 5a. Normalization Strategy (λ_bc=10000)

| Dataset | rescaled_norm | norm | no_norm | Supervised |
|---------|--------------|------|---------|-----------|
| Taylor-Green | **0.0070** | 0.0294 | 0.0875 | 0.0014 |
| Wave 2D | **0.0275** | 0.1007 | 0.0478 | 0.0097 |
| AdvDiff 2D | **0.0049** | 0.0133 | 0.0102 | 0.0023 |
| Burgers 2D | **0.0034** | 0.0109 | 0.1303 | 0.0022 |

> **rescaled_norm wins on ALL 4 datasets.** Key idea: set λ_pde = min(eq_scales) to prevent normalized PDE loss from drowning BC guidance, while keeping per-equation balance via eq_scales.

### 5b. lambda_bc Ablation (rescaled_norm, λ_pde=min(eq_scales))

| Dataset | λ_bc=100 | λ_bc=1000 | λ_bc=10000 | λ_bc=100000 | Supervised |
|---------|----------|-----------|-----------|-------------|-----------|
| Taylor-Green | 0.0232 | 0.0170 | **0.0070** | 0.0098 | 0.0014 |
| Wave 2D | 0.0676 | 0.0489 | 0.0275 | **0.0243** | 0.0097 |
| AdvDiff 2D | 0.0123 | **0.0046** | 0.0049 | — | 0.0023 |
| Burgers 2D | 0.0092 | 0.0041 | **0.0034** | 0.0043 | 0.0022 |

> **Best λ_bc per dataset**: TG=10000, Wave=100000, AdvDiff=1000, Burgers=10000.
> General trend: λ_bc=10000 is a strong default. Wave benefits from even stronger BC (100000). Too high (100000) slightly hurts TG/Burgers.
> **Best physics-only config (rescaled_norm)**: TG 0.70%, Wave 2.43%, AdvDiff 0.46%, Burgers 0.34%.

---

## 6. Per-Channel Detail

### Taylor-Green 2D (u, v, p)

| Variant | RMSE(Vx) | RMSE(Vy) | RMSE(p) | VRMSE(Vx) | VRMSE(Vy) | VRMSE(p) |
|---------|----------|----------|---------|-----------|-----------|----------|
| supervised | 0.0006 | 0.0006 | 0.0005 | 0.0012 | 0.0011 | 0.0020 |
| norm | 0.0055 | 0.0055 | 0.0152 | 0.0114 | 0.0114 | 0.0653 |
| no_norm | 0.0240 | 0.0291 | 0.0356 | 0.0496 | 0.0603 | 0.1525 |
| scratch | 0.0377 | 0.0418 | 0.2356 | 0.0783 | 0.0866 | 1.0112 |

### Wave 2D (u, w=u_t)

| Variant | RMSE(u) | RMSE(w) | VRMSE(u) | VRMSE(w) |
|---------|---------|---------|----------|----------|
| supervised | 0.0133 | 0.0322 | 0.0120 | 0.0074 |
| norm | 0.1268 | 0.3598 | 0.1185 | 0.0829 |
| no_norm | 0.0383 | 0.2602 | 0.0359 | 0.0596 |
| scratch | 0.6482 | 4.1303 | 0.6134 | 0.9672 |

### AdvDiff 2D (u)

| Variant | RMSE(u) | VRMSE(u) | nRMSE(u) |
|---------|---------|----------|----------|
| supervised | 0.0029 | 0.0023 | 0.0023 |
| norm | 0.0169 | 0.0133 | 0.0133 |
| no_norm | 0.0131 | 0.0102 | 0.0102 |
| scratch | 0.0833 | 0.0673 | 0.0109 |

### Burgers 2D Cole-Hopf (u, v)

| Variant | RMSE(Vx) | RMSE(Vy) | VRMSE(Vx) | VRMSE(Vy) |
|---------|----------|----------|-----------|-----------|
| supervised | 0.0002 | 0.0001 | 0.0021 | 0.0022 |
| norm | 0.0006 | 0.0003 | 0.0114 | 0.0104 |
| no_norm | 0.0060 | 0.0037 | 0.1210 | 0.1396 |
| scratch | 0.0055 | 0.0016 | 0.0872 | 0.0559 |

---

## 7. Key Takeaways for Paper

1. **Pretrain is critical**: Scratch degrades 6.6–16.5× across all PDEs (Table 2).
2. **Physics-only finetune is competitive**: Best VRMSE 1–5%, using only BC (~3% GT info).
3. **eq_scales normalization**: Helps multi-equation PDEs with large scale differences; less important for single-equation.
4. **BC width matters**: bcw1 consistently outperforms bcw1 on The Well datasets (19–82% improvement).
5. **Cross-PDE transfer**: Unlike PINO (same PDE pretrain+finetune), we pretrain on diverse PDEs and finetune on unseen ones.

---

## 8. Missing Experiments (TODO)

- [x] ~~Supervised LoRA finetune baseline (upper bound)~~ — Done for 4 exact-solution datasets
- [ ] Zero-shot baseline (pretrain -> direct inference, no finetune)
- [ ] The Well official baseline numbers (FNO/TFNO/U-Net exact values)
- [ ] BC width systematic ablation (1/4/8/16) on 2D exact-solution datasets
- [ ] Data efficiency: supervised finetune with 5%/10%/50% labels vs physics-only (0%)
- [ ] Supervised LoRA for The Well datasets (Gray-Scott, Rayleigh-Benard, Shear Flow, Active Matter)

---

## References

[1] Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. arXiv:2010.08895.

[2] Li, Z. et al. *Physics-Informed Neural Operator for Learning Partial Differential Equations*. ACM/IMS Journal of Data Science, 2024. arXiv:2111.03794.

[3] Herde, M. et al. *Poseidon: Efficient Foundation Models for PDEs*. NeurIPS 2024. arXiv:2405.19101.

[4] Takamoto, M. et al. *PDEBench: An Extensive Benchmark for Scientific Machine Learning*. NeurIPS 2022 (Datasets & Benchmarks). arXiv:2210.07182.

[5] Koehler, F. et al. *APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs*. NeurIPS 2024 (Datasets & Benchmarks). arXiv:2411.00180.

[6] Hao, Z. et al. *DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training*. ICML 2024. arXiv:2403.03542.

[7] Raonic, B. et al. *Convolutional Neural Operators for Robust and Accurate Learning of PDEs*. NeurIPS 2023. arXiv:2302.01178.

[8] Raissi, M. et al. *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*. Journal of Computational Physics, 378:686-707, 2019.

[9] Yu, R. *An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning*. arXiv:2512.19439, 2025.

[10] McCabe, M. et al. *Multiple Physics Pretraining for Physical Surrogate Models*. NeurIPS 2024. arXiv:2310.02994.

[11] Li, Y. et al. *Frequency-Separable Hamiltonian Neural Network for Multi-Timescale Dynamics*. arXiv:2603.06354, 2026.


去掉这个apebench，保留原来的the well。但我准备这样做，先对表格进行转置，有4列，上面标记the well，另外7列标记Exact-Solution Datasets，两列是1D burgers，1D advection，Taylor-Green 2D，Wave 2D，AdvDiff 2D，Burgers 2D，3D advection。\                                 
其中4个2D的数据可以在我的eval_results_2d.md里面找到。ours填入除Supervised LoRA最好的数据。然后行，也就是原来的列保持不变。新的7列，只用填入我们的数据，其他都是用-来代替，我后面的数值慢慢填入\               
关于表格的caption你可以稍后和我沟通


| Dataset | FNO | TFNO | UNetClassic | UNetConvNext |
|---------|-----|------|-------------|--------------|
| burgers | 0.0066 | **0.0029** | 0.0905 | 0.0092 |
| Taylor-Green | **0.0013** | 0.0014 | 0.0195 | 0.0027 |
| AdvDiff | 0.0010 | **0.0002** | 0.0296 | 0.0014 |
| Wave | 0.0029 | **0.0009** | 0.0293 | 0.0015 |

好的，把我们的eval_results_2d.md里面表格的内容，全部写到main.tex里面。