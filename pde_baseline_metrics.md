# PDE Baseline Metrics Survey

> 所有数值均来自论文原文表格，未做任何编造。不同论文的实验设置不同，数值不可跨论文直接对比。

---

## 1D+t PDEs

### 1. 1D Burgers

#### Source A: PDEBench (NeurIPS 2022) — Table 6-7

| 设置 | 分辨率 1024, T=200 步, 10000 samples, periodic BC, domain x∈(0,1) t∈(0,2] |
|---|---|

| Method | nu=0.001 nRMSE | nu=0.01 nRMSE | nu=0.1 nRMSE | nu=1.0 nRMSE |
|--------|---------------|--------------|-------------|-------------|
| FNO | **2.9e-2** | **7.8e-3** | **2.9e-3** | **4.0e-3** |
| U-Net | 3.7e-1 | 2.2e-1 | 2.3e-1 | 2.4e-1 |
| PINN | 3.9e-1 | 8.5e-1 | 4.6e-1 | 1.9e-2 |

Metric: **nRMSE** = ||u_pred - u_true||₂ / ||u_true||₂

#### Source B: FNO 原文 (ICLR 2021) — Table 3

| 设置 | 分辨率 256–8192, N=1000 train / 200 test, nu=0.1, 预测 u(x, t=1) |
|---|---|

| Method | s=256 | s=1024 | s=8192 |
|--------|-------|--------|--------|
| **FNO** | **0.0149** | **0.0160** | **0.0139** |
| LNO | 0.0212 | 0.0217 | 0.0189 |
| PCANN | 0.0398 | 0.0391 | 0.0393 |
| GNO | 0.0555 | 0.0651 | 0.0699 |

Metric: **Relative L2 error**

#### Source C: APEBench (NeurIPS 2024) — Table 13

| 设置 | 分辨率 160, 50 IC, 50 time steps train, 100-step rollout test, one-step supervised |
|---|---|

| Method | GMean nRMSE (100-step rollout) |
|--------|-------------------------------|
| **ResNet** | **0.013** |
| Conv | 0.026 |
| Dil (DilResNet) | 0.057 |
| UNet | 0.065 |
| FNO | 0.070 |

Metric: **GMean of Mean nRMSE** over 100 rollout steps

#### Source D: PINNacle (NeurIPS 2024) — Table 3

| 设置 | x∈[-1,1], t∈[0,1], nu=0.01/π, IC=-sin(πx), BC=0, MLP 5×100, 20000 epochs, 8192 collocation |
|---|---|

| Method | L2RE |
|--------|------|
| **LBFGS** | **1.33e-2** |
| LAAF | 1.43e-2 |
| PINN | 1.45e-2 |
| PINN-NTK | 1.84e-2 |
| FBPINN | 2.32e-2 |
| PINN-w | 2.63e-2 |
| RAR | 3.32e-2 |
| MultiAdam | 4.85e-2 |
| gPINN | 2.16e-1 |
| vPINN | 3.47e-1 |

Metric: **L2RE** (L2 Relative Error)

---

### 2. 1D Advection

#### Source A: PDEBench (NeurIPS 2022) — Table 6

| 设置 | 分辨率 1024, T=200 步, 10000 samples, periodic BC, x∈(0,1) t∈(0,2] |
|---|---|

| Method | β=0.1 nRMSE | β=0.4 nRMSE | β=1.0 nRMSE | β=4.0 nRMSE |
|--------|------------|------------|------------|------------|
| **FNO** | **7.7e-3** | **1.0e-2** | **9.7e-3** | **6.7e-3** |
| U-Net | 5.0e-2 | 2.3e-1 | 2.3e-1 | 2.4e-2 |
| PINN | 7.8e-3 | 3.0e-2 | 1.3e-2 | 7.7e-1 |

Metric: **nRMSE**

#### Source B: APEBench (NeurIPS 2024) — Table 9 (bridging)

| 设置 | 分辨率 160, γ₁=0.5, Mean nRMSE at specific rollout steps |
|---|---|

| Method | t=1 | t=10 | t=50 |
|--------|-----|------|------|
| **ResNet** | **0.001** | **0.005** | **0.023** |
| Conv | 0.001 | 0.009 | 0.040 |
| FNO | 0.001 | 0.010 | 0.049 |
| UNet | 0.005 | 0.024 | 0.090 |
| Dil | 0.004 | 0.024 | 0.095 |

Metric: **Mean nRMSE** at each rollout step

---

### 3. 1D Allen-Cahn

> 标准 setup: u_t = 0.0001·u_xx + 5u - 5u³, x∈[-1,1], t∈[0,1], IC=x²cos(πx), periodic BC

#### Source A: PirateNet (JMLR 2024) — Table 2 (最全面对比)

| 设置 | 同上标准 setup, 256 neurons, tanh, batch 1024, 1e5 Adam steps |
|---|---|

| Method | Relative L2 Error |
|--------|------------------|
| Vanilla PINN | 4.98e-1 |
| Adaptive time sampling | 2.33e-2 |
| SA-PINN | 2.10e-2 |
| Time marching | 1.68e-2 |
| Causal PINN | 1.39e-4 |
| Dirac delta causal | 6.29e-5 |
| JAX-PI | 5.37e-5 |
| RBA-PINNs | 4.55e-5 |
| **PirateNet** | **2.24e-5** |

Metric: **Relative L2 Error**

#### Source B: TL-DPINN (IJCAI 2024) — Table 1

| 设置 | 同上标准 setup, 4×128 MLP, Fourier embedding, Nt=200 Nr=512 |
|---|---|

| Method | Relative L2 Error | Train Time (s) |
|--------|-------------------|----------------|
| Causal PINN | 1.66e-3 | 9264 |
| **TL-DPINN1** | **5.92e-4** | 2328 |
| TL-DPINN2 | 9.82e-4 | 1100 |
| PINN | 8.23e-1 | 1412 |

Metric: **Relative L2 Error**

> **注意**: Allen-Cahn 主要由 PINN 类方法 benchmark，无 FNO/U-Net/DeepONet 的 operator learning baseline。

---

### 4. 1D KdV (Korteweg-de Vries)

#### Source A: IS-FNO (arXiv 2512.19439) — Table I

| 设置 | 分辨率 256, t∈[0,1.22], dt=0.0024, 350 trajectories, 20-step prediction, single A40 |
|---|---|

| Method | Train Rel-L2 | Val Rel-L2 |
|--------|-------------|-----------|
| **IS-FNO°** | **5.9e-4** | **5.3e-4** |
| kFNO° | 7.3e-4 | 7.1e-4 |
| IS-FNO* | 7.0e-4 | 7.0e-4 |
| IS-FNO' | 8.7e-4 | 9.2e-4 |
| kFNO* | 1.0e-4 | 1.1e-3 |
| kFNO' | 1.1e-3 | 1.3e-3 |
| FNO | 1.3e-3 | 1.4e-3 |

Metric: **Relative L2 Error** (20-step average)

#### Source B: APEBench (NeurIPS 2024) — Table 13

| 设置 | 分辨率 160, 50 IC, 100-step rollout |
|---|---|

| Method | GMean nRMSE |
|--------|-------------|
| **DilResNet** | **0.077** |
| UNet | 0.096 |
| Conv | 0.123 |
| FNO | 0.210 |
| ResNet | 0.876 |

Metric: **GMean of Mean nRMSE** (100-step rollout)

---

### 5. 1D Fisher-KPP

#### Source A: PINN Retraining Study (arXiv 2601.11406)

| 设置 | D=0.01, R=1.0, PINN vs FDM vs exact solution |
|---|---|

| Method | Relative L2 Error |
|--------|--------------------|
| **FDM** | **1.42e-4** |
| Best PINN | 5.57e-2 |
| PINN (retrained) | 9.79e-2 |

Metric: **Relative L2 Error**

#### Source B: Wave-PINN (arXiv 2402.08313)

| 设置 | Fisher equation with varying sharpness ρ |
|---|---|

| Method | ρ=1000 L2 Error | ρ=10000 L2 Error |
|--------|-----------------|------------------|
| **wave-PINN** | **0.06e-4** | best |
| Standard PINN | 19.5e-4 | 324e-4 (fails) |
| Data-driven ANN | 0.74e-4 | — |

Metric: **L2 Error**

> **注意**: Fisher-KPP 在 APEBench 中定义但 Table 13 不含数值。Operator learning baseline 缺失。

---

## 2D+t PDEs

### 6. 2D Taylor-Green Vortex

> u = cos(x)sin(y)e^{-2νt}, v = -sin(x)cos(y)e^{-2νt}

#### Source A: FS-HNN (arXiv 2603.06354)

| 设置 | 2D TGV vorticity, periodic, RK4, train 5 steps rollout 50 |
|---|---|

| Method | Rollout MSE |
|--------|-----------|
| PHNN | 3.43e-4 |
| FNO | 3.43e-4 |
| **FS-HNN (High)** | **8.33e-8** |
| FS-HNN (Combined) | 5.01e-8 |

Metric: **MSE** (rollout)

#### Source B: MindSpore Flow PINN Tutorial

| 设置 | [0,2π]², Re=100, t∈(0,2), 6-layer FC, tanh, 500 epochs |
|---|---|

| Channel | Relative L2 Error |
|---------|-------------------|
| u | 0.0211 |
| v | 0.0133 |
| p | 0.0452 |
| **Total** | **0.0213** |

Metric: **Relative L2 Error**

> **注意**: Taylor-Green 不是主流 operator learning benchmark，FNO/DeepONet 等大规模对比缺失。CFDONEval (IJCAI 2025) 有 12 种 operator learning 方法对比但数值在 PDF 图中。

---

### 7. 2D KP-II (Kadomtsev-Petviashvili)

#### Source: IS-FNO (arXiv 2512.19439) — Table I

| 设置 | 128×128, domain [0,20]², t∈[0,1.22], dt=0.0024, 350 trajectories, 20-step prediction |
|---|---|

| Method | Train Rel-L2 | Val Rel-L2 |
|--------|-------------|-----------|
| **IS-FNO°** | **4.3e-4** | **4.3e-4** |
| kFNO* | 5.0e-4 | 5.0e-4 |
| FNO | 3.1e-3 | 3.1e-3 |

Metric: **Relative L2 Error** (20-step average)

> **注意**: 仅 3 个方法 (IS-FNO, kFNO, FNO)，无 U-Net/ResNet/DeepONet baseline。

---

### 8. 2D Wave

#### Source A: CNO (arXiv 2302.01178) — Table 1 (**最全面**)

| 设置 | 64×64, periodic (T²), c=0.1, T=5, K=24 modes, 512 train / 256 test, map IC→u(T=5) |
|---|---|

| Method | In-dist Median Rel-L1 (%) | OOD Median Rel-L1 (%) |
|--------|--------------------------|----------------------|
| **CNO** | **0.63** | **1.17** |
| ResNet | 0.79 | 1.36 |
| FNO | 1.02 | 1.77 |
| GT | 1.44 | 1.79 |
| UNet | 1.51 | 2.03 |
| DeepONet | 2.26 | 2.83 |
| FFNN | 2.51 | 3.01 |

Metric: **Median Relative L1 Error (%)**

#### Source B: Poseidon (NeurIPS 2024) — Table 1

| 设置 | 128×128, [0,1]², absorbing BC, ~10000 trajectories, direct rollout |
|---|---|

| Method | Wave-Layer AG (vs FNO) | Wave-Gauss AG (vs FNO) |
|--------|----------------------|----------------------|
| **Poseidon-L** | **6.1×** | **5.6×** |
| CNO | 3.0× | 2.6× |
| scOT | 2.9× | 2.1× |
| CNO-FM | 2.2× | 1.8× |
| FNO | 1× (baseline) | 1× (baseline) |

Metric: **Accuracy Gain** (= FNO_error / Method_error, higher=better). 原始误差值未列表，仅在 log-scale 图中。

---

### 9. 2D Advection-Diffusion

#### Source: SpecB-FNO (arXiv 2404.07200) — PDEBench 2D Diffusion-Reaction (最接近)

| 设置 | 128×128, Nt=101, FitzHugh-Nagumo, 1000 samples, Neumann BC |
|---|---|

| Method | NRMSE |
|--------|-------|
| **SpecB-FNO** | **0.0066** |
| FFNO | 0.0072 |
| ResNet | 0.0138 |
| FNO | 0.0190 |
| CNO | 0.0257 |
| U-Net | 0.1160 |
| DeepONet | NaN (failed) |

Metric: **NRMSE**

> **注意**: 这是 2D Diffusion-Reaction (FitzHugh-Nagumo)，非纯 Advection-Diffusion。纯 2D AdvDiff 在主流 benchmark 中缺失。

---

### 10. 2D Burgers

#### Source: APEBench (NeurIPS 2024) — Table 13

| 设置 | 160×160, 50 IC, 50 time steps train, 100-step rollout test |
|---|---|

| Method | GMean nRMSE |
|--------|-------------|
| **ResNet** | **0.053** |
| Dil | 0.139 |
| Conv | 0.146 |
| UNet | 0.162 |
| FNO | 0.328 |

Metric: **GMean of Mean nRMSE** (100-step rollout)

---

## 3D+t PDEs

### 11. 3D Beltrami (Ethier-Steinman)

#### Source A: NSFnets (JCP 2021) — velocity-velocity formulation

| 设置 | [-1,1]³, t∈[0,1], Re=1 (ν=1), a=d=1, 10×100 MLP, tanh, 31×31 boundary per face |
|---|---|

| Method | u L2RE (t=1) | v L2RE (t=1) | w L2RE (t=1) | p L2RE (t=1) |
|--------|-------------|-------------|-------------|-------------|
| VP-NSFnet | 0.426% | 0.366% | 0.587% | 4.766% |
| **VV-NSFnet** | **0.255%** | **0.284%** | **0.263%** | N/A |

Metric: **Relative L2 Error (%)**

#### Source B: WAN3DNS (arXiv 2509.26034)

| 设置 | [-1,1]³, t∈[0,1], ν=10 (不同于 NSFnets!), 6×50 MLP, 20000 epochs |
|---|---|

| Method | u error | v error | w error | p error |
|--------|---------|---------|---------|---------|
| DeepXDE | 0.021 | 0.018 | 0.017 | 10.54 |
| NSFnets | 0.046 | 0.073 | 0.062 | 27.66 |
| **WAN3DNS** | **0.010** | **0.008** | **0.016** | **0.011** |

Metric: **L2 Relative Error** (注意 ν=10 与 Source A 不同)

> **注意**: 仅 PINN 类方法，无 FNO/U-Net/DeepONet baseline。

---

### 12. 3D Advection

#### Source: APEBench (NeurIPS 2024) — Table 13

| 设置 | 分辨率未明确 (推测 32³), 50 IC, 100-step rollout |
|---|---|

| Method | GMean nRMSE |
|--------|-------------|
| **ResNet** | **0.134** |
| UNet | 0.183 |
| Dil | 0.205 |
| Conv | 0.308 |
| FNO | 0.895 |

Metric: **GMean of Mean nRMSE**

---

### 13. 3D Heat

> APEBench 中有定义 (diffusion scenario)，但 Table 13 无直接数值。最接近的是 "2D Anisotropic Diffusion":

#### Source: APEBench — Table 13 (2D Anisotropic Diffusion, 作为参考)

| 设置 | 160×160, 50 IC, 100-step rollout |
|---|---|

| Method | GMean nRMSE |
|--------|-------------|
| **Conv** | **0.016** |
| ResNet | 0.022 |
| Dil | 0.041 |
| UNet | 0.044 |
| FNO | 0.077 |

Metric: **GMean of Mean nRMSE**

> **注意**: 这是 2D 各向异性扩散，非 3D Heat。3D Heat 精确数值在现有论文中缺失。

---

## 覆盖情况总结

| # | PDE | Operator Learning (FNO/UNet等) | PINN 类 | 数据充分度 |
|---|-----|-------------------------------|---------|-----------|
| 1 | 1D Burgers | ✅ PDEBench + FNO + APEBench | ✅ PINNacle | 充分 |
| 2 | 1D Advection | ✅ PDEBench + APEBench | ✅ PDEBench | 充分 |
| 3 | 1D Allen-Cahn | ❌ 无 operator learning | ✅ PirateNet (9种PINN) | 仅 PINN |
| 4 | 1D KdV | ✅ IS-FNO + APEBench | ❌ | 中等 |
| 5 | 1D Fisher-KPP | ❌ 无 operator learning 数值 | ⚠️ 少量 PINN | 不足 |
| 6 | 2D Taylor-Green | ⚠️ 仅 FS-HNN (MSE) | ⚠️ 仅 tutorial | 不足 |
| 7 | 2D KP-II | ✅ IS-FNO (3方法) | ❌ | 勉强 |
| 8 | 2D Wave | ✅ CNO (7方法) | ❌ | 充分 |
| 9 | 2D Advection-Diffusion | ⚠️ 仅 Diffusion-Reaction 近似 | ❌ | 不足 |
| 10 | 2D Burgers | ✅ APEBench (5方法) | ❌ | 中等 |
| 11 | 3D Beltrami | ❌ | ✅ NSFnets + WAN3DNS | 仅 PINN |
| 12 | 3D Advection | ✅ APEBench (5方法) | ❌ | 中等 |
| 13 | 3D Heat | ❌ 无精确数值 | ❌ | 不足 |

---

## 论文来源索引

| 简称 | 论文 | ArXiv | Metric |
|------|------|-------|--------|
| PDEBench | PDEBench: An Extensive Benchmark for SciML | 2210.07182 | nRMSE |
| FNO | Fourier Neural Operator for Parametric PDEs | 2010.08895 | Relative L2 |
| APEBench | APEBench: Autoregressive PDE Emulator Benchmark | 2411.00180 | GMean nRMSE |
| IS-FNO | Inverse Scattering Inspired FNO | 2512.19439 | Relative L2 |
| PINNacle | PINNacle: Comprehensive PINN Benchmark | 2306.08827 | L2RE |
| PirateNet | PirateNets: Residual Adaptive Networks | JMLR 2024 | Relative L2 |
| CNO | Convolutional Neural Operators | 2302.01178 | Median Rel-L1 |
| Poseidon | Efficient Foundation Models for PDEs | 2405.19101 | AG/EG (relative) |
| DPOT | Auto-Regressive Denoising Operator Transformer | 2403.03542 | L2RE |
| NSFnets | NS Flow Nets | 2003.06496 | Relative L2 (%) |
| SpecB-FNO | Spectral Perspective on FNO | 2404.07200 | NRMSE |
| FS-HNN | Fourier-Space Hamiltonian NN | 2603.06354 | MSE |
