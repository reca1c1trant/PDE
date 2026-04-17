# PDE Baseline Numerical Results Survey

## 1. 1D Fisher-KPP Equation

**Equation**: u_t = nu * u_xx + r * u * (1 - u)

---

### Paper A: PINN Comprehensive Retraining Study (2025)

**Title**: "Solving the Fisher nonlinear differential equations via Physics-Informed Neural Networks: A Comprehensive Retraining Study and Comparative Analysis with the Finite Difference Method"
**arXiv ID**: 2601.11406

**Methods compared**: PINN (various training configs) vs FDM (Finite Difference Method) vs Exact analytical solution

**Metric**: Relative L2 Error

**Experimental Setup**:
- Spatial domain: x in [0, 1], temporal: t in [0, 1]
- Diffusion coefficient D = 0.01, Reaction rate R = 1.0
- FDM grid: Nx = 201 (dx = 0.005), Nt = 1600 (dt = 0.000625)
- PINN: 7 hidden layers x 50 neurons, Tanh activation
- Collocation points: 10,000; BC/IC points: 1,000 each

**Results (Relative L2 Error at t=1.0)**:

| Comparison                        | Relative L2 Error |
|-----------------------------------|-------------------|
| Exact vs FDM                      | 1.42 x 10^-4      |
| Exact vs PINN (initial, 10k iter) | 5.57 x 10^-2      |
| Exact vs PINN (retrained, 40k)    | 9.79 x 10^-2      |
| PINN (final) vs FDM               | 9.81 x 10^-2      |

**Training Details**:

| Config               | Iterations | LR      | L2 Error     | Time (s) |
|----------------------|-----------|---------|--------------|----------|
| Initial training     | 10,000    | 1e-3    | 5.57 x 10^-2 | 150.32   |
| Retraining phase 1   | +20,000   | 1e-4    | 9.80 x 10^-2 | 351.04   |
| Retraining phase 2   | +40,000   | 1e-4    | 9.79 x 10^-2 | 703.48   |

**Key finding**: Initial PINN training (5.57% error) outperformed retrained versions (~9.8%), indicating optimizer reset during retraining degrades performance.

---

### Paper B: Approximating Families of Sharp Solutions to Fisher's Equation with PINNs (2024)

**Title**: "Approximating Families of Sharp Solutions to Fisher's Equation with Physics-Informed Neural Networks"
**arXiv ID**: 2402.08313

**Methods compared**: standard-ANN, wave-ANN, standard-PINN, wave-PINN (with varying lambda)

**Metric**: L2 Error (x 10^-4)

**Experimental Setup**:
- Spatial: x in [-5, 5], Temporal: t in [0, 0.004]
- Diffusion coefficient mu = 10
- Discrete-rho networks: 2 hidden layers, 20 neurons; Continuous-rho: 3 layers, 20 neurons
- Activation: tanh (hidden), sigmoid (output)
- Training: Adam, LR=0.001, decay 0.95/1000 epochs, 50k-100k epochs

**Results - Discrete rho (L2 Error x 10^-4)**:

| rho   | standard-ANN  | wave-ANN     | standard-PINN | wave-PINN (best) |
|-------|---------------|--------------|---------------|------------------|
| 10^2  | 0.82 +/- 0.17 | 0.64 +/- 0.11 | 8.33 +/- 1.66  | **0.14 +/- 0.09**  |
| 10^3  | 0.74 +/- 0.09 | 0.64 +/- 0.08 | 19.5 +/- 4.4   | **0.06 +/- 0.03**  |
| 10^4  | 1.72 +/- 0.24 | 1.35 +/- 0.45 | 324 +/- 171    | **0.31 +/- 0.28**  |

**Results - Continuous rho interpolation (L2 Error x 10^-4)**:

| Domain     | standard-ANN   | wave-ANN       | standard-PINN  | wave-PINN (best) |
|------------|----------------|----------------|----------------|------------------|
| 10^2-10^3  | 110 +/- 51     | 1.83 +/- 1.54  | 5.16 +/- 0.66  | **1.46 +/- 1.79**  |
| 10^3-10^4  | 559 +/- 101    | 3.21 +/- 2.96  | 173 +/- 23     | **1.83 +/- 1.27**  |
| 10^2-10^4  | 1354 +/- 120   | 2.15 +/- 2.23  | 363 +/- 45     | **1.30 +/- 1.69**  |

**Key finding**: wave-PINN with lambda=1 consistently achieves the lowest error, even outperforming data-driven ANNs. Standard PINNs degrade dramatically for sharp (high rho) waves.

---

### Paper C: Residual Weighted PINN for Reaction-Diffusion (2025)

**Title**: "A residual weighted physics informed neural network for forward and inverse problems of reaction diffusion equations"
**arXiv ID**: 2504.07058

**Methods compared**: PINN, RWa-PINN, RWb-PINN

**Metric**: Training Error (epsilon_T) and Generalization Error (epsilon_G)

**Results - 1D Extended Fisher-Kolmogorov (gamma=0.001)**:

| Method    | Training Error | Generalization Error |
|-----------|---------------|---------------------|
| PINN      | 0.00080       | 1.8 x 10^-4         |
| RWa PINN  | 0.00055       | 3.5 x 10^-5         |
| RWb PINN  | 0.00059       | 3.8 x 10^-5         |

**Results - 2D Extended Fisher-Kolmogorov (gamma=0.01)**:

| Method    | Training Error | Generalization Error |
|-----------|---------------|---------------------|
| PINN      | 0.0024        | 1.1 x 10^-3         |
| RWa PINN  | 0.0011        | 5.3 x 10^-4         |
| RWb PINN  | 0.0013        | 4.6 x 10^-4         |

**Key finding**: Residual weighting reduces generalization error by 4-5x on Fisher-Kolmogorov equations.

---

### Paper D: APEBench (NeurIPS 2024)

**Title**: "APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs"
**arXiv ID**: 2411.00180

Fisher-KPP is one of 46 PDEs in the APEBench suite. Results are presented as figures (not numerical tables in accessible HTML) comparing FNO, ResNet, U-Net, and hybrid architectures. Key qualitative findings:
- **ResNet and local convolutional architectures** perform best on reaction-diffusion PDEs (including Fisher-KPP)
- **FNO was the least suitable** architecture for reaction-diffusion due to its low-frequency bias
- Supervised unrolling consistently improved ResNet performance
- FNO performance was almost unaffected by unrolling (likely due to global receptive field)

*Note: Exact nRMSE values are in the paper's figures (not extractable via web scraping). The paper uses mean nRMSE over rollout as primary metric.*

---

## 2. 2D Advection-Diffusion / Diffusion-Reaction Equation

**Equation**: u_t + a*u_x + b*u_y = nu*(u_xx + u_yy) and related diffusion-reaction forms

---

### Paper E: SpecB-FNO / Understanding Fourier Neural Operators from Spectral Perspective (2024)

**Title**: "Toward a Better Understanding of Fourier Neural Operators from a Spectral Perspective"
**arXiv ID**: 2404.07200

**Methods compared**: DeepONet, ResNet, U-Net, CNO, FNO, FFNO, SpecB-FNO

**Metric**: NRMSE (Normalized Root Mean Square Error)

**Experimental Setup**: Uses PDEBench datasets; 2D Diffusion-Reaction (Fitzhugh-Nagumo equation, coupled activator-inhibitor dynamics)

**Results - NRMSE on Multiple PDEs**:

| Model       | Darcy Flow      | NS (nu=1e-3)   | NS (nu=1e-5)   | Shallow Water   | **Diffusion-Reaction** |
|-------------|-----------------|-----------------|-----------------|-----------------|------------------------|
| DeepONet    | .0428 +/- .0007 | .0716 +/- .0018 | .2484 +/- .0027 | .1576 +/- .0216 | NaN                    |
| ResNet      | .2455 +/- .0011 | .9946 +/- .2337 | .3926 +/- .0007 | 1.501 +/- .1519 | **.0138 +/- .0016**    |
| U-Net       | .0098 +/- .0005 | .1105 +/- .0547 | .1334 +/- .0071 | 2.088 +/- .2135 | .1160 +/- .0068        |
| CNO         | .0075 +/- .0014 | .0512 +/- .0017 | .1203 +/- .0072 | .0326 +/- .0021 | .0257 +/- .0088        |
| FNO         | .0067 +/- .0001 | .0039 +/- .0004 | .0576 +/- .0004 | .0050 +/- .0001 | **.0190 +/- .0003**    |
| FFNO        | .0096 +/- .0001 | .0317 +/- .0023 | .1499 +/- .0219 | .0540 +/- .0119 | .0072 +/- .0001        |
| **SpecB-FNO** | **.0036 +/- .0002** | **.0014 +/- .0001** | **.0351 +/- .0018** | **.0004 +/- .0002** | **.0066 +/- .0003**    |

**Key finding for Diffusion-Reaction**: ResNet (0.0138) and SpecB-FNO (0.0066) lead; standard FNO achieves 0.0190. U-Net (0.1160) is much worse on this task. DeepONet failed (NaN).

---

### Paper F: PDEBench (NeurIPS 2022)

**Title**: "PDEBench: An Extensive Benchmark for Scientific Machine Learning"
**arXiv ID**: 2210.07182

**Methods compared**: FNO, U-Net, (PINN partial)

**Metric**: RMSE, nRMSE, cRMSE, bRMSE, fRMSE (low/mid/high frequency)

**PDEs tested**: 1D Advection, 1D Burgers, 1D/2D Diffusion-Reaction, 1D Diffusion-Sorption, 1D/2D/3D Compressible NS, 2D Darcy Flow, 2D Shallow Water

**Key reported numbers** (from various citing papers):
- 1D Advection: FNO RMSE = 0.00530 (baseline)
- FNO has consistent error of ~4 x 10^-4 across frequency spectrum for many problems
- 2D Diffusion-Reaction: resolution 128^2 x 20 time steps

*Note: Full tables are in PDF (not extractable via web). Qualitative finding: FNO provides best predictions for most metrics; U-Net sometimes better for Darcy Flow.*

---

### Paper G: Self-supervised Pretraining for PDEs (2024)

**Title**: "Self-supervised Pretraining for Partial Differential Equations"
**arXiv ID**: 2407.06209

**Methods compared**: FNO (single-task), FNO (multi-task), ViT/PDE-T

**Metric**: MSE (Mean Squared Error)

**Results (MSE, PDEBench problems)**:

| System           | FNO [single] | FNO [multi]  | ViT (PDE-T)  |
|------------------|-------------|-------------|--------------|
| 1D Advection     | 1.08E-04    | 1.02E-04    | 3.73E-03     |
| 1D Burgers       | 7.72E-02    | 8.40E-06    | 6.16E-04     |
| 2D Navier-Stokes | 2.96E-04    | 3.26E-04    | 2.81E-01     |

**Key finding**: Multi-task FNO drastically improves Burgers (7.72e-2 -> 8.40e-6) but FNO remains strong on NS and Advection.

---

### Paper H: PINO - Physics-Informed Neural Operator (2021/2024)

**Title**: "Physics-Informed Neural Operator for Learning Partial Differential Equations"
**arXiv ID**: 2111.03794 (published in ACM/IMS Journal of Data Science, 2024)

**Methods compared**: PINO, FNO, PI-DeepONet, DeepONet

**Metric**: Relative L2 Error (%)

**Key reported numbers**:

| PDE               | PINO    | PI-DeepONet | Notes                              |
|-------------------|---------|-------------|-------------------------------------|
| Burgers Equation  | 0.38%   | 1.38%       | PINO 3.6x better                    |
| Darcy Flow        | < FNO   | -           | PINO outperforms FNO with physics   |
| NS (transient)    | ~7% lower than FNO | - | 400x speedup vs spectral solver |
| NS (Kolmogorov)   | ~7% lower than FNO | - | Matches FNO speed                |

**Key finding**: With physics constraints, PINO requires fewer/no training data and generalizes better than data-driven FNO.

---

### Paper I: PINTO - Physics-Informed Transformer Neural Operator (2024)

**Title**: "PINTO: Physics-informed transformer neural operator for learning generalized solutions of PDEs for any IC and BC"
**arXiv ID**: 2412.09009

**Methods compared**: PINTO vs PI-DeepONet

**Metric**: Mean Relative Error (%)

| Test Case         | PINTO (seen)       | PINTO (unseen)     | PI-DeepONet (seen) | PI-DeepONet (unseen) |
|-------------------|--------------------|--------------------|--------------------|-----------------------|
| 1D Advection      | 2.11% (std 4.01%)  | 2.85% (std 4.73%)  | 1.35% (std 3.75%)  | 11.26% (std 11.42%)   |
| 1D Burgers        | 4.81% (std 4.43%)  | 5.24% (std 4.51%)  | 12.81% (std 11.85%)| 15.03% (std 10.78%)   |
| Kovasznay Flow    | 0.037% (std 0.033%)| 0.41% (std 2.55%)  | 0.08% (std 0.066%) | 2.26% (std 6.54%)     |
| Beltrami Flow     | 0.53% (std 0.9%)   | 0.6% (std 0.92%)   | 2.62% (std 4.19%)  | 4.89% (std 12.14%)    |
| Lid Driven Cavity | 1.36% (std 1.44%)  | 2.78% (std 2.49%)  | 1.96% (std 2.31%)  | 6.08% (std 6.61%)     |

**Key finding**: PINTO generalizes much better to unseen conditions (2-5x lower error than PI-DeepONet on unseen cases).

---

### Paper J: P2INNs - Parameterized PINNs for CDR Equations (2024)

**Title**: "Parameterized Physics-informed Neural Networks for Parameterized PDEs"
**arXiv ID**: 2408.09446

**Methods compared**: PINN vs P2INN (parameterized PINN)

**Metric**: Absolute L2 Error

**Results on Convection-Diffusion-Reaction (CDR) equations**:

| PDE Type               | PINN Abs. Error | P2INN Abs. Error | Improvement |
|------------------------|-----------------|------------------|-------------|
| Convection             | 0.0496          | 0.0330           | 33.43%      |
| Diffusion              | 0.3611          | 0.1592           | 55.91%      |
| Reaction               | 0.5825          | 0.0041           | 99.30%      |
| Convection-Diffusion   | 0.1493          | 0.0532           | 64.34%      |
| Reaction-Diffusion     | 0.4744          | 0.1319           | 72.21%      |
| Conv.-Diff.-Reaction   | 0.4811          | 0.0391           | 91.88%      |

**Extended results (parameter range 1-20)**:
- Reaction: P2INN relative error 0.0092 vs PINN 0.4932
- Conv.-Diff.-Reac.: P2INN abs error 0.0353 vs PINN 0.4080

---

### Paper K: WHNO - Walsh-Hadamard Neural Operator (2025)

**Title**: "Walsh-Hadamard Neural Operators for Solving PDEs with Discontinuous Coefficients"
**arXiv ID**: 2511.07347

**Methods compared**: WHNO vs FNO

**Metric**: MSE, MAE, Mean Relative Error

**Results**:

| Problem        | WHNO MSE           | FNO MSE            | WHNO Advantage |
|----------------|--------------------|--------------------|----------------|
| Heat Conduction| 0.000113           | 0.000148           | 24% lower      |
| Burgers        | 0.000113 +/- 4.9e-5| 0.000148 +/- 9.2e-5| 24% lower      |
| Darcy Flow     | 0.88% rel. error   | > 0.88%            | -              |

---

### Paper L: FC-PINO (2022)

**Title**: "FC-PINO: High-Precision Physics-Informed Neural Operators via Fourier Continuation"
**arXiv ID**: 2211.15960

**Methods compared**: FC-PINO (Gram/Legendre), Pad-PINO, Standard PINO

**Metric**: PDE Residual Loss

**Results - 1D Burgers (self-similar, lambda=0.5)**:

| Model                  | PDE Residual | BC Loss    | Total Loss |
|------------------------|-------------|------------|------------|
| FC-Gram FC-PINO        | 1.9e-11     | 9.2e-15    | 6.7e-10    |
| FC-Legendre FC-PINO    | 1.8e-12     | 1.9e-14    | 2.8e-09    |
| Pad-PINO               | 1.4e-08     | 1.1e-10    | 4.9e-06    |
| Standard PINO          | 2.0e+00     | 2.0e-06    | 2.0e+00    |

**Key finding**: FC-PINO achieves machine-precision accuracy (10^-11) vs standard PINO (O(1) error).

---

## Summary: Best Known Error Ranges

### Fisher-KPP (1D):
| Approach              | Best Reported Error | Reference |
|-----------------------|---------------------|-----------|
| FDM (vs exact)        | 1.42 x 10^-4 (rel L2) | Paper A |
| PINN (best case)      | 5.57 x 10^-2 (rel L2) | Paper A |
| wave-PINN             | 0.06 x 10^-4 (L2)     | Paper B (rho=10^3) |
| RW-PINN (EFK)         | 3.5 x 10^-5 (gen. err) | Paper C |
| Autoregressive (ResNet)| Best among neural (qualitative) | Paper D |

### Advection-Diffusion / Diffusion-Reaction (2D):
| Approach              | Best Reported Error | Problem Type | Reference |
|-----------------------|---------------------|-------------|-----------|
| SpecB-FNO             | 0.0066 nRMSE        | 2D Diff-Reac | Paper E |
| FFNO                  | 0.0072 nRMSE        | 2D Diff-Reac | Paper E |
| ResNet                | 0.0138 nRMSE        | 2D Diff-Reac | Paper E |
| FNO                   | 0.0190 nRMSE        | 2D Diff-Reac | Paper E |
| FNO (PDEBench)        | 0.00530 RMSE        | 1D Advection | Paper F |
| FNO (MSE)             | 1.08e-4 MSE         | 1D Advection | Paper G |
| PINO                  | 0.38% rel error      | 1D Burgers   | Paper H |
| PINTO                 | 2.11% mean rel err   | 1D Advection | Paper I |
| P2INN                 | 0.0532 abs L2        | Conv-Diff    | Paper J |
