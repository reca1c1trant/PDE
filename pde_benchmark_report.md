# PDE Benchmark Problems with Analytical Solutions for Neural PDE Solvers

## Overview

This report catalogs **13 PDE benchmark problems** with analytical/exact solutions, sourced from recent papers (2023-2026) published in top venues (NeurIPS, ICML, ICLR, IJCAI, IMA Journal). Each problem can be used to generate datasets independently and has been used to benchmark neural PDE solvers such as FNO, TFNO, UNet, DeepONet, PINN, etc.

---

## 1. 1D Viscous Burgers Equation

**Source Paper:** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (Raissi et al., Journal of Computational Physics, 2019; widely used as baseline in NeurIPS 2021/2022/2024 papers)
- Used in: PINNacle (NeurIPS 2024), PDEBench (NeurIPS 2022), RoPINN (NeurIPS 2024), "Characterizing failure modes in PINNs" (NeurIPS 2021)
- Paper URLs:
  - https://arxiv.org/abs/1711.10561
  - https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf
  - https://arxiv.org/abs/2306.08827

**PDE Equation:**
```
u_t + u * u_x = ν * u_xx
```

**Domain:** x ∈ [-1, 1], t ∈ [0, 1]

**Boundary Conditions:** Dirichlet: u(t, -1) = u(t, 1) = 0

**Initial Condition:** u(0, x) = -sin(πx)

**Parameters:** ν = 0.01/π (Raissi), or ν = 0.001, 0.01, 0.1 (PDEBench)

**Analytical Solution (via Cole-Hopf Transform):**
```
u(x, t) = -2ν * (∂/∂x) ln φ(x, t)

where φ(x, t) = (1/√(4πνt)) ∫_{-∞}^{∞} exp[-(x-ξ)²/(4νt) - (1/(2ν))∫_0^ξ u₀(η)dη] dξ

For u₀(x) = -sin(πx), this can be computed via Fourier-Bessel series:
φ(x,t) = Σ_{n=-∞}^{∞} aₙ(t) exp(inπx)
```
In practice, high-resolution numerical solutions serve as ground truth, but the Cole-Hopf transform gives the exact solution.

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, FNO, U-Net, PINO, DeepONet, KAN, PINNsFormer

---

## 2. 1D Advection Equation

**Source Paper:** PDEBench (NeurIPS 2022); also used in "Characterizing failure modes in PINNs" (NeurIPS 2021)
- Paper URLs:
  - https://arxiv.org/abs/2210.07182
  - https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf

**PDE Equation:**
```
u_t + β * u_x = 0
```

**Domain:** x ∈ [0, 2π], t ∈ [0, T]

**Boundary Conditions:** Periodic: u(t, 0) = u(t, 2π)

**Initial Condition:** u(0, x) = sin(x) (or other smooth functions)

**Parameters:** β = 1, 10, 30, 40 (NeurIPS 2021 failure modes paper tests increasingly difficult β)

**Analytical Solution:**
```
u(x, t) = sin(x - β*t)
```
(More generally: u(x, t) = u₀(x - βt) for any initial condition u₀)

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, FNO, U-Net, causal PINN, sequence-to-sequence PINN

---

## 3. 1D Reaction Equation

**Source Paper:** "Characterizing possible failure modes in PINNs" (NeurIPS 2021); RoPINN (NeurIPS 2024)
- Paper URLs:
  - https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf
  - https://arxiv.org/abs/2405.14369

**PDE Equation:**
```
u_t = ρ * u * (1 - u)
```

**Domain:** x ∈ [0, 2π], t ∈ [0, T]

**Boundary Conditions:** Periodic

**Initial Condition:** u(0, x) = sin(x) (shifted/scaled to be in [0,1] as needed)

**Parameters:** ρ = 1, 5, 10 (higher ρ → sharper transitions, harder for PINNs)

**Analytical Solution:**
```
u(x, t) = u₀(x) * exp(ρt) / (1 - u₀(x) + u₀(x) * exp(ρt))

where u₀(x) = (sin(x) + 1) / 2  (mapped to [0,1])
```
This is the logistic growth ODE solution applied pointwise.

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, KAN, PINNsFormer, RoPINN, causal PINN

---

## 4. 1D Diffusion (Heat) Equation

**Source Paper:** PDEBench (NeurIPS 2022); PINNacle (NeurIPS 2024); SciML Benchmarks
- Paper URLs:
  - https://arxiv.org/abs/2210.07182
  - https://arxiv.org/abs/2306.08827
  - https://docs.sciml.ai/SciMLBenchmarksOutput/stable/PINNOptimizers/1d_diffusion/

**PDE Equation:**
```
u_t = ν * u_xx
```

**Domain:** x ∈ [0, L], t ∈ [0, T] (commonly L=1 or L=2π)

**Boundary Conditions:** Dirichlet: u(t, 0) = u(t, L) = 0

**Initial Condition:** u(0, x) = sin(nπx/L)

**Parameters:** ν = diffusion coefficient (e.g., ν = 1, 0.5, 0.1); n = mode number

**Analytical Solution:**
```
u(x, t) = exp(-n²π²νt/L²) * sin(nπx/L)

General (superposition):
u(x, t) = Σ_n Bₙ exp(-n²π²νt/L²) sin(nπx/L)
where Bₙ are Fourier coefficients of the initial condition.
```

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, FNO, U-Net, DeepONet, DRM, WAN

---

## 5. 1D Wave Equation

**Source Paper:** PINNacle (NeurIPS 2024); RoPINN (NeurIPS 2024)
- Paper URLs:
  - https://arxiv.org/abs/2306.08827
  - https://arxiv.org/abs/2405.14369

**PDE Equation:**
```
u_tt = c² * u_xx
```

**Domain:** x ∈ [0, L], t ∈ [0, T]

**Boundary Conditions:** Dirichlet: u(t, 0) = u(t, L) = 0

**Initial Conditions:** u(0, x) = sin(πx/L), u_t(0, x) = 0

**Parameters:** c = wave speed (e.g., c = 1, 2)

**Analytical Solution (d'Alembert / Separation of Variables):**
```
u(x, t) = cos(cπt/L) * sin(πx/L)

General:
u(x, t) = Σ_n [Aₙ cos(nπct/L) + Bₙ sin(nπct/L)] sin(nπx/L)
```

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, KAN, PINNsFormer, RoPINN, causal PINN

---

## 6. 2D Poisson Equation

**Source Paper:** "A hands-on introduction to PINNs for solving PDEs" (arXiv 2024); PINNacle (NeurIPS 2024); "Can PINNs beat FEM?" (IMA J. Appl. Math. 2024)
- Paper URLs:
  - https://arxiv.org/html/2403.00599v1
  - https://arxiv.org/abs/2306.08827
  - https://arxiv.org/abs/2302.04107

**PDE Equation:**
```
Δu = u_xx + u_yy = f(x, y)
```

**Domain:** Ω = [0, 1]² or [-1, 1]²

**Multiple test cases with different analytical solutions:**

**Case (a) - Exponential:**
```
f(x,y) = (x² + y²) * exp(xy)
u(x,y) = exp(xy)
BC: Dirichlet, matching exact solution on boundary
```

**Case (b) - Mixed:**
```
f(x,y) = 1
u(x,y) = exp(kx)*sin(ky) + (1/4)(x² + y²)
BC: Dirichlet
```

**Case (c) - Sinh:**
```
f(x,y) = sinh(x)
u(x,y) = sinh(x)
BC: Neumann
```

**Case (d) - Gaussian-like:**
```
f(x,y) = 4(x² + y² + 1) * exp(x² + y²)
u(x,y) = exp(x² + y²)
BC: Mixed Neumann-Dirichlet
```

**Parameters:** Various BC types tested (Dirichlet, Neumann, mixed, Cauchy)

**Dimensionality:** 2D (steady-state)

**Baselines Compared:** PINN, DRM, WAN, DFLM, RFM, DFVM (PDENNEval, IJCAI 2024), FNO, U-Net, DeepONet (PDEBench)

---

## 7. 2D Laplace Equation

**Source Paper:** "A hands-on introduction to PINNs" (arXiv 2024)
- Paper URL: https://arxiv.org/html/2403.00599v1

**PDE Equation:**
```
Δu = u_xx + u_yy = 0
```

**Domain:** Ω = [-1, 1]²

**Boundary Conditions:** Dirichlet, matching exact solution values

**Analytical Solution:**
```
u(x, y) = x² - y²
```
(harmonic polynomial)

**Dimensionality:** 2D (steady-state)

**Baselines Compared:** PINN variants with different BC enforcement strategies

---

## 8. 2D Helmholtz Equation

**Source Paper:** "A hands-on introduction to PINNs" (arXiv 2024); PINNacle (NeurIPS 2024)
- Paper URLs:
  - https://arxiv.org/html/2403.00599v1
  - https://arxiv.org/abs/2306.08827

**PDE Equation:**
```
Δu + k²u = 0    (or with source: Δu + k²u = f)
```

**Domain:** Ω = [-L/2, L/2]² (rectangular domain)

**Boundary Conditions:** Dirichlet on all four boundaries

**Analytical Solution (magnetic arcade / standing wave):**
```
u(x, z) = Σ_{k=1}^{3} exp(-νz) * aₖ * cos(kπx/L)

where ν² = k²π²/L² - c²
```

**Parameters:** k = wavenumber, c = constant

**Dimensionality:** 2D (steady-state)

**Baselines Compared:** PINN, PINN with loss reweighting, PINN with adaptive sampling

---

## 9. 3D Helmholtz Equation

**Source Paper:** "Feasibility study on solving the Helmholtz equation in 3D with PINNs" (arXiv 2024)
- Paper URL: https://arxiv.org/abs/2403.06623

**PDE Equation:**
```
Δp + k²p = f(x, y, z)

where Δ = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
```

**Domain:** Ω = [0, 1]³ (unit cube)

**Test Case 1 (Dirichlet BC):**
```
f(x,y,z) = 2k² cos(kx) cos(ky) cos(kz)
p(x,y,z) = cos(kx) cos(ky) cos(kz)

BC: Homogeneous Dirichlet (p = 0 on ∂Ω)
k = 4π (wavelength λ = 1/2)
```

**Test Case 2 (Neumann BC):**
```
f(x,y,z) = 2k² sin(kx) sin(ky) sin(kz)
p(x,y,z) = sin(kx) sin(ky) sin(kz)

BC: Homogeneous Neumann (∇p · n = 0 on ∂Ω)
k = 4π
```

**Parameters:** k = 2π/λ, wavelength λ, speed of sound c

**Dimensionality:** 3D (steady-state)

**Baselines Compared:** PINN (DeepXDE framework), FEM (openCFS)

---

## 10. 2D Taylor-Green Vortex (Navier-Stokes with Exact Solution)

**Source Paper:** PINO (NeurIPS 2024 / JDS 2024); NSFnets (JCP 2021); Multiple PINN benchmarks
- Paper URLs:
  - https://arxiv.org/abs/2111.03794
  - https://www.sciencedirect.com/science/article/abs/pii/S0021999120307257

**PDE Equation (Incompressible Navier-Stokes):**
```
u_t + (u·∇)u = -∇p + ν Δu
∇·u = 0
```

**Domain:** Ω = [0, 2π]² (or [-1, 1]²), t ∈ [0, T]

**Boundary Conditions:** Periodic

**Analytical Solution:**
```
u(x, y, t) = -cos(πx) sin(πy) exp(-2π²νt)
v(x, y, t) =  sin(πx) cos(πy) exp(-2π²νt)
p(x, y, t) = -(1/4)(cos(2πx) + cos(2πy)) exp(-4π²νt)
```

**Parameters:** ν = kinematic viscosity (e.g., ν = 0.01, 0.001); Re = 1/ν

**Dimensionality:** 2D + time

**Baselines Compared:** PINN, FNO, PINO, U-Net, DeepONet, FEM

---

## 11. 2D Kovasznay Flow (Steady Navier-Stokes)

**Source Paper:** NSFnets (JCP 2021); Multiple PINN benchmarks; KAN-PINN (2024)
- Paper URLs:
  - https://www.sciencedirect.com/science/article/abs/pii/S0021999120307257
  - https://arxiv.org/html/2411.04516v1

**PDE Equation (Steady Incompressible Navier-Stokes):**
```
(u·∇)u = -∇p + (1/Re) Δu
∇·u = 0
```

**Domain:** Ω = [-0.5, 1.0] × [-0.5, 1.5] (or similar rectangular domain)

**Boundary Conditions:** Dirichlet, matching exact solution

**Analytical Solution (Kovasznay, 1948):**
```
u(x, y) = 1 - exp(λx) cos(2πy)
v(x, y) = (λ/(2π)) exp(λx) sin(2πy)
p(x, y) = (1/2)(1 - exp(2λx))

where λ = Re/2 - √((Re/2)² + 4π²)
```

**Parameters:** Re = 20, 40, 100 (Reynolds number)

**Dimensionality:** 2D (steady-state)

**Baselines Compared:** PINN, NSFnets (VP and VV formulations), ChebPIKAN, KAN

---

## 12. 3D Beltrami Flow (Unsteady Navier-Stokes)

**Source Paper:** NSFnets (JCP 2021); Ethier & Steinman (IJNMF 1994, widely used in 2024 PINN benchmarks)
- Paper URLs:
  - https://www.sciencedirect.com/science/article/abs/pii/S0021999120307257
  - https://pinnx.readthedocs.io/unit-examples-forward/Beltrami_flow.html

**PDE Equation (3D Incompressible Navier-Stokes):**
```
ρ[u_t + (u·∇)u] = -∇p + μ Δu
∇·u = 0
```

**Domain:** [-1, 1]³, t ∈ [0, 1]

**Boundary Conditions:** Dirichlet, matching exact solution

**Analytical Solution (Ethier-Steinman):**
```
u(x,y,z,t) = -a[exp(ax)sin(ay+dz) + exp(az)cos(ax+dy)] exp(-d²t)
v(x,y,z,t) = -a[exp(ay)sin(az+dx) + exp(ax)cos(ay+dz)] exp(-d²t)
w(x,y,z,t) = -a[exp(az)sin(ax+dy) + exp(ay)cos(az+dx)] exp(-d²t)

p(x,y,z,t) = -(a²/2)[exp(2ax) + exp(2ay) + exp(2az)
  + 2sin(ax+dy)cos(az+dx)exp(a(y+z))
  + 2sin(ay+dz)cos(ax+dy)exp(a(z+x))
  + 2sin(az+dx)cos(ay+dz)exp(a(x+y))] exp(-2d²t)
```

**Parameters:** a = 1, d = 1, Re = 1, ρ = 1, μ = 1

**Dimensionality:** 3D + time

**Baselines Compared:** PINN, NSFnets, EPINN, FEM

---

## 13. 1D Allen-Cahn Equation

**Source Paper:** "Can PINNs beat FEM?" (IMA J. Appl. Math. 2024); "Benchmark problems for PINNs: Allen-Cahn" (AIMS Math. 2025); PINNacle (NeurIPS 2024)
- Paper URLs:
  - https://arxiv.org/abs/2302.04107
  - https://www.aimspress.com/article/doi/10.3934/math.2025335
  - https://arxiv.org/abs/2306.08827

**PDE Equation:**
```
u_t = ε² u_xx + u - u³

(equivalently: u_t = ε² u_xx - f'(u), where f(u) = (1/4)(u²-1)²)
```

**Domain:** x ∈ [-1, 1], t ∈ [0, T]

**Boundary Conditions:** Dirichlet or periodic

**Analytical Solution (Traveling Wave / Heteroclinic):**
```
u(x, t) = tanh((x - ct)/(ε√2))

where c is the wave speed (c = 0 for symmetric double-well)
```
For the stationary kink: u(x) = tanh(x/(ε√2))

Note: For general initial conditions, exact closed-form solutions are not available; high-resolution numerical solutions (FEM on fine mesh) serve as ground truth.

**Parameters:** ε = 0.01, 0.1 (interface width); smaller ε → sharper interface → harder problem

**Dimensionality:** 1D + time

**Baselines Compared:** PINN, FEM, energy-dissipation-preserving PINN, causal PINN

---

## Summary Table

| # | PDE | Dim | Has Analytical Solution | Key Venues | Key Baselines |
|---|-----|-----|------------------------|------------|---------------|
| 1 | 1D Viscous Burgers | 1D+t | Yes (Cole-Hopf) | NeurIPS 2021/2022/2024 | PINN, FNO, U-Net, DeepONet, PINO |
| 2 | 1D Advection | 1D+t | Yes (exact) | NeurIPS 2021/2022 | PINN, FNO, U-Net |
| 3 | 1D Reaction | 1D+t | Yes (logistic) | NeurIPS 2021/2024 | PINN, KAN, PINNsFormer, RoPINN |
| 4 | 1D Diffusion/Heat | 1D+t | Yes (Fourier series) | NeurIPS 2022/2024 | PINN, FNO, U-Net, DeepONet |
| 5 | 1D Wave | 1D+t | Yes (d'Alembert) | NeurIPS 2024 | PINN, KAN, PINNsFormer, RoPINN |
| 6 | 2D Poisson | 2D | Yes (manufactured) | NeurIPS 2024, IJCAI 2024 | PINN, DRM, WAN, FNO, U-Net, DeepONet |
| 7 | 2D Laplace | 2D | Yes (harmonic) | arXiv 2024 | PINN variants |
| 8 | 2D Helmholtz | 2D | Yes (series) | NeurIPS 2024 | PINN variants |
| 9 | 3D Helmholtz | 3D | Yes (cos/sin products) | arXiv 2024 | PINN, FEM |
| 10 | 2D Taylor-Green Vortex | 2D+t | Yes (exponential decay) | NeurIPS 2024, JDS 2024 | PINN, FNO, PINO, U-Net, DeepONet |
| 11 | 2D Kovasznay Flow | 2D | Yes (Kovasznay 1948) | JCP 2021, multiple 2024 | PINN, NSFnets, KAN |
| 12 | 3D Beltrami Flow | 3D+t | Yes (Ethier-Steinman) | JCP 2021, multiple 2024 | PINN, NSFnets, EPINN, FEM |
| 13 | 1D Allen-Cahn | 1D+t | Partial (traveling wave) | IMA 2024, NeurIPS 2024 | PINN, FEM |

---

## Key Benchmark Papers Referenced

1. **PINNacle** (NeurIPS 2024) - 22 PDE tasks, 10 PINN methods
   - https://arxiv.org/abs/2306.08827
   - https://github.com/i207M/PINNacle

2. **RoPINN** (NeurIPS 2024) - 19 PDE tasks, region optimization
   - https://arxiv.org/abs/2405.14369
   - https://github.com/thuml/RoPINN

3. **PDEBench** (NeurIPS 2022) - 10 PDE families (1D/2D/3D)
   - https://arxiv.org/abs/2210.07182
   - https://github.com/pdebench/PDEBench

4. **APEBench** (NeurIPS 2024) - 46+ PDEs in 1D/2D/3D, autoregressive
   - https://arxiv.org/abs/2411.00180
   - https://github.com/tum-pbs/apebench

5. **PDENNEval** (IJCAI 2024) - 19 PDE tasks, 12 NN methods
   - https://www.ijcai.org/proceedings/2024/573
   - https://github.com/Sysuzqs/PDENNEval

6. **Characterizing Failure Modes in PINNs** (NeurIPS 2021) - convection/reaction/diffusion
   - https://arxiv.org/abs/2109.01050
   - https://github.com/a1k12/characterizing-pinns-failure-modes

7. **Can PINNs Beat FEM?** (IMA J. Appl. Math. 2024) - Poisson 1D/2D/3D, Allen-Cahn, Schrodinger
   - https://arxiv.org/abs/2302.04107
   - https://github.com/TamaraGrossmann/FEM-vs-PINNs

8. **NSFnets** (JCP 2021) - Kovasznay, Taylor-Green, Beltrami flows
   - https://www.sciencedirect.com/science/article/abs/pii/S0021999120307257

9. **PINO** (JDS/NeurIPS 2024) - Physics-Informed Neural Operator
   - https://arxiv.org/abs/2111.03794
   - https://github.com/neuraloperator/physics_informed

10. **3D Helmholtz with PINNs** (arXiv 2024)
    - https://arxiv.org/abs/2403.06623

---

## Recommendations for Dataset Generation

### Easy to generate (exact closed-form solutions):
- **1D Advection** (#2): Trivial - just shift the initial condition
- **1D Reaction** (#3): Pointwise logistic ODE solution
- **1D Diffusion** (#4): Fourier series with exponential decay
- **1D Wave** (#5): Superposition of standing waves
- **2D Poisson** (#6): Manufacture solutions by choosing u, computing f = Δu
- **2D Laplace** (#7): Harmonic polynomials
- **2D Taylor-Green** (#10): Fully explicit formula
- **2D Kovasznay** (#11): Fully explicit formula
- **3D Helmholtz** (#9): Fully explicit formula
- **3D Beltrami** (#12): Fully explicit formula

### Requires numerical integration:
- **1D Burgers** (#1): Cole-Hopf gives exact solution but requires integral evaluation
- **1D Allen-Cahn** (#13): Exact for traveling wave only; general IC needs numerics

### Dataset generation strategy:
1. Choose a range of PDE parameters (viscosity, wave speed, etc.)
2. Generate initial conditions (fixed or random from a distribution)
3. Compute exact solutions on a fine grid
4. Optionally add noise for robustness testing
5. Store as (parameter, IC, solution) triplets for operator learning, or (x, t, u) for PINN benchmarking
