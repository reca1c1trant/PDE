# PDE Benchmark Problems with Exact Analytical Solutions

> Last updated: 2026-03-21
> Goal: Identify 12+ time-dependent PDEs with closed-form exact solutions for ML benchmarking

---

## Existing Problems (8 total, already selected)

| # | PDE | Dim | Type |
|---|-----|-----|------|
| 1 | 1D Burgers (Cole-Hopf) | 1D+t | Nonlinear |
| 2 | 1D Advection | 1D+t | Linear |
| 3 | 2D Taylor-Green Vortex | 2D+t | Nonlinear (NS) |
| 4 | 3D Beltrami/Ethier-Steinman | 3D+t | Nonlinear (NS) |
| 5 | 1D Allen-Cahn (traveling wave tanh) | 1D+t | Nonlinear |
| 6 | 1D KdV (IST, N-soliton) | 1D+t | Nonlinear |
| 7 | 2D KP-II (IST extension) | 2D+t | Nonlinear |
| 8 | 2D Wave equation (d'Alembert/Fourier) | 2D+t | Linear |

---

## NEW Problem #9: 1D Nonlinear Schrodinger Equation (NLS) — Bright Soliton

### PDE Equation
```
i ψ_t + ψ_xx + 2|ψ|² ψ = 0
```
(focusing NLS, cubic nonlinearity; can also write as `i u_t + (1/2) u_xx + |u|^2 u = 0`)

### Exact Analytical Solution (1-soliton)
```
ψ(x,t) = η · sech(η(x - 2ξt - x₀)) · exp(-i[ξx + (ξ² - η²)t + θ₀])
```
where:
- `η` = amplitude parameter (real, > 0)
- `ξ` = velocity parameter (real)
- `x₀` = initial position
- `θ₀` = initial phase

**N-soliton solutions** also available via inverse scattering transform (IST), Zakharov & Shabat (1972).

**Breather / rogue wave solutions** are also exactly known (Peregrine soliton, Akhmediev breather, Kuznetsov-Ma breather).

### Domain, BCs, Dimensionality
- **Domain**: x ∈ [-L, L], t ∈ [0, T] (typically L=20-50, T=5-20π)
- **BCs**: Periodic or decaying (soliton → 0 at boundaries)
- **Dim**: 1D+t (complex-valued, so effectively 2 real channels)
- **Note**: The output is complex-valued; for ML benchmarks, split into real/imaginary or modulus/phase

### Parameters
- `η` (amplitude): typically 1.0
- `ξ` (velocity): typically 0.0 (stationary) or varied
- Can vary η, ξ to generate parametric families

### ML Papers & Baselines
- **FNO**: Yan et al. (2022, Chaos Solitons Fractals) — FNO for fractional NLS soliton mappings
- **PINN**: Raissi et al. (2019, JCP) — original PINN paper uses NLS as benchmark; multiple PINN variants since
- **PINN variants**: Mix-training PINN (2022), CV-PINN (2025, Phys Rev Research), SCTgNN (2024), phPINN (2024)
- **Neural operator**: Data-driven soliton parameter discovery via FNO
- **Baseline count**: ≥3 (FNO, PINN, DeepONet variants)
- **Status**: Well-established benchmark in PINN community; FNO/DeepONet usage exists but fewer multi-baseline operator comparisons

### Data Self-Generation
**YES** — trivially computed from the closed-form soliton formula. N-soliton solutions can be computed via Darboux transform or explicit formulas.

---

## NEW Problem #10: 1D Sine-Gordon Equation — Kink Soliton & Breather

### PDE Equation
```
u_tt - u_xx + sin(u) = 0
```
(relativistic, Lorentz-invariant nonlinear wave equation)

### Exact Analytical Solutions

**1-Soliton (Kink):**
```
u(x,t) = 4 arctan(exp(± (x - βt) / √(1 - β²)))
```
where β ∈ (-1, 1) is the soliton velocity. (+) = kink, (-) = antikink.

**2-Soliton (Kink-Kink collision):**
```
u(x,t) = 4 arctan[ (β · sinh(x/√(1-β²))) / cosh(βt/√(1-β²)) ]
```

**Breather (oscillating bound state):**
```
u(x,t) = -4 arctan[ (m/√(1-m²)) · sin(√(1-m²) · t) / cosh(m · x) ]
```
where m ∈ (0, 1) is the breather frequency parameter.

**N-soliton solutions**: Available via IST / Backlund transformation. Explicit formulas for arbitrary N.

### Domain, BCs, Dimensionality
- **Domain**: x ∈ [-L, L], t ∈ [0, T]
- **BCs**: For kink — u → 0 as x → -∞, u → 2π as x → +∞ (or periodic with large L)
- **BCs**: For breather — decaying (u → 0 at boundaries)
- **Dim**: 1D+t (second-order in time, so needs both u(x,0) and u_t(x,0))

### Parameters
- `β` (kink velocity): β ∈ (-1, 1)
- `m` (breather frequency): m ∈ (0, 1)
- Rich parametric family for operator learning

### ML Papers & Baselines
- **PINN**: Deresse (2024, Wiley) — generalized nonlinear sine-Gordon with PINNs
- **PINN**: Li et al. (2021, Commun Theor Phys) — physics-constrained deep residual network
- **FNO**: Yan et al. (2022) — FNO for fractional nonlinear wave equations including sine-Gordon
- **Neural operator**: Limited direct multi-baseline comparisons (mostly PINN-based studies)
- **Baseline count**: ~2-3 (PINN, FNO, FDM baselines)
- **Status**: Established PINN benchmark; neural operator usage exists but opportunity to be first comprehensive multi-operator benchmark

### Data Self-Generation
**YES** — exact closed-form formulas for all solution types. Zero computational cost.

---

## NEW Problem #11: 2D Advection-Diffusion Equation — Gaussian Pulse

### PDE Equation
```
u_t + v⃗ · ∇u = ν ∇²u
```
where v⃗ = (a, b) is the constant advection velocity and ν is the diffusion coefficient.

Expanded in 2D:
```
u_t + a·u_x + b·u_y = ν(u_xx + u_yy)
```

### Exact Analytical Solution (Gaussian pulse)
```
u(x, y, t) = 1/(4νt + 1) · exp(-[(x - x₀ - at)² + (y - y₀ - bt)²] / [4ν(4νt + 1)σ₀²])
```
More generally, for arbitrary initial Gaussian with width σ₀:
```
u(x, y, t) = σ₀² / (σ₀² + 2νt) · exp(-[(x - at)² + (y - bt)²] / [2(σ₀² + 2νt)])
```

**Alternative (Fourier modes on periodic domain):**
```
u(x, y, t) = Σ_{m,n} A_{mn} · exp(-ν(k_m² + k_n²)t) · sin(k_m(x - at)) · sin(k_n(y - bt))
```
where k_m = 2πm/L, k_n = 2πn/L.

### Domain, BCs, Dimensionality
- **Domain**: [0, L]² × [0, T] or ℝ² × [0, T]
- **BCs**: Periodic (Fourier modes) or unbounded (Gaussian on large domain)
- **Dim**: 2D+t
- **Peclet number** Pe = |v⃗|L/ν controls advection-diffusion balance

### Parameters
- `a, b` (advection velocity): vary to create parametric families
- `ν` (diffusion coefficient): typically 0.001-0.1
- `σ₀` (initial Gaussian width)
- Peclet number: Pe = |v⃗|L/ν (low Pe → diffusion-dominated, high Pe → advection-dominated)

### ML Papers & Baselines
- **PINN**: He et al. (2021, Water Resources Research) — forward/backward advection-dispersion
- **PINN**: NTK analysis of PINN for advection-diffusion (2022)
- **FNO**: Used in PDEBench (1D diffusion-reaction variant)
- **PINO**: Physics-Informed Neural Operator includes advection-diffusion
- **DeepONet**: Physics-informed DeepONet (Science Advances 2023) tested on advection-diffusion
- **Baseline count**: ≥3 (PINN, FNO/PINO, DeepONet)
- **Status**: Widely used but often in 1D; 2D version with exact solution is straightforward extension

### Data Self-Generation
**YES** — direct evaluation of Gaussian formula or Fourier series. Zero error by construction.

---

## NEW Problem #12: 1D Fisher-KPP Equation — Traveling Wave

### PDE Equation
```
u_t = D · u_xx + r · u(1 - u)
```
where D is diffusion coefficient and r is reaction rate.

Dimensionless form (D=1, r=1):
```
u_t = u_xx + u(1 - u)
```

### Exact Analytical Solution (special wave speed c = 5/√6)

For the dimensionless Fisher-KPP equation, at the special wave speed c = 5/√6 ≈ 2.0412:
```
u(x, t) = [1 + C · exp(-(x - ct)/√6)]^(-2)
```
where:
- c = 5/√6
- C > 0 is a free constant (sets initial position)
- The solution is a traveling front connecting u=1 (left) to u=0 (right)

This was derived by Ablowitz & Zeppetella (1979) using Painleve analysis.

**General traveling wave** for any c ≥ 2√(rD):
- No closed-form for arbitrary c, but the c = 5/√6 case is exactly solvable

### Domain, BCs, Dimensionality
- **Domain**: x ∈ [-L, L], t ∈ [0, T] (large L to avoid boundary effects)
- **BCs**: u(-L, t) → 1, u(L, t) → 0 (or periodic on extended domain)
- **Dim**: 1D+t
- **Note**: u ∈ [0, 1] always (biological interpretation: population density)

### Parameters
- `D` (diffusion): scales space
- `r` (reaction rate): scales time
- `C` (position constant): shifts the wave
- Can vary D, r to create parametric families (c_min = 2√(rD))
- Large r produces steep fronts (challenging for ML)

### ML Papers & Baselines
- **PINN**: Comprehensive retraining study (arXiv 2601.11406, 2026) — PINN vs FDM comparison
- **PINN**: Approximating sharp solutions (CMAME 2024) — residual weighting for steep fronts
- **FNO**: APEBench includes Fisher-KPP in its 46 PDEs (1D, 2D, 3D variants)
- **UNet**: APEBench baseline comparison
- **Multiple baselines in APEBench**: FNO, UNet, ConvResNet, dilated ResNet tested on Fisher-KPP
- **Baseline count**: ≥4 via APEBench (FNO, UNet, ResNet variants, dilated ResNet)
- **Status**: Excellent benchmark — included in APEBench (NeurIPS 2024) with multiple baselines

### Data Self-Generation
**YES** — closed-form traveling wave at c = 5/√6. For other speeds, use spectral methods (exact for periodic linear part).

---

## NEW Problem #13 (Bonus): 2D Heat/Diffusion Equation — Fourier Modes

### PDE Equation
```
u_t = α(u_xx + u_yy)
```
where α is thermal diffusivity.

### Exact Analytical Solution
On [0, L]² with homogeneous Dirichlet BCs:
```
u(x, y, t) = Σ_{m,n} B_{mn} · exp(-α π²(m²+n²)/L² · t) · sin(mπx/L) · sin(nπy/L)
```

On [0, L]² with periodic BCs:
```
u(x, y, t) = Σ_{m,n} A_{mn} · exp(-α(k_m² + k_n²)t) · exp(i(k_m·x + k_n·y))
```
where k_m = 2πm/L.

### Domain, BCs, Dimensionality
- **Domain**: [0, L]² × [0, T]
- **BCs**: Periodic or Dirichlet
- **Dim**: 2D+t
- **IC**: Any L²-integrable function (Fourier decomposition gives exact solution)

### Parameters
- `α` (thermal diffusivity): typically 0.01-1.0
- Initial condition complexity (number of Fourier modes)
- Domain size L

### ML Papers & Baselines
- **FNO**: Heat Equation 2D FNO notebook (GitHub: abelsr/Fourier-Neural-Operator)
- **PINN**: PINNacle benchmark includes Heat2D variants (VaryingCoef, Multiscale, ComplexGeometry, LongTime)
- **DeepONet**: Physics-informed DeepONet (Science Advances 2023) — tested on diffusion
- **PDEBench**: Includes 1D/2D diffusion-reaction
- **APEBench**: Diffusion available in 1D, 2D, 3D
- **Baseline count**: ≥4 (FNO, PINN, DeepONet, UNet all tested)
- **Status**: Extremely well-established benchmark; every major paper uses some form of heat/diffusion

### Data Self-Generation
**YES** — trivially computed from Fourier series. Zero error.

---

## NEW Problem #14 (Bonus): 3D Advection Equation

### PDE Equation
```
u_t + a·u_x + b·u_y + c·u_z = 0
```

### Exact Analytical Solution
```
u(x, y, z, t) = u₀(x - at, y - bt, z - ct)
```
where u₀ is any initial condition. The solution is just the initial condition translated.

### Domain, BCs, Dimensionality
- **Domain**: [0, L]³ × [0, T] with periodic BCs
- **Dim**: **3D+t** (satisfies the 3D requirement)

### Parameters
- `a, b, c` (advection velocities): parametric family
- Initial condition u₀: Gaussian, sinusoidal, or arbitrary

### ML Papers & Baselines
- **APEBench**: Advection available in 1D, 2D, 3D; specifically "Unbalanced Advection" in 3D
- **PDEBench**: 1D Advection is a core benchmark
- **FNO/UNet/ResNet**: All tested in APEBench framework
- **Baseline count**: ≥4 via APEBench
- **Status**: Simple but important — tests pure transport without diffusion in 3D

### Data Self-Generation
**YES** — trivial translation of initial condition. Zero error.

---

## NEW Problem #15 (Bonus): 3D Heat/Diffusion Equation

### PDE Equation
```
u_t = α(u_xx + u_yy + u_zz)
```

### Exact Analytical Solution
On [0, L]³ with periodic BCs:
```
u(x, y, z, t) = Σ_{l,m,n} A_{lmn} · exp(-α(k_l² + k_m² + k_n²)t) · exp(i(k_l·x + k_m·y + k_n·z))
```

### Domain, BCs, Dimensionality
- **Domain**: [0, L]³ × [0, T]
- **BCs**: Periodic
- **Dim**: **3D+t**

### ML Papers & Baselines
- **APEBench**: Diffusion in 3D available
- **FNO**: 3D FNO demonstrated
- **Baseline count**: ≥3 via APEBench
- **Status**: Standard linear PDE in 3D; important for testing scalability

### Data Self-Generation
**YES** — Fourier series evaluation. Zero error.

---

## Summary Table

| # | PDE | Dim | Type | Exact Solution | ML Baselines (≥3?) | Novel? |
|---|-----|-----|------|---------------|---------------------|--------|
| 1 | 1D Burgers | 1D+t | Nonlinear | Cole-Hopf | Yes (FNO,PINN,DeepONet,UNet) | No |
| 2 | 1D Advection | 1D+t | Linear | Translation | Yes (PDEBench,APEBench) | No |
| 3 | 2D Taylor-Green Vortex | 2D+t | Nonlinear | Exp decay | Yes (PINN,FNO) | No |
| 4 | 3D Beltrami/Ethier-Steinman | 3D+t | Nonlinear | Exact NS | Partial (PINN, NSFnets) | Semi-novel |
| 5 | 1D Allen-Cahn | 1D+t | Nonlinear | tanh wave | Yes (APEBench,PINNacle) | No |
| 6 | 1D KdV | 1D+t | Nonlinear | IST soliton | Yes (FNO,PINN,IS-FNO) | No |
| 7 | 2D KP-II | 2D+t | Nonlinear | IST soliton | Partial (IS-FNO) | Semi-novel |
| 8 | 2D Wave | 2D+t | Linear | d'Alembert/Fourier | Yes (PINNacle) | No |
| **9** | **1D NLS (bright soliton)** | **1D+t** | **Nonlinear** | **sech soliton (IST)** | **Yes (FNO,PINN,DeepONet)** | **No** |
| **10** | **1D Sine-Gordon** | **1D+t** | **Nonlinear** | **4arctan kink/breather** | **Partial (~2-3)** | **Semi-novel** |
| **11** | **2D Advection-Diffusion** | **2D+t** | **Linear** | **Gaussian pulse** | **Yes (PINN,PINO,DeepONet)** | **No** |
| **12** | **1D Fisher-KPP** | **1D+t** | **Nonlinear** | **[1+Ce^z]^(-2), c=5/√6** | **Yes (APEBench ≥4)** | **No** |
| **13** | **2D Heat** | **2D+t** | **Linear** | **Fourier series** | **Yes (FNO,PINN,DeepONet,UNet)** | **No** |
| **14** | **3D Advection** | **3D+t** | **Linear** | **Translation** | **Yes (APEBench ≥4)** | **No** |
| **15** | **3D Heat** | **3D+t** | **Linear** | **Fourier series** | **Partial (APEBench ≥3)** | **Semi-novel** |

---

## Dimension Coverage

| Dim | Count | Problems |
|-----|-------|----------|
| 1D+t | 7 | Burgers, Advection, Allen-Cahn, KdV, NLS, Sine-Gordon, Fisher-KPP |
| 2D+t | 5 | Taylor-Green, KP-II, Wave, Advection-Diffusion, Heat |
| 3D+t | 3 | Beltrami, 3D Advection, 3D Heat |

---

## Type Coverage

| Type | Count | Problems |
|------|-------|----------|
| Linear | 6 | Advection(1D), Wave(2D), Adv-Diff(2D), Heat(2D), 3D Advection, 3D Heat |
| Nonlinear | 9 | Burgers, Taylor-Green, Beltrami, Allen-Cahn, KdV, KP-II, NLS, Sine-Gordon, Fisher-KPP |

---

## Recommended Priority for NEW Problems (#9-#15)

### Tier 1: Must Include (strong ML baseline coverage + exact solution)
1. **#9 NLS Bright Soliton** — Rich physics (complex-valued), IST-solvable, well-studied in PINN/FNO
2. **#12 Fisher-KPP** — Already in APEBench with ≥4 baselines, reaction-diffusion class
3. **#11 2D Advection-Diffusion** — Fills 2D linear gap, exact Gaussian, tested in PINN/PINO/DeepONet
4. **#13 2D Heat** — Ubiquitous benchmark, exact Fourier series

### Tier 2: Include for Completeness (fills dimensional gaps)
5. **#14 3D Advection** — Only 3D+t linear problem with trivial exact solution
6. **#10 Sine-Gordon** — Kink + breather diversity, but fewer ML baselines (<3 operator methods)

### Tier 3: Bonus/Future (less unique)
7. **#15 3D Heat** — Similar to 2D Heat extended; fills 3D gap

---

## Rejected / Not Recommended

| PDE | Reason |
|-----|--------|
| 2D Kovasznay Flow | Steady-state (no time dimension) |
| Cahn-Hilliard | No simple closed-form time-dependent solution (4th order, complex dynamics) |
| Telegraph equation | Less popular in ML community, few baselines |
| Euler-Bernoulli Beam | Structural mechanics niche, few ML operator baselines |
| Maxwell's Equations | Few neural operator benchmarks; PINNs struggle with EM |
| Linear Schrodinger (free particle) | Dispersive spreading is trivial; NLS is more interesting |
| mKdV | Too similar to KdV already included |
| Klein-Gordon | Mostly studied with PINNs; less operator learning literature |

---

## Key References

### Benchmark Suites
- **APEBench** (NeurIPS 2024): 46 PDEs in 1D/2D/3D, includes Fisher-KPP, advection, diffusion, Burgers, KdV, Allen-Cahn, KS. [GitHub](https://github.com/tum-pbs/apebench)
- **PDEBench** (NeurIPS 2022): 1D advection, 1D Burgers, 1D/2D diffusion-reaction, compressible NS, Darcy flow, shallow water. [GitHub](https://github.com/pdebench/PDEBench)
- **PINNacle** (NeurIPS 2024): 20+ PDEs for PINN evaluation (Heat2D, NS2D, Poisson, Wave, Burgers, etc.). [GitHub](https://github.com/i207M/PINNacle)
- **DeepXDE**: Library with many PINN examples with exact solutions. [Docs](https://deepxde.readthedocs.io/)
- **Exponax**: JAX library powering APEBench, 46+ ETDRK-based PDE solvers. [GitHub](https://github.com/Ceyron/exponax)

### Neural Operator Papers
- Li et al. (2021, ICLR): FNO — Burgers, Darcy, NS benchmarks
- Lu et al. (2022, CMAME): Comprehensive FNO vs DeepONet comparison, 16 benchmarks
- Li et al. (2024, JDS): PINO — physics-informed neural operator
- Yan et al. (2022, Chaos Solitons Fractals): FNO for NLS/sine-Gordon soliton mappings
- IS-FNO (arXiv 2512.19439, 2025): Inverse scattering FNO for KdV, KP, MS, KS

### PINN Papers
- Raissi et al. (2019, JCP): Original PINN — Burgers, Schrodinger, NS benchmarks
- Ablowitz & Zeppetella (1979): Fisher-KPP exact solution at c=5/√6
- NSFnets (2020): Beltrami flow PINN benchmark
