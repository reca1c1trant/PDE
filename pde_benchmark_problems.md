# PDE Benchmark Problems for Neural PDE Solvers

> Compiled 2026-03-21. All problems are **time-dependent** with **>=3 non-PINN baselines** compared.
> Problems already covered (excluded): 1D Burgers, 1D Advection, 1D Reaction, 1D Diffusion/Heat, 1D Wave, 2D Taylor-Green Vortex, 3D Beltrami/Ethier-Steinman Flow, 1D Allen-Cahn.

---

## Problem 1: 1D Kuramoto-Sivashinsky (KS) Equation

**Sources:**
- IS-FNO paper (arXiv:2512.19439, 2024)
- APEBench (NeurIPS 2024, arXiv:2411.00180)

**PDE:**
$$\partial_t u + u \partial_x u + \partial_{xx} u + \partial_{xxxx} u = 0$$

Alternate form (from IS-FNO):
$$\partial_t \phi + \frac{1}{2\beta^2}(\nabla\phi)^2 = -\frac{1}{\beta^2}\nabla^2\phi - \frac{1}{\beta^4}\nabla^4\phi$$

**Domain:** $x \in [0, 2\pi]$ (or $[0, L]$), periodic BCs, $t \in [0, T]$

**Dimensionality:** 1D + t (also available in 2D + t)

**Analytical solution:** None (chaotic dynamics). Ground truth via high-resolution pseudo-spectral simulation (ETDRK4).

**Parameters varied:**
- Domain length $L$ (controls chaos level)
- $\beta = 10, 40$ (IS-FNO)
- Viscosity-like parameter

**Baselines compared (>=3):**
- IS-FNO paper: FNO, kFNO (Koopman), IS-FNO, kFNO variants (6+ models total)
- APEBench: ConvNet, ResNet, U-Net, Dilated ResNet, FNO (5 architectures)

**Data generation:** Self-generated via pseudo-spectral methods (ETDRK4, exponential time differencing). APEBench provides JAX code; IS-FNO provides code.

**Why interesting:** Canonical chaotic PDE. Tests long-horizon stability of autoregressive models. Available in 1D and 2D.

---

## Problem 2: 1D Korteweg-de Vries (KdV) Equation

**Sources:**
- IS-FNO (arXiv:2512.19439, 2024)
- APEBench (NeurIPS 2024)

**PDE:**
$$\partial_t u + 6u\partial_x u + \partial_{xxx} u = 0$$

**Domain:** $x \in [0, 20]$, periodic BCs, $t \in [0, T]$ (T ~ 1.22 in IS-FNO)

**Dimensionality:** 1D + t

**Analytical solution:** YES -- exactly solvable via **Inverse Scattering Transform (IST)**. N-soliton solutions are known in closed form:
$$u(x,t) = -2\partial_{xx} \ln \det(I + A(x,t))$$
where $A$ is constructed from scattering data.

**Parameters varied:**
- Initial condition type: low-wavenumber randomization, random soliton superposition
- Number of solitons

**Baselines compared (>=3):**
- FNO (baseline), kFNO*, kFNO_o, kFNO', IS-FNO*, IS-FNO_o, IS-FNO', IS-FNO_kappa' (9 models)

**Data generation:** Self-generated. Pseudo-spectral methods on periodic domain. 256 grid points, 350 trajectories.

**Why interesting:** Integrable PDE with exact analytical solutions. Tests whether models can learn conserved quantities and soliton interactions.

---

## Problem 3: 2D Kadomtsev-Petviashvili (KP-II) Equation

**Sources:**
- IS-FNO (arXiv:2512.19439, 2024)

**PDE:**
$$\partial_{x_1}\left(\partial_t u + 6u\partial_{x_1} u + \partial_{x_1 x_1 x_1} u\right) + \partial_{x_2 x_2} u = 0$$

**Domain:** $[0, 20]^2$, periodic BCs, $t \in [0, 1.22]$

**Dimensionality:** 2D + t

**Analytical solution:** YES -- integrable via IST extension to 2D. Line-soliton solutions and their interactions are known analytically.

**Constraint:** $\int_0^{20} (\partial_{x_2}^2 u_0) dx_1 = 0$

**Parameters varied:**
- Initial conditions (random perturbations satisfying constraint)

**Baselines compared (>=3):**
- FNO, kFNO*, kFNO_o, kFNO', IS-FNO*, IS-FNO_o, IS-FNO' (7+ models)

**Data generation:** Self-generated. Pseudo-spectral methods, $128^2$ grid, 350 trajectories.

**Why interesting:** 2D generalization of KdV with exact solutions. Tests 2D operator learning on an integrable system. Rare example of 2D+t with analytical solutions.

---

## Problem 4: 2D Shallow Water Equations (SWE) -- Radial Dam Break

**Sources:**
- PDEBench (NeurIPS 2022, arXiv:2210.07182)
- DPOT (arXiv:2403.03542, ICML 2024)
- BCAT (arXiv:2501.18972, 2025)
- MPP (NeurIPS 2024)
- The Well (NeurIPS 2024)

**PDE:**
$$\partial_t h + \partial_x(hu) + \partial_y(hv) = 0$$
$$\partial_t(hu) + \partial_x\left(hu^2 + \frac{1}{2}gh^2\right) + \partial_y(huv) = 0$$
$$\partial_t(hv) + \partial_x(huv) + \partial_y\left(hv^2 + \frac{1}{2}gh^2\right) = 0$$

**Domain:** $[0,1]^2$ or $[-2.5, 2.5]^2$, periodic or open BCs

**Dimensionality:** 2D + t

**Analytical solution:** Semi-analytical radial 1D reference solution exists for circular dam break (via Riemann solver). Not fully closed-form, but well-established high-accuracy numerical ground truth.

**Initial conditions:**
- $h_{in} = 2.0$ (inner), $h_{out} = 1.0$ (outer), dam radius = 0.5
- Zero initial velocities

**Parameters varied:**
- $g$ (gravitational acceleration)
- Inner/outer water depth ratio
- Dam radius

**Baselines compared (>=3):**
- PDEBench: FNO, U-Net, PINN (3 baselines)
- BCAT: DeepONet, FNO, UNet, ViT, MPP-B, MPP-L, DPOT-M, DPOT-L, BCAT (9 baselines)
- The Well: FNO, TFNO, U-Net, CNextU-Net (4 baselines)

**Data generation:** Self-generated via Clawpack (finite volume solver). PDEBench provides `gen_radial_dam_break.py`.

**Why interesting:** Hyperbolic conservation law with shocks and rarefaction waves. Widely used across PDEBench/DPOT/BCAT/MPP ecosystems.

---

## Problem 5: 2D Incompressible Navier-Stokes (Vorticity Form)

**Sources:**
- Li et al. FNO (ICLR 2021, arXiv:2010.08895)
- DPOT (ICML 2024)
- Poseidon (NeurIPS 2024)
- BCAT (2025)
- PDEArena (2023)
- Virtually every neural operator paper

**PDE:**
$$\partial_t w + \mathbf{u} \cdot \nabla w = \nu \Delta w + f(\mathbf{x})$$
$$\nabla \cdot \mathbf{u} = 0, \quad w = \nabla \times \mathbf{u}$$

**Domain:** $[0,1]^2$, periodic BCs, $t \in [0, T]$

**Dimensionality:** 2D + t

**Analytical solution:** No closed-form for turbulent regime. Ground truth via high-resolution spectral solver (Crank-Nicolson, $\Delta t = 10^{-4}$, generated on $256 \times 256$, downsampled to $64 \times 64$).

**Initial conditions:** $w_0 \sim \mathcal{N}(0, 7^{3/2}(-\Delta + 49I)^{-2.5})$ (Gaussian random field)

**Parameters varied:**
- Viscosity $\nu$: $10^{-3}$, $10^{-4}$, $10^{-5}$ (controls Reynolds number)
- Forcing function $f(\mathbf{x})$
- Multiple sub-benchmarks: piecewise constant (PwC), buoyancy-driven (BB), shear layer (SL), single vortex sheet (SVS)

**Baselines compared (>=3):**
- Original FNO paper: ResNet, U-Net, TF-Net, FNO-2D, FNO-3D (5 baselines)
- DPOT: FNO, UNet, FFNO, GK-Transformer, GNOT, OFormer, MPP, DPOT (8+ baselines)
- BCAT: DeepONet, FNO, UNet, ViT, MPP, DPOT, BCAT (7+ baselines)
- Poseidon: FNO, CNO, scOT, CNO-FM, MPP-B (5 baselines)

**Data generation:** Self-generated via pseudo-spectral methods. Code available from neuraloperator GitHub.

**Why interesting:** THE canonical benchmark for neural operators. Turbulent dynamics, multi-scale features. Extensive baseline coverage.

---

## Problem 6: 2D/3D Compressible Navier-Stokes (Riemann Problems)

**Sources:**
- PDEBench (NeurIPS 2022)
- DPOT (ICML 2024)
- BCAT (2025)
- MPP (NeurIPS 2024)

**PDE:**
$$\partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0$$
$$\partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p\mathbf{I}) = \nabla \cdot \boldsymbol{\sigma}$$
$$\partial_t E + \nabla \cdot ((E+p)\mathbf{v}) = \nabla \cdot (\boldsymbol{\sigma} \cdot \mathbf{v})$$

where $E = \frac{p}{\Gamma - 1} + \frac{1}{2}\rho|\mathbf{v}|^2$, $\Gamma = 5/3$

**Domain:** $[0,1]^d$, $d \in \{1, 2, 3\}$, periodic BCs

**Dimensionality:** 1D/2D/3D + t (all available in PDEBench)

**Analytical solution:** For 1D Riemann problems (Sod shock tube), exact solutions exist. For 2D/3D with random ICs, high-resolution numerical ground truth via WENO/finite-volume methods.

**Initial conditions:** Riemann problems with random left/right states ($Q_L, Q_R$), random discontinuity location.

**Parameters varied:**
- Shear viscosity $\eta$, bulk viscosity $\zeta$
- Mach number (via initial conditions)
- Spatial dimension (1D, 2D, 3D)

**Baselines compared (>=3):**
- PDEBench: FNO, U-Net, PINN (3 baselines)
- BCAT: DeepONet, FNO, UNet, ViT, MPP-B, MPP-L, DPOT-M, DPOT-L, BCAT (9 baselines)
- DPOT: FNO, FFNO, GK-Transformer, GNOT, OFormer, MPP, DPOT (7+ baselines)

**Data generation:** Self-generated. PDEBench provides scripts in `data_gen_NLE/CompressibleFluid/`.

**Why interesting:** Multi-variable system (density, velocity, pressure, energy). Contains shocks and discontinuities. Available in 3D -- one of the few 3D+t benchmarks with extensive baseline comparisons.

---

## Problem 7: 2D Diffusion-Reaction (FitzHugh-Nagumo)

**Sources:**
- PDEBench (NeurIPS 2022)
- DPOT (ICML 2024)
- MPP (NeurIPS 2024)

**PDE (FitzHugh-Nagumo type):**
$$\partial_t u = D_u \nabla^2 u + R_u(u, v)$$
$$\partial_t v = D_v \nabla^2 v + R_v(u, v)$$

where:
$$R_u(u,v) = u - u^3 - k - v, \quad R_v(u,v) = u - v$$
$k = 5 \times 10^{-3}$, $D_u = 1 \times 10^{-3}$, $D_v = 5 \times 10^{-3}$

**Domain:** $[-1, 1]^2$, no-flux Neumann BCs, $t \in [0, T]$

**Dimensionality:** 2D + t (also 1D variant in PDEBench)

**Analytical solution:** No closed-form. High-resolution numerical ground truth via 4th-order Runge-Kutta + finite volume ($512^2$ grid, downsampled to $128^2$).

**Initial conditions:** $u(0, x, y) \sim \mathcal{N}(0, 1)$ (random Gaussian noise)

**Parameters varied:**
- Diffusion coefficients $D_u, D_v$
- Reaction parameter $k$
- Initial condition distributions

**Baselines compared (>=3):**
- PDEBench: FNO, U-Net, PINN (3 baselines)
- DPOT: FNO, FFNO, UNet, GK-Transformer, GNOT, OFormer, MPP, DPOT (8+ baselines)

**Data generation:** Self-generated. PDEBench provides `gen_diff_react.py`. SciPy RK4 + finite volume.

**Why interesting:** Nonlinear coupled system producing biological pattern formation (Turing patterns). Neumann BCs (non-periodic). Two-channel output.

---

## Problem 8: 2D Compressible Euler (Multi-Quadrant Riemann)

**Sources:**
- The Well (NeurIPS 2024)
- Poseidon (NeurIPS 2024) -- CE-RPUI, CE-RM variants

**PDE:**
$$\partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0$$
$$\partial_t (\rho \mathbf{v}) + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v} + p\mathbf{I}) = 0$$
$$\partial_t E + \nabla \cdot ((E+p)\mathbf{v}) = 0$$

(inviscid -- no viscous terms)

**Domain:** $[0,1]^2$, open or periodic BCs

**Dimensionality:** 2D + t

**Analytical solution:** No general closed-form, but 1D Riemann sub-problems have exact solutions. High-resolution numerical ground truth.

**Initial conditions:** Multi-quadrant piecewise constant states (density, velocity, pressure differ per quadrant).

**Parameters varied:**
- Quadrant initial states (random)
- Boundary condition type (open vs periodic)
- Adiabatic index $\gamma$

**Baselines compared (>=3):**
- The Well: FNO, TFNO, U-Net, CNextU-Net (4 baselines)
- Poseidon: FNO, CNO, scOT, CNO-FM, MPP-B (5 baselines)

**Data generation:** Self-generated via finite-volume Godunov-type solvers (e.g., Athena++, Clawpack).

**Why interesting:** Rich wave interactions (shocks, contact discontinuities, rarefaction fans). Multi-quadrant setup creates complex 2D patterns from simple initial data.

---

## Problem 9: 2D Gray-Scott Reaction-Diffusion

**Sources:**
- APEBench (NeurIPS 2024)
- The Well (NeurIPS 2024)

**PDE:**
$$\partial_t u = D_u \nabla^2 u - uv^2 + F(1-u)$$
$$\partial_t v = D_v \nabla^2 v + uv^2 - (F+k)v$$

**Domain:** Periodic or no-flux BCs, square domain

**Dimensionality:** 1D/2D/3D + t (APEBench provides all three)

**Analytical solution:** No closed-form for general ICs. Numerical ground truth via spectral methods (APEBench) or finite-difference (The Well).

**Parameters varied:**
- Feed rate $F$
- Kill rate $k$
- Diffusion coefficients $D_u, D_v$
- Different $(F, k)$ regimes produce qualitatively different patterns (spots, stripes, spirals, chaos)

**Baselines compared (>=3):**
- APEBench: ConvNet, ResNet, U-Net, Dilated ResNet, FNO (5 architectures)
- The Well: FNO, TFNO, U-Net, CNextU-Net (4 baselines)

**Data generation:** Self-generated. APEBench provides JAX code. The Well provides data.

**Why interesting:** Produces diverse Turing patterns depending on parameters. Two-channel coupled system. Available in 1D/2D/3D.

---

## Problem 10: 2D Wave Equation

**Sources:**
- Poseidon (NeurIPS 2024) -- Wave-Gauss, Wave-Layer variants

**PDE:**
$$\partial_{tt} u = c^2 \nabla^2 u$$

or equivalently (first-order system):
$$\partial_t u = v, \quad \partial_t v = c^2 \nabla^2 u$$

**Domain:** $[0,1]^2$, periodic BCs

**Dimensionality:** 2D + t

**Analytical solution:** YES -- for periodic domain, d'Alembert-type solutions exist. For Gaussian initial conditions on periodic domain, the solution is a superposition of plane waves (Fourier modes evolving independently).

**Initial conditions:**
- Gaussian pulse(s) with random positions and widths
- Layer-type initial conditions

**Parameters varied:**
- Wave speed $c$
- Initial condition type (Gaussian vs layer)
- Pulse positions, widths

**Baselines compared (>=3):**
- Poseidon: FNO, CNO, scOT, CNO-FM, MPP-B (5 baselines)

**Data generation:** Self-generated via spectral methods or finite differences.

**Why interesting:** Linear PDE with known exact solution. Tests basic wave propagation learning. Good sanity check for architectures.

---

## Problem 11: 3D Compressible Navier-Stokes (PDEBench)

**Sources:**
- PDEBench (NeurIPS 2022)

**PDE:** Same as Problem 6 but in 3D.

**Domain:** $[0,1]^3$, periodic BCs

**Dimensionality:** 3D + t

**Analytical solution:** No closed-form. High-resolution numerical ground truth via WENO schemes.

**Variables:** $(\rho, \mathbf{v}, p)$ -- density, 3D velocity, pressure (5 channels total)

**Parameters varied:**
- Shear/bulk viscosity
- Initial condition type (Riemann problems, turbulent velocity)
- Resolution

**Baselines compared (>=3):**
- PDEBench: FNO, U-Net, PINN (3 baselines)
- DPOT: FNO, FFNO, GNOT, OFormer, MPP, DPOT (6+ baselines, for downstream 3D tasks)

**Data generation:** Self-generated. PDEBench provides scripts.

**Why interesting:** One of the very few 3D+t benchmarks with standardized baselines. Multi-channel output (5 fields). Discontinuous solutions.

---

## Summary Table

| # | Problem | Dim | Analytical? | Key Baselines | Source Papers |
|---|---------|-----|------------|---------------|---------------|
| 1 | Kuramoto-Sivashinsky | 1D+t | No (chaotic) | FNO, kFNO, IS-FNO, ResNet, U-Net, ConvNet, DilResNet | IS-FNO, APEBench |
| 2 | Korteweg-de Vries | 1D+t | YES (IST) | FNO, kFNO, IS-FNO (9 models) | IS-FNO |
| 3 | Kadomtsev-Petviashvili (KP-II) | 2D+t | YES (IST) | FNO, kFNO, IS-FNO (7+ models) | IS-FNO |
| 4 | Shallow Water Eqs (Radial Dam Break) | 2D+t | Semi-analytical | FNO, U-Net, PINN, DeepONet, ViT, MPP, DPOT, BCAT, TFNO | PDEBench, DPOT, BCAT, MPP, The Well |
| 5 | Incompressible NS (Vorticity) | 2D+t | No (numerical) | FNO, ResNet, U-Net, TF-Net, DeepONet, ViT, MPP, DPOT, BCAT, CNO, scOT | FNO (Li 2021), DPOT, BCAT, Poseidon |
| 6 | Compressible NS (Riemann) | 2D/3D+t | 1D exact, 2D/3D numerical | FNO, U-Net, PINN, DeepONet, ViT, MPP, DPOT, BCAT | PDEBench, DPOT, BCAT |
| 7 | Diffusion-Reaction (FitzHugh-Nagumo) | 2D+t | No (numerical) | FNO, U-Net, PINN, FFNO, GNOT, OFormer, MPP, DPOT | PDEBench, DPOT, MPP |
| 8 | Compressible Euler (Multi-Quadrant) | 2D+t | Partial (1D sub-problems) | FNO, TFNO, U-Net, CNextU-Net, CNO, scOT, MPP | The Well, Poseidon |
| 9 | Gray-Scott Reaction-Diffusion | 2D+t | No (numerical) | FNO, TFNO, U-Net, CNextU-Net, ConvNet, ResNet, DilResNet | APEBench, The Well |
| 10 | Wave Equation | 2D+t | YES (d'Alembert) | FNO, CNO, scOT, CNO-FM, MPP-B | Poseidon |
| 11 | Compressible NS 3D | 3D+t | No (numerical) | FNO, U-Net, PINN, FFNO, GNOT, OFormer, MPP, DPOT | PDEBench, DPOT |

---

## Key Source Papers

1. **PDEBench** -- NeurIPS 2022 (Datasets & Benchmarks)
   - Paper: https://arxiv.org/abs/2210.07182
   - GitHub: https://github.com/pdebench/PDEBench
   - Equations: 1D Advection, 1D Burgers, 1D/2D Diffusion-Reaction, 1D Diffusion-Sorption, 1D/2D/3D Compressible NS, 2D Darcy, 2D SWE
   - Baselines: FNO, U-Net, PINN

2. **APEBench** -- NeurIPS 2024 (Datasets & Benchmarks)
   - Paper: https://arxiv.org/abs/2411.00180
   - GitHub: https://github.com/tum-pbs/apebench
   - Equations: 46+ PDEs (Advection, Diffusion, Dispersion, Burgers, KdV, KS, Fisher-KPP, Gray-Scott, Swift-Hohenberg, NS) in 1D/2D/3D
   - Baselines: ConvNet, ResNet, U-Net, Dilated ResNet, FNO

3. **IS-FNO** -- arXiv 2024
   - Paper: https://arxiv.org/abs/2512.19439
   - Equations: KS (1D/2D), Michelson-Sivashinsky (1D/2D), KdV (1D), KP-II (2D)
   - Baselines: FNO, kFNO, IS-FNO (multiple variants, 6-9 models)

4. **DPOT** -- ICML 2024
   - Paper: https://arxiv.org/abs/2403.03542
   - 12 pretraining datasets from FNO/PDEBench/PDEArena/CFDBench
   - Baselines: FNO, UNet, FFNO, GK-Transformer, GNOT, OFormer, MPP, DPOT

5. **Poseidon** -- NeurIPS 2024
   - Paper: https://arxiv.org/abs/2405.19101
   - 15 downstream tasks (NS variants, Euler, Wave, Allen-Cahn, Schrodinger, Poisson, Helmholtz)
   - Baselines: FNO, CNO, scOT, CNO-FM, MPP-B

6. **BCAT** -- arXiv 2025
   - Paper: https://arxiv.org/abs/2501.18972
   - Datasets from PDEBench + PDEArena + CFDBench
   - Baselines: DeepONet, FNO, UNet, ViT, MPP-B, MPP-L, DPOT-M, DPOT-L, BCAT

7. **MPP** -- NeurIPS 2024
   - Paper: https://arxiv.org/abs/2310.02994
   - CNS, INS, SWE, DiffReact from PDEBench
   - Baselines: FNO, UNet, task-specific transformers

8. **The Well** -- NeurIPS 2024
   - Paper: https://arxiv.org/abs/2412.00568
   - GitHub: https://github.com/PolymathicAI/the_well
   - 23 spatiotemporal physics datasets (15TB)
   - Baselines: FNO, TFNO, U-Net, CNextU-Net

---

## Recommendations for Selection

### Must-include (strong analytical/semi-analytical + many baselines):
1. **KdV (1D+t)** -- exact IST solution, 9 baselines
2. **KP-II (2D+t)** -- exact IST solution (2D!), 7+ baselines
3. **2D Wave Equation** -- exact d'Alembert solution, 5 baselines
4. **2D SWE (Radial Dam Break)** -- semi-analytical reference, 9 baselines across papers

### Strong picks (numerical ground truth + extensive baselines):
5. **2D Incompressible NS (Vorticity)** -- THE standard benchmark, 10+ baselines
6. **2D/3D Compressible NS** -- multi-dimensional, 9 baselines, 3D available
7. **2D Diffusion-Reaction (FitzHugh-Nagumo)** -- pattern formation, 8+ baselines

### Good additions:
8. **1D Kuramoto-Sivashinsky** -- chaotic dynamics, well-studied
9. **2D Gray-Scott** -- Turing patterns, available in 3D
10. **2D Compressible Euler (Multi-Quadrant)** -- shocks, 5+ baselines
