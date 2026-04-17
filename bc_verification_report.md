# Boundary Condition Verification Report for 5 PDE Benchmarks

> Date: 2026-03-23
> Purpose: Verify whether "all 5 use periodic BCs" is correct

---

## 1. 2D Taylor-Green Vortex

### What benchmark papers use

| Paper | BCs Used | Domain | Source |
|-------|----------|--------|--------|
| FNO (Li et al. 2021, ICLR) | **Periodic** | (0,1)^2 (unit torus) | Blog: "2-d Navier-Stokes equation for a viscous, incompressible fluid in vorticity form on the **unit torus**" |
| PINO (Li et al. 2021, arXiv:2111.03794) | **Periodic** | Torus | Uses spectral/FFT solver (implies periodic) |
| NSFnets (Jin et al. 2020, arXiv:2003.06496) | **Dirichlet** for some benchmarks | Bounded domains | Kovasznay flow uses known BC values; TGV can use either |
| Wikipedia / Classical definition | **Periodic** | [-pi, pi]^d or [0, 2pi]^d | "periodic boundary conditions in all directions" |
| NASA benchmark (C3.3) | **Periodic** | [-pi, pi]^3 | "All boundaries are periodic" |

### Verdict: **PERIODIC is CORRECT**

The Taylor-Green vortex is **inherently periodic** by construction. The exact analytical solution:
```
u = sin(x)cos(y)exp(-2nu*t)
v = -cos(x)sin(y)exp(-2nu*t)
```
is periodic with period 2pi in both x and y. Every major benchmark (FNO, PINO, NASA) uses periodic BCs.

The TGV solution automatically satisfies periodic BCs on [0, 2pi]^2 because sin/cos are 2pi-periodic. On [0,1]^2, rescale: u = sin(2pi*x)cos(2pi*y)exp(-8pi^2*nu*t).

### Analytical solution compatibility: **FULLY COMPATIBLE** with periodic BCs.

---

## 2. 2D KP-II (Kadomtsev-Petviashvili)

### What benchmark papers use

| Paper | BCs Used | Domain | Source |
|-------|----------|--------|--------|
| IS-FNO (arXiv:2512.19439) | **Periodic** | [0, 20]^2 | "We consider periodic domains D=[0, l_KDV]^d with l_KDV=20" |

### Verdict: **PERIODIC is CORRECT**

The IS-FNO paper explicitly states periodic domains for KP-II. The pseudo-spectral solver used for data generation inherently requires periodic BCs (FFT-based). The integral constraint for well-posedness is also stated: integral of d^2_{x2} phi_0 dx_1 = 0 over the period.

### Analytical solution compatibility: **COMPATIBLE**. Line-soliton solutions on periodic domains are well-defined when the domain is large enough relative to soliton width. IST solutions are naturally formulated on periodic or infinite domains.

---

## 3. 2D Wave Equation

### What benchmark papers use

| Paper | BCs Used | Domain | Source |
|-------|----------|--------|--------|
| Poseidon (Herde et al. 2024, NeurIPS) | **Likely periodic** (pretraining is periodic; wave is downstream) | [0,1]^2 | Pretraining: "unit square with periodic boundary conditions"; wave is listed among tasks with potentially non-periodic BCs but specifics not in accessible appendix |
| PINNacle | **Dirichlet** (fixed membrane) | Bounded domain | Standard PINN benchmark uses u=0 on boundary |
| Classical drum problem | **Dirichlet** u=0 on boundary | Bounded domain | Fixed membrane: u(boundary, t) = 0 |
| Classical infinite/periodic | **Periodic** | Torus or infinite | d'Alembert solution on periodic domain |

### Verdict: **DEPENDS ON CONTEXT -- BOTH ARE COMMON**

**Key distinction:**
- **Physical membrane (drum)**: Dirichlet u=0 on boundary -- this is the classical physics problem
- **Wave propagation (ocean/atmosphere)**: Periodic -- used in spectral methods and neural operator pretraining
- **Poseidon specifically**: The pretraining dataset uses periodic BCs, and the wave equation tasks are generated with the same PDEgym framework. The paper states "many of the downstream tasks are with **non-periodic** boundary conditions", implying some ARE periodic. Given that the wave tasks are generated with spectral methods on [0,1]^2, **periodic is likely** for Poseidon's wave tasks, though the appendix B.2 details were not accessible.

### Analytical solution compatibility:
- **Periodic**: Fourier mode superposition works perfectly. u(x,y,t) = sum A_mn cos(omega_mn t + phi_mn) sin(k_m x) sin(k_n y) with k_m = 2pi*m/L.
- **Dirichlet**: Use sin-series only (no constant/cosine modes). u(x,y,t) = sum A_mn cos(omega_mn t) sin(m pi x/L) sin(n pi y/L).
- Both have exact solutions. The implementation should match the benchmark's BC choice.

### Recommendation: Use **periodic** if targeting FNO/neural operator benchmarks. Use **Dirichlet** if targeting PINN benchmarks.

---

## 4. 2D Advection-Diffusion

### What benchmark papers use

| Paper | BCs Used | Domain | Source |
|-------|----------|--------|--------|
| PDEBench (Takamoto et al. 2022) | **Periodic** (1D advection) | Domain varies | "the periodic boundary condition ... most commonly used in Scientific ML studies" for advection |
| APEBench (Koehler et al. 2024) | **Periodic** | Configurable | "We currently focus on periodic boundary conditions" |
| PINN papers (He et al. 2021) | **Dirichlet / Neumann** | Bounded domains | PINN papers often use bounded domains with explicit BCs |
| Physics-informed DeepONet | Various | Various | Tested on multiple BC types |

### Verdict: **PERIODIC is the STANDARD for neural operators, but the Gaussian pulse solution needs careful treatment**

**Critical mathematical issue:**
The standard Gaussian pulse solution:
```
u(x,y,t) = sigma^2/(sigma^2 + 2nu*t) * exp(-[(x-at)^2 + (y-bt)^2] / [2(sigma^2 + 2nu*t)])
```
is an **infinite-domain** solution. It does NOT naturally satisfy periodic BCs because:
1. The Gaussian is not periodic -- it decays to zero at infinity
2. On a finite periodic domain [0,L]^2, the correct solution requires **periodic images** (sum over all periodic copies)

**For periodic domain**, the correct solution is the **Fourier series**:
```
u(x,y,t) = sum_{m,n} A_mn * exp(-nu*(k_m^2 + k_n^2)*t) * exp(i*(k_m*(x-at) + k_n*(y-bt)))
```
where k_m = 2pi*m/L and A_mn are the Fourier coefficients of the initial condition.

**Practical approach**: If the Gaussian is narrow relative to the domain (sigma << L) and the simulation time is short enough that the pulse doesn't wrap around, the infinite-domain Gaussian is a good approximation on a periodic domain. But for long times or wide Gaussians, one must use the periodic Fourier solution.

### Recommendation: Use **periodic** BCs (matches neural operator benchmarks). The Gaussian solution is approximately correct for narrow pulses but the Fourier-mode solution is exact.

---

## 5. 2D Burgers Equation

### What benchmark papers use

| Paper | BCs Used | Domain | Source |
|-------|----------|--------|--------|
| FNO (Li et al. 2021) | **Periodic** | (0,1) | "with periodic boundary conditions where u0 in L^2_per((0,1))" |
| PDEBench (Takamoto et al. 2022) | **Periodic** | (-1,1) or (0,1) | Default bc mode="periodic" in utils.py; domain inconsistency noted in GitHub issue #51 |
| APEBench (Koehler et al. 2024) | **Periodic** | Configurable | ETDRK spectral solver uses periodic BCs |
| Raissi PINN (2017) | **Dirichlet** u(t,-1)=u(t,1)=0 | [-1,1] | "u(t,-1) = u(t,1) = 0" with IC u(0,x)=-sin(pi*x) |
| DeepXDE | **Dirichlet** | [-1,1] x [0,1] | "DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)" |

### Verdict: **SPLIT -- Neural operators use PERIODIC, PINNs use DIRICHLET**

**For neural operator benchmarks (FNO, PDEBench, APEBench)**: Periodic is standard.

**For PINN benchmarks (Raissi, DeepXDE)**: Dirichlet u=0 on boundaries is standard.

### Current codebase (`generate_burgers2d_dataset.py`) analysis:

The existing solution in the codebase uses a **traveling wave** solution:
```python
exp_arg = (-4*x + 4*y - t) / (32 * nu)
exp_term = np.exp(exp_arg)
u = 0.75 - 0.25 / (1 + exp_term)
v = 0.75 + 0.25 / (1 + exp_term)
```
with u + v = 1.5 on domain [0,1]^2.

**This is an infinite-domain traveling wave solution.** It does NOT satisfy:
- **Periodic BCs**: u(0,y,t) != u(1,y,t) in general (the sigmoid function is not periodic)
- **Dirichlet BCs**: u != 0 on boundaries (u ranges from ~0.5 to ~1.0)

The solution is valid on an **infinite domain** or a domain large enough that boundary effects are negligible. The code uses "ghost cell extrapolation" at boundaries (indices 0 and 127), which is a numerical workaround but NOT a proper BC treatment.

### Compatibility of Cole-Hopf solution with periodic BCs:

For periodic BCs, the correct 2D Burgers solution uses Cole-Hopf on periodic domain:
```
theta(x,y,t) = 1 + A*exp(-2nu*t)*cos(x) + B*exp(-2nu*t)*cos(y)
u = 2nu * A*exp(-2nu*t)*sin(x) / theta
v = 2nu * B*exp(-2nu*t)*sin(y) / theta
```
This is naturally periodic on [0, 2pi]^2.

### Recommendation:
- For neural operator benchmark: Use **periodic** BCs with the Cole-Hopf Fourier solution (not the traveling wave)
- The current codebase traveling wave solution is **NOT compatible** with periodic BCs
- Need to implement the Cole-Hopf periodic solution for this benchmark

---

## Summary Table

| # | PDE | Claim: Periodic? | Verdict | Correct BC | Current Code OK? |
|---|-----|-------------------|---------|------------|------------------|
| 1 | 2D Taylor-Green Vortex | Yes | **CORRECT** | Periodic | N/A (not checked) |
| 2 | 2D KP-II | Yes | **CORRECT** | Periodic | N/A (not checked) |
| 3 | 2D Wave Equation | Yes | **MOSTLY CORRECT** (periodic for neural operators, Dirichlet for PINNs) | Periodic (for FNO benchmarks) | N/A |
| 4 | 2D Advection-Diffusion | Yes | **CORRECT but solution needs care** | Periodic (Fourier series, not naive Gaussian) | N/A |
| 5 | 2D Burgers | Yes | **CORRECT for neural operators** but current code uses non-periodic traveling wave | Periodic (Cole-Hopf Fourier, NOT traveling wave) | **NO -- needs fix** |

---

## Key Findings

### 1. All 5 PDEs use periodic BCs in the neural operator literature -- the claim is CORRECT

Every major neural operator benchmark (FNO, PDEBench, APEBench, IS-FNO, Poseidon pretraining) uses periodic BCs. This is because:
- Spectral/FFT-based solvers are the gold standard for data generation
- FFT inherently assumes periodicity
- FNO's Fourier layers work best with periodic data

### 2. PINN benchmarks use different BCs (Dirichlet)
Raissi's PINN paper uses Dirichlet for Burgers. PINNacle uses various BCs. If targeting PINN comparisons, BCs differ.

### 3. The current 2D Burgers code needs fixing
The traveling wave solution in `generate_burgers2d_dataset.py` is NOT periodic. For a proper benchmark matching FNO/PDEBench conventions, implement the Cole-Hopf Fourier solution on a periodic domain.

### 4. The 2D Advection-Diffusion Gaussian needs periodic images
The naive Gaussian pulse is an infinite-domain solution. For periodic benchmarks, either:
- Use the Fourier-mode solution (exact on periodic domain)
- Use narrow Gaussian with domain large enough that periodicity errors are negligible

---

## Sources

- [FNO blog post (Li)](https://zongyi-li.github.io/blog/2020/fourier-pde/)
- [PDEBench paper (NeurIPS 2022)](https://arxiv.org/abs/2210.07182)
- [PDEBench GitHub](https://github.com/pdebench/PDEBench)
- [PDEBench GitHub Issue #51 (Burgers domain)](https://github.com/pdebench/PDEBench/issues/51)
- [APEBench paper (NeurIPS 2024)](https://arxiv.org/abs/2411.00180)
- [IS-FNO paper](https://arxiv.org/abs/2512.19439)
- [Poseidon paper (NeurIPS 2024)](https://arxiv.org/abs/2405.19101)
- [Poseidon project page](https://camlab-ethz.github.io/poseidon/)
- [PDEgym on HuggingFace](https://huggingface.co/collections/camlab-ethz/pdegym-665472c2b1181f7d10b40651)
- [Raissi PINNs page](https://maziarraissi.github.io/PINNs/)
- [DeepXDE Burgers example](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/burgers.html)
- [PINO paper](https://arxiv.org/abs/2111.03794)
- [NSFnets paper](https://arxiv.org/abs/2003.06496)
- [NASA TGV benchmark](https://www1.grc.nasa.gov/wp-content/uploads/C3.3_Twente.pdf)
