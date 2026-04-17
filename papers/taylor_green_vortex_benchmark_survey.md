# Neural Network / Operator Learning Benchmarks on 2D Taylor-Green Vortex

## Summary

The 2D Taylor-Green vortex (TGV) is a canonical exact-solution benchmark for incompressible
Navier-Stokes:
- u = cos(x)sin(y)e^{-2nu*t}
- v = -sin(x)cos(y)e^{-2nu*t}

Unlike Darcy flow or Kolmogorov flow, the 2D TGV is **not** widely adopted as a standard
benchmark in the major neural operator papers (FNO, DeepONet, PINO). Instead, it appears
primarily in PINN-focused studies and a few recent operator learning papers. Below are the
papers found with actual or near-actual numerical error values.

---

## Paper 1: Experience Report of PINNs in Fluid Simulations (Chuang & Barba, SciPy 2022)

- **Paper**: "Experience report of physics-informed neural networks in fluid simulations: pitfalls and frustration"
- **ArXiv**: 2205.14249
- **Methods**: PINN (NVIDIA Modulus) vs PetIBM (finite-difference CFD solver)

### Setup
- Domain: [-pi, pi] x [-pi, pi]
- Re = 100 (nu = 0.01)
- Time range: t in (0, 100]
- Evaluation mesh: 512 x 512
- Metric: L2 norm against analytical solution

### Key Numerical Results
- PINN accuracy at t=2 reaches error magnitude ~10^{-3}, requiring **>8 hours** on 1x A100 GPU
- PetIBM achieves the same 10^{-3} error in **<1 second** on 1x K40 GPU + 6 CPU cores
- ~32 hours of PINN training needed to match accuracy of a 16x16 finite-difference simulation (which takes <20 seconds)
- PetIBM tested at resolutions 2^k x 2^k for k=4,...,10

**Note**: This paper primarily presents results graphically (figures, not tables). Exact per-timestep
L2 values are shown in plots rather than enumerated in tables.

---

## Paper 2: MindSpore Flow PINN Tutorial (Huawei MindSpore)

- **Source**: MindSpore Flow documentation (not a research paper, but provides concrete numbers)
- **Method**: PINN (6-layer fully-connected, tanh activation)

### Setup
- Domain: 2pi x 2pi
- Re = 100
- Time range: t in (0, 2)
- Training epochs: 500

### Numerical Results (Relative L2 Error by Epoch)

| Epoch | U Error | V Error | P Error | Total Error |
|-------|---------|---------|---------|-------------|
| 20    | 0.7096  | 0.7081  | 1.0046  | 0.7376      |
| 40    | 0.0918  | 0.1450  | 1.0219  | 0.3150      |
| 60    | 0.0865  | 0.0788  | 0.7114  | 0.2187      |
| 80    | 0.0869  | 0.1062  | 0.3270  | 0.1320      |
| 460   | 0.0338  | 0.0258  | 0.0878  | 0.0382      |
| 480   | 0.0224  | 0.0211  | 0.0621  | 0.0274      |
| 500   | 0.0211  | 0.0133  | 0.0452  | 0.0213      |

Final total relative L2 error: **~2.1%** (within 5% error range as documented).

---

## Paper 3: Frequency-Separable Hamiltonian Neural Network (FS-HNN, arXiv 2603.06354, March 2026)

- **Paper**: "Frequency-Separable Hamiltonian Neural Network for Multi-Timescale Dynamics"
- **ArXiv**: 2603.06354
- **Methods**: FS-HNN vs PHNN (Pseudo-Hamiltonian NN) vs FNO

### Setup
- System: 2D Incompressible Taylor-Green Vortex (vorticity formulation)
- Domain: Periodic square domain
- Integration: 4th-order Runge-Kutta (RK4)
- Training: first 5 timesteps; Rollout: subsequent 50 timesteps
- Metric: Rollout MSE

### Numerical Results (Table 2)

| Model              | Resolution | Rollout MSE     |
|--------------------|-----------|-----------------|
| PHNN               | High      | 3.43 x 10^{-4}  |
| FNO                | High      | 3.43 x 10^{-4}  |
| FS-HNN             | Low       | 4.95 x 10^{-7}  |
| FS-HNN             | Medium    | 1.91 x 10^{-7}  |
| FS-HNN             | High      | 8.33 x 10^{-8}  |
| FS-HNN (Combined)  | Combined  | **5.01 x 10^{-8}** |

FS-HNN improves over PHNN/FNO baselines by ~3-4 orders of magnitude.

**Note**: Viscosity, domain size, and grid resolution details are in Appendix B.6 (not
extracted from HTML).

---

## Paper 4: Transport-Embedded Neural Architecture (TENN, arXiv 2410.04114, 2024)

- **Paper**: "Transport-Embedded Neural Architecture: Redefining the Landscape of physics aware neural models in fluid mechanics"
- **ArXiv**: 2410.04114
- **Methods**: Vanilla PINN vs TENN

### Setup
- Domain: 1 x 1 periodic domain (unit toroid T^2)
- Reynolds numbers tested: Re = 0.1, 1, 10, 100
- Activation functions tested: sin, tanh, softplus

### Numerical Results
- TENN at high Re: **relative error ~4%** (stated in text, no table)
- Vanilla PINN: **fails** at high Re (returns static/initial condition as solution)
- Results presented as error heatmaps in figures, not tabulated

---

## Paper 5: Spectral-Refiner / ST-FNO (arXiv 2405.17211, 2024)

- **Paper**: "Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows"
- **ArXiv**: 2405.17211
- **Methods**: ST-FNO3d vs FNO3d (roll-out) vs IMEX-RK4 solver

### Setup
- Domain: Unit torus (T^2), periodic BCs
- Viscosity: nu = 0.001 (~Re = 1000)
- Grid: 256 x 256
- 10 trajectories with varying vortices per wavelength
- Prediction interval: t = 4.5 to t = 5.5 (40 steps)

### Computational Efficiency (Table 6 from paper)

| Method           | Runtime (ms)     | GFLOPs/eval |
|------------------|-----------------|-------------|
| ST-FNO3d         | 22.7 +/- 4.0    | 1.06        |
| FNO3d (roll-out) | 45.6 +/- 6.7    | 28.7        |
| IMEX-RK4 solver  | 450 +/- 8.7     | -           |

**Note**: Relative error values and residual norms are in the paper's Table 1 but could not
be extracted from the HTML/PDF. The paper reports improvements in both accuracy and
efficiency vs non-fine-tuned FNO3d.

---

## Paper 6: Examining Robustness of PINNs to Noise (arXiv 2509.20191, 2025)

- **Paper**: "Examining the robustness of Physics-Informed Neural Networks to noise for Inverse Problems"
- **ArXiv**: 2509.20191
- **Methods**: PINN vs FEM/SLSQP vs PINN/FEM hybrid

### Setup (2D Taylor-Green Vortex, Inverse Problem)
- Domain: x,y in [0, 2pi], t in [0, 2.5]
- Spatial resolution: dx = dy = 0.05
- Temporal resolution: dt = 0.1
- Ground truth viscosity: nu = 0.1 (Re ~ 100)
- Total observations: 396,900
- Training: 5,000 samples (~1.26%)

### Key Findings
- FEM/SLSQP outperforms PINN for **all tested noise levels** on 2D TGV
- PINN systematically underestimates viscosity up to noise level sigma=0.25
- Results are in figures (not tables); exact error values not enumerated

---

## Paper 7: CFDONEval (IJCAI 2025)

- **Paper**: "CFDONEval: A Comprehensive Evaluation of Operator-Learning Neural Network Models for Computational Fluid Dynamics"
- **Venue**: IJCAI 2025 Proceedings (paper 640)
- **Methods**: 12 operator-learning models including FNO, U-Net, GNOT, DeepONet, GFormer, NU-FNO
- **Benchmarks**: 7 problems including Taylor-Green vortex

### Key Info
- 22 datasets, 18 newly generated
- 8 metrics including accuracy, efficiency, and kinetic energy spectra
- Attention-based models handle most challenges well
- U-shaped models excel at multiscale problems
- NU-FNO has smallest relative L2 error for nonuniform grid data

**Note**: This is the most comprehensive operator-learning benchmark that includes TGV, but
the full error table is in the PDF (binary, could not extract). This paper would be the most
valuable reference for your purpose.

---

## Paper 8: PINO (Li et al., arXiv 2111.03794)

- **Paper**: "Physics-Informed Neural Operator for Learning Partial Differential Equations"
- **ArXiv**: 2111.03794
- **Venue**: ACM/IMS Journal of Data Science

### Taylor-Green Vortex Status
**NOT included as a benchmark in this paper.** PINO benchmarks on:
- Burgers equation
- Darcy flow
- Navier-Stokes (Kolmogorov flow, lid cavity flow, long temporal transient flow)

### Relevant NS Results for Reference

| Method | Relative Error | Notes |
|--------|---------------|-------|
| FNO    | 3.04%         | Kolmogorov flow |
| PINO (no fine-tune) | 2.87% | Kolmogorov flow |
| PINO (fine-tuned)   | 1.84% | Kolmogorov flow, 400x speedup |
| PINN   | 18.7%         | Kolmogorov flow, fails to converge |

---

## Papers That Do NOT Include 2D Taylor-Green Vortex

These major benchmark papers were checked and confirmed to NOT include TGV:

| Paper | ArXiv | What They Benchmark Instead |
|-------|-------|-----------------------------|
| PDEBench (NeurIPS 2022) | 2210.07182 | Advection, Burgers, Darcy, NS (compressible/incompressible), Diffusion-Reaction, Shallow Water |
| FNO (Li et al. 2020) | 2010.08895 | Burgers, Darcy, NS (vorticity) |
| DeepONet-FNO comparison | 2111.05512 | 16 benchmarks, no TGV |
| PINNacle (NeurIPS 2024) | 2306.08827 | ~20 PDEs including Burgers, NS, KS, but TGV not confirmed in main text |
| SpecB-FNO (2024) | 2404.07200 | NS, Darcy, Shallow Water, Diffusion-Reaction |

---

## Conclusions

1. **The 2D Taylor-Green vortex is NOT a standard benchmark in the major neural operator papers** (FNO, DeepONet, PINO, PDEBench). These papers prefer Kolmogorov flow, Darcy flow, or vorticity-based NS.

2. **The best tabulated results come from FS-HNN (2026)**: FNO achieves MSE 3.43e-4, while FS-HNN achieves 5.01e-8 on the vorticity formulation.

3. **For PINNs**: MindSpore tutorial shows final relative L2 ~2.1% at 500 epochs. Chuang & Barba show PINN needs orders of magnitude more compute than CFD for similar accuracy (~10^{-3}).

4. **CFDONEval (IJCAI 2025) is the most comprehensive operator-learning benchmark including TGV**, comparing 12 models. The full error table in the paper PDF is the best resource but could not be extracted here.

5. **For your own benchmarking**, the most relevant comparison would be:
   - PINN baseline: relative L2 ~2-4% (from MindSpore/Barba results)
   - FNO baseline: MSE ~3.4e-4 (from FS-HNN paper)
   - Newer methods: FS-HNN achieves MSE ~5e-8 (but different formulation - vorticity)
