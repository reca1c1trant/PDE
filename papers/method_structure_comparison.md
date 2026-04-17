# PDE Foundation Model Papers: Method Section Structure Comparison

## 1. DPOT (ICML 2024)

### Subsection Headings
- **3. Proposed Method**
  - 3.1. Overview of DPOT
  - 3.2. Auto-regressive Denoising Pre-training
  - 3.3. Data Preprocessing and Sampling
  - 3.4. Model Architecture
  - 3.5. Analysis of Our Model

### Logic Chain
Problem formulation (general PDE form) -> Training paradigm (AR denoising) -> Data pipeline (padding, masking, balanced sampling) -> Architecture (Fourier attention) -> Theoretical analysis (universal approximation)

### How They Handle Key Elements
- **Problem setup**: Starts with general parametric PDE formulation (Eq.1), defines operator learning as predicting next T frames from previous frames. Mathematical and concise.
- **Architecture**: Detailed subsection (3.4) covering input encoding, temporal aggregation, Fourier attention layer, multi-head structure. Equations 5-11 give full forward pass.
- **Pretrain**: Core innovation = denoising AR pretrain (Sec 3.2). Noise injection into inputs, one-step prediction loss (Eq.3). No separate "pretrain section" -- it IS the method.
- **Finetune**: NOT described in method. Only mentioned in experiments (Sec 4) as "fine-tuning on each subset for 200/500 epochs."
- **Loss function**: Simple MSE with noise (Eq.3). Loss introduced early in Sec 3.2, right after training paradigm is motivated.

### Key Structural Pattern
**"Training trick first, architecture second."** The denoising pretrain strategy (the core contribution) comes BEFORE the architecture description. This is unusual -- most papers describe architecture first. DPOT prioritizes "how we train" over "what we train."

---

## 2. OmniArch (ICML 2025)

### Subsection Headings
- **3. (no explicit section title -- starts with three challenge paragraphs: Multi-Scale, Multi-Physics, Physical Alignment)**
  - 3.1. Pre-training OmniArch: Flexibly Learning from Different Dynamic Systems
    - 3.1.1. Encoder/Decoder in Fourier Domain
    - 3.1.2. Transformer as an Integral Neural Operator
  - 3.2. Fine-tuning OmniArch: Enabling Physics-Informed Learning via Equation Supervision

### Logic Chain
Three challenges (motivation) -> Pretrain framework (Fourier encoder/decoder + Transformer backbone + Temporal Mask + nRMSE loss) -> Finetune framework (PDE-Aligner for physics alignment, contrastive loss in frequency domain)

### How They Handle Key Elements
- **Problem setup**: NO formal PDE formulation. Instead, frames the problem as three challenges (Multi-Scale, Multi-Physics, Physical Alignment). Motivation-driven, not math-driven.
- **Architecture**: Interleaved with pretrain description. Fourier encoder/decoder (3.1.1) and Transformer backbone (3.1.2) are sub-subsections under "Pre-training." Architecture is presented as serving the pretrain goal.
- **Pretrain**: Section 3.1 = pretrain. Covers architecture + training in one unified section. Loss = nRMSE (Eq.8).
- **Finetune**: Explicitly separated as Section 3.2. Introduces PDE-Aligner: a contrastive learning module that aligns frequency-domain physics signatures with textual PDE descriptions. Fine-tune loss = L_sim - L_eq (data accuracy + physics consistency).
- **Loss function**: Two-stage losses clearly separated:
  - Pretrain: nRMSE similarity loss L_sim (Eq.8)
  - Finetune: Alignment loss L_Align = L_eq + lambda * L_E (Eq.9), with energy conservation term
  - Final finetune loss: L_ft = L_sim - L_eq

### Key Structural Pattern
**"Pretrain and finetune as two parallel pillars."** OmniArch is the only paper that gives finetune its own top-level method section (3.2) at equal weight to pretrain (3.1). The PDE-Aligner is a separate trainable module, not just "run more epochs with a different loss." This makes the method section read as two distinct contributions.

---

## 3. PDE-Transformer (ICML 2025)

### Subsection Headings
- **3. PDE-Transformer** (short notation paragraph first)
  - 3.1. Design Space
    - Patching
    - Multi-scale architecture
    - Shifted Windows
    - Mixed and separate channel representations
    - Conditioning mechanism
    - Boundary conditions
    - Algorithmic improvements
  - 3.2. Supervised and Diffusion Training

### Logic Chain
Notation -> Design space (7 design choices, each as a bold paragraph) -> Training modes (supervised MSE + diffusion/flow matching)

### How They Handle Key Elements
- **Problem setup**: Minimal. A short "Notation" paragraph defines the spatiotemporal system S and autoregressive prediction task. No formal PDE equation. Just input/output notation.
- **Architecture**: Dominates the method (Sec 3.1 is ~80% of the method). Structured as a "design space exploration" -- each design choice (patching, multi-scale, windows, channels, conditioning, BCs, algorithmic tricks) is a bold-headed paragraph. No sub-subsection numbers.
- **Pretrain**: NOT separated from training. Section 3.2 describes both supervised and diffusion training in ~0.5 page total. Pretrain and finetune are the same architecture, just different datasets.
- **Finetune**: Mentioned only in experiments (Sec 4.2) as downstream tasks. No method description for finetune.
- **Loss function**: Two losses in Sec 3.2:
  - Supervised: MSE loss L_S (Eq.1)
  - Diffusion: Flow matching loss L_FM (Eq.3)
  - Both are standard, presented without much discussion.

### Key Structural Pattern
**"Architecture-as-design-space."** PDE-Transformer treats the method as a systematic exploration of design choices, not a single coherent model description. Each design decision gets its own paragraph with justification. This mirrors the DiT paper's style. The training is almost an afterthought (~10% of method text).

---

## 4. MORPH (Preprint 2026)

### Subsection Headings
- **3. Proposed Method**
  - 3.1. Unified Physics Tensor Format (UPTF-7)
  - 3.2. Overview of MORPH
  - 3.3. Transfer across data modalities

### Logic Chain
Data format (UPTF-7) -> Architecture overview (3 key mechanisms: component-wise convolutions, inter-field cross-attention, 4D axial attention) -> Cross-modality transfer analysis (Gap-Closure Ratio)

### How They Handle Key Elements
- **Problem setup**: NO formal PDE formulation. Starts with the data heterogeneity problem and proposes UPTF-7 format (B,T,F,C,D,H,W) as the solution. Problem = "how to handle diverse data shapes."
- **Architecture**: Section 3.2. Three mechanisms described in flowing prose (no sub-subsections): (a) component-wise convolutions for scalar/vector channels, (b) field-wise multi-head cross-attention for inter-field fusion, (c) 4D factorized axial attention (time, depth, height, width). 
- **Pretrain**: Described briefly in Sec 4 (Experiments), not in Method. AR(1) self-supervised, MSE loss. "We train MORPH in a self-supervised AR(1) setting where we learn a map F_theta: X -> X such that X_{t+1} approx F_theta(X_t)."
- **Finetune**: Mentioned in experiments. LoRA-based fine-tuning discussed as a result, not a method contribution. "77M of the 480M parameters comprising LoRA adapters on attention and MLP blocks."
- **Loss function**: Implicitly MSE. No explicit loss equation in the method section. Loss is treated as obvious/standard.

### Key Structural Pattern
**"Data format as first-class contribution."** MORPH is unique in dedicating its first method subsection (3.1) entirely to the data representation format (UPTF-7). The paper argues that the right data format is as important as the right architecture. Section 3.3 on cross-modality transfer is also unusual -- it's an analysis/evaluation framing embedded in the method.

---

## 5. BCAT (Preprint 2025)

### Subsection Headings
- **3. Methods**
  - 3.1. Problem Setup
  - 3.2. Model Overview
  - 3.3. Image Tokenization via Patch Embedding
  - 3.4. Preliminary: Next Token Prediction
  - 3.5. Next Frame Prediction
  - 3.6. Muon Optimizer
  - 3.7. Implementation Details

### Logic Chain
Problem setup -> Model overview -> Tokenization -> Background (NTP) -> Core method (next frame prediction with block causal mask) -> Optimizer choice -> Implementation details

### How They Handle Key Elements
- **Problem setup**: Clean and formal (Sec 3.1). Defines u(x,t) in R^d, input T_0 frames, predict T future frames. Short and precise.
- **Architecture**: Described across multiple sections (3.2-3.5). GPT-2 style decoder-only transformer. Patch embedding (3.3) -> block causal attention mask (3.5, with explicit mask matrix Eq.2). Architecture is simple; the mask design is the key innovation.
- **Pretrain**: Not explicitly separated. The model is trained end-to-end on mixed PDE datasets. "Training is performed end-to-end by minimizing the MSE between predicted and ground truth sequences."
- **Finetune**: NOT discussed in method. Only in experiments.
- **Loss function**: MSE between normalized predicted and ground truth. Stated in one sentence in Sec 3.2, no equation. Remarkably minimal.

### Key Structural Pattern
**"Preliminary before core method."** BCAT uniquely includes a "Preliminary" section (3.4) on next token prediction BEFORE presenting its actual method (next frame prediction, 3.5). This creates a contrast: "here's what LLMs do (NTP) -> here's why that's suboptimal for PDE -> here's our solution (NFP)." Also unique: dedicates a full subsection (3.6) to the optimizer (Muon), treating optimizer choice as a method contribution.

---

## 6. PINO (ICLR 2024) -- CRITICAL FOR OUR APPROACH

### Subsection Headings
- **2. Preliminaries and problem settings**
  - 2.1. Problem settings
  - 2.2. Solving equation using the physics-informed neural networks
  - 2.3. Learning the solution operator via neural operator
  - 2.4. Neural operators
- **3. Physics-informed neural operator (PINO)**
  - 3.1. Physics-informed operator learning
  - 3.2. Instance-wise fine-tuning of trained operator ansatz
  - 3.3. Derivatives of neural operators

### Logic Chain
Problem settings (stationary + dynamic PDEs) -> PINN background (physics-informed loss) -> Neural operator background (data loss) -> PINO framework (combine both) -> Operator learning phase (J_data + J_pde) -> Instance-wise fine-tuning (L_pde + L_op anchor loss) -> How to compute derivatives (numerical, autograd, function-wise)

### How They Handle Key Elements
- **Problem setup**: Extensive and rigorous (Sec 2.1-2.4, ~3.5 pages). Defines both stationary and dynamic PDE systems with full equations (Eq.1-2). Then separately defines PINN loss (Eq.3-4), data-driven operator loss (Eq.5-6), PDE operator loss (Eq.7). This is the most thorough problem setup of all 6 papers.
- **Architecture**: Minimal in method. FNO is described via Definition 1-3 in Sec 2.4. Architecture is NOT a contribution -- it's infrastructure.
- **Pretrain** (= operator learning): Section 3.1. Uses J_data AND/OR J_pde together. Key insight: "one can sample unlimited virtual PDE instances by drawing additional initial conditions." Physics constraints make it semi-supervised.
- **Finetune** (= instance-wise fine-tuning): Section 3.2. Uses learned operator as ansatz, optimizes L_pde + alpha * L_op (anchor loss). Explicitly separated from operator learning. Discusses optimization landscape advantages.
- **Loss function**: The most detailed treatment of any paper:
  - Data loss L_data (Eq.5): ||u - G_theta(a)||^2
  - PDE loss L_pde (Eq.3-4): residual + BC + IC terms, with hyperparameters alpha, beta
  - Operator data loss J_data (Eq.6): expectation over instances
  - Operator PDE loss J_pde (Eq.7): expectation of L_pde
  - Anchor loss L_op: ||G_theta_i(a) - G_theta_0(a)||^2
  - Total fine-tune loss: L_pde + alpha * L_op
- **PDE residual equations**: YES, explicitly. Eq.3 (stationary) and Eq.4 (dynamic) give the full PDE residual loss with interior, boundary, and initial condition terms.

### Key Structural Pattern
**"Preliminaries do the heavy lifting."** PINO has the most extensive preliminaries section (~4 pages), which defines ALL the building blocks (PINN loss, operator loss, FNO) BEFORE the actual PINO method. The method itself (Sec 3) is then remarkably concise because all notation is established. Also unique: Section 3.3 on computing derivatives of neural operators is a standalone technical contribution -- no other paper has this.

---

## Cross-Paper Comparison Matrix

| Aspect | DPOT | OmniArch | PDE-Trans | MORPH | BCAT | PINO |
|--------|------|----------|-----------|-------|------|------|
| **Starts with** | PDE formulation | Motivation (3 challenges) | Notation | Data format | Problem setup | Problem settings (formal) |
| **Problem formulation** | Formal (Eq.1) | None (challenge-based) | Minimal notation | None (data-centric) | Formal but brief | Most rigorous (Eq.1-2) |
| **Architecture depth** | Detailed (Sec 3.4) | Interleaved with pretrain | Dominates (~80%) | 1 section, prose | Spread across 3.2-3.5 | Minimal (in prelim) |
| **Pretrain in method?** | Yes (core, Sec 3.2) | Yes (Sec 3.1) | Brief (Sec 3.2) | No (in experiments) | No (implicit) | Yes (Sec 3.1) |
| **Finetune in method?** | No | Yes (Sec 3.2, equal weight) | No | No | No | Yes (Sec 3.2) |
| **Loss equation count** | 1 (Eq.3) | 2 (Eq.8-9) | 2 (Eq.1,3) | 0 | 0 | 7+ (Eq.3-7 + anchor) |
| **PDE residual loss?** | No | Partial (freq domain) | No | No | No | Yes, explicit (Eq.3-4) |
| **Theoretical analysis** | Yes (Thm 3.1) | No | No | No | No | Convergence discussion |
| **Unique element** | Denoising before arch | PDE-Aligner module | Design space format | UPTF-7 data format | Optimizer as method | Derivative computation |

## Key Takeaways for Our Paper

1. **Problem formulation**: PINO-style (formal PDE + operator learning framing) is strongest when physics loss is a contribution. If our contribution is physics-informed finetuning, we need rigorous problem setup like PINO, not the minimal style of BCAT/PDE-Transformer.

2. **Pretrain vs Finetune separation**: Only OmniArch and PINO give finetune equal method-section weight. Since our contribution involves physics-informed finetuning of a pretrained model, we should follow OmniArch/PINO's structure with explicit pretrain and finetune subsections.

3. **Loss function placement**: PINO introduces all loss components in preliminaries, then combines them in the method. OmniArch introduces losses inline with each stage. For a hybrid data+physics approach, PINO's style (define components first, combine later) is cleaner.

4. **PDE residual loss**: Only PINO writes out the full PDE residual loss with BC/IC terms. If we use physics-informed loss, we MUST include explicit residual equations -- this is what distinguishes our work from pure data-driven approaches.

5. **Architecture vs Training split**: If our architecture is not novel (using existing backbone), follow MORPH/PINO style (architecture in ~1 paragraph, training pipeline gets the space). If architecture has modifications, follow DPOT style (but still put training contribution first).

6. **Data pipeline**: DPOT and MORPH both dedicate significant space to data preprocessing (padding, normalization, sampling). If multi-dataset training is part of our story, this deserves its own subsection.
