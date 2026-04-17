# Orthogonal Fine-Tuning & Spectral Methods: Structured Literature Survey

---

## 1. OFT: Controlling Text-to-Image Diffusion by Orthogonal Finetuning

**Authors:** Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf
**Venue:** NeurIPS 2023 (arXiv:2306.07280)
**Links:** [Paper](https://arxiv.org/abs/2306.07280) | [Project](https://oft.wyliu.com/)

### Key Mathematical Insight

Fine-tuning 可以看成对 neuron direction 的变换。OFT 的核心洞察是：**orthogonal transformation 是 flexibility 和 regularity 之间的 sweet spot** —— 它足够灵活来学习新任务，又足够受约束来保留预训练的几何结构。

### Core Formulation

**Weight Update:**
$$W = R \cdot W^0, \quad R^T R = R R^T = I$$

**Forward pass:**
$$z = W^T x = (R \cdot W^0)^T x$$

这等价于一个 additive update: $W = W^0 + (R - I)W^0$，但 $(R-I)$ 受正交约束，不像 LoRA 的低秩 additive update。

### Hyperspherical Energy (核心理论贡献)

定义 neuron 的 hyperspherical energy:
$$\text{HE}(W) := \sum_{i \neq j} \|\hat{w}_i - \hat{w}_j\|^{-1}, \quad \hat{w}_i = w_i / \|w_i\|$$

**Theorem:** 对所有 neuron 施加相同的正交变换 $R$，pairwise angular similarity 被精确保持：
$$\cos\theta_{ij} = \hat{w}_i \cdot \hat{w}_j = \widehat{Rw_i} \cdot \widehat{Rw_j}$$

因为正交变换保持内积: $\langle Rx, Ry \rangle = \langle x, y \rangle$。

**物理直觉:** Hyperspherical energy 编码了 neuron 之间的相对关系（angular structure）。Fine-tuning 如果破坏这些关系，就会丢失预训练学到的 feature representation。OFT 通过正交约束 provably 保持这些关系。

### Block-Diagonal Parameterization

为减少参数量，使用 block-diagonal 结构：
$$R = \text{diag}(R_1, R_2, \ldots, R_r), \quad R_i \in O(d/r)$$

- **参数量:** $O(d^2/r)$（通过 Cayley 参数化 + block sharing 可降至 $O(d^2/r^2)$）
- **Cayley 参数化:** $R_i = (I + Q_i)(I - Q_i)^{-1}$，其中 $Q_i$ 是 skew-symmetric

### COFT (Constrained OFT) 变体

加额外的 radius constraint: $\|R - I\| \leq \epsilon$，通过 Neumann series 近似实现，限制 fine-tuning 偏离预训练模型的程度。

### 与 LoRA 的数学对比

| 方面 | OFT | LoRA |
|------|-----|------|
| Update 形式 | $W = R \cdot W^0$（乘法） | $W = W^0 + BA$（加法） |
| 约束 | $R^TR = I$（正交） | $\text{rank}(BA) = r$（低秩） |
| Angular 保持 | **精确保持** | 不保证 |
| 参数量 | $O(d^2/r)$ | $O(r(d+n))$ |
| Spectral norm | 保持 | 不保证 |

### Why Orthogonality Helps

1. **Angular preservation:** neuron 之间的 cosine similarity 不变 → 保持 feature 的判别性
2. **Spectral norm preservation:** $\sigma_i(RW) = \sigma_i(W)$ → 训练稳定性
3. **Hyperspherical energy invariance:** 全局 feature 结构被保持
4. **不 collapse:** 不会让不同 neuron 变得更相似（避免 representation collapse）

---

## 2. BOFT: Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization

**Authors:** Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, et al.
**Venue:** ICLR 2024 (arXiv:2311.06243)
**Links:** [Paper](https://arxiv.org/abs/2311.06243) | [Project](https://boft.wyliu.com/)

### Key Mathematical Insight

OFT 的 block-diagonal 结构限制了表达力（只能在 block 内旋转，block 间不通信）。BOFT 把正交矩阵生成看成 **信息传输问题**：如何用最少的边（参数）实现全连接（dense orthogonal matrix）？答案来自 FFT 的 butterfly 结构。

### Core Formulation

**Butterfly Factorization:**
$$R(m, b) = \prod_{i=1}^{m} \tilde{B}^b_{(d,i)}$$

每个 butterfly factor $\tilde{B}^b_{(d,i)}$ 本身是正交的，它们的乘积也是正交的。

**Butterfly Factor 结构:**
$$B^F(k) = \begin{bmatrix} \text{diag}(d_1) & \text{diag}(d_2) \\ \text{diag}(d_3) & \text{diag}(d_4) \end{bmatrix} \in \mathbb{R}^{k \times k}$$

### 与 FFT 的连接

Cooley-Tukey FFT 把 $N$-point DFT 递归分解为两个 $N/2$-point DFT，产生的 butterfly topology 恰好实现了：
- **Dense connectivity:** 每个 source node 都能到达每个 receiver node
- **Logarithmic edges:** 只需 $O(d \log d)$ 条边（vs $O(d^2)$ for dense）

### Parameter Efficiency

| 方法 | 参数量 | Dense? |
|------|--------|--------|
| Full orthogonal | $O(d^2)$ | Yes |
| OFT (block-diagonal) | $O(bd)$ | No (block-sparse) |
| **BOFT** | $O(d \log d)$ | **Yes** |
| LoRA | $O(2rd)$ | Low-rank |

当 $m = \log d, b = 2$ 时，BOFT 参数量为 $\frac{1}{2}(b-1)dm$。例如 $d=768$：OFT ~590K → BOFT ~5.3K。

### Key Theorem: Expressiveness

BOFT subsumes OFT as special case。通过 "kaleidoscope hierarchy"：
$$B_{d-1,1}(d) B_{d-1,2}^T(d) \cdots B_{1,1}(d) B_{1,2}^T(d)$$
可以逼近任意正交矩阵（完整正交群的 universal approximation）。

### Spectral Preservation

OFT 和 BOFT 都保持 singular values:
$$RU\Sigma V^T \implies \sigma_i(\text{new}) = \sigma_i(\text{old})$$

左乘正交矩阵只改变 left singular vectors，不改变 singular values 或 right singular vectors。

### 实验结果

- **GLUE (DeBERTaV3-base):** BOFT 89.89 vs LoRA 88.50 vs OFT 89.77
- **VTAB-1K (DINOv2-large, 19 tasks):** BOFT 77.4% vs OFT 77.3%
- **ControlNet:** BOFT error 5.667 vs OFT 6.407 vs LoRA 8.038
- **独特优势:** Weight interpolation — 将 butterfly component 设为 identity 可 progressively 回退到预训练模型

---

## 3. Muon Optimizer: Momentum Orthogonalized by Newton-Schulz

**Authors:** Keller Jordan, Jeremy Bernstein, Laker Newhouse, Vlado Boza, Yuchen Jin, Franz Cesista, Jiacheng You
**Date:** December 2024
**Links:** [Blog](https://kellerjordan.github.io/posts/muon/) | [Derivation by Bernstein](https://jeremybernste.in/writing/deriving-muon)

**Scaling Paper:** "Muon is Scalable for LLM Training"
**Authors:** Jingyuan Liu, Jianlin Su, Zhilin Yang et al. (Moonshot AI)
**Venue:** arXiv:2502.16982, Feb 2025
**Links:** [Paper](https://arxiv.org/abs/2502.16982) | [GitHub (Moonlight)](https://github.com/MoonshotAI/Moonlight)

### Key Mathematical Insight

Muon 把 SGD momentum 的 update matrix 替换为距它最近的 semi-orthogonal matrix。这等价于 **spectral norm 下的 steepest descent**：在给定 update magnitude 预算下，最大化 loss 的下降。

### Core Algorithm

```
M_t = μ·M_{t-1} + ∇L_t(W_{t-1})          # Momentum accumulation
O_t = NewtonSchulz(M_t)                     # Orthogonalize: O_t ≈ UV^T where M=UΣV^T
W_t = W_{t-1} - η_t(O_t + λ·W_{t-1})      # Update with weight decay
```

### Newton-Schulz Iteration

**目标:** 近似计算 $(MM^T)^{-1/2}M = UV^T$（polar decomposition 的正交因子）

**迭代公式:**
$$X_k = aX_{k-1} + b(X_{k-1}X_{k-1}^T)X_{k-1} + c(X_{k-1}X_{k-1}^T)^2 X_{k-1}$$

**系数:** $(a, b, c) = (3.4445, -4.7750, 2.0315)$

**初始化:** $X_0 = M_t / \|M_t\|_F$

这是一个 quintic polynomial $\varphi(x) = ax + bx^3 + cx^5$，在 singular value 空间上迭代逼近 sign function（将所有 singular values 映射到 1）。

### Why Orthogonalization Helps (Bernstein 的推导)

**问题:** minimize linearized loss subject to update magnitude constraint:
$$\min_{\Delta W} \langle \nabla_W L, \Delta W \rangle \quad \text{s.t.} \quad \|\Delta W\|_{\text{RMS}\to\text{RMS}} \leq \epsilon$$

**解:** $\Delta W \propto UV^T$，即只保留梯度的 singular vectors，丢弃 singular values 的 magnitude。

**直觉:** 普通梯度可能在某些方向上 magnitude 很大但不重要，在其他方向上很小但很重要。Orthogonalization "amplifies rare but important directions"——将所有方向的 update scale 统一化。

### Scaling (Moonshot AI Paper)

**关键发现:**
1. **Weight decay 必须加:** Vanilla Muon 的权重会无限增长
2. **Per-parameter scale 校正:** Update RMS = $1/\sqrt{\max(A,B)}$，需要乘以 $\sqrt{\max(A,B)}$ 来对齐不同 shape 的参数

**Scaling Law 结果:** Muon 达到 AdamW 同等性能只需 **~52% FLOPs**（~2x computational efficiency）

**Moonlight 模型:** 3B/16B MoE 模型，5.7T tokens，用 Muon 训练。

---

## 4. Spectral/SVD-Based Methods

### 4a. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

**Authors:** Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
**Venue:** ICML 2024 Oral (arXiv:2403.03507)
**Links:** [Paper](https://arxiv.org/abs/2403.03507) | [GitHub](https://github.com/jiaweizzhao/GaLore)

**Core Insight:** 不要对 weight 做低秩约束（like LoRA），而是对 gradient 做低秩投影。这样可以 full-parameter learning，但 optimizer state 内存大幅减少。

**核心公式:**
$$\tilde{G}_t = P_t \cdot \rho_t(P_t^T G_t Q_t) \cdot Q_t^T$$

其中 $P_t \in \mathbb{R}^{m \times r}$, $Q_t \in \mathbb{R}^{n \times r}$ 来自 gradient 的 SVD:
$$G_t = U \Sigma V^T \implies P_t = U_{:,:r}, \quad Q_t = V_{:,:r}$$

每 $T$ 步重新计算一次 SVD（subspace switching）。

**Convergence (Theorem 3.6):**
$$\|R_t\|_F \leq [1 - \eta(\kappa_{t-1} - L_A - L_B L_C D^2)] \|R_{t-1}\|_F$$

**Memory:** optimizer states 从 $O(mn)$ 降至 $O(mr + nr)$，实现 65.5% reduction。

**vs LoRA:**

| 方面 | GaLore | LoRA |
|------|--------|------|
| 投影对象 | Gradient | Weight |
| Learning trajectory | 精确（$r=\min(m,n)$ 时） | 被低秩约束改变 |
| Subspace | 动态切换 | 固定 |
| Pre-training | 有效 | 需要 full-rank warmup |

### 4b. LoRA+: Efficient Low Rank Adaptation of Large Models

**Authors:** Soufiane Hayou, Nikhil Ghosh, Bin Yu
**Venue:** ICML 2024 (arXiv:2402.12354)
**Links:** [Paper](https://arxiv.org/abs/2402.12354) | [GitHub](https://github.com/nikhil-ghosh-berkeley/loraplus)

**Core Insight:** LoRA 中 $A$ 和 $B$ 用相同 learning rate 是 suboptimal 的。通过 scaling argument（large width limit），$B$ 的 learning rate 应该远大于 $A$。

**公式:** 设 $\eta_A$ 和 $\eta_B$ 分别为 $A$ 和 $B$ 的 learning rate：
$$\eta_B = \lambda \cdot \eta_A, \quad \lambda \gg 1$$

**数学依据:** 在 large width $d$ 的 regime 下，$A$ 和 $B$ 的 gradient scale 不同。同一 learning rate 导致 $B$ 的更新不充分（feature learning 效率低）。

**结果:** 1-2% accuracy improvement + up to 2x speedup，无额外计算开销。

### 4c. SVDiff: Compact Parameter Space for Diffusion Fine-Tuning

**Authors:** Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, Feng Yang
**Venue:** ICCV 2023 (arXiv:2303.11305)
**Links:** [Paper](https://arxiv.org/abs/2303.11305) | [Project](https://svdiff.github.io/)

**Core Insight:** 只 fine-tune weight matrix 的 singular values，保持 singular vectors 不变。

**公式:** 预训练权重 $W = U \Sigma V^T$，只训练 $\Delta\Sigma$:
$$W' = U(\Sigma + \Delta\Sigma)V^T$$

**参数量:** 只需每层 $\min(m,n)$ 个参数（singular values），极其紧凑（StableDiffusion: 1.7MB vs DreamBooth: 3.66GB）。

**为什么有效:** Singular vectors 编码了 feature directions，singular values 编码了 feature importance。Fine-tuning 只调整 importance 而不改变 directions，保持了预训练的 feature space 结构。

---

## 5. Theoretical Analysis: Why Orthogonal Constraints Help

### 5a. MOFT: Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation

**Authors:** Fei Wu, Jia Hu, Geyong Min, Shiqiang Wang
**Date:** May 2025 (arXiv:2505.11235)
**Links:** [Paper](https://arxiv.org/abs/2505.11235)

**关键理论贡献:** 建立了 low-rank subspace 中正交变换保持 hyperspherical energy 的 **充要条件**。

**Theorem 4.1:** Angular preservation 要求 $R^T G R = G$，其中 $G = A^T A$。

**MOFT Update:**
$$W = A \cdot R \cdot B + W^{\text{res}}$$

其中 $W^{\text{pre}} = U\Sigma V^T$，$A = U_{:,:r}$，$B = \Sigma_{:r,:r} V_{:,:r}^T$，只训练 $r \times r$ 的正交矩阵 $R$。

**等价于:** full-space 的 $R_{\text{full}} = \begin{bmatrix} R & 0 \\ 0 & I_{d-r} \end{bmatrix}$

**Memory:** 43.4% average peak memory reduction vs OFT/BOFT.

**Insight:** **Bridge between OFT and LoRA** —— MOFT 在 principal subspace 中做正交变换，兼具 LoRA 的参数效率和 OFT 的几何保持性。

### 5b. HOFT: Householder Orthogonal Fine-tuning

**Authors:** Alejandro Moreno Arcas, Albert Sanchis
**Date:** May 2025, submitted to NeurIPS 2025 (arXiv:2505.16531)
**Links:** [Paper](https://arxiv.org/abs/2505.16531)

**关键理论贡献:** 证明了 **需要两个正交矩阵** 才能达到 full expressivity。

**Theorem:** 为了覆盖所有可能的 adapted matrix（保持相同 singular values），需要:
$$\hat{M} = Q_U M Q_V, \quad Q_U \in O(m), Q_V \in O(n)$$

单个正交矩阵（如 OFT）只改变 left singular vectors，存在 approximation gap。

**Householder 参数化:**
$$Q_U = I - U S^{-1} U^T$$

其中 $S$ 由 $U^T U$ 的上三角部分推导，用 Neumann series 近似逆: $S^{-1} \approx D^{-1} - D^{-1}AD^{-1}$。

**SHOFT 变体:** 加入 element-wise scaling $m$:
$$\hat{M} = Q_U \cdot \text{diag}(m) \cdot M \cdot Q_V$$

解耦 direction（正交）和 magnitude（scaling），$m$ 初始化为 1。

---

## Summary: Mathematical Taxonomy

### 为什么正交约束有效 —— 五个层面的解释

| 层面 | 解释 | 对应方法 |
|------|------|---------|
| **Angular** | 保持 neuron pairs 的 cosine similarity | OFT, BOFT, MOFT |
| **Spectral** | 保持 singular values → 训练稳定性 | OFT, BOFT, SVDiff |
| **Energetic** | Hyperspherical energy invariant → 全局 feature 结构保持 | OFT, COFT |
| **Optimization** | Spectral norm steepest descent → 最大化每步 learning | Muon |
| **Information** | Butterfly topology → 最少参数实现 dense connectivity | BOFT |

### 方法分类

```
Parameter-Efficient Fine-Tuning
├── Additive (Low-Rank)
│   ├── LoRA:     W = W₀ + BA          (低秩加法)
│   ├── LoRA+:    不同 lr for A, B      (修正 scaling)
│   └── GaLore:   Gradient projection   (投影梯度而非权重)
│
├── Multiplicative (Orthogonal)
│   ├── OFT:      W = R·W₀             (block-diagonal orthogonal)
│   ├── BOFT:     W = R(m,b)·W₀        (butterfly orthogonal, O(d log d))
│   ├── MOFT:     W = A·R·B + W_res    (principal subspace orthogonal)
│   └── HOFT:     W = Q_U·W₀·Q_V       (dual Householder orthogonal)
│
├── Spectral (SVD-Based)
│   └── SVDiff:   W = U(Σ+ΔΣ)V^T       (只调 singular values)
│
└── Optimizer-Level Orthogonalization
    └── Muon:     Update ← UV^T of momentum  (正交化 update direction)
```

### 关键 Trade-off

- **LoRA 系列:** 参数最少，但不保证几何保持；适合 "快速适应"
- **OFT 系列:** 精确保持 angular structure，但参数量较大（BOFT/MOFT 缓解）
- **Muon:** 不是 fine-tuning 方法，而是 training optimizer；正交化 update 而非 weight
- **GaLore:** 不是 PEFT，而是 memory-efficient full-parameter training
- **SVDiff:** 最 compact，但表达力受限于 singular value 空间

### Open Questions

1. **OFT + Muon?** 用 Muon optimizer 来训练 OFT 的正交参数，是否双重获益？
2. **MOFT 的理论边界:** principal subspace rank $r$ 太小是否会丢失关键信息？
3. **Butterfly 结构 for PDE:** BOFT 的 information transmission 框架是否适用于 PDE solver 的 fine-tuning？
4. **Spectral norm 约束 vs 正交约束:** Muon 只在 optimizer level 做正交化，OFT 在 weight level 做约束，哪个对 PDE task 更合适？

---

*Survey compiled: 2026-03-30*

## Sources

- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [BOFT Paper](https://arxiv.org/abs/2311.06243)
- [Muon Blog (Keller Jordan)](https://kellerjordan.github.io/posts/muon/)
- [Deriving Muon (Jeremy Bernstein)](https://jeremybernste.in/writing/deriving-muon)
- [Muon Scalability Paper (Moonshot AI)](https://arxiv.org/abs/2502.16982)
- [GaLore Paper](https://arxiv.org/abs/2403.03507)
- [LoRA+ Paper](https://arxiv.org/abs/2402.12354)
- [SVDiff Paper](https://arxiv.org/abs/2303.11305)
- [MOFT Paper](https://arxiv.org/abs/2505.11235)
- [HOFT Paper](https://arxiv.org/abs/2505.16531)
- [OFT v2 (Scalable)](https://arxiv.org/abs/2506.19847)
