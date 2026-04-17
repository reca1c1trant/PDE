# NeurIPS 2026 投稿完整攻略

> 最后更新：2026-03-24
> 来源：NeurIPS 官方网站 (neurips.cc/Conferences/2026)

---

## 1. 关键时间节点

### 主会议论文 & Evaluations/Datasets Track

| 事件 | 日期 | 时区 |
|------|------|------|
| 投稿系统开放 | 2026年4月5日 | - |
| **摘要提交截止** | **2026年5月4日** | AOE（地球上任何地方） |
| **全文 + 补充材料提交** | **2026年5月6日** | AOE |
| 作者通知（录用/拒稿） | **2026年9月24日** | AOE |

### 审稿流程时间线（根据 NeurIPS 2025 推算）

NeurIPS 2026 **尚未公布**完整审稿/rebuttal 时间线，以下为基于 2025 年的估算：

| 阶段 | NeurIPS 2025 日期 | NeurIPS 2026 预计 |
|------|-------------------|-------------------|
| 审稿开始 | 5月29日 | ~5月底 / 6月初 |
| 审稿结束 | 7月2日 | ~7月初 |
| 作者 rebuttal 开始 | 7月24日 | ~7月下旬 |
| 作者 rebuttal 结束 | 7月30日 | ~7月底 / 8月初 |
| 审稿人-作者讨论 | 7月31日-8月6日 | ~8月初 |
| 结果通知 | 9月18日 (2025) | **9月24日 (2026 已确认)** |
| Camera-ready 截止 | 10月23日 (2025) | ~10月下旬 (待公布) |

### 会议日期（已确认）

| 活动 | 日期 |
|------|------|
| Expo Talks | 2026年12月6日（周日） |
| 线上通行证 | 12月6-12日 |
| Tutorials | 12月7日（周一） |
| **正式会议** | **12月8日（周二）— 12月10日（周四）** |
| Workshops | 12月11日（周五）— 12月12日（周六） |

**地点：** 澳大利亚悉尼 — 国际会议中心（ICC Sydney）

---

## 2. 投稿格式要求

### 页数限制
- **正文最多 9 页**（含图表）
- **不计入 9 页的部分：** 参考文献、致谢（仅 camera-ready）、论文 checklist、技术附录
- **附录：** 无页数限制
- 超过 9 页的论文 **不会被审稿**

### LaTeX 模板
- **只接受 LaTeX 投稿**（Word 模板已停用）
- 样式文件：`neurips_2026.sty`
- 官方 Overleaf 模板：https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc
- 官方 ZIP 下载：NeurIPS 2026 投稿系统中提供

### 样式文件选项
```latex
\usepackage[<选项>]{neurips_2026}
```

| 选项 | 用途 |
|------|------|
| （默认 / `main`） | Main Track，双盲匿名 |
| `final` | Camera-ready 版本（去匿名化） |
| `preprint` | ArXiv 预印本（底部显示"Preprint. Work in progress."） |
| `nonatbib` | 避免 natbib 包冲突 |
| `position` | Position Paper Track |
| `eandd` | Evaluations & Datasets Track |
| `creativeai` | Creative AI Track |
| `sglblindworkshop` | 单盲 Workshop |
| `dblblindworkshop` | 双盲 Workshop |

### 排版与布局
- **文本区域：** 5.5 英寸宽 × 9 英寸长
- **左边距：** 1.5 英寸；上边距：1 英寸
- **正文字体：** Times New Roman，10pt，行距 11pt
- **标题：** 17pt，加粗，居中
- **参考文献：** 9pt 字体，natbib 包（默认）
- **纸张大小：** US Letter
- **PDF 字体：** 仅限 Type 1 或 Embedded TrueType
- 使用 `pdflatex` 生成

### 补充材料
- **大小限制：** 100MB（单个 ZIP 文件）
- 提交时必须匿名化
- 小数据集可直接包含；大数据集通过匿名 URL 提供
- Camera-ready 时去匿名化

---

## 3. 审稿流程

### 双盲审稿
- Main Track 和 E&D Track **均为双盲**（默认）
- 引用自己已发表的工作时使用**第三人称**
- 提交时不要使用 `final` 和 `preprint` 选项（行号会自动添加）
- E&D Track 的数据集论文可选择**单盲**审稿（`\usepackage[eandd, nonanonymous]{neurips_2026}`）

### 审稿结构
- **每篇论文审稿人数：** 通常 3-4 人（由 Area Chair 分配）
- **Area Chairs (ACs)：** 管理约 20 篇论文，撰写 meta-review
- **Senior Area Chairs (SACs)：** 监督约 10 个 AC，确保评分一致性
- **Program Chairs：** 最终决定

### 审稿阶段

1. **论文竞选（Bidding）：** 审稿人对匹配专业领域的论文进行竞选
2. **撰写审稿意见：** 审稿人提交实质性评审
3. **预回复 Meta-review（2026 新增）：** AC 在作者 rebuttal **之前**就撰写初步 meta-review，将论文分类为：
   - 倾向拒稿
   - 需要澄清
   - 计划接收
4. **作者回复：** 每条 review 限 10,000 字符（markdown 格式），不可上传新文件
5. **第一阶段：** 作者撰写回复（审稿人/AC 暂不可见）
6. **第二阶段：** 各方讨论；作者可回应追问
7. **第三阶段：** 审稿人和 AC 私下讨论；作者无法参与
8. **最终 Meta-review：** AC 在讨论后修改 meta-review

### 评分系统

**四个评估维度**（各评 1-4 分）：
- Quality（技术可靠性）
- Clarity（写作与组织）
- Significance（影响力、被采用的可能性）
- Originality（新颖见解、方法）

**总分（1-6 分）：**

| 分数 | 含义 |
|------|------|
| 6 | Strong Accept — 突破性影响 |
| 5 | Accept — 对至少一个子领域有高影响 |
| 4 | Borderline Accept — 接收理由多于拒稿理由 |
| 3 | Borderline Reject — 拒稿理由多于接收理由 |
| 2 | Reject — 技术缺陷或评估薄弱 |
| 1 | Strong Reject — 已知结果或未处理的伦理问题 |

**置信度评分：** 1-5（对相关工作的熟悉程度）

### 贡献类型
作者需指定贡献类型；审稿人会获得相应的补充指南：
- **General**：大多数论文
- **Theory**：强调证明/分析
- **Use-Inspired**：实际应用影响
- **Concept & Feasibility**：高风险初步工作（门槛更高）
- **Negative Results**：理解失败案例（门槛更高）

### 伦理审查
- 审稿人可随时将论文**标记**进行伦理审查
- 伦理审稿人加入委员会；其意见在 rebuttal 期间对作者可见
- 伦理审稿人**不能单独拒稿**，但 Program Chair 可基于伦理原因拒稿（不论科学质量）
- 遵循 NeurIPS 伦理准则

---

## 4. 2026 年新政策

### 相比 2025 年的重大变化

1. **Evaluations & Datasets (E&D) Track** — 从"Datasets & Benchmarks Track"更名，范围大幅扩展。现在将评估视为科学研究对象（审计、红队测试、benchmark 分析、文档方法论、负面结果）

2. **预回复 Meta-review（新增）** — AC 在作者 rebuttal **之前**撰写 meta-review，给作者明确信号，让其知道需要重点回应什么

3. **LLM 审稿实验（新增）** — NeurIPS 2026 正在进行对照实验：
   - 每位审稿人的部分论文允许使用官方认可的 LLM 支持（通过 OpenReview）
   - 其他论文禁止使用任何 LLM
   - 审稿人在被允许时**只能使用官方 LLM**，不能用外部 LLM
   - 旨在研究 LLM 对审稿质量的影响

4. **审稿人-作者问责制** — 同时身为审稿人的作者，必须先完成**所有**分配的审稿任务，才能查看自己论文的审稿意见。严重失职的审稿人-作者可能面临 **desk rejection**

5. **会议地点变更** — 澳大利亚悉尼（ICC Sydney），而非往年的北美场地

### 作者使用 LLM 的政策
- 作者可以使用 LLM 作为工具
- 如果 LLM 作为**核心方法**使用，必须详细描述
- 使用 LLM 进行写作/编辑/排版**不需要**声明
- 作者对所有内容**完全负责**，无论使用了什么工具
- 如果使用方式"重要、原创或非标准"，需在 checklist 第 16 项声明

---

## 5. 论文 Checklist（强制要求）

NeurIPS 论文 checklist 是**所有投稿的必填项**，共 16 项：

1. **声明（Claims）** — 摘要/引言中的声明是否与贡献匹配？
2. **局限性（Limitations）** — 专门章节讨论假设、范围限制、失败案例
3. **理论、假设、证明** — 完整的证明（正文或补充材料），所有假设已陈述
4. **实验可复现性** — 算法描述、架构规格、模型访问、checkpoint
5. **数据和代码的开放获取** — 代码、数据、复现说明（补充材料或 URL）
6. **实验设置** — 训练细节、数据划分、超参数、选择方法
7. **统计显著性** — 误差棒、置信区间、显著性检验
8. **计算资源** — GPU/CPU 类型、内存、每次运行和总运行时间
9. **伦理准则** — 遵守 NeurIPS 伦理准则
10. **更广泛影响** — 负面社会影响、恶意用途、公平性、隐私
11. **安全保障** — 高风险模型的控制措施、数据集安全
12. **许可证** — 引用创建者、尊重许可证、包含 URL 和版本号
13. **资产** — 记录新发布的资产、训练细节、局限性
14. **众包 / 人类受试者** — 完整说明、最低工资合规
15. **IRB 批准** — 需要时提供机构审查委员会批准
16. **LLM 使用声明** — 如果 LLM 的使用对核心方法来说"重要/原创/非标准"则必须声明

---

## 6. 可投 Track

| Track | 模板选项 | 审稿类型 | 备注 |
|-------|---------|---------|------|
| **Main Track** | `main`（默认） | 双盲 | 标准研究论文 |
| **Evaluations & Datasets** | `eandd` | 双盲（默认）或数据集论文单盲 | 2026 新名称/新范围 |
| **Position Papers** | `position` | 双盲 | ML 的元视角 |
| **Creative AI** | `creativeai` | 待公布 | AI/ML 用于创意和艺术 |
| **Journal-to-Conference** | 无 | 无 | 截止：9月26日；容量 150 篇 |
| **Workshops** | `sglblindworkshop` / `dblblindworkshop` | 单盲或双盲 | 各种主题 |

### Evaluations & Datasets Track 细则
- 与 Main Track 相同截止日期（5月4日/6日）
- 数据集必须托管在 ML 平台（HuggingFace、Kaggle、Dataverse、OpenML）
- 必须使用 **Croissant** 机器可读元数据格式
- 数据集必须在 camera-ready 前公开可用
- Benchmark/工具类论文必须提交代码；分析类论文建议提交

---

## 7. LaTeX 模板详情

### 下载地址
- **官方：** NeurIPS 2026 投稿系统中的 ZIP 下载
- **Overleaf：** https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc

### 关键 LaTeX 命令
```latex
% 匿名投稿（审稿阶段）
\usepackage{neurips_2026}

% Camera-ready（去匿名化）
\usepackage[final]{neurips_2026}

% ArXiv 预印本
\usepackage[preprint]{neurips_2026}

% E&D Track（双盲）
\usepackage[eandd]{neurips_2026}

% E&D Track（单盲，数据集论文）
\usepackage[eandd, nonanonymous]{neurips_2026}

% 致谢（匿名模式下自动隐藏）
\begin{ack}
资助和利益冲突声明。
\end{ack}
```

### 图表最佳实践
- 使用 `\includegraphics`，宽度设为 `\linewidth` 的倍数
- 表格使用 `booktabs` 包（不用竖线）
- 图片：标题在下方；表格：标题在上方
- 必须在黑白打印下可辨认

---

## 8. 重复投稿政策

### 禁止
- 将实质性相似的工作**同时投到其他正式会议/期刊**（有论文集的）
- "切片投稿"（Thin slicing）— 投两篇非常相似的论文，期望其中一篇被录用
- 在 NeurIPS 审稿期间投到其他场所

### 允许
- **ArXiv 预印本：** 在 arXiv（或其他非同行评审平台）发布**不会**导致拒稿
- 已在 arXiv 上的匿名化工作可以直接投稿，无需引用
- ArXiv 版本使用 `\usepackage[preprint]{neurips_2026}`
- **Workshop 论文**（非正式出版）通常允许 — 具体看各 workshop 政策

### 同期工作
- 投稿截止后才出现的论文视为同期工作
- 作者无需与同期工作进行对比

---

## 9. 代码与可复现性要求

### 投稿时
- 提交代码**强烈建议**但非严格必须
- 必须在补充材料 ZIP 中**匿名化**（< 100MB）
- 应该自包含且可执行
- 如不可执行（特殊硬件、专有库），需提供解释
- 大数据集：使用匿名 URL

### Camera-Ready 时
- **去匿名化**所有代码和 URL
- 最佳实践：包含可复现主要实验结果的代码
- 遵循 Papers with Code 指南：https://github.com/paperswithcode/releasing-research-code

### 审稿人代码执行
- 审稿人**可以**运行提交的代码
- 必须使用安全环境（Docker、VM、隔离实例）
- 代码保密 — 不可分享或输入 LLM

### Checklist 相关要求
- 第 4 项：提供可复现途径（算法描述、架构、checkpoint）
- 第 5 项：主要结果的代码/数据/说明
- 第 6 项：所有训练细节（数据划分、超参数、选择方法）
- 第 7 项：误差棒、置信区间
- 第 8 项：计算资源（GPU 型号、内存、时间）

---

## 10. 伦理声明要求

### 论文中
- **不强制要求单独的"Broader Impact"章节**，但应讨论负面社会影响
- 局限性章节**必须有**（checklist 第 2 项）
- 讨论：恶意用途、公平性问题、隐私/安全问题
- 在适用时包含缓解策略

### 伦理审查流程
- 审稿人可随时标记论文进行伦理审查
- 伦理审稿人加入审稿委员会
- 其意见在 rebuttal 期间对作者可见
- 伦理审稿人**不能单独拒稿**
- Program Chair 可基于伦理原因拒稿（不论科学质量）

### 具体要求
- 高风险模型：实施安全保障（使用指南、访问限制）
- 网络爬取的数据集：解决安全问题
- 人类受试者：IRB 批准、众包工人最低工资
- LLM 使用：如果是方法论核心则须声明

---

## 投稿前 Checklist（快速参考）

- [ ] 摘要在 **2026年5月4日 AOE** 前提交
- [ ] 全文（最多 9 页）在 **2026年5月6日 AOE** 前提交
- [ ] 所有共同作者都有 OpenReview 账号
- [ ] 使用 `neurips_2026.sty` 并选择正确的 track 选项
- [ ] 论文已匿名化（未使用 `final` 或 `preprint` 选项）
- [ ] 正文、图表、元数据中无作者身份信息
- [ ] 引用自己的已发表工作时使用第三人称
- [ ] 16 项强制 checklist 已全部完成
- [ ] 包含局限性章节
- [ ] 补充材料已匿名化且 < 100MB
- [ ] PDF 使用 US Letter 纸张，Type 1/Embedded TrueType 字体
- [ ] 使用 `pdflatex` 生成
- [ ] 未同时投到其他正式会议/期刊
- [ ] 代码已匿名化（如包含）
- [ ] LLM 使用声明已填写（如适用，checklist 第 16 项）

---

## 重要链接

- NeurIPS 2026 主页：https://neurips.cc/Conferences/2026
- 征稿启事：https://neurips.cc/Conferences/2026/CallForPapers
- 日期与截止时间：https://neurips.cc/Conferences/2026/Dates
- Main Track 手册：https://neurips.cc/Conferences/2026/MainTrackHandbook
- E&D Track 征稿：https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
- E&D Track FAQ：https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ
- 论文 Checklist 指南：https://neurips.cc/public/guides/PaperChecklist
- 代码提交政策：https://neurips.cc/public/guides/CodeSubmissionPolicy
- OpenReview 投稿入口：https://openreview.net/group?id=NeurIPS.cc/2026/Conference
- Overleaf 模板：https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc
- Journal-to-Conference Track：https://neurips.cc/public/JournalToConference
