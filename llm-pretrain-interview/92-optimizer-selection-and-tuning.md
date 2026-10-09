# 92. 优化器选择与调优

> **文档编号**: 92
> **所属部分**: 第十二部分 - 优化器理论与实现 (81-92)
> **代码位置**: `megatron/core/optimizer/__init__.py`, `megatron/core/optimizer/optimizer_config.py`, `megatron/core/optimizer/optimizer.py`, `megatron/core/optimizer/distrib_optimizer.py`, `megatron/core/optimizer_param_scheduler.py`, `megatron/training/arguments.py`, `megatron/training/training.py`
> **前置文档**: 81-91 优化器系列, 93-96 混合精度训练
> **后续文档**: 97-100 数据工程与完整训练流程

---

## 目录

1. [引言 (Introduction)](#1-引言-introduction)
2. [相关工作 (Related Work)](#2-相关工作-related-work)
3. [符号定义 (Notation)](#3-符号定义-notation)
4. [数学原理 (Mathematical Foundations)](#4-数学原理-mathematical-foundations)
5. [算法伪代码 (Pseudocode)](#5-算法伪代码-pseudocode)
6. [代码实现详解 (Implementation)](#6-代码实现详解-implementation)
7. [实验结果 (Experiments)](#7-实验结果-experiments)
8. [消融研究 (Ablation Studies)](#8-消融研究-ablation-studies)
9. [超参数分析 (Hyperparameters)](#9-超参数分析-hyperparameters)
10. [深入探讨 (Advanced Topics)](#10-深入探讨-advanced-topics)
11. [总结 (Conclusion)](#11-总结-conclusion)
12. [参考文献 (References)](#12-参考文献-references)

---

## 1. 引言 (Introduction)

### 1.1 概述

优化器选择与调优的目标不是找到一个“永远最好”的配置，而是在给定模型规模、数据质量、训练预算、精度模式和并行策略的情况下，构造一个可解释、可恢复、可扩展的训练方案。

在 Megatron-LM 中，优化器调优至少涉及三层决策：

1. **算法层**：AdamW、Adam、SGD、DistributedOptimizer。
2. **调度层**：学习率、warmup、decay、weight decay 调度。
3. **系统层**：FP16/BF16、梯度裁剪、分布式状态、checkpoint恢复。

本文档给出工程化选择矩阵和调参流程，帮助把文档81-91中的理论知识落到实际训练脚本。

### 1.2 前置知识

- Adam/AdamW/SGD 的数学更新。
- 学习率调度和 weight decay 的作用。
- 分布式优化器与混合精度训练。
- Megatron-LM 参数解析和训练循环。

### 1.3 文档组织

第2-4节总结优化器选择依据；第5-6节给出决策算法和代码入口；第7-9节给出实验、消融和调参表；第10节讨论前沿优化器和生产风险。

### 1.4 代码位置

关键代码路径：

- `megatron/core/optimizer/optimizer_config.py`：优化器配置数据类。
- `megatron/core/optimizer/__init__.py`：构造 Adam/AdamW/SGD、CPU offload、DistributedOptimizer。
- `megatron/core/optimizer/optimizer.py`：FP32/FP16/BF16 optimizer wrapper。
- `megatron/core/optimizer/distrib_optimizer.py`：分布式 optimizer state 与 checkpoint。
- `megatron/core/optimizer_param_scheduler.py`：LR/WD scheduler。
- `megatron/training/arguments.py`：命令行参数解析和校验。
- `megatron/training/training.py`：optimizer config 与 scheduler 创建。

---

## 2. 相关工作 (Related Work)

### 2.1 历史发展

深度学习优化器的发展可以概括为：

- **SGD + Momentum**：简单、泛化强、状态少，但对学习率敏感。
- **AdaGrad/RMSProp**：引入自适应学习率，改善稀疏或非平稳梯度。
- **Adam**：结合 momentum 与 RMSProp，成为 Transformer 训练基础。
- **AdamW**：解耦 weight decay，成为现代 LLM 预训练默认起点。
- **内存高效优化器**：Adafactor、ZeRO/DistributedOptimizer 等降低状态内存。
- **新型一阶/二阶优化器**：Lion、Shampoo/SOAP 等尝试改善更新效率或收敛。

### 2.2 技术对比

| 优化器/包装 | 适用场景 | 优势 | 风险 |
|-------------|----------|------|------|
| AdamW | LLM预训练默认起点 | 稳定、经验充分 | optimizer state大 |
| Adam | 兼容旧实验或特殊正则 | 与历史配置一致 | weight decay语义容易混淆 |
| SGD | 小模型、对照实验 | 状态少、简单 | Transformer预训练通常收敛慢 |
| DistributedOptimizer | 大模型DP训练 | 降低状态显存 | checkpoint/恢复更复杂 |
| CPU offload optimizer | 显存极限场景 | 减少GPU状态占用 | PCIe/CPU带宽可能成瓶颈 |

### 2.3 Megatron-LM中的实现

Megatron-LM 并不是只暴露 `torch.optim`。它通过 `OptimizerConfig` 和 `get_megatron_optimizer()` 将优化器选择、param group、混合精度 wrapper、DistributedOptimizer 和 scheduler 组合起来。因此调优时应看最终生成的 config 和日志，而不是只看训练脚本中的单个参数。

---

## 3. 符号定义 (Notation)

### 3.1 数学符号表

| 符号 | 含义 | 备注 |
|------|------|------|
| $P$ | 参数量 | 决定 optimizer state 内存 |
| $T$ | 总训练步数或 token budget | 决定 scheduler |
| $B_g$ | global batch size | 影响梯度噪声 |
| $\eta_{\max}$ | peak learning rate | warmup 后达到 |
| $\eta_{\min}$ | min learning rate | decay 末端 |
| $w$ | warmup steps | 早期稳定 |
| $\lambda$ | weight decay | AdamW中解耦 |
| $C$ | clip grad 阈值 | 全局梯度范数上限 |
| $\beta_1,\beta_2$ | Adam动量参数 | 控制一阶/二阶矩平滑 |

### 3.2 代码变量约定

- `args.optimizer` 与 `config.optimizer` 控制 optimizer 类型。
- `args.lr`, `args.min_lr`, `args.lr_decay_style` 控制 scheduler。
- `args.adam_beta1`, `args.adam_beta2`, `args.adam_eps` 控制 Adam 系列。
- `args.weight_decay`, `args.start_weight_decay`, `args.end_weight_decay` 控制 WD。
- `args.use_distributed_optimizer` 控制 DistributedOptimizer。
- `args.fp16`, `args.bf16` 控制 mixed precision wrapper。

---

## 4. 数学原理 (Mathematical Foundations)

### 4.1 核心理论

优化器选择可以视为约束优化问题：

$$
\max_{\mathcal{O}, h} \; Q(\mathcal{O}, h)
$$

其中 $\mathcal{O}$ 是优化器族，$h$ 是超参数集合，目标 $Q$ 同时考虑：

- validation loss 或 downstream 指标；
- 每 token 训练成本；
- 显存占用；
- 恢复和复现实验的可靠性；
- 在目标硬件上的吞吐。

实际工程中常用多目标约束形式：

$$
\min_{\mathcal{O}, h} \; \mathcal{L}_{val}
\quad
\text{s.t.}\quad
M(\mathcal{O}, h) \leq M_{GPU},
\quad
S(\mathcal{O}, h) \geq S_{min}
$$

其中 $M$ 是显存占用，$S$ 是训练稳定性指标，例如 skipped step 比例、grad norm异常比例。

### 4.2 优化器内存模型

对于参数量 $P$ 的模型：

- FP16/BF16模型参数约 $2P$ bytes。
- FP32 master参数约 $4P$ bytes。
- Adam一阶/二阶矩约 $8P$ bytes。
- 梯度约 $2P$ 或 $4P$ bytes，取决于存储精度。

不使用分布式优化器时，DP rank 上 optimizer state 完整复制。使用 DistributedOptimizer 后，主要 optimizer state 可按 data-parallel 维度近似降为 $1/N_d$，但会引入通信和checkpoint复杂度。

### 4.3 学习率与batch size

扩大 $B_g$ 会降低梯度噪声：

$$
\text{Var}(g) \propto \frac{1}{B_g}
$$

但学习率不能无限线性放大，因为大模型训练同时受 loss landscape 曲率、warmup、token分布和数值精度约束。实用策略是先固定 token budget 和模型结构，做短程 LR range test，再选择能稳定下降且不过度裁剪的最大安全 LR。

### 4.4 复杂度分析

| 决策 | 主要影响 | 复杂度变化 |
|------|----------|------------|
| AdamW vs SGD | 收敛与状态内存 | AdamW多 $2P$ 状态 |
| DistributedOptimizer | 显存与通信 | 状态约降为 $1/N_d$ |
| BF16 vs FP16 | 数值稳定 | BF16通常少 loss scaling |
| 更大 batch | 吞吐与泛化 | 梯度噪声降低，但调参更敏感 |
| 更长 warmup | 稳定性 | 消耗更多高LR前训练步 |

---

## 5. 算法伪代码 (Pseudocode)

### 5.1 优化器选择决策树

```text
Algorithm 92.1: Optimizer Selection for Megatron-LM Pretraining
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  model size P, GPU memory M, hardware dtype support,
  data quality, target token budget, parallel sizes
Output:
  optimizer family and initial hyperparameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: if training is LLM pretraining then
2:     choose AdamW as baseline
3: else if running small ablation or legacy baseline then
4:     consider SGD or Adam to match baseline
5: end if
6: if optimizer state does not fit GPU memory then
7:     enable DistributedOptimizer
8:     if still OOM then consider CPU offload or smaller parallel shard
9: end if
10: if hardware supports BF16 reliably then
11:     choose BF16
12: else
13:     choose FP16 with dynamic loss scaling
14: end if
15: set clip_grad = 1.0 as protection baseline
16: choose LR schedule based on token budget:
17:     long fixed budget -> cosine or WSD
18:     short debug run -> constant or linear
19: run short stability test
20: tune LR, warmup, betas, weight decay using diagnostics
```

### 5.2 调参循环

```text
Algorithm 92.2: Safe Tuning Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: Start from known AdamW + BF16 baseline.
2: Run 100-1000 step smoke test.
3: If loss spikes early, reduce LR or increase warmup.
4: If grad norm is always clipped, reduce LR or increase clip threshold cautiously.
5: If validation loss worsens while training loss falls, increase regularization or improve data.
6: If training is stable but slow, test larger LR or batch size one at a time.
7: If memory is the blocker, enable DistributedOptimizer before changing model quality knobs.
8: Lock config only after resume-from-checkpoint test succeeds.
```

---

## 6. 代码实现详解 (Implementation)

### 6.1 核心类与函数

**`OptimizerConfig`**

该 dataclass 是优化器选择的中心结构。调参时应把训练脚本参数最终解析到 config 的结果打印或记录下来，避免 YAML、CLI 和默认值覆盖造成误判。

**`get_megatron_optimizer()`**

该函数负责：

- 按参数分组设置 weight decay 和 lr multiplier。
- 根据 `config.optimizer` 选择 Adam/SGD。
- 根据 `config.decoupled_weight_decay` 切换 AdamW 语义。
- 根据混合精度和分布式选项包裹 optimizer。

**`OptimizerParamScheduler`**

调优 LR/WD 时要检查 `state_dict()` 和 `load_state_dict()` 行为。resume 训练时，如果 scheduler 状态未恢复，loss 曲线可能出现突变。

**`megatron/training/arguments.py`**

训练参数会在这里做互斥和默认值校验，例如 iter-based 与 sample-based scheduler 不能混用。调参计划中必须明确使用哪一种计数口径。

### 6.2 关键实现细节

**AdamW并不等于Adam加L2**

在自适应优化器中，把 L2 正则加入梯度会被二阶矩缩放；AdamW 的解耦衰减直接作用在参数上，调参语义更清晰。

**scheduler按step还是sample**

Megatron-LM 同时支持按 iteration 和 sample 设置 decay/warmup。大规模训练中，如果 global batch size 变化，sample-based 语义通常更便于保持 token budget 一致。

**param group multiplier**

部分参数可能通过 `lr_mult` 或 `wd_mult` 使用不同学习率/weight decay。调优报告需要记录这些 multiplier，否则同一个 `--lr` 并不代表所有参数的实际 LR。

### 6.3 单元测试

调优配置至少需要以下验证：

- 小模型训练 20-50 step 无 NaN/Inf。
- checkpoint 保存后恢复，LR、WD、optimizer state、loss scale 与恢复前连续。
- 开启 DistributedOptimizer 后，显存下降符合预期。
- BF16/FP16 配置在目标硬件上不触发 dtype 不支持错误。

---

## 7. 实验结果 (Experiments)

### 7.1 实验设置

推荐实验矩阵：

| 实验 | 变量 | 固定项 | 目标 |
|------|------|--------|------|
| LR sweep | `lr` | batch、WD、betas | 找最大稳定LR |
| Warmup sweep | warmup steps/fraction | LR | 消除早期spike |
| WD sweep | `weight_decay` | LR schedule | 控制泛化 |
| Beta sweep | $\beta_2$ | LR/WD | 控制梯度噪声响应 |
| Memory test | DistributedOptimizer on/off | 模型和batch | 验证显存收益 |

### 7.2 性能指标

选择优化器时需要同时看：

- 训练 loss 和 validation loss。
- grad norm 与裁剪比例。
- skipped step 比例。
- tokens/sec 或 samples/sec。
- GPU memory peak。
- checkpoint save/load 时间。

### 7.3 可视化分析

建议每次调参至少绘制：

- `loss` vs `tokens`，而不仅是 `loss` vs `iterations`。
- `lr` 与 `grad_norm` 同图，判断学习率阶段与梯度峰值关系。
- `validation loss` 与 `weight_decay`，判断正则是否过强。
- `memory peak` 与 `use_distributed_optimizer`，判断系统策略是否有效。

---

## 8. 消融研究 (Ablation Studies)

### 8.1 组件消融

| 组件 | 消融方式 | 观察重点 |
|------|----------|----------|
| AdamW | 换成Adam或SGD | 收敛速度、validation loss |
| Warmup | 减半或移除 | 前期loss spike |
| Weight decay | 0、0.01、0.1 | 过拟合和参数范数 |
| Clip grad | 0、1、2 | 梯度爆炸和更新幅度 |
| DistributedOptimizer | 开/关 | 显存、吞吐、checkpoint |

### 8.2 设计选择的合理性

一次消融只改变一个主要变量。尤其不要同时改变 LR 和 batch size，否则无法判断收益来自噪声尺度变化还是步长变化。对于 LLM 预训练，应优先用 consumed tokens 对齐不同实验，而不是只对齐 iteration。

---

## 9. 超参数分析 (Hyperparameters)

### 9.1 关键超参数

| 超参数 | 默认建议 | 调整信号 |
|--------|----------|----------|
| `optimizer` | `adam` + AdamW语义 | 仅对照实验才换 |
| `lr` | 从历史同规模模型迁移 | spike降，过慢升 |
| `min_lr` | peak LR的1%-10% | 后期停滞升 |
| `lr_warmup_fraction` | 0.005-0.02 | 早期不稳升 |
| `adam_beta1` | 0.9 | 噪声大可降到0.85 |
| `adam_beta2` | 0.95或0.999 | 大模型常用0.95起步 |
| `adam_eps` | 1e-8 | 数值异常时谨慎改 |
| `weight_decay` | 0.01-0.1 | 过拟合升，欠拟合降 |
| `clip_grad` | 1.0 | 长期裁剪说明LR/阈值需重审 |

### 9.2 超参数交互

**LR、batch、warmup**

扩大 batch 后，噪声降低但早期不稳定风险可能上升。常见做法是提高 LR 的同时增加 warmup，随后用短程实验确认 grad norm 和 loss 曲线。

**Betas与梯度噪声**

较低的 $\beta_2$ 对梯度方差变化响应更快，适合很多 Transformer 预训练配置；较高的 $\beta_2$ 更平滑，但可能在分布变化时响应慢。

**Weight decay与训练长度**

训练越长，weight decay 累计效果越明显。长训练不能只看单步 $\lambda$，要看 $\sum_t \eta_t\lambda_t$ 的总衰减量。

---

## 10. 深入探讨 (Advanced Topics)

### 10.1 理论深化

优化器调优的核心是控制有效更新量：

$$
\rho_t = \frac{\|\Delta\theta_t\|}{\|\theta_t\|}
$$

当 $\rho_t$ 在早期异常大时，常见表现是 loss spike 或 NaN；当 $\rho_t$ 长期过小时，训练会停滞。LR、betas、weight decay、gradient clipping 和 loss scaling 最终都通过不同路径影响 $\rho_t$。

### 10.2 与其他技术的关系

- **数据质量**：低质量或重复数据会改变 validation loss，对 optimizer 调参造成误导。
- **并行策略**：TP/PP/DP 改变通信和 batch 组织，可能改变吞吐和有效 batch。
- **MoE**：专家负载不均会使局部梯度统计不同于 dense 模型。
- **FP8训练**：需要额外关注 scale 更新和 transformer engine 的数值边界。

### 10.3 常见问题与解决方案

| 问题 | 可能原因 | 处理 |
|------|----------|------|
| 刚开始就NaN | LR过大、warmup过短、FP16溢出 | 降LR、增warmup、查loss scale |
| loss下降慢 | LR过小、batch过大、min_lr过低 | 做LR sweep |
| validation变差 | WD不足、数据问题、训练过长 | 调WD并审查数据 |
| resume后曲线不连续 | scheduler或optimizer state丢失 | 检查checkpoint加载参数 |
| 显存不够 | optimizer state过大 | 启用DistributedOptimizer |

### 10.4 最佳实践

- 默认从 AdamW + BF16 + cosine/WSD + clip=1.0 起步。
- 调参用 token 对齐，不只用 iteration 对齐。
- 所有实验记录 optimizer config、scheduler state 和并行配置。
- 大规模训练前必须做 checkpoint resume 验证。
- 显存问题优先考虑 DistributedOptimizer，而不是牺牲模型结构或训练稳定性参数。

### 10.5 前沿研究方向

Adafactor、Lion、Shampoo/SOAP 等优化器值得关注，但生产引入前需要回答四个问题：

1. 是否有与 Megatron-LM 并行状态兼容的实现。
2. 是否支持混合精度和动态 loss scaling。
3. checkpoint 是否可恢复、可reshard。
4. 在相同 token budget 下是否稳定优于 AdamW baseline。

---

## 11. 总结 (Conclusion)

### 11.1 核心要点回顾

- LLM 预训练默认优先 AdamW，而不是从零搜索优化器。
- 调优顺序应先稳定，再加速，最后压榨显存和吞吐。
- DistributedOptimizer 是系统级优化选择，会影响checkpoint和恢复语义。

### 11.2 技术优势

系统化选择和调优能减少无效实验，让训练曲线、显存、吞吐和恢复行为同时可控。

### 11.3 局限性

没有单一超参数表能覆盖所有模型和数据。最终配置仍必须通过目标数据和目标硬件上的短程实验确认。

### 11.4 适用场景

适用于从小模型验证到多节点 LLM 预训练的完整流程，尤其适合需要在 Megatron-LM 中维护可复现实验配置的团队。

### 11.5 与其他文档的联系

本文档收束 81-92 优化器系列，并连接后续 93-100 的混合精度、数据工程和完整训练工作流。

---

## 12. 参考文献 (References)

### 12.1 核心论文

- Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization". ICLR. arXiv:1412.6980.
- Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization". ICLR. arXiv:1711.05101.
- Shazeer & Stern (2018). "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost". ICML. arXiv:1804.04235.
- Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv:1909.08053.
- Rajbhandari et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models". SC. arXiv:1910.02054.

### 12.2 相关论文

- Ma & Yarats (2019). "On the adequacy of untuned warmup for adaptive optimization". arXiv:1910.04209.
- Chen et al. (2023). "Symbolic Discovery of Optimization Algorithms". arXiv:2302.06675.
- Vyas et al. (2024). "SOAP: Improving and Stabilizing Shampoo using Adam". arXiv:2409.11321.

### 12.3 官方文档

- NVIDIA Megatron-LM GitHub repository: https://github.com/NVIDIA/Megatron-LM
- PyTorch Optimizers documentation: https://pytorch.org/docs/stable/optim.html

### 12.4 博客与教程

- NVIDIA ADLR Megatron-LM project page: https://research.nvidia.com/labs/adlr/MegatronLM/

---

## 附录 (Appendices)

### 附录 A：推荐起点

```bash
--optimizer adam \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--adam-eps 1e-8 \
--lr 3e-4 \
--min-lr 3e-5 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--weight-decay 0.1 \
--clip-grad 1.0 \
--bf16 \
--use-distributed-optimizer
```

### 附录 B：调参记录模板

| 字段 | 示例 |
|------|------|
| model size | 7B |
| tokens | 1T |
| global batch | 4M tokens |
| optimizer | AdamW |
| lr schedule | cosine, 1% warmup |
| precision | BF16 |
| distributed optimizer | enabled |
| primary metric | validation loss |

### 附录 C：上线前检查

- smoke test 无 NaN。
- resume test 连续。
- grad norm 不长期被裁剪。
- LR 和 WD 曲线符合预算。
- checkpoint 能在目标并行配置恢复。

### 附录 D：场景化选择表

优化器选择不能只看模型参数量，还要看训练目标、数据成熟度、硬件精度和恢复要求。下面的表给出保守起点：

| 场景 | 推荐起点 | 暂不优先 | 关键原因 |
|------|----------|----------|----------|
| 新 GPT 预训练 | AdamW + BF16 + cosine/WSD | 新型研究优化器 | 先建立稳定 baseline |
| 小模型教学实验 | AdamW 或 SGD 对照 | DistributedOptimizer | 简化调试 |
| 复现实验 | 与论文一致 | 自行换优化器 | 保持可比性 |
| 显存受限大模型 | AdamW + DistributedOptimizer | 直接降模型宽度 | 先减少状态冗余 |
| FP16 硬件 | AdamW + dynamic loss scaling | 静态大 loss scale | 需要处理 overflow |
| 长上下文继续训练 | AdamW + 更保守 LR/warmup | 直接沿用短上下文 LR | 激活和梯度峰值不同 |
| MoE 预训练 | AdamW + expert 监控 | 只看全局 grad norm | 专家负载不均会隐藏局部异常 |
| 数据质量未稳定 | AdamW baseline | 大规模调 LR/WD | 数据噪声会污染调参结论 |
| 生产长期训练 | AdamW + resume验证 | 无checkpoint消融 | 恢复可靠性是一等目标 |
| 研究新优化器 | AdamW 对照 + 小规模验证 | 直接全量训练 | 需要证明收益超过系统风险 |

### 附录 E：从模型规模推导初始配置

下面是选择初始配置的工程化步骤，重点是约束推导，而不是给固定魔法数：

1. 确定目标 token budget 和 global batch tokens。
2. 确定硬件是否稳定支持 BF16；若不支持，再设计 FP16 loss scaling。
3. 用参数量估算 AdamW state 和 FP32 main 参数显存。
4. 如果 optimizer state 是主要瓶颈，优先启用 DistributedOptimizer。
5. 从同族模型迁移 peak LR，并用 100-1000 step 稳定性实验验证。
6. warmup 先按总训练步或 token 的小比例设置，再根据早期 grad norm 调整。
7. weight decay 先沿用同数据域 baseline，再用 validation loss 和参数范数校准。
8. 固定 optimizer 后再调 batch size；不要在第一轮同时改 batch 和 LR。

用于记录推导过程的表：

| 项目 | 记录值 | 判断 |
|------|--------|------|
| 参数量 P |  | 是否需要分布式优化器 |
| DP size |  | optimizer state 分片收益 |
| TP/PP size |  | 每 rank 参数与激活 |
| dtype |  | 是否需要 dynamic loss scaling |
| global batch tokens |  | 是否改变梯度噪声 |
| target tokens |  | scheduler 总预算 |
| peak LR 来源 |  | 迁移自哪个 baseline |
| warmup 口径 |  | iter/sample/token |
| WD 来源 |  | 数据域是否一致 |
| resume 策略 |  | 是否跨并行配置恢复 |

### 附录 F：调参优先级

调参时按“稳定性、收敛、泛化、吞吐、显存”的顺序推进：

| 优先级 | 问题 | 首先调整 | 暂缓调整 |
|--------|------|----------|----------|
| 1 | NaN/Inf | LR、warmup、loss scale | weight decay |
| 2 | early spike | warmup、peak LR | min_lr |
| 3 | grad norm 长期裁剪 | peak LR、clip_grad | batch size |
| 4 | loss 下降慢 | LR sweep、batch | optimizer family |
| 5 | validation 差 | WD、数据、训练长度 | loss scale |
| 6 | 后期停滞 | min_lr、decay style | beta1 |
| 7 | 显存不够 | DistributedOptimizer、activation checkpoint | 降 LR |
| 8 | 吞吐低 | profile、并行策略 | 盲目改 optimizer |

一个保守的调参循环：

```text
baseline = AdamW + BF16 + known schedule
while not stable:
    inspect early loss, grad_norm, skipped_step
    adjust lr or warmup only
while stable but slow:
    run lr sweep at fixed batch
    choose largest safe lr
while validation not acceptable:
    tune weight_decay and data mixture
if memory blocks scale:
    enable distributed optimizer
    repeat smoke + resume tests
```

### 附录 G：学习率选择细化

学习率不是单个数，而是曲线：

| 曲线阶段 | 目的 | 主要风险 | 观察指标 |
|----------|------|----------|----------|
| warmup | 让 Adam 矩估计稳定 | 太短导致 spike，太长浪费预算 | early loss, grad_norm |
| plateau | 保持高效学习 | LR 过高导致抖动 | smoothed loss, clip ratio |
| decay | 收敛与泛化 | 过早衰减导致欠训练 | validation loss |
| min_lr | 保留后期更新 | 太低停滞，太高震荡 | late loss slope |

常见判断：

- 如果 warmup 结束时 loss 突然抖动，peak LR 可能过高。
- 如果 warmup 期间 loss 几乎不降，warmup 可能过长或 LR 太低。
- 如果 decay 后 validation 明显改善，说明前期主要在探索。
- 如果 decay 后训练 loss 停滞但 validation 不改善，可能需要数据或正则审查。
- 如果不同 batch 的最优 LR 不按线性缩放，优先相信短程 sweep，而不是公式。

### 附录 H：Adam 参数选择细化

| 参数 | 作用 | 调整信号 | 注意事项 |
|------|------|----------|----------|
| `adam_beta1` | 一阶动量 | loss 抖动、响应速度 | 改动通常小于 LR 改动 |
| `adam_beta2` | 二阶矩平滑 | 梯度方差、早期稳定 | 大模型常需要经验验证 |
| `adam_eps` | 分母稳定项 | 极少数数值异常 | 不应作为首要调参旋钮 |
| `decoupled_weight_decay` | AdamW语义 | 泛化、参数范数 | 与 L2 正则不同 |
| `weight_decay` | 参数收缩 | validation gap | 受 LR 曲线累计影响 |

AdamW 的有效衰减量与 LR 曲线相关：

$$
\theta \leftarrow \theta - \eta_t \lambda_t \theta
$$

因此两次实验即使 `weight_decay` 相同，只要 LR 或训练长度不同，累计正则强度也不同。调参记录中应写明：

```text
peak_lr:
min_lr:
decay_style:
total_tokens:
weight_decay:
start_weight_decay:
end_weight_decay:
```

### 附录 I：DistributedOptimizer 决策

启用 DistributedOptimizer 前先回答：

| 问题 | 如果答案是是 | 如果答案是否 |
|------|--------------|--------------|
| optimizer state 是否 OOM | 启用分片收益大 | 先保持简单 |
| DP size 是否足够 | 分片收益明显 | 可能收益有限 |
| checkpoint 是否要跨规模恢复 | 需要提前验证 reshard | 同配置恢复即可 |
| step time 是否通信受限 | profile 后再启用 | 可以优先尝试 |
| 团队是否有恢复演练 | 可进入长训 | 先补 resume test |

显存估算时不要只写“AdamW 需要 12P bytes”。更完整的记录应区分：

- 模型参数是否 TP/PP/FSDP 切分。
- optimizer state 是否 DP 分片。
- FP32 main 参数是否存在。
- gradients 是否以 FP32 或 BF16/FP16 保留。
- activation checkpointing 是否影响峰值显存。
- checkpoint 保存是否需要临时聚合状态。

### 附录 J：调参日志规范

每个实验目录建议包含以下机器可读字段：

```yaml
model:
  hidden_size:
  num_layers:
  num_attention_heads:
  seq_length:
data:
  blend:
  tokenizer:
  consumed_tokens_target:
parallel:
  tensor_model_parallel_size:
  pipeline_model_parallel_size:
  data_parallel_size:
optimizer:
  optimizer: adam
  decoupled_weight_decay: true
  lr:
  min_lr:
  lr_decay_style:
  lr_warmup_fraction:
  adam_beta1:
  adam_beta2:
  adam_eps:
  weight_decay:
  clip_grad:
precision:
  bf16:
  fp16:
  initial_loss_scale:
system:
  use_distributed_optimizer:
  cpu_offload:
validation:
  smoke_test:
  resume_test:
  known_risks:
```

日志图建议统一使用 tokens 横轴。iteration 横轴在 micro-batch、DP size 或 gradient accumulation 变化后会误导对比。

### 附录 K：研究优化器引入门槛

Adafactor、Lion、Shampoo/SOAP 等优化器只有在满足以下条件时才适合进入大规模预训练候选：

| 门槛 | 必须回答的问题 |
|------|----------------|
| 数学收益 | 同 token budget 下是否优于 AdamW |
| 系统实现 | 是否支持 Megatron 并行参数和 param group |
| 混合精度 | 是否支持 BF16/FP16/FP8 的稳定更新 |
| 分布式状态 | optimizer state 是否能分片 |
| checkpoint | 是否能保存、恢复、reshard |
| 监控 | 是否有 update norm、state norm、异常检测 |
| 回滚 | 失败时能否恢复 AdamW baseline |

若任何一项无法回答，应把该优化器标为研究候选，而不是生产默认。

### 附录 L：最终决策记录模板

```text
decision_date:
owner:
baseline_run:
selected_optimizer:
selected_precision:
selected_scheduler:
selected_batch_tokens:
selected_parallel_config:
reason_for_selection:
experiments_considered:
stability_evidence:
validation_evidence:
memory_evidence:
resume_evidence:
risks_accepted:
rollback_plan:
```

决策记录的价值在于减少后续误读：当几周后 loss 曲线变化或并行度变化时，团队可以知道当前配置是为稳定性、显存还是吞吐做出的取舍。

### 附录 M：异常处理流程

当训练异常时，按以下顺序处理，避免无效调参：

| 异常 | 第一步 | 第二步 | 第三步 |
|------|--------|--------|--------|
| NaN/Inf | 查数据和loss | 查grad norm/loss scale | 降LR或增warmup |
| loss spike | 对齐LR曲线 | 看clip比例 | 调warmup |
| loss不降 | 做LR sweep | 查数据重复 | 查batch过大 |
| validation差 | 查数据分布 | 调weight decay | 调训练长度 |
| 显存OOM | 开DistributedOptimizer | 降micro-batch | 再考虑结构变化 |
| resume不连续 | 查scheduler state | 查optimizer state | 查rng/data state |
| 吞吐低 | profile | 查通信/IO | 再改optimizer |

每次异常只允许修改一个主变量，并在实验记录中写明假设。没有假设的调参会快速污染结论。

### 附录 N：发布配置冻结标准

优化器配置可以冻结的条件：

1. smoke test 通过。
2. 100-1000 step 稳定性测试通过。
3. resume test 连续。
4. 关键监控字段完整。
5. 显存峰值低于预算并留有余量。
6. validation loss 达到 baseline 要求。
7. 分布式优化器和 checkpoint 格式匹配。
8. 已写明回滚配置。

### 附录 O：配置变更分级

| 变更 | 风险等级 | 需要重新验证 |
|------|----------|--------------|
| 修改日志字段 | 低 | smoke test |
| 调整 `min_lr` | 中 | 短程稳定性和后期loss |
| 调整 peak `lr` | 高 | warmup、grad norm、validation |
| 调整 batch size | 高 | LR、吞吐、泛化 |
| 开启 DistributedOptimizer | 高 | 显存、吞吐、checkpoint |
| FP16/BF16切换 | 高 | 数值稳定、loss曲线 |
| 更换优化器族 | 最高 | 全量消融和回滚 |

配置冻结后只能接受低风险变更。中高风险变更必须重新走调参闭环，而不是直接复用旧结论。

### 附录 P：最终自检

1. 是否明确 AdamW 是默认 baseline，并说明其他优化器的适用边界。
2. 是否把优化器选择、scheduler、precision、distributed state 分层。
3. 是否说明调参先稳定再加速。
4. 是否要求 token 对齐而非只按 iteration 对齐。
5. 是否要求记录最终解析后的 config。
6. 是否为研究优化器设置引入门槛。
7. 是否包含异常处理流程。
8. 是否包含配置冻结标准。

---

**文档状态**: ✅ 已完成
**最后更新**: 2026-05-10
