# 91. 混合优化策略

> **文档编号**: 91
> **所属部分**: 第十二部分 - 优化器理论与实现 (81-92)
> **代码位置**: `megatron/core/optimizer/optimizer.py`, `megatron/core/optimizer/optimizer_config.py`, `megatron/core/optimizer/distrib_optimizer.py`, `megatron/core/optimizer/clip_grads.py`, `megatron/core/optimizer/grad_scaler.py`, `megatron/core/optimizer_param_scheduler.py`, `megatron/training/training.py`
> **前置文档**: 84-Adam优化器详解, 85-AdamW解耦权重衰减, 86-学习率调度策略, 88-分布式优化器, 90-梯度裁剪, 93-96混合精度训练
> **后续文档**: 92-优化器选择与调优

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

混合优化策略不是一个单独的优化器，而是一组共同决定训练稳定性和收敛效率的机制：

- **基础更新规则**：AdamW、Adam、SGD 等参数更新方法。
- **学习率控制**：warmup、cosine/linear/WSD 衰减、最小学习率。
- **正则与约束**：解耦权重衰减、梯度裁剪、QK裁剪。
- **数值稳定机制**：FP32 master 参数、loss scaling、溢出检测。
- **分布式状态管理**：DistributedOptimizer、ZeRO/FSDP式分片、checkpoint resharding。
- **训练节奏控制**：梯度累积、micro-batch 切分、global batch size 扩展。

在大语言模型预训练中，单一技巧通常无法单独解决问题。比如 AdamW 提供稳定的自适应更新，但如果 warmup 太短，早期二阶矩估计不足会放大更新；如果梯度裁剪过小，会抑制有效学习；如果 loss scale 策略错误，会造成频繁 skipped step。混合优化策略的目标，就是把这些机制组合成一个可解释、可调试、可扩展的训练系统。

### 1.2 前置知识

- Adam 与 AdamW 的动量、二阶矩和解耦 weight decay。
- 学习率调度的 warmup/decay/min-lr 设计。
- 混合精度训练中的梯度缩放与溢出检测。
- 数据并行和分布式优化器的梯度同步语义。
- Megatron-LM 训练循环中的 forward-backward-update 三阶段。

### 1.3 文档组织

本文档先建立组合优化的数学视角，再映射到 Megatron-LM 的代码路径；随后给出完整训练 step 的伪代码、配置模板、诊断表和调优流程。重点不是重新讲 AdamW 或 LR scheduler，而是解释这些组件如何在同一个训练 step 中交互。

### 1.4 代码位置

核心代码路径：

- `megatron/core/optimizer/optimizer_config.py`：`OptimizerConfig` 定义优化器、权重衰减、裁剪、loss scaling 等参数。
- `megatron/core/optimizer/__init__.py`：`get_megatron_optimizer()` 构造 Adam/AdamW/SGD、CPU offload 和 DistributedOptimizer。
- `megatron/core/optimizer/optimizer.py`：混合精度优化器、FP32 master 参数、unscale、step、clip grad。
- `megatron/core/optimizer/distrib_optimizer.py`：分布式优化器状态分片、参数/梯度 buffer 映射、checkpoint。
- `megatron/core/optimizer/clip_grads.py`：全局梯度范数计算与裁剪。
- `megatron/core/optimizer/grad_scaler.py`：静态和动态 loss scaling。
- `megatron/core/optimizer_param_scheduler.py`：学习率与 weight decay 调度。
- `megatron/training/training.py`：优化器与调度器创建、训练循环中的更新顺序。

---

## 2. 相关工作 (Related Work)

### 2.1 历史发展

LLM 训练优化策略大致经历了四个阶段：

1. **SGD/动量时代**：依赖 momentum、学习率阶梯衰减和人工调参，适合视觉模型和中小规模网络。
2. **Adam/自适应优化时代**：Adam 使用一阶和二阶矩估计，提高稀疏梯度和非平稳目标下的鲁棒性。
3. **AdamW与warmup实践时代**：解耦权重衰减解决 Adam 中 L2 正则和 weight decay 不等价的问题；线性 warmup 成为 Transformer 训练默认配置。
4. **系统级混合优化时代**：当模型进入十亿到万亿参数规模，优化器状态内存、通信、混合精度和 checkpoint 变成同等重要的设计变量。

### 2.2 技术对比

| 策略 | 主要收益 | 主要风险 | Megatron-LM入口 |
|------|----------|----------|-----------------|
| AdamW | 稳定、泛化好、LLM默认选择 | 状态内存高，LR/WD仍需调优 | `decoupled_weight_decay=True` |
| LR warmup | 抑制早期大步长 | 太长会浪费训练预算 | `lr_warmup_iters/samples/fraction` |
| Cosine decay | 平滑收敛，后期稳定 | min-lr过低可能过早停滞 | `lr_decay_style=cosine` |
| WSD | 长平台期+短衰减，适合长训练 | 需要明确训练预算 | `lr_decay_style=WSD` |
| Gradient clipping | 防止梯度爆炸 | 阈值太小会持续削弱更新 | `clip_grad` |
| Dynamic loss scaling | FP16下减少 underflow/overflow | 频繁溢出会跳步 | `DynamicGradScaler` |
| DistributedOptimizer | 降低DP冗余状态内存 | checkpoint和通信更复杂 | `use_distributed_optimizer` |

### 2.3 Megatron-LM中的实现

Megatron-LM 的设计原则是把“数学优化器”和“系统优化器”分层：

- 底层 PyTorch 或 Apex optimizer 负责 AdamW/Adam/SGD 的参数更新。
- `Float16Optimizer` 或 `FP32Optimizer` 负责精度、梯度缩放、梯度范数和 skipped step。
- `DistributedOptimizer` 在 data-parallel 维度切分 optimizer state、main parameter 和 gradient buffer。
- `OptimizerParamScheduler` 独立调整每个 param group 的 `lr` 与 `weight_decay`。

因此，混合优化策略的正确性取决于更新顺序：先累积并同步梯度，再 unscale 和检查溢出，再裁剪，随后 optimizer step，最后 scheduler step。

---

## 3. 符号定义 (Notation)

### 3.1 数学符号表

| 符号 | 含义 | 维度/类型 | 备注 |
|------|------|-----------|------|
| $\theta_t$ | 第 $t$ 步模型参数 | $P$ | 可为分片参数 |
| $g_t$ | 当前全局梯度 | $P$ | 梯度累积与DP同步后的结果 |
| $\eta_t$ | 学习率 | 标量/param group | 由 scheduler 决定 |
| $\lambda_t$ | 解耦权重衰减 | 标量/param group | 可随时间调度 |
| $m_t, v_t$ | Adam一阶/二阶矩 | $P$ | optimizer state |
| $s_t$ | loss scale | 标量 | FP16训练使用 |
| $G$ | global batch size | 标量 | $G=m \cdot b_\mu \cdot N_d$ |
| $C$ | clip grad 阈值 | 标量 | `clip_grad` |
| $N_d$ | data parallel size | 标量 | DP/ZeRO维度 |

### 3.2 代码变量约定

- `config.lr` 对应基础学习率 $\eta_{\max}$。
- `param_group['lr']` 对应当前 step 的 $\eta_t$。
- `config.weight_decay` 和 scheduler 写入的 `param_group['weight_decay']` 对应 $\lambda_t$。
- `config.clip_grad` 对应 $C$。
- `grad_scaler.scale` 对应 $s_t$。
- `found_inf` 表示本 step 是否发生溢出。

---

## 4. 数学原理 (Mathematical Foundations)

### 4.1 核心理论

混合优化策略可以写成受约束的随机优化过程：

$$
\min_{\theta} \; \mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)] + R(\theta)
$$

其中 $R(\theta)$ 在 AdamW 中不直接作为梯度项加入 $g_t$，而是在参数更新阶段解耦执行：

$$
\theta_{t+\frac{1}{2}} = \theta_t - \eta_t \lambda_t \theta_t
$$

然后对数据损失梯度执行 Adam 更新：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\theta_{t+1} =
\theta_{t+\frac{1}{2}} -
\eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

混合策略的关键是 $g_t$ 并不是单个 micro-batch 的梯度，而是经过累积、同步、反缩放、裁剪后的梯度。

### 4.2 梯度累积与全局梯度

设一次 optimizer step 包含 $K$ 个 micro-batch，每个 micro-batch 梯度为 $g_{t,k}^{(r)}$，其中 $r$ 是 data parallel rank。同步后的全局梯度为：

$$
g_t = \frac{1}{N_d K} \sum_{r=1}^{N_d}\sum_{k=1}^{K} g_{t,k}^{(r)}
$$

在实践中，loss 可能已经按 micro-batch 或 token 数归一化，因此代码实现必须避免重复除以 $K$ 或 $N_d$。Megatron-LM 将归一化语义分散在 forward-backward、DDP buffer 和 optimizer 阶段，写文档或改配置时必须以实际代码路径为准。

### 4.3 反缩放、溢出与裁剪

FP16 训练使用 scaled loss：

$$
\tilde{\ell} = s_t \ell
$$

对应梯度为 $\tilde{g}_t = s_t g_t$。optimizer step 前需要反缩放：

$$
g_t = \frac{\tilde{g}_t}{s_t}
$$

随后检查 `inf/nan`。若发生溢出，本 step 不应更新参数，也不应推进调度器的有效训练状态。若没有溢出，则执行全局梯度裁剪：

$$
g_t^{clip} = g_t \cdot \min\left(1, \frac{C}{\|g_t\|_2}\right)
$$

裁剪必须在 unscale 之后，否则阈值会依赖 loss scale。

### 4.4 复杂度分析

| 组件 | 时间复杂度 | 显存复杂度 | 通信复杂度 |
|------|------------|------------|------------|
| AdamW | $O(P)$ | $O(2P)$ optimizer state | 无额外通信 |
| FP32 master参数 | $O(P)$ cast/copy | $O(P)$ | 无额外通信 |
| Grad norm | $O(P)$ | $O(1)$ | DP/TP/PP 归约标量 |
| DistributedOptimizer | $O(P/N_d)$ state update | $O(P/N_d)$ state | reduce-scatter/all-gather |
| Scheduler | $O(\#groups)$ | $O(1)$ | 无 |

---

## 5. 算法伪代码 (Pseudocode)

### 5.1 完整混合优化训练步

```text
Algorithm 91.1: Megatron-style Hybrid Optimization Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  model parameters theta
  optimizer state m, v
  micro-batch count K
  data-parallel size Nd
  loss scale s
  learning-rate scheduler S
Output:
  updated theta, optimizer state, scheduler state
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: zero_grad_buffer()
2: for k = 1 ... K do
3:     loss_k = forward_backward_microbatch(k)
4:     accumulate scaled gradients into grad buffers
5: end for
6: finish gradient synchronization across DP/TP/PP groups
7: if mixed_precision then
8:     unscale gradients by current loss scale s
9:     found_inf = check_nan_inf(gradients)
10:    if found_inf then
11:        decrease/update loss scale
12:        skip parameter update
13:        return
14:    end if
15: end if
16: if clip_grad > 0 then
17:     grad_norm = compute_global_norm(gradients)
18:     gradients = gradients * min(1, clip_grad / grad_norm)
19: end if
20: apply decoupled weight decay if AdamW mode
21: update Adam moments and parameters
22: update loss scale after successful step
23: advance lr/weight-decay scheduler
24: clear or reuse gradient buffers
```

### 5.2 配置组合模板

```bash
--optimizer adam \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--adam-eps 1e-8 \
--lr 3.0e-4 \
--min-lr 3.0e-5 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--weight-decay 0.1 \
--clip-grad 1.0 \
--bf16 \
--use-distributed-optimizer
```

该模板表达的是常见 LLM 预训练起点，不是所有模型的最优配置。调优逻辑见文档92。

---

## 6. 代码实现详解 (Implementation)

### 6.1 核心类与函数

**`OptimizerConfig`**

`megatron/core/optimizer/optimizer_config.py` 集中定义混合优化策略的关键参数：

- `optimizer`: 选择 `adam` 或 `sgd`。
- `lr`, `min_lr`: 学习率上下界。
- `weight_decay`, `decoupled_weight_decay`: AdamW语义。
- `fp16`, `bf16`, `loss_scale`, `initial_loss_scale`: 混合精度语义。
- `clip_grad`: 全局梯度裁剪阈值。
- `use_distributed_optimizer`: 分布式优化器入口。

**`get_megatron_optimizer()`**

`megatron/core/optimizer/__init__.py` 根据 config 构造底层 optimizer，并决定是否包裹为 `Float16OptimizerWithFloat16Params` 或 `DistributedOptimizer`。这里是数学优化器与系统优化器的结合点。

**`MixedPrecisionOptimizer.step()`**

`megatron/core/optimizer/optimizer.py` 的 step 流程负责：

- 将 scaled gradient 反缩放。
- 检查 `found_inf`。
- 执行 gradient clipping。
- 调用底层 optimizer 的 `step()`。
- 同步 FP16/BF16 模型参数与 FP32 master 参数。

**`OptimizerParamScheduler.step()`**

`megatron/core/optimizer_param_scheduler.py` 根据训练步数更新 `lr` 和 `weight_decay`。关键点是它操作 param group，而不是直接改 optimizer 算法。

### 6.2 关键实现细节

**更新顺序不可交换**

`unscale -> found_inf -> clip -> optimizer.step -> scheduler.step` 是稳定训练的核心顺序。把裁剪放在 unscale 前会改变裁剪阈值；把 scheduler 放在 skipped step 前会让学习率状态和实际参数更新数不一致。

**BF16与FP16策略不同**

BF16通常不需要动态 loss scaling，因为指数范围接近 FP32；FP16更依赖 `DynamicGradScaler`。因此同一套 AdamW/LR 配置迁移到 FP16 时，首先要检查 skipped step 比例，而不是直接降低学习率。

**DistributedOptimizer改变内存与checkpoint语义**

启用 `use_distributed_optimizer` 后，optimizer state 不再在每个 DP rank 完整复制。文档、脚本和 checkpoint 恢复说明必须明确是否依赖分布式 optimizer，否则容易出现恢复失败或显存估算错误。

### 6.3 单元测试

相关测试应覆盖：

- `OptimizerParamScheduler` 的 warmup、decay、WSD、state_dict/load_state_dict。
- `clip_grads.py` 中全局 norm 的 DP/TP 归约语义。
- 混合精度 optimizer 在 `found_inf=True` 时是否跳过参数更新。
- DistributedOptimizer checkpoint 是否能在相同并行配置下恢复。

如果没有现成端到端测试，至少应通过小模型 dry run 验证 loss、grad norm、lr、loss scale 和 skipped step 日志字段随训练合理变化。

---

## 7. 实验结果 (Experiments)

### 7.1 实验设置

推荐用三档配置验证混合优化策略：

| 档位 | 目的 | 建议配置 |
|------|------|----------|
| Smoke | 验证训练循环可跑通 | 单机小 GPT，几十步 |
| Stability | 验证 loss scale/grad norm/lr 曲线 | 小规模 DP/TP，数千步 |
| Scale | 验证 DistributedOptimizer 和并行状态 | 真实并行度，短时间 profile |

### 7.2 性能指标

应记录以下指标：

- `loss` 与 smoothed loss。
- `learning_rate`、`weight_decay`。
- `grad_norm`、`num_zeros_in_grad`。
- `loss_scale` 与 skipped step 数。
- tokens/sec、TFLOPs/GPU、MFU。
- optimizer state 显存占用与 checkpoint 大小。

### 7.3 可视化分析

最有价值的图不是单独的 loss 曲线，而是联合曲线：

- `lr` 与 `loss`：判断 warmup 是否过短或 decay 是否过早。
- `grad_norm` 与 `clip coefficient`：判断裁剪是否长期生效。
- `loss_scale` 与 `found_inf`：判断 FP16 稳定性。
- `tokens/sec` 与并行配置：判断 DistributedOptimizer 是否造成通信瓶颈。

---

## 8. 消融研究 (Ablation Studies)

### 8.1 组件消融

| 消融项 | 预期现象 | 解释 |
|--------|----------|------|
| 移除 warmup | 早期 loss spike 或溢出 | Adam二阶矩估计尚未稳定 |
| AdamW改Adam+L2 | 泛化或训练稳定性下降 | 自适应缩放破坏L2与weight decay等价性 |
| 关闭 gradient clipping | 偶发梯度爆炸更难恢复 | 长序列或大batch下风险更高 |
| FP16关闭 dynamic scaling | underflow/overflow 增多 | FP16指数范围有限 |
| 关闭 DistributedOptimizer | 显存占用显著上升 | optimizer state 在DP rank复制 |

### 8.2 设计选择的合理性

混合优化策略的设计目标不是“每个组件都最大化”，而是让每个组件在合适区间工作：

- warmup 负责早期稳定，不负责最终收敛。
- weight decay 负责隐式正则，不负责控制梯度爆炸。
- gradient clipping 负责异常步保护，不应长期主导更新幅度。
- loss scaling 负责数值表示，不应掩盖真实学习率过大的问题。

---

## 9. 超参数分析 (Hyperparameters)

### 9.1 关键超参数

| 参数 | 常见起点 | 调整方向 |
|------|----------|----------|
| `lr` | $1e^{-4}$ 到 $6e^{-4}$ | loss spike 降低；收敛慢或欠拟合升高 |
| `min_lr` | `lr` 的 1%-10% | 后期过早停滞则升高 |
| `lr_warmup_fraction` | 0.5%-2% | 早期不稳则增加 |
| `adam_beta1` | 0.9 | 噪声大可略降 |
| `adam_beta2` | 0.95 或 0.999 | LLM常用0.95；小数据可更高 |
| `weight_decay` | 0.01-0.1 | 过拟合升高；欠拟合降低 |
| `clip_grad` | 1.0 | 长期裁剪则升高或降LR |
| `loss_scale` | 动态或静态 | FP16优先动态；BF16通常不需要 |

### 9.2 超参数交互

**LR与weight decay**

AdamW 中 weight decay 与梯度更新解耦，但实际参数衰减量仍含 $\eta_t\lambda_t$。提高 LR 后，即使 $\lambda$ 不变，参数收缩也会变强。

**batch size与LR**

扩大 global batch size 会降低梯度噪声，通常可以提高学习率，但线性缩放不是无条件成立；需要同时观察 validation loss 和 grad norm。

**clip grad与loss scale**

如果 FP16 出现频繁溢出，先看 unscale 后的 grad norm。如果 grad norm 本身异常大，问题通常是 LR/warmup；如果 grad norm 正常但仍溢出，才优先调整 loss scale。

---

## 10. 深入探讨 (Advanced Topics)

### 10.1 理论深化

混合优化策略可以理解为对 update norm 的多重控制：

$$
\|\Delta\theta_t\| \leq
\eta_t \left\|
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\right\|
+ \eta_t \lambda_t\|\theta_t\|
$$

warmup 控制 $\eta_t$，Adam 二阶矩控制坐标尺度，gradient clipping 控制 $g_t$ 的全局范数，weight decay 控制参数范数。稳定训练需要这些约束彼此一致。

### 10.2 与其他技术的关系

- **与混合精度**：FP16/BF16改变数值安全边界，不改变优化目标。
- **与ZeRO/FSDP**：分片改变状态存储和通信，不改变 AdamW 的数学更新。
- **与MoE**：专家路由带来更高梯度方差，通常需要更关注 load balance loss、expert gradient norm 和 capacity factor。
- **与长上下文训练**：序列长度增大会放大激活内存和注意力梯度峰值，warmup、clip 和 FP8/BF16策略需要联合评估。

### 10.3 常见问题与解决方案

| 症状 | 可能原因 | 优先检查 |
|------|----------|----------|
| 前100步loss spike | warmup过短、LR过高 | `lr`, warmup, grad norm |
| loss scale频繁下降 | FP16溢出或梯度异常 | `found_inf`, unscaled grad norm |
| grad norm长期等于clip阈值 | 裁剪过强或LR过高 | clip coefficient, LR曲线 |
| 后期loss停滞 | min_lr过低或decay过早 | scheduler状态 |
| resume后loss突变 | scheduler/optimizer state未恢复 | checkpoint中的 optimizer 和 scheduler |

### 10.4 最佳实践

- 新模型先用 AdamW + cosine/WSD + warmup + clip=1.0 + BF16 建立 baseline。
- FP16 训练时把 skipped step 比例作为一等指标。
- 大规模训练前先跑短程稳定性实验，确认 lr、grad norm、loss scale 曲线正常。
- 启用 DistributedOptimizer 后，checkpoint、恢复脚本和显存估算都按分片状态处理。
- 不在同一次实验中同时大幅修改 LR、batch size、weight decay 和 clip grad。

### 10.5 前沿研究方向

前沿优化器包括 Adafactor、Lion、Shampoo/SOAP 等。它们可能减少状态内存或改善大batch收敛，但在 Megatron-LM 当前路径中并不是默认生产实现。用于 LLM 预训练时应先明确：

- 是否有可靠的分布式 optimizer state 支持。
- 是否支持混合精度和 checkpoint resharding。
- 是否在目标模型规模和数据分布上优于 AdamW baseline。

---

## 11. 总结 (Conclusion)

### 11.1 核心要点回顾

- 混合优化策略是 AdamW、scheduler、gradient clipping、loss scaling、distributed optimizer 和 batch 组织的组合系统。
- 训练稳定性取决于更新顺序：累积/同步、反缩放、溢出检查、裁剪、参数更新、调度器推进。
- 大模型中 optimizer state 和 checkpoint 语义与数学更新同等重要。

### 11.2 技术优势

- 能在大规模分布式环境中保持稳定更新。
- 能把数值稳定、泛化、吞吐和显存放在同一套配置中调节。
- 与 Megatron-LM 的模块化 optimizer/scheduler 设计匹配。

### 11.3 局限性

- 组件交互复杂，单个指标不能解释所有训练问题。
- 调参仍依赖短程实验和监控曲线。
- 新优化器在系统支持、checkpoint和混合精度方面需要额外验证。

### 11.4 适用场景

适用于 GPT/BERT/T5/MoE 等大规模预训练，也适用于从小模型 smoke test 逐步扩展到多节点训练的配置设计。

### 11.5 与其他文档的联系

本文档连接 81-90 的优化器理论与 93-96 的混合精度实践。下一篇文档92给出面向具体场景的优化器选择和调参流程。

---

## 12. 参考文献 (References)

### 12.1 核心论文

- Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization". ICLR. arXiv:1412.6980.
- Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization". ICLR. arXiv:1711.05101.
- Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv:1909.08053.
- Rajbhandari et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models". SC. arXiv:1910.02054.

### 12.2 相关论文

- Shazeer & Stern (2018). "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost". ICML. arXiv:1804.04235.
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

### 附录 A：诊断顺序速查

1. loss 是否在前几百步 spike。
2. grad norm 是否异常大或长期被裁剪。
3. FP16 下 loss scale 是否频繁下降。
4. scheduler 的当前 step、lr、min_lr 是否符合预期。
5. checkpoint resume 是否同时恢复 optimizer 和 scheduler。

### 附录 B：最小训练日志字段

```text
iteration, consumed_samples, learning_rate, loss, grad_norm,
num_zeros_in_grad, loss_scale, skipped_iter, elapsed_time_per_iter,
tokens_per_second
```

### 附录 C：配置审查清单

- AdamW 是否启用解耦 weight decay。
- BF16/FP16 是否和硬件匹配。
- global batch size 是否与 token budget 匹配。
- warmup 是按 iter 还是 sample 计算。
- DistributedOptimizer 是否和 checkpoint 格式匹配。

### 附录 D：Megatron 更新顺序代码映射

混合优化最容易出错的地方不是单个公式，而是多个组件的执行顺序。下面的映射把训练 step 拆成可审查的阶段：

| 阶段 | 主要语义 | 代码锚点 | 审查问题 |
|------|----------|----------|----------|
| forward/backward | micro-batch 前向反向、梯度进入 buffer | `megatron/training/training.py` | loss 是否已经按 token 或 micro-batch 归一化 |
| gradient sync | DP/TP/PP 相关同步 | `megatron/core/distributed/distributed_data_parallel.py` | 是否所有 DP rank 使用同一归一化口径 |
| prepare grads | copy/unscale/check inf | `megatron/core/optimizer/optimizer.py` | FP16 是否在裁剪前完成 unscale |
| clip grad | 计算全局范数并缩放 | `megatron/core/optimizer/clip_grads.py` | clip 阈值是否长期生效 |
| optimizer step | AdamW/SGD 更新参数 | `megatron/core/optimizer/__init__.py` | param group 的 lr/weight_decay 是否正确 |
| scheduler step | 更新 lr 和 weight decay | `megatron/core/optimizer_param_scheduler.py` | skipped step 时是否推进调度状态 |
| checkpoint | 保存 optimizer/scheduler/rng | `megatron/training/checkpointing.py` | resume 后曲线是否连续 |

建议在每次新增训练脚本时记录以下最小审查结论：

```text
optimizer_step_order:
  gradients_accumulated: yes/no
  data_parallel_sync_finished: yes/no
  fp16_unscale_before_clip: yes/no/not-applicable
  found_inf_skips_optimizer: yes/no/not-applicable
  scheduler_advances_only_after_update: yes/no
  optimizer_state_checkpointed: yes/no
```

### 附录 E：配置字段交互矩阵

`OptimizerConfig` 中很多字段不是独立旋钮。下面的矩阵用于避免“单点调参”误判：

| 字段 | 直接影响 | 交互字段 | 常见误判 |
|------|----------|----------|----------|
| `lr` | AdamW 梯度更新和解耦衰减强度 | `weight_decay`, `clip_grad`, warmup | 降 loss spike 只改 loss scale |
| `min_lr` | 后期最小步长 | decay style, token budget | min_lr 太低导致后期停滞 |
| `lr_decay_style` | 中后期训练节奏 | total iters/samples | resume 后预算口径不一致 |
| `weight_decay` | 参数范数收缩 | `lr`, WD scheduler | 只比较相同 WD，忽略不同 LR |
| `adam_beta1` | 一阶矩惯性 | batch size, gradient noise | 噪声大时仍维持过强 momentum |
| `adam_beta2` | 二阶矩平滑 | warmup, LR | beta2 高导致早期响应慢 |
| `adam_eps` | 分母下界 | dtype, grad scale | 把数值问题误当成 eps 问题 |
| `clip_grad` | 全局梯度范数上限 | `lr`, loss scale | clip 长期触发但继续升 LR |
| `fp16` | 动态 loss scaling 路径 | `loss_scale`, `initial_loss_scale` | 用 BF16 经验直接迁移到 FP16 |
| `bf16` | 更宽指数范围 | hardware support | 以为 BF16 完全不需要监控 NaN |
| `use_distributed_optimizer` | optimizer state 分片 | DP size, checkpoint | 显存估算仍按完整状态复制 |
| CPU offload | GPU optimizer state 下降 | PCIe/NVLink/CPU内存 | 只看显存不看 step time |

### 附录 F：skipped step 与 scheduler 语义

FP16 动态 loss scaling 下，`found_inf=True` 的 step 应被视为“无效参数更新”。调优时必须区分三种计数：

| 计数 | 含义 | 应用于 |
|------|------|--------|
| wall-clock iteration | 训练循环跑过的 iteration | 性能、吞吐 |
| consumed samples/tokens | 已读入的数据量 | 数据预算、日志横轴 |
| effective optimizer steps | 真正更新参数的次数 | scheduler、Adam bias correction、实验对齐 |

如果 skipped step 很多，会出现以下现象：

- loss 曲线横轴按 tokens 看似正常推进，但参数实际少更新。
- LR scheduler 如果按 iteration 继续推进，会让有效更新数和 LR 阶段错位。
- Adam 的动量状态停在旧参数附近，恢复后可能出现短暂不连续。

建议日志中至少保留：

```text
iteration
consumed_samples
consumed_tokens
learning_rate
loss_scale
found_inf
skipped_iter
effective_optimizer_step
grad_norm
```

排障顺序：

1. 如果 `found_inf` 连续出现，先看 unscaled `grad_norm` 是否异常。
2. 如果 `grad_norm` 异常，优先降低 LR 或增加 warmup。
3. 如果 `grad_norm` 正常但仍 overflow，调整 initial/dynamic loss scale。
4. 如果 skipped step 只在开始阶段出现，确认 scheduler 没有过快进入高 LR。
5. 如果 resume 后 skipped step 激增，检查 loss scale 和 optimizer state 是否恢复。

### 附录 G：不同精度模式的优化策略

| 精度模式 | 优势 | 主要风险 | 推荐监控 |
|----------|------|----------|----------|
| FP32 | 数值最稳 | 显存和吞吐成本高 | loss, grad_norm |
| BF16 | 指数范围接近 FP32，适合大模型 | 硬件要求高，尾数精度低 | loss, grad_norm, NaN |
| FP16 dynamic | 吞吐高，兼容广 | overflow/underflow，跳步 | loss_scale, found_inf, skipped_iter |
| FP8 training | 激活/权重通信更省 | scale 管理复杂 | amax, scale, overflow, validation loss |

混合优化配置迁移时要遵守两个原则：

- 从 BF16 迁移到 FP16，不应只复制 LR；必须重新验证 loss scale 和 skipped step。
- 从 FP16 迁移到 BF16，不能因为没有动态 loss scaling 就移除 grad norm、NaN 和 checkpoint resume 监控。

### 附录 H：分布式优化器内存审查

AdamW 的内存压力主要来自：

```text
model parameters      ~ 2P bytes  (BF16/FP16)
main FP32 parameters  ~ 4P bytes  (视实现而定)
Adam first moment     ~ 4P bytes
Adam second moment    ~ 4P bytes
gradients             ~ 2P or 4P bytes
```

普通 DP 会在每个 DP rank 上复制 optimizer state。启用 DistributedOptimizer 后，主要 optimizer state 可在 DP 维度分片，但以下开销仍要审查：

| 项目 | 是否一定按 DP 分片 | 备注 |
|------|--------------------|------|
| Adam moments | 通常是 | 取决于 optimizer wrapper |
| FP32 main params | 通常是 | 与参数 buffer 映射有关 |
| model params | 否 | 仍由 TP/PP/FSDP 等决定 |
| gradients | 部分 | 取决于 reduce-scatter/all-gather 路径 |
| checkpoint metadata | 否 | rank mapping 和 resharding 需要额外信息 |

启用分布式优化器前的检查：

1. DP size 是否足够大，能抵消额外通信。
2. checkpoint 是否需要跨不同并行配置恢复。
3. optimizer state 是否被正确保存和加载。
4. 日志中是否能看到每 rank 显存下降。
5. step time 是否因为 all-gather/reduce-scatter 明显变差。

### 附录 I：典型故障案例

| 症状 | 第一怀疑 | 第二怀疑 | 建议动作 |
|------|----------|----------|----------|
| 前 50 step loss spike | warmup 太短 | LR 太高 | warmup 翻倍或 LR 降 20%-50% |
| grad norm 长期等于阈值 | clip 太小 | LR 太高 | 看 clip coefficient 分布 |
| loss scale 一直下降 | FP16 overflow | 数据异常 | 查 unscaled grad norm 和 batch |
| validation loss 上升 | WD 不足或数据问题 | LR 过高 | 做 WD sweep，并审查重复数据 |
| tokens/sec 下降 | 分布式优化器通信 | CPU offload 瓶颈 | 拆分 optimizer step profile |
| resume 后 LR 跳变 | scheduler state 丢失 | iter/sample 口径变化 | 对比 checkpoint 中 scheduler 字段 |
| resume 后 loss scale 变回初值 | grad scaler state 丢失 | FP16配置变化 | 检查 optimizer wrapper state |
| 多节点才 NaN | 通信/归一化不一致 | rank 数据差异 | 比对 DP rank grad norm |
| MoE 专家不稳定 | load balance loss 弱 | expert 梯度方差大 | 单独监控 expert grad norm |
| 长上下文才 OOM | activation/KV/sequence并行不足 | micro-batch 过大 | 增加 checkpointing 或降低 micro-batch |

### 附录 J：上线前最小实验闭环

生产级混合优化策略至少要通过以下闭环：

| 步骤 | 目标 | 通过标准 |
|------|------|----------|
| 20 step smoke | 验证脚本/数据/并行组 | 无 NaN，能保存 checkpoint |
| 200 step stability | 验证 warmup、loss scale、grad norm | skipped step 可解释，loss 下降 |
| resume test | 验证状态恢复 | loss/lr/loss_scale 连续 |
| distributed test | 验证分布式优化器 | 显存下降且 step time 可接受 |
| profile test | 验证瓶颈 | optimizer step、通信、IO 有明确占比 |
| ablation test | 验证关键旋钮 | 每次只改一个主变量 |

最小闭环报告模板：

```text
run_id:
model:
data_snapshot:
parallel_config:
precision:
optimizer:
lr_schedule:
global_batch_tokens:
warmup_tokens:
weight_decay:
clip_grad:
distributed_optimizer:
smoke_result:
stability_result:
resume_result:
known_risks:
decision:
```

### 附录 K：审查问答

**为什么 scheduler 不能无条件按 wall-clock iteration 推进？**

因为 FP16 overflow 可能导致 skipped step。若参数没有更新但 scheduler 继续衰减，LR曲线会和有效优化步数错位。

**为什么 clip grad 必须在 unscale 后？**

FP16 scaled gradient 的范数包含 loss scale。若先裁剪，阈值实际变成依赖当前 loss scale 的动态值。

**为什么 DistributedOptimizer 不改变 AdamW 数学？**

它改变 optimizer state 和梯度 buffer 的存储/通信方式；只要分片和聚合正确，每个参数的 AdamW 更新语义应保持一致。

**为什么 BF16 仍需监控数值稳定？**

BF16 指数范围更宽，但尾数精度较低，且 NaN 也可能来自数据、mask、除零或 kernel bug。

**为什么 weight decay 要和 LR 曲线一起看？**

AdamW 的解耦衰减量含 $\eta_t\lambda_t$。相同 `weight_decay` 在不同 LR 曲线和训练长度下累计效果不同。

### 附录 L：代码评审检查项

| 检查项 | 通过标准 |
|--------|----------|
| optimizer config | 日志中打印最终解析值 |
| param groups | `lr_mult`/`wd_mult` 有记录 |
| skipped step | 不推进无效更新状态 |
| grad norm | unscale 后计算 |
| scheduler state | checkpoint 中保存并恢复 |
| grad scaler | FP16 resume 后连续 |
| distributed optimizer | state 分片和恢复测试通过 |
| CPU offload | profile 证明可接受 |
| monitoring | loss/lr/grad_norm/loss_scale齐全 |

### 附录 M：发布门禁

混合优化策略进入长训前必须满足：

1. 训练脚本打印最终 optimizer config。
2. loss、lr、grad_norm、loss_scale、skipped step 全部入日志。
3. FP16/BF16 模式分别有 smoke test。
4. checkpoint resume 后 scheduler 和 optimizer state 连续。
5. DistributedOptimizer 配置有保存/恢复测试。
6. 梯度裁剪比例不是长期饱和。
7. warmup 结束附近没有不可解释 spike。
8. 监控面板能同时按 iteration 和 consumed tokens 查看。

如果任何一项失败，不应把问题归因于“随机种子不好”；必须先补齐证据。

### 附录 N：最终自检

1. 是否写清 `unscale -> found_inf -> clip -> step -> scheduler` 顺序。
2. 是否说明 skipped step 与 scheduler 的关系。
3. 是否说明 BF16 和 FP16 监控差异。
4. 是否把 DistributedOptimizer 描述为系统分片，而非新优化算法。
5. 是否把 weight decay 与 LR 曲线关联。
6. 是否列出最小日志字段。
7. 是否包含 resume test。
8. 是否避免把新型优化器写成默认生产选择。
9. 是否把异常处理写成可执行排查顺序。

---

**文档状态**: ✅ 已完成
**最后更新**: 2026-05-10
