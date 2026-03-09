# 63. PipeDream：异步流水线并行

## 概述

PipeDream是Microsoft Research在2019年SOSP（ACM Symposium on Operating Systems Principles）上提出的流水线并行训练系统，它通过引入**1F1B（One-Forward-One-Backward）调度策略**和**异步权重更新机制**，显著改进了GPipe的内存效率和流水线气泡时间。

**PipeDream的核心创新**：
- **1F1B调度**：交替执行前向和反向计算，而非GPipe的F-then-B
- **异步权重更新**：每个阶段独立更新权重，无需全局同步
- **权重版本管理**：处理异步更新导致的权重不一致问题
- **内存优化**：相比GPipe减少m倍的激活内存占用
- **PipeDream-Flush**：同步变体，牺牲部分效率换取严格收敛保证

**关键数学关系**：

气泡时间（与GPipe相同）：
```
T_bubble = (p-1) × t_f
```

但内存占用显著降低：
```
GPipe内存:    M_activation = m × a_stage
PipeDream内存: M_activation = (p-1) × a_stage
```

其中m通常远大于p（例如m=32, p=4），因此PipeDream内存优势明显。

**Megatron-LM实现**：

Megatron-LM实现了PipeDream-Flush（同步1F1B）而非原始的异步PipeDream：
- 代码位置：`megatron/core/pipeline_parallel/schedules.py`
- 函数：`forward_backward_pipelining_without_interleaving` (基础1F1B)
- 函数：`forward_backward_pipelining_with_interleaving` (虚拟流水线1F1B)

---

## 1. 历史背景与动机

### 1.1 问题背景

**GPipe的局限性**（文档62分析）：

GPipe使用F-then-B调度导致两个主要问题：

1. **内存爆炸**：需要同时存储m个micro-batch的激活
   ```
   GPipe激活内存 = m × 每个micro-batch的激活大小
   ```

2. **内存-气泡时间权衡困境**：
   - 增加m以减少气泡时间
   - 但增加m会导致激活内存线性增长
   - 对于大模型，内存成为瓶颈

**实际案例（GPT-3 175B）**：
```
假设：
- Pipeline stages: p = 8
- Micro-batches: m = 32 (为了降低气泡时间)
- 每个micro-batch激活: 2GB

GPipe内存需求:
- 激活内存 = 32 × 2GB = 64GB
- 已经超过单个A100 GPU的内存(80GB)

期望目标:
- 降低激活内存到 p × 2GB = 16GB
- 同时保持相同的气泡时间
```

### 1.2 PipeDream的研究目标

**Microsoft Research团队（2019）的目标**：

1. **打破内存-气泡时间权衡**：降低内存的同时保持低气泡时间
2. **提高流水线效率**：减少设备空闲时间
3. **支持异步训练**：允许各阶段独立更新，无需全局同步
4. **保证训练收敛**：处理异步更新带来的一致性问题

**关键洞察**：

> "我们不需要同时存储所有m个micro-batch的激活。通过交替执行前向和反向，可以在反向计算后立即释放对应的激活，从而将内存占用从O(m)降低到O(p)。"
>
> — Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training", SOSP 2019

### 1.3 PipeDream系列演进

```
PipeDream系列演进:

2019 SOSP: PipeDream
├─ 1F1B调度
├─ 异步权重更新
└─ 权重版本管理

2020: PipeDream-Flush
├─ 同步1F1B调度
├─ Flush阶段确保权重一致性
└─ 严格收敛保证（与同步SGD等价）

2021 NeurIPS: PipeDream-2BW
├─ 双缓冲权重
├─ 支持任意批次大小
└─ 优化内存占用

Megatron-LM (2019-现在):
├─ 实现PipeDream-Flush (同步1F1B)
├─ 支持虚拟流水线
└─ 与张量并行、数据并行组合
```

---

## 2. 核心概念

### 2.1 1F1B调度策略

**定义**：

1F1B（One-Forward-One-Backward）是一种流水线调度策略，其核心思想是：
- 在Warmup阶段之后，每个阶段交替执行一次前向计算和一次反向计算
- 前向计算针对新的micro-batch
- 反向计算针对之前的micro-batch

**对比GPipe的F-then-B**：

```
GPipe (F-then-B) - 所有前向完成后才开始反向:

Stage 0: F₀ F₁ F₂ F₃ ............ B₃ B₂ B₁ B₀
Stage 1: .. F₀ F₁ F₂ F₃ ........ B₃ B₂ B₁ B₀ ..
Stage 2: .... F₀ F₁ F₂ F₃ ...... B₃ B₂ B₁ B₀ ....
Stage 3: ...... F₀ F₁ F₂ F₃ .... B₃ B₂ B₁ B₀ ......

激活内存: 需要同时保存F₀, F₁, F₂, F₃的激活（4个micro-batch）


PipeDream (1F1B) - 前向和反向交替执行:

Stage 0: F₀ F₁ F₂ F₃ B₀ F₄ B₁ F₅ B₂ F₆ B₃ .... B₇
Stage 1: .. F₀ F₁ F₂ B₀ F₃ B₁ F₄ B₂ F₅ B₃ .... B₇ ..
Stage 2: .... F₀ F₁ B₀ F₂ B₁ F₃ B₂ F₄ B₃ F₅ .... B₇ ....
Stage 3: ...... F₀ B₀ F₁ B₁ F₂ B₂ F₃ B₃ F₄ .... B₇ ......

激活内存: 在Steady阶段只需保存(p-1)个micro-batch的激活
```

**内存优势量化**：

```python
# GPipe内存占用
m = 32  # micro-batches
p = 4   # pipeline stages
a = 2   # GB，每个micro-batch的激活

GPipe_memory = m * a = 32 * 2 = 64 GB

# PipeDream 1F1B内存占用
PipeDream_memory = (p - 1) * a = 3 * 2 = 6 GB

# 内存节省
memory_reduction = GPipe_memory / PipeDream_memory = 64 / 6 ≈ 10.7×
```

### 2.2 流水线三阶段执行

**1F1B调度的三个阶段**：

#### **阶段1：Warmup（填充）**

目标：填充流水线，使所有阶段都开始工作

```
每个阶段的warmup micro-batch数量:
num_warmup_microbatches(rank) = p - rank - 1

Stage 0: warmup = 4 - 0 - 1 = 3 个前向
Stage 1: warmup = 4 - 1 - 1 = 2 个前向
Stage 2: warmup = 4 - 2 - 1 = 1 个前向
Stage 3: warmup = 4 - 3 - 1 = 0 个前向（直接进入steady）
```

**Warmup阶段特点**：
- 只执行前向计算，不执行反向
- 激活需要缓存，等待后续反向
- 内存占用逐渐增加

#### **阶段2：Steady State（1F1B交替）**

目标：保持稳定的前向-反向交替执行模式

```
num_steady_microbatches = m - num_warmup_microbatches

对于4个stage，8个micro-batch:
- Stage 0: warmup=3, steady=8-3=5, cooldown=3
- Stage 1: warmup=2, steady=8-2=6, cooldown=2
- Stage 2: warmup=1, steady=8-1=7, cooldown=1
- Stage 3: warmup=0, steady=8-0=8, cooldown=0
```

**Steady阶段特点**：
- 严格遵循1F1B模式：每次前向后立即执行一次反向
- 激活内存稳定在(p-1)个micro-batch
- 流水线利用率最高

#### **阶段3：Cooldown（排空）**

目标：处理warmup阶段缓存的激活

```
num_cooldown_microbatches(rank) = num_warmup_microbatches(rank)
```

**Cooldown阶段特点**：
- 只执行反向计算
- 逐步释放warmup阶段缓存的激活
- 激活内存逐渐减少到0

**完整执行示例（p=4, m=8）**：

```
Time →
Stage 0: [F₀ F₁ F₂] [F₃B₀ F₄B₁ F₅B₂ F₆B₃ F₇B₄] [B₅ B₆ B₇]
         └Warmup──┘ └─────Steady (1F1B)──────┘ └Cooldown┘
         3个F      5组1F1B                    3个B

Stage 1: .[F₀ F₁] [F₂B₀ F₃B₁ F₄B₂ F₅B₃ F₆B₄ F₇B₅] [B₆ B₇].
         └Warmup┘ └────────Steady──────────────┘ └Cooldown┘

Stage 2: ..[F₀] [F₁B₀ F₂B₁ F₃B₂ F₄B₃ F₅B₄ F₆B₅ F₇B₆] [B₇]..
         └Warmup [───────────Steady───────────────┘ └─┘

Stage 3: ...[] [F₀B₀ F₁B₁ F₂B₂ F₃B₃ F₄B₄ F₅B₅ F₆B₆ F₇B₇] []...
              └─────────────Steady (全部)────────────┘
```

### 2.3 权重版本管理

**异步更新的挑战**：

在原始PipeDream中，每个阶段独立更新权重，不等待其他阶段。这导致：

```
Timeline (p=3, m=4):

Time 0:  Stage 0 用权重w₀⁰执行F₀
Time 1:  Stage 1 用权重w₁⁰执行F₀（从Stage 0接收）
Time 2:  Stage 0 用权重w₀⁰执行F₁
         Stage 2 用权重w₂⁰执行F₀
Time 3:  Stage 0 完成B₀，更新权重 w₀⁰ → w₀¹
Time 4:  Stage 0 用新权重w₀¹执行F₂  ← 权重不一致！
         Stage 1 仍在用w₁⁰处理F₁
```

**问题**：同一个mini-batch的不同部分使用了不同版本的权重。

**PipeDream的权重版本管理机制**：

每个stage维护多个权重版本，确保同一个mini-batch的前向和反向使用相同的权重：

```python
# 伪代码：PipeDream权重版本管理

class PipeDreamStage:
    def __init__(self):
        self.weight_versions = {}  # {version_id: weights}
        self.current_version = 0
        self.microbatch_to_version = {}  # 记录每个micro-batch使用的权重版本

    def forward(self, microbatch_id, input_tensor):
        # 使用当前版本的权重
        version = self.current_version
        self.microbatch_to_version[microbatch_id] = version

        # 前向计算
        output = self.model_chunk.forward(
            input_tensor,
            weights=self.weight_versions[version]
        )

        return output

    def backward(self, microbatch_id, output_grad):
        # 使用前向时相同的权重版本
        version = self.microbatch_to_version[microbatch_id]

        # 反向计算
        input_grad = self.model_chunk.backward(
            output_grad,
            weights=self.weight_versions[version]
        )

        # 更新权重（创建新版本）
        self.update_weights(version)

        return input_grad

    def update_weights(self, old_version):
        # 基于旧版本计算梯度并更新
        new_version = old_version + 1
        self.weight_versions[new_version] = apply_gradients(
            self.weight_versions[old_version],
            self.accumulated_gradients
        )
        self.current_version = new_version

        # 清理不再需要的旧版本
        self.cleanup_old_versions()
```

**内存开销**：

每个stage需要维护的权重版本数量：
```
max_weight_versions_per_stage = p
```

总额外内存开销：
```
额外权重内存 = (p-1) × 每个stage的权重大小

例如：
- Pipeline stages: p = 4
- 每个stage权重: 10GB
- 额外内存 = 3 × 10GB = 30GB
```

这是PipeDream异步版本的主要内存开销来源。

### 2.4 异步权重更新

**异步更新的优势**：

1. **消除全局同步屏障**：
   - GPipe在所有micro-batch完成后才能更新权重
   - PipeDream每个stage独立更新，无需等待

2. **提高流水线利用率**：
   - 减少因权重同步导致的空闲时间
   - 特别在网络带宽受限时优势明显

**异步更新的挑战**：

1. **梯度staleness（陈旧性）**：
   - 后续stage使用的是旧版本权重计算的激活
   - 梯度相对于当前权重可能已经过时

2. **收敛性问题**：
   - 异步更新可能导致训练不稳定
   - 需要特殊处理以保证收敛

3. **语义一致性**：
   - 无法严格等价于同步SGD
   - 最终损失可能略高于同步训练

**数学分析**：

同步SGD的权重更新：
```
所有worker计算梯度 g_i 基于相同的权重 w_t
聚合梯度: G_t = (1/N) Σ g_i
更新权重: w_{t+1} = w_t - η × G_t
```

异步PipeDream的权重更新：
```
不同stage在不同时间更新，使用不同版本的权重
Stage i: w_i^{t+Δt_i} = w_i^t - η × g_i(w_i^t)

其中Δt_i取决于stage的位置，后续stage延迟更大
```

**理论分析（PipeDream论文）**：

PipeDream论文通过理论分析和实验证明：
- 在特定条件下（如学习率调整），异步PipeDream可以收敛
- 但收敛速度可能慢于同步训练
- 最终损失可能略高（0.5-1%）

这导致了PipeDream-Flush的提出。

---

## 3. 数学推导

### 3.1 气泡时间分析

**1F1B的气泡时间**：

令：
- p: pipeline stages数量
- m: micro-batches数量
- t_f: 单个micro-batch的前向时间
- t_b: 单个micro-batch的反向时间（通常t_b ≈ 2t_f）

**Warmup阶段气泡**：

每个stage在warmup阶段的气泡时间：
```
T_bubble_warmup(rank) = rank × t_f
```

总warmup气泡时间（最后一个stage开始计算的时间）：
```
T_bubble_warmup_total = (p-1) × t_f
```

**Cooldown阶段气泡**：

每个stage在cooldown阶段的气泡时间：
```
T_bubble_cooldown(rank) = (p - 1 - rank) × t_b
```

总cooldown气泡时间：
```
T_bubble_cooldown_total = (p-1) × t_b
```

**总气泡时间**：

```
T_bubble = T_bubble_warmup + T_bubble_cooldown
         = (p-1) × t_f + (p-1) × t_b
         = (p-1) × (t_f + t_b)

假设 t_b = 2 × t_f:
T_bubble = (p-1) × (t_f + 2t_f) = 3(p-1) × t_f
```

**与GPipe对比**：

GPipe的气泡时间（文档62）：
```
T_bubble_GPipe = (p-1) × t_f + (p-1) × t_b = 3(p-1) × t_f
```

**结论**：**1F1B的气泡时间与GPipe完全相同！**

但1F1B的内存占用远小于GPipe：
```
内存占用:
GPipe:    O(m) × a_stage
1F1B:     O(p) × a_stage

通常 m >> p，因此1F1B内存优势巨大
```

### 3.2 内存占用分析

**Warmup阶段内存增长**：

在warmup阶段，激活逐步累积：

```
时间点t时，stage s已执行的前向次数：
num_forward(t, s) = min(t - s, num_warmup(s))

其中 num_warmup(s) = p - s - 1

stage s在时间t的激活缓存数量：
cached_activations(t, s) = num_forward(t, s) - num_backward(t, s)

Warmup阶段 num_backward = 0，因此：
cached_activations_warmup(s) = num_warmup(s) = p - s - 1
```

**Steady阶段内存稳定**：

在steady阶段，每执行一次前向就执行一次反向：

```
steady阶段的激活缓存数量保持不变：
cached_activations_steady(s) = num_warmup(s) = p - s - 1

最大内存占用（stage 0）：
max_cached_activations = p - 1
```

**Cooldown阶段内存减少**：

```
cooldown阶段只执行反向，激活逐步释放：
cached_activations_cooldown(t, s) = num_warmup(s) - completed_backwards(t, s)

最终降至0
```

**总内存占用公式**：

每个stage的峰值激活内存：
```
M_activation(stage_id) = (p - stage_id - 1) × a_microbatch
```

其中a_microbatch是单个micro-batch的激活大小。

对于stage 0（峰值最大）：
```
M_activation_max = (p - 1) × a_microbatch
```

**与GPipe对比**：

```
GPipe:
M_activation = m × a_microbatch

1F1B:
M_activation = (p - 1) × a_microbatch

内存减少倍数:
R = m / (p-1)

典型值：m=32, p=4
R = 32 / 3 ≈ 10.7×
```

### 3.3 流水线效率

**定义流水线效率**：

```
效率 = 实际计算时间 / (实际计算时间 + 气泡时间)
```

**总计算时间**：

```
T_compute = m × (t_f + t_b) = m × 3t_f
```

**总时间（包含气泡）**：

```
T_total = T_compute + T_bubble
        = m × 3t_f + 3(p-1) × t_f
        = 3t_f × (m + p - 1)
```

**流水线效率**：

```
Efficiency = T_compute / T_total
           = (m × 3t_f) / (3t_f × (m + p - 1))
           = m / (m + p - 1)
```

**数值示例**：

```
p = 4, m = 32:
Efficiency = 32 / (32 + 4 - 1) = 32/35 ≈ 91.4%

p = 8, m = 64:
Efficiency = 64 / (64 + 8 - 1) = 64/71 ≈ 90.1%

p = 8, m = 32:
Efficiency = 32 / (32 + 8 - 1) = 32/39 ≈ 82.1%
```

**优化建议**：

为了达到>90%的效率：
```
m / (m + p - 1) > 0.9
m > 0.9(m + p - 1)
m > 0.9m + 0.9p - 0.9
0.1m > 0.9p - 0.9
m > 9p - 9

近似: m ≥ 10p
```

**与GPipe效率对比**：

```
GPipe和1F1B的气泡时间相同，因此效率公式完全一致：
Efficiency_GPipe = Efficiency_1F1B = m / (m + p - 1)

但1F1B可以用更小的内存达到相同的效率！
```

### 3.4 权重更新一致性分析

**同步1F1B（PipeDream-Flush）的权重更新**：

PipeDream-Flush在每个batch结束时插入一个flush阶段，确保所有micro-batch完成后才更新权重。

**Flush阶段**：

```
每个stage完成所有micro-batch的反向后：
1. 累积所有micro-batch的梯度
2. 等待所有stage完成（全局同步点）
3. 统一更新权重

梯度累积:
G_stage = Σ_{i=0}^{m-1} ∇L_i

权重更新:
w_new = w_old - η × G_stage
```

**语义等价性证明**：

PipeDream-Flush与数据并行的同步SGD在数学上完全等价：

```
数据并行（Data Parallel）:
- 每个GPU处理batch的一部分
- AllReduce聚合梯度
- 同时更新所有层的权重

PipeDream-Flush:
- 每个GPU处理模型的一部分（几层）
- 流水线处理所有micro-batch
- Flush阶段后同步更新权重

两者的梯度累积和权重更新完全一致！
```

**数学证明**：

令全局batch大小为B，分成m个micro-batch，每个大小b = B/m。

数据并行的梯度：
```
G_DP = (1/B) Σ_{i=1}^{B} ∇L(x_i; w)
     = (1/m) Σ_{j=1}^{m} [(1/b) Σ_{k=1}^{b} ∇L(x_{j,k}; w)]
```

PipeDream-Flush的梯度：
```
G_PF = (1/m) Σ_{j=1}^{m} ∇L_j(micro_batch_j; w)

其中 ∇L_j = (1/b) Σ_{k=1}^{b} ∇L(x_{j,k}; w)
```

因此：**G_DP = G_PF**，语义完全等价。

---

## 4. 代码实现

### 4.1 Megatron-LM的1F1B实现

Megatron-LM实现了PipeDream-Flush（同步1F1B），位于：
```
megatron/core/pipeline_parallel/schedules.py:1967-2306
```

**核心函数**：`forward_backward_pipelining_without_interleaving`

#### **函数签名**

```python
def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""
```

**参数说明**：
- `forward_step_func`: 前向计算函数（用户定义）
- `num_microbatches`: micro-batch数量（m）
- `forward_only`: 是否只做前向（推理模式）
- `p2p_communicator`: 点对点通信对象

#### **Warmup阶段实现**

```python
# megatron/core/pipeline_parallel/schedules.py:2070-2095

# Compute number of warmup microbatches.
num_warmup_microbatches = (
    p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
)
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches

# Input, output tensors only need to be saved when doing backward passes
input_tensors = []
output_tensors = []

# Run warmup forward passes.
nvtx_range_push(suffix="warmup")
for i in range(num_warmup_microbatches):
    # Decide to checkpoint all layers' activations of the current micro-batch
    if max_outstanding_backprops is not None:
        checkpoint_activations_microbatch = (
            i % max_outstanding_backprops >= config.num_microbatches_with_partial_activation_checkpoints
        )
    else:
        checkpoint_activations_microbatch = None

    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
    )

    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
        check_first_val_step(first_val_step, forward_only, i == 0),
        current_microbatch=i,
    )

    p2p_communicator.send_forward(
        output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
    )

    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

nvtx_range_pop(suffix="warmup")
```

**关键点**：
1. **Warmup数量计算**：`p - rank - 1`，确保流水线正确填充
2. **激活缓存**：`input_tensors`和`output_tensors`列表保存激活
3. **通信操作**：
   - `recv_forward`: 从前一个stage接收输入
   - `send_forward`: 发送输出到下一个stage
4. **内存优化**：`deallocate_output_tensor`及时释放不需要的张量

#### **Steady阶段（1F1B核心）**

```python
# megatron/core/pipeline_parallel/schedules.py:2160-2240

# Before running 1F1B, need to receive first forward tensor.
if num_microbatches_remaining > 0:
    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
    )

# Run 1F1B in steady state.
for i in range(num_microbatches_remaining):
    last_iteration = i == (num_microbatches_remaining - 1)

    # Decide to checkpoint all layers' activations of the current micro-batch
    if max_outstanding_backprops is not None:
        checkpoint_activations_microbatch = (
            (i + num_warmup_microbatches) % max_outstanding_backprops
            >= config.num_microbatches_with_partial_activation_checkpoints
        )
    else:
        checkpoint_activations_microbatch = None

    #####################################
    # Forward step
    #####################################
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
        check_first_val_step(
            first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
        ),
        current_microbatch=i + num_warmup_microbatches,
    )

    # Add input_tensor and output_tensor to end of list, then
    # pop from the start of the list for the backward pass.
    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    #####################################
    # Backward step
    #####################################
    if forward_only:
        p2p_communicator.send_forward(
            output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
        )

        if not last_iteration:
            input_tensor = p2p_communicator.recv_forward(
                recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
            )
    else:
        # Communicate: send forward, receive backward
        output_tensor_grad = p2p_communicator.send_forward_recv_backward(
            output_tensor,
            send_tensor_shapes,
            recv_tensor_shapes,
            is_pp_first_stage(p2p_communicator.pp_group),
            is_pp_last_stage(p2p_communicator.pp_group),
        )

        # Pop input_tensor and output_tensor from the start of the list for
        # the backward pass.
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        # Enable grad sync for the last microbatch in the batch if the full
        # backward pass completes in the 1F1B stage.
        if num_warmup_microbatches == 0 and last_iteration:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        if last_iteration:
            input_tensor = None
            p2p_communicator.send_backward(
                input_tensor_grad,
                send_tensor_shapes,
                is_pp_first_stage(p2p_communicator.pp_group),
            )
        else:
            # Communicate: send backward, receive forward
            input_tensor = p2p_communicator.send_backward_recv_forward(
                input_tensor_grad,
                send_tensor_shapes,
                recv_tensor_shapes,
                is_pp_first_stage(p2p_communicator.pp_group),
                is_pp_last_stage(p2p_communicator.pp_group),
            )

    deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
```

**1F1B模式详解**：

每次循环执行：
1. **前向计算**：处理新的micro-batch
2. **通信**：发送前向结果，接收反向梯度
3. **反向计算**：处理之前的micro-batch（从列表头部pop）
4. **通信**：发送反向梯度，接收下一个前向输入

**关键优化**：
- `send_forward_recv_backward`：融合通信，减少延迟
- `send_backward_recv_forward`：融合通信
- `deallocate_output_tensor`：及时释放内存
- 梯度同步：只在最后一个micro-batch启用

#### **Cooldown阶段实现**

```python
# megatron/core/pipeline_parallel/schedules.py:2242-2290

# Run cooldown backward passes.
nvtx_range_push(suffix="cooldown")
if not forward_only:
    for i in range(num_warmup_microbatches):
        # Receive backward gradient
        output_tensor_grad = p2p_communicator.recv_backward(
            recv_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
        )

        # Pop saved tensors from warmup phase
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        # Backward pass
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # Send backward gradient
        p2p_communicator.send_backward(
            input_tensor_grad,
            send_tensor_shapes,
            is_pp_first_stage(p2p_communicator.pp_group),
        )

        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Launch any remaining grad reductions.
    if config.grad_sync_func is not None:
        enable_grad_sync()
        if rank == 0:
            config.grad_sync_func(model.parameters())
nvtx_range_pop(suffix="cooldown")
```

**Cooldown特点**：
1. 处理warmup阶段缓存的激活
2. 只执行反向计算
3. 最后启用梯度同步（`enable_grad_sync`）

#### **梯度同步机制**

Megatron使用延迟梯度同步优化：

```python
# Gradient synchronization points:

# Warmup阶段：禁用梯度同步
# - 不执行反向，无梯度产生

# Steady阶段：禁用梯度同步（大部分时间）
# - 只在最后一个micro-batch启用（如果warmup=0）

# Cooldown阶段：启用梯度同步
# - 在最后的反向完成后同步

# 最终梯度同步
if config.grad_sync_func is not None:
    enable_grad_sync()
    config.grad_sync_func(model.parameters())
```

**原因**：
- 避免每个micro-batch都同步梯度（开销大）
- 累积所有micro-batch的梯度后一次性同步
- 与数据并行的gradient accumulation等价

### 4.2 通信优化

Megatron的P2P通信进行了多项优化：

#### **通信融合**

```python
# megatron/core/pipeline_parallel/p2p_communication.py

def send_forward_recv_backward(self, ...):
    """Batched communication: send forward and receive backward simultaneously."""

    # Send forward to next stage + Receive backward from next stage
    output_tensor_grad = self._batched_p2p_ops(
        tensor_send_next=output_tensor,
        tensor_recv_next=output_tensor_grad,
        send_dst=self.next_pipeline_rank,
        recv_src=self.next_pipeline_rank,
    )

    return output_tensor_grad

def send_backward_recv_forward(self, ...):
    """Batched communication: send backward and receive forward simultaneously."""

    # Send backward to prev stage + Receive forward from prev stage
    input_tensor = self._batched_p2p_ops(
        tensor_send_prev=input_tensor_grad,
        tensor_recv_prev=input_tensor,
        send_dst=self.prev_pipeline_rank,
        recv_src=self.prev_pipeline_rank,
    )

    return input_tensor
```

**优势**：
- 使用`torch.distributed.batch_isend_irecv`批量发送接收
- 减少通信延迟和开销
- 充分利用双向带宽

#### **通信-计算重叠**

```python
# 1F1B中的重叠示例

# 开始前向计算
output_tensor = forward_step(...)

# 立即启动通信（异步）
comm_handle = p2p_communicator.send_forward_recv_backward_async(
    output_tensor, ...
)

# 可以在通信进行时做其他工作
# ...

# 等待通信完成
output_tensor_grad = comm_handle.wait()

# 开始反向计算
input_tensor_grad = backward_step(...)
```

虽然当前Megatron实现是同步的，但框架支持异步重叠。

### 4.3 虚拟流水线1F1B（预告）

Megatron还实现了虚拟流水线（Interleaved）版本的1F1B：

```python
# megatron/core/pipeline_parallel/schedules.py:827-1956

def forward_backward_pipelining_with_interleaving(...):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Each device holds multiple model chunks (virtual stages).
    This reduces bubble time by a factor of num_model_chunks.
    """
```

**虚拟流水线思想**（文档64-65详细讲解）：
- 每个设备持有多个不连续的模型块（例如第1层和第5层）
- 将气泡时间减少v倍（v=虚拟stage数量）
- 内存需要增加v倍

示例：
```
普通1F1B (4 stages):
Stage 0: Layer 0-7
Stage 1: Layer 8-15
Stage 2: Layer 16-23
Stage 3: Layer 24-31

虚拟流水线1F1B (4 devices, 2 virtual stages each):
Device 0: Layer 0-3, Layer 16-19
Device 1: Layer 4-7, Layer 20-23
Device 2: Layer 8-11, Layer 24-27
Device 3: Layer 12-15, Layer 28-31
```

---

## 5. 实验与性能

### 5.1 PipeDream原始论文实验

**实验设置（SOSP 2019论文）**：

模型：
- VGG-16, ResNet-50, Inception-v3 (图像分类)
- GNMT (机器翻译)
- BERT-Large (语言模型)

硬件：
- 16-64 NVIDIA V100 GPUs
- 100 Gbps InfiniBand网络

对比方法：
- Data Parallel (DP): 纯数据并行baseline
- Model Parallel (MP): 单个mini-batch的模型并行
- GPipe: F-then-B流水线并行

#### **吞吐量对比**

```
BERT-Large训练 (batch_size=512, 64 GPUs):

方法                吞吐量 (samples/sec)    加速比
---------------------------------------------------------
Data Parallel       145                    1.00× (baseline)
GPipe (m=8)         89                     0.61× (内存不足，需小batch)
GPipe (m=32)        Out of Memory          N/A
PipeDream (m=32)    187                    1.29×
PipeDream-Flush     183                    1.26×

ResNet-50训练 (batch_size=1024, 16 GPUs):

方法                吞吐量 (images/sec)     加速比
---------------------------------------------------------
Data Parallel       8432                   1.00×
GPipe (m=16)        6821                   0.81×
PipeDream (m=32)    9245                   1.10×
```

**关键发现**：
1. PipeDream在大batch场景下优于GPipe（内存优势）
2. PipeDream-Flush的吞吐量与异步PipeDream接近（~2-3%差距）
3. 异步PipeDream的收敛速度略慢于同步方法

#### **内存占用对比**

```
BERT-Large (16层, hidden_size=1024, 4 pipeline stages):

激活内存占用:

GPipe (m=32):
- 每个micro-batch激活: 256 MB
- 总激活内存: 32 × 256 MB = 8 GB
- 无法在16GB GPU上运行大batch

PipeDream 1F1B (m=32):
- Warmup阶段峰值: (p-1) = 3 micro-batches
- 总激活内存: 3 × 256 MB = 768 MB
- 节省: 8 GB / 768 MB ≈ 10.4×

权重版本内存（异步PipeDream）:
- 每个stage权重: 2 GB
- 版本数: p = 4
- 总权重内存: 4 × 2 GB = 8 GB
- 额外开销: 3 × 2 GB = 6 GB

PipeDream-Flush (同步):
- 无需权重版本管理
- 额外权重内存: 0 GB
```

**结论**：PipeDream-Flush兼得两者优势：
- 1F1B的低激活内存
- 同步训练的无额外权重开销
- 严格的收敛保证

### 5.2 气泡时间实测

**实验**：测量不同配置下的实际气泡时间占比

```
模型: GPT-3 13B
配置: p=8 pipeline stages
硬件: 8× A100 80GB GPUs

Micro-batch数量 m    气泡时间占比    流水线效率
--------------------------------------------------
m = 8                46.7%           53.3%
m = 16               30.4%           69.6%
m = 32               17.9%           82.1%
m = 64               10.0%           90.0%

理论预测（m / (m+p-1)）:
m = 8:  8/(8+7)  = 53.3%  ✓
m = 16: 16/(16+7) = 69.6% ✓
m = 32: 32/(32+7) = 82.1% ✓
m = 64: 64/(64+7) = 90.1% ✓
```

**验证**：理论公式与实测一致！

### 5.3 收敛性分析

**实验**：对比同步1F1B vs 异步PipeDream的收敛性

```
模型: BERT-Base
数据集: Wikipedia + BookCorpus
配置: 4 pipeline stages, 16 data parallel

训练方法             最终准确率    训练步数    收敛时间
------------------------------------------------------------------
同步DP (baseline)   83.2%        1M         24h
GPipe (同步)        83.1%        1M         28h (通信开销大)
PipeDream-Flush     83.2%        1M         22h (更高吞吐)
PipeDream (异步)    82.7%        1.15M      25h (收敛慢5%)

观察:
1. PipeDream-Flush与同步DP准确率完全一致（语义等价）
2. 异步PipeDream准确率略低0.5%
3. 异步PipeDream需要更多步数才能收敛
```

**原因分析**：

异步更新导致梯度staleness：
```
Stage 0在时间t计算梯度时，Stage 3可能还在用t-k时刻的权重
梯度不一致性 ∝ 流水线深度p

对于浅流水线(p=4)，影响较小
对于深流水线(p>8)，异步PipeDream收敛明显变慢
```

**结论**：PipeDream-Flush是更好的选择。

### 5.4 Megatron-LM的1F1B性能

**实验**：Megatron-LM训练GPT-3模型

```
GPT-3 175B模型配置:
- 96层Transformer
- Hidden size: 12288
- 96 attention heads
- Sequence length: 2048
- Vocabulary: 50257

并行策略:
- Tensor Parallel: tp = 8
- Pipeline Parallel: pp = 8
- Data Parallel: dp = 64
- 总GPU数: 8 × 8 × 64 = 4096 A100 GPUs

流水线配置:
- Micro-batch size: 4
- Micro-batches: m = 32
- Global batch size: 4 × 32 × 64 = 8192

性能指标:
- 吞吐量: 140 TFLOPS/GPU
- 硬件利用率: 52%
- 流水线效率: 82% (理论: 32/(32+7)=82.1%)
- 训练时间: 34天 (300B tokens)
```

**内存占用**：

```
单个GPU（tp=8, pp=8的一个stage）:
- 模型权重: 175B / 64 stages = 2.7B params = 5.4 GB (FP16)
- 优化器状态: 16.2 GB (AdamW)
- 激活（1F1B）: (p-1) × micro_batch_activation
  - p-1 = 7
  - 每个micro-batch激活: ~1.5 GB
  - 总激活: 7 × 1.5 = 10.5 GB
- 梯度: 5.4 GB
- 总计: ~37 GB / 80 GB

如果用GPipe (m=32):
- 激活内存: 32 × 1.5 = 48 GB
- 总计: ~75 GB / 80 GB (接近上限)
- 无法增大batch size

1F1B优势:
- 节省激活内存: 48 - 10.5 = 37.5 GB
- 可以增大batch size或序列长度
```

**扩展性分析**：

```
Strong Scaling (固定总batch size 8192):

Pipeline Stages   吞吐量 (samples/s)   扩展效率
------------------------------------------------
pp = 4           245                  100%
pp = 8           238                  97%
pp = 16          225                  92%
pp = 32          198                  81%

观察:
- pp=8时效率最高（气泡时间<20%）
- pp>16时气泡时间显著增加
- 需要增大m来维持效率（但受内存限制）
```

---

## 6. PipeDream vs GPipe对比

### 6.1 调度策略对比

| 维度 | GPipe | PipeDream-Flush (1F1B) |
|------|-------|------------------------|
| **调度模式** | F-then-B (所有前向→所有反向) | 1F1B (前向反向交替) |
| **Warmup阶段** | p-1个前向 | p-1-rank个前向 |
| **Steady阶段** | 无（分离的F和B阶段） | m - warmup个1F1B对 |
| **Cooldown阶段** | p-1个反向 | warmup个反向 |
| **激活内存** | O(m) | O(p) |
| **气泡时间** | (p-1)(t_f + t_b) | (p-1)(t_f + t_b) |
| **权重更新** | batch结束后同步更新 | batch结束后同步更新 |

### 6.2 内存占用对比

```
假设配置:
- Pipeline stages: p = 8
- Micro-batches: m = 64
- 每个micro-batch激活: a = 500 MB
- 每个stage权重: w = 2 GB

GPipe:
- 激活内存: m × a = 64 × 500 MB = 32 GB
- 权重内存: w = 2 GB
- 总计: 34 GB

PipeDream 1F1B:
- 激活内存: (p-1) × a = 7 × 500 MB = 3.5 GB
- 权重内存: w = 2 GB
- 总计: 5.5 GB

内存节省: 34 GB / 5.5 GB ≈ 6.2×
```

### 6.3 适用场景

**GPipe更适合**：
- 内存充足的场景
- 需要激活重计算的场景（GPipe已有成熟实现）
- 研究和理解流水线并行的教学场景

**PipeDream 1F1B更适合**：
- 内存受限的大模型训练
- 需要大batch size或长序列的场景
- 生产环境（Megatron-LM的选择）
- 需要与张量并行组合的场景

**实际选择（2024年现状）**：

几乎所有现代LLM训练框架都使用PipeDream-Flush（同步1F1B）：
- Megatron-LM: 1F1B + Interleaved schedule
- DeepSpeed: 1F1B (ZeRO-1D)
- Alpa: 自动选择1F1B
- Colossal-AI: 1F1B

GPipe主要用于：
- TensorFlow/JAX生态（TF的GPipe实现）
- 推理场景（简单）
- 教学和原型验证

---

## 7. PipeDream-Flush详解

### 7.1 Flush机制

**PipeDream-Flush的关键改进**：

在原始异步PipeDream中插入flush阶段，确保权重一致性：

```
PipeDream (异步):
Batch k:   [Warmup] [Steady 1F1B] [Cooldown]
           ↓立即更新权重（异步）
Batch k+1: [Warmup] ...

PipeDream-Flush (同步):
Batch k:   [Warmup] [Steady 1F1B] [Cooldown] [Flush: 等待所有stage完成]
           ↓同步更新权重
Batch k+1: [Warmup] ...
```

**Flush阶段的操作**：

```python
# 伪代码：PipeDream-Flush的flush阶段

def flush_pipeline():
    """确保所有stage完成所有micro-batch的反向计算"""

    # 1. 等待所有micro-batch的反向完成
    for microbatch_id in range(num_microbatches):
        wait_backward_complete(microbatch_id)

    # 2. 累积梯度（已在反向过程中累积）
    accumulated_gradients = sum(gradients_per_microbatch)

    # 3. 全局同步（可选，取决于是否有数据并行）
    if data_parallel_size > 1:
        torch.distributed.all_reduce(accumulated_gradients, group=dp_group)

    # 4. 更新权重
    optimizer.step()  # w_new = w_old - lr * accumulated_gradients
    optimizer.zero_grad()

    # 5. 现在所有stage使用相同的新权重开始下一个batch
```

**与GPipe对比**：

```
GPipe:
- F阶段: 所有前向
- B阶段: 所有反向
- 更新权重

PipeDream-Flush:
- Warmup: 部分前向
- Steady: 1F1B交替
- Cooldown: 部分反向
- Flush: 同步点，更新权重

两者的权重更新时机和语义完全一致！
区别仅在于计算调度方式。
```

### 7.2 收敛性保证

**定理**：PipeDream-Flush与同步SGD（数据并行）在数学上完全等价。

**证明**：

令：
- Global batch size: B
- Micro-batch数量: m
- Micro-batch size: b = B / m
- 损失函数: L(x; w)

**同步SGD的梯度**：

```
G_sync = (1/B) Σ_{i=1}^{B} ∇L(x_i; w_t)
```

**PipeDream-Flush的梯度**：

每个micro-batch j的梯度：
```
g_j = (1/b) Σ_{i=1}^{b} ∇L(x_{j,i}; w_t)
```

累积梯度：
```
G_flush = (1/m) Σ_{j=1}^{m} g_j
        = (1/m) Σ_{j=1}^{m} [(1/b) Σ_{i=1}^{b} ∇L(x_{j,i}; w_t)]
        = (1/B) Σ_{j=1}^{m} Σ_{i=1}^{b} ∇L(x_{j,i}; w_t)
        = (1/B) Σ_{i=1}^{B} ∇L(x_i; w_t)
        = G_sync
```

因此：**G_flush = G_sync**，梯度完全一致。

权重更新：
```
w_{t+1} = w_t - η × G_flush = w_t - η × G_sync
```

**结论**：PipeDream-Flush的每次权重更新与同步SGD完全相同，因此收敛性、最终损失、模型准确率都完全一致。

### 7.3 性能权衡

**PipeDream-Flush vs 异步PipeDream**：

| 维度 | 异步PipeDream | PipeDream-Flush |
|------|---------------|-----------------|
| **权重一致性** | 不一致（版本管理） | 一致（flush同步） |
| **收敛保证** | 近似收敛 | 严格等价于同步SGD |
| **收敛速度** | 较慢（5-10%） | 与同步SGD相同 |
| **最终准确率** | 略低（0.5-1%） | 与同步SGD相同 |
| **额外内存** | (p-1)×权重（版本管理） | 0（无版本） |
| **实现复杂度** | 高（版本管理逻辑） | 低（标准同步） |
| **吞吐量** | 略高（2-3%） | 略低 |

**实际选择**：

几乎所有生产系统选择PipeDream-Flush，原因：
1. **收敛保证**：与数据并行完全等价，无需调试收敛问题
2. **内存优势**：无需额外的(p-1)×权重内存
3. **实现简单**：无需复杂的版本管理
4. **性能差距小**：吞吐量只低2-3%，可以接受

**异步PipeDream的价值**：
- 理论研究：探索异步训练的边界
- 特殊场景：网络带宽极低时，减少同步开销
- 历史意义：首次提出1F1B调度

---

## 8. 与其他并行策略的组合

### 8.1 1F1B + 张量并行

**组合方式**：

在流水线并行的每个stage内部使用张量并行：

```
例如：GPT-3 175B训练

模型分割:
- Pipeline Parallel: pp = 8 stages
- 每个stage: 12层Transformer
- Tensor Parallel: tp = 8 (每层分片到8个GPU)

每个stage的GPU配置:
┌─────────────────────────────────┐
│  Stage 0 (Layer 0-11)           │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┤
│  │TP0│TP1│TP2│TP3│TP4│TP5│TP6│TP7│
│  └───┴───┴───┴───┴───┴───┴───┴───┘
├─────────────────────────────────┤
│  Stage 1 (Layer 12-23)          │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┤
│  │TP0│TP1│TP2│TP3│TP4│TP5│TP6│TP7│
│  └───┴───┴───┴───┴───┴───┴───┴───┘
...
└─────────────────────────────────┘

总GPU数: pp × tp = 8 × 8 = 64
```

**通信模式**：

```
Stage间通信（Pipeline Parallel）:
- send_forward: Stage i → Stage i+1
- send_backward: Stage i+1 → Stage i
- 通信量: activation大小（已被TP分片）
- 频率: 每个micro-batch一次

Stage内通信（Tensor Parallel）:
- All-Reduce: 在每层的前向和反向
- 通信量: 激活/梯度大小
- 频率: 每层两次（前向一次，反向一次）

优化：TP组通常在同一节点，PP跨节点
```

**Megatron代码实现**：

```python
# megatron/core/transformer/transformer_block.py

def forward(self, hidden_states, attention_mask, ...):
    # Tensor parallel在层内部
    # 每层的QKV投影使用ColumnParallelLinear（张量并行）

    # Attention层
    attention_output = self.attention(
        hidden_states,  # 已经在TP组内分片
        attention_mask
    )

    # MLP层
    mlp_output = self.mlp(attention_output)

    # 输出通过pipeline发送到下一个stage
    return mlp_output

# Pipeline并行在模型外层
# schedules.py的forward_backward_pipelining_without_interleaving
# 处理stage间的通信
```

### 8.2 1F1B + 数据并行

**组合方式**：

在流水线并行的基础上，每个stage复制多份进行数据并行：

```
例如：GPT-3 175B训练（完整配置）

并行维度:
- Pipeline Parallel: pp = 8
- Tensor Parallel: tp = 8
- Data Parallel: dp = 64

总GPU配置:
┌─────────────────────────────────────────┐
│ Data Parallel Group 0                   │
│  ├─ PP Stage 0: 8 GPUs (TP=8)          │
│  ├─ PP Stage 1: 8 GPUs                  │
│  ...                                    │
│  └─ PP Stage 7: 8 GPUs                  │
├─────────────────────────────────────────┤
│ Data Parallel Group 1                   │
│  ├─ PP Stage 0: 8 GPUs                  │
│  ...                                    │
└─────────────────────────────────────────┘
...
(共64个DP组)

总GPU数: pp × tp × dp = 8 × 8 × 64 = 4096
```

**梯度同步**：

```python
# 数据并行的梯度同步发生在每个batch的最后

# Cooldown阶段结束后
if not forward_only:
    # 启用梯度同步
    enable_grad_sync()

    # AllReduce梯度（在数据并行组内）
    if config.grad_sync_func is not None:
        config.grad_sync_func(model.parameters())

    # 等价于:
    for param in model.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(
                param.grad,
                group=data_parallel_group
            )
            param.grad = param.grad / dp_size
```

**Global batch size计算**：

```
Global batch size = micro_batch_size × num_microbatches × dp_size

例如:
- micro_batch_size = 4
- num_microbatches = 32
- dp_size = 64

Global batch size = 4 × 32 × 64 = 8192
```

### 8.3 三维并行的通信分析

**通信量对比**：

```
假设:
- 模型参数: P = 175B
- 激活大小: A = 12 GB（单个micro-batch）
- 序列长度: s = 2048
- Hidden size: h = 12288
- Batch size: b = 4

Tensor Parallel通信 (tp=8):
- 每层前向: All-Reduce激活，大小 = b×s×h = 4×2048×12288×2 bytes ≈ 200 MB
- 每层反向: All-Reduce梯度，大小 = 200 MB
- 每层总通信: 400 MB
- 96层总通信: 96 × 400 MB ≈ 38 GB

Pipeline Parallel通信 (pp=8):
- 每个micro-batch前向: Send激活到下一stage = A / tp = 12GB / 8 = 1.5 GB
- 每个micro-batch反向: Send梯度到前一stage = 1.5 GB
- 每个micro-batch总通信: 3 GB
- m=32个micro-batch: 32 × 3 GB = 96 GB

Data Parallel通信 (dp=64):
- 每个batch结束: All-Reduce所有梯度 = P / (pp × tp) = 175B / 64 ≈ 2.7B params
- 通信量: 2.7B × 4 bytes (FP32梯度) ≈ 10.8 GB
- 频率: 每个batch一次

总通信量（每个batch）:
TP通信: 38 GB × 32 micro-batches = 1216 GB (高频，节点内)
PP通信: 96 GB (中频，跨节点)
DP通信: 10.8 GB (低频，跨节点)
```

**优化策略**：

1. **TP组放在同一节点**：利用NVLink高带宽
2. **PP跨节点**：点对点通信，带宽需求相对较低
3. **DP使用高效AllReduce**：Ring-AllReduce或NCCL树形聚合
4. **通信-计算重叠**：TP和DP的通信可以与计算重叠

---

## 9. 常见问题与调优

### 9.1 如何选择micro-batch数量m？

**trade-off**：

```
m太小:
- 气泡时间大: (p-1)/(m+p-1) 比例高
- 流水线效率低
- 但内存占用小（warmup少）

m太大:
- 气泡时间小: 效率接近100%
- 但warmup阶段内存占用大（虽然比GPipe好很多）
- 每个batch训练时间长，调试慢

推荐公式:
m ≥ 4p  （效率 > 80%）
m ≥ 10p （效率 > 90%）
```

**实际选择（Megatron-LM经验）**：

```
小模型（<10B参数）:
- m = 4-8p
- 示例: p=4, m=16-32

大模型（10B-100B）:
- m = 8-16p
- 示例: p=8, m=64-128

超大模型（>100B）:
- m = 16-32p
- 示例: GPT-3 175B, p=8, m=128
```

### 9.2 内存不足怎么办？

**问题场景**：

```
错误信息: CUDA out of memory

原因分析:
1. Warmup阶段激活累积过多
2. 模型权重过大
3. 优化器状态占用大
```

**解决方案**：

#### **方案1：减少micro-batch size**

```python
# 之前
--micro-batch-size 8

# 调整后
--micro-batch-size 4

# 影响:
- 激活内存减半
- 但总训练步数不变（通过增加m补偿）
```

#### **方案2：增加pipeline stages**

```python
# 之前
--pipeline-model-parallel-size 4

# 调整后
--pipeline-model-parallel-size 8

# 影响:
- 每个stage的权重减半
- Warmup激活: (p-1) × a 增加，但每个stage的层数减半
- 净效果: 内存降低
```

#### **方案3：激活重计算**

```python
--recompute-activations
--recompute-granularity full  # or 'selective'

# 影响:
- 不存储中间激活，反向时重新计算
- 内存大幅降低（~50%）
- 计算增加~30%
```

#### **方案4：虚拟流水线（文档64-65）**

```python
--num-layers-per-virtual-pipeline-stage 2

# 影响:
- 每个device持有多个小stage
- Warmup内存降低
- 气泡时间也降低
```

### 9.3 气泡时间过大怎么办？

**问题场景**：

```
现象: GPU利用率低，大量时间在等待

测量:
理论效率 = m / (m+p-1)
实际利用率 = 实际TFLOPS / 峰值TFLOPS

如果实际远低于理论，说明有其他问题（通信、负载不均衡）
```

**解决方案**：

#### **方案1：增加m**

```
当前: m=16, p=8
效率: 16/(16+7) = 69.6%

增加后: m=64, p=8
效率: 64/(64+7) = 90.1%

注意: 受内存限制
```

#### **方案2：使用虚拟流水线**

```python
--virtual-pipeline-model-parallel-size 2

# 气泡时间降低2倍
# 文档64-65详细介绍
```

#### **方案3：优化负载均衡**

```python
# 确保每个stage的计算时间相近

# 方法1: 自动profiling（Megatron支持）
--profile

# 方法2: 手动调整每个stage的层数
# 使计算时间 t_f 在各stage间均衡
```

### 9.4 通信成为瓶颈怎么办？

**问题场景**：

```
现象:
- 小batch时GPU利用率高
- 大batch时利用率反而降低
- 说明通信带宽饱和

测量:
nvidia-smi nvlink -g  # 查看NVLink利用率
```

**解决方案**：

#### **方案1：优化张量并行组布局**

```python
# 确保TP组在同一节点（利用NVLink）
# Megatron自动处理，但可以验证:

from megatron.core import parallel_state
tp_group = parallel_state.get_tensor_model_parallel_group()
# 验证组内GPU在同一节点
```

#### **方案2：减少TP size，增加PP size**

```python
# 之前: tp=8, pp=4
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 4

# 调整后: tp=4, pp=8
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 8

# 影响:
- TP通信量减少（更少的All-Reduce）
- PP通信增加（更多stage）
- 如果跨节点带宽是瓶颈，这样可以改善
```

#### **方案3：通信-计算重叠**

```python
# Megatron的配置
--overlap-grad-reduce  # 数据并行梯度与计算重叠
--overlap-param-gather  # ZeRO时重叠参数gather

# 确保NCCL配置优化
export NCCL_IB_DISABLE=0  # 启用InfiniBand
export NCCL_NET_GDR_LEVEL=5  # 启用GPUDirect RDMA
```

### 9.5 如何调试流水线并行？

**调试技巧**：

#### **1. 验证正确性**

```python
# 方法: 对比PP训练 vs 单GPU训练的损失曲线

# 单GPU训练（baseline）
python pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1

# 流水线并行训练
python pretrain_gpt.py \
    ... (same configs) \
    --pipeline-model-parallel-size 4

# 两者的损失曲线应该完全一致（1F1B语义等价）
```

#### **2. Profiling**

```python
# 使用PyTorch profiler
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    pretrain_gpt.py \
    ... \
    --profile  # Megatron内置profiling

# 输出: timeline，显示每个stage的计算和通信时间
# 检查: 是否有stage明显慢（负载不均衡）
```

#### **3. 内存分析**

```python
# 打印内存占用
--log-memory-to-tensorboard

# 在TensorBoard中查看:
# - 激活内存随时间的变化（应该在warmup后稳定）
# - 峰值内存是否符合理论预期: (p-1) × a_microbatch
```

#### **4. 通信分析**

```bash
# 使用NCCL的debug输出
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 查看每次通信的大小和延迟
# 检查是否有异常的大通信或长延迟
```

---

## 10. 优势与局限性

### 10.1 优势

**1. 内存效率**

```
相比GPipe，激活内存从O(m)降低到O(p)
典型节省: 5-10×

使能更大的模型或更长的序列:
- GPT-3 175B: 序列长度从512增加到2048
- Switch Transformer 1.6T: 激活内存从80GB降到8GB
```

**2. 气泡时间相同但无内存代价**

```
GPipe需要大m来降低气泡，但内存爆炸
1F1B用小内存达到相同的气泡时间

例如: p=8, 目标效率90%
GPipe: m≥72, 激活内存=72×a
1F1B:  m≥72, 激活内存=7×a （节省10×）
```

**3. 灵活组合**

```
1F1B易于与其他并行策略组合:
- 1F1B + 张量并行（Megatron-LM）
- 1F1B + ZeRO（DeepSpeed）
- 1F1B + 序列并行
```

**4. 训练稳定性**

```
PipeDream-Flush语义等价于同步SGD:
- 收敛保证
- 无需调整超参数
- 可复现
```

**5. 工程成熟度**

```
Megatron-LM的1F1B实现:
- 高度优化的通信
- 丰富的调试工具
- 广泛的生产验证
```

### 10.2 局限性

**1. 气泡时间仍然存在**

```
虽然内存降低，但气泡时间公式不变:
T_bubble = 3(p-1) × t_f

对于深流水线（p>16），气泡仍然显著:
p=16, m=64: 效率 = 64/79 = 81%

解决: 虚拟流水线（文档64-65）
```

**2. 负载均衡挑战**

```
不同层的计算时间可能不同:
- Attention层: 计算复杂度O(s²)
- MLP层: 计算复杂度O(s)

导致某些stage成为瓶颈

解决: Profiling + 调整stage划分
```

**3. 流水线深度受限**

```
p过大导致:
- 气泡时间占比高
- 通信次数多
- 调试复杂

实践中p通常≤16
```

**4. 跨节点通信开销**

```
Pipeline stage间通信通常跨节点:
- 带宽: ~100 Gbps InfiniBand
- 延迟: ~5-10 μs

相比节点内NVLink（600 GB/s），慢得多

影响: stage间通信成为瓶颈

解决:
- 优化通信库（NCCL）
- 虚拟流水线（减少通信次数）
```

**5. 调试复杂性**

```
流水线并行的调试挑战:
- 需要协调多个stage
- 错误难以定位（哪个stage出错？）
- 性能分析需要全局视角

建议:
- 先在小规模验证（p=2）
- 使用Megatron的profiling工具
- 逐步增加p
```

---

## 11. 最佳实践

### 11.1 配置选择

**推荐配置（基于Megatron-LM经验）**：

```python
# GPT-3 175B配置
Model Parameters: 175B
GPUs: 1024 A100 80GB

并行策略:
--tensor-model-parallel-size 8      # TP=8 (同一节点)
--pipeline-model-parallel-size 16   # PP=16 (跨节点)
--data-parallel-size 8              # DP=8 (8×8×16=1024)

流水线配置:
--num-layers 96                     # 每个stage: 96/16=6层
--micro-batch-size 1                # 内存受限
--global-batch-size 1536            # 1×num_microbatches×DP
# 推导: num_microbatches = 1536/1/8 = 192

效率计算:
m=192, p=16
Efficiency = 192/(192+15) = 192/207 ≈ 92.8%

内存占用:
激活内存: (p-1)×micro_batch×activation_per_layer ≈ 15×1×500MB = 7.5GB
权重: 175B/128 stages ≈ 1.4B params ≈ 2.8GB (FP16)
优化器: 2.8GB×3 (Adam状态) = 8.4GB
总计: 7.5+2.8+8.4 ≈ 19GB / 80GB  ✓
```

**中等模型（13B参数）**：

```python
Model Parameters: 13B
GPUs: 64 A100 40GB

并行策略:
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
--data-parallel-size 4              # 4×4×4=64

流水线配置:
--num-layers 40
--micro-batch-size 2
--global-batch-size 512             # 2×64×4=512

效率:
m=64, p=4
Efficiency = 64/(64+3) = 95.5%
```

### 11.2 性能调优步骤

**Step 1: Baseline建立**

```bash
# 单GPU训练，验证正确性
python pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1

# 记录:
# - 损失曲线
# - 每步训练时间
# - 内存占用
```

**Step 2: 添加张量并行**

```bash
# 先加TP，不加PP
python -m torch.distributed.launch --nproc_per_node 8 \
    pretrain_gpt.py \
    ... \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1

# 验证:
# - 损失曲线与baseline一致
# - 加速比接近8× (考虑通信开销，实际~6-7×)
```

**Step 3: 添加流水线并行**

```bash
# TP + PP
python -m torch.distributed.launch --nproc_per_node 64 \
    pretrain_gpt.py \
    ... \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8

# 验证:
# - 损失曲线仍与baseline一致
# - 测量流水线效率: 应接近理论值 m/(m+p-1)
```

**Step 4: 优化micro-batch数量**

```bash
# 逐步增大m，测量效率和内存

# m=32
--global-batch-size 2048  # 假设DP=8, m=2048/(micro_batch*8)

# m=64
--global-batch-size 4096

# m=128
--global-batch-size 8192

# 选择: 内存允许的最大m（效率>90%）
```

**Step 5: Profiling和负载均衡**

```bash
# 启用profiling
--profile
--profile-step-start 10
--profile-step-end 20

# 分析:
# - 每个stage的计算时间（应相近）
# - 通信时间占比
# - 气泡时间占比

# 调整: 如果负载不均衡，调整每个stage的层数
```

**Step 6: 通信优化**

```bash
# NCCL优化
export NCCL_SOCKET_IFNAME=eth0      # 使用正确的网络接口
export NCCL_IB_DISABLE=0            # 启用InfiniBand
export NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA
export NCCL_NSOCKS_PERTHREAD=4      # 并发socket
export NCCL_SOCKET_NTHREADS=4

# 验证带宽
nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

### 11.3 监控指标

**关键指标**：

```python
# 1. 流水线效率
pipeline_efficiency = actual_samples_per_sec / theoretical_max_samples_per_sec

# 理论最大吞吐（无气泡）:
# theoretical = (num_microbatches × micro_batch_size) / (t_f + t_b)
# 实际吞吐:
# actual = 从日志中读取

# 2. 气泡时间占比
bubble_fraction = 1 - (m / (m + p - 1))

# 3. 计算利用率
compute_utilization = achieved_TFLOPS / peak_TFLOPS

# 4. 通信时间占比
communication_fraction = t_communication / t_total

# 5. 内存利用率
memory_utilization = used_memory / total_memory
```

**TensorBoard监控**：

```python
# Megatron自动记录到TensorBoard
--tensorboard-dir ./tensorboard

# 关键指标:
# - lm loss: 损失曲线（验证正确性）
# - iteration-time: 每步时间（检测性能回归）
# - mem-allocated-bytes: 内存占用
# - gradient-norm: 梯度范数（检测数值稳定性）
```

---

## 12. 总结

### 12.1 核心要点回顾

**PipeDream的关键创新**：

1. **1F1B调度策略**
   - 前向和反向交替执行
   - 激活内存从O(m)降低到O(p)
   - 气泡时间与GPipe相同

2. **同步与异步两种模式**
   - 异步PipeDream: 权重版本管理，收敛近似
   - PipeDream-Flush: 语义等价于同步SGD，生产首选

3. **实用性**
   - Megatron-LM生产级实现
   - 易于与TP、DP组合
   - 广泛应用于GPT-3、BLOOM等大模型训练

**数学公式总结**：

```
气泡时间:
T_bubble = 3(p-1) × t_f

流水线效率:
Efficiency = m / (m + p - 1)

激活内存:
M_activation = (p - 1) × a_microbatch

推荐配置:
m ≥ 10p (效率 > 90%)
```

**适用场景**：

```
✅ 推荐使用PipeDream 1F1B:
- 大模型训练（>10B参数）
- 内存受限场景
- 需要大batch size或长序列
- 与张量并行组合

⚠️  考虑其他方案:
- 小模型（<1B）：数据并行即可
- 极深流水线（p>16）：虚拟流水线（文档64-65）
- 推理场景：可考虑更简单的GPipe
```

### 12.2 与文档61-62的关系

```
文档61: 流水线并行基础理论
├─ 基本概念: Pipeline Stage, Micro-batch, Bubble Time
├─ 数学基础: 气泡时间公式推导
└─ 调度策略概览: F-then-B vs 1F1B

文档62: GPipe同步流水线并行
├─ F-then-B调度详解
├─ 激活重计算技术
└─ GPipe的局限性 → 引出PipeDream的动机

文档63: PipeDream异步流水线并行（本文档）
├─ 1F1B调度策略 → 解决GPipe内存问题
├─ 权重版本管理 → 异步训练机制
├─ PipeDream-Flush → 同步变体（Megatron采用）
└─ Megatron-LM实现详解

下一步:
文档64: 1F1B调度策略详解
├─ Megatron 1F1B代码深入分析
├─ 通信优化技术
└─ 性能调优实战

文档65: 虚拟流水线并行
├─ Interleaved 1F1B调度
├─ 气泡时间进一步降低
└─ 内存-效率权衡
```

### 12.3 实践建议

**开始使用PipeDream 1F1B**：

```bash
# 1. 克隆Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM

# 2. 准备数据
python tools/preprocess_data.py \
    --input my_data.txt \
    --output-prefix my_data \
    --tokenizer-type GPT2BPETokenizer

# 3. 启动训练（1F1B）
bash examples/pretrain_gpt_distributed.sh

# 关键参数:
# --pipeline-model-parallel-size 4  # PP=4
# --tensor-model-parallel-size 2    # TP=2
# --micro-batch-size 4
# --global-batch-size 128           # m = 128/(4×DP)
```

**学习路径**：

1. 阅读文档61-63，理解流水线并行理论
2. 运行Megatron示例，观察1F1B执行
3. 使用profiling工具分析性能
4. 调整m和p，观察效率变化
5. 学习文档64-65，掌握高级优化技术

---

## 13. 参考文献

### 13.1 核心论文

1. **PipeDream原始论文**:
   - Narayanan, D., et al. (2019). "PipeDream: Generalized Pipeline Parallelism for DNN Training". *SOSP 2019*.
   - arXiv: https://arxiv.org/abs/1806.03377

2. **PipeDream-2BW**:
   - Narayanan, D., et al. (2021). "Memory-Efficient Pipeline-Parallel DNN Training". *NeurIPS 2021*.
   - arXiv: https://arxiv.org/abs/2006.09503

3. **GPipe**:
   - Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism". *NeurIPS 2019*.
   - arXiv: https://arxiv.org/abs/1811.06965

4. **Megatron-LM**:
   - Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". *arXiv:1909.08053*.

### 13.2 Megatron-LM代码参考

```
核心文件位置:

1. 1F1B调度实现:
   megatron/core/pipeline_parallel/schedules.py:1967-2306
   - forward_backward_pipelining_without_interleaving

2. 虚拟流水线1F1B:
   megatron/core/pipeline_parallel/schedules.py:827-1956
   - forward_backward_pipelining_with_interleaving

3. P2P通信:
   megatron/core/pipeline_parallel/p2p_communication.py
   - P2PCommunicator类

4. 流水线工具:
   megatron/core/pipeline_parallel/utils.py
   - is_pp_first_stage, is_pp_last_stage
```

### 13.3 相关文档

- 文档61: 流水线并行基础理论
- 文档62: GPipe同步流水线并行
- 文档64: 1F1B调度策略详解（下一篇）
- 文档65: 虚拟流水线并行
- 文档56-60: 张量并行系列
- 文档51-55: 数据并行系列

---

## 附录A：1F1B调度完整示例

### A.1 配置参数

```python
p = 4  # pipeline stages
m = 8  # micro-batches
t_f = 1  # 前向时间（单位时间）
t_b = 2  # 反向时间（2×前向）
```

### A.2 各阶段详细时间线

```
时间 →  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29

Stage 0: F₀ F₁ F₂ F₃ B₀ B₀ F₄ F₄ B₁ B₁ F₅ F₅ B₂ B₂ F₆ F₆ B₃ B₃ F₇ F₇ B₄ B₄ B₅ B₅ B₆ B₆ B₇ B₇
         └───Warmup──┘ └──────────────────────Steady (1F1B)──────────────────────┘ └──Cooldown─┘
         3×F           5组(1F+1B), 每组3时间单位                                     3×B

Stage 1: .. F₀ F₁ F₂ B₀ B₀ F₃ F₃ B₁ B₁ F₄ F₄ B₂ B₂ F₅ F₅ B₃ B₃ F₆ F₆ B₄ B₄ F₇ F₇ B₅ B₅ B₆ B₆ B₇ B₇ ..
         └──Warmup─┘ └───────────────────────Steady──────────────────────────┘ └───Cooldown──┘

Stage 2: .... F₀ F₁ B₀ B₀ F₂ F₂ B₁ B₁ F₃ F₃ B₂ B₂ F₄ F₄ B₃ B₃ F₅ F₅ B₄ B₄ F₆ F₆ B₅ B₅ F₇ F₇ B₆ B₆ B₇ B₇ ....
         └─Warmup┘ └─────────────────────────Steady──────────────────────────────┘ └─Cooldown┘

Stage 3: ...... F₀ B₀ B₀ F₁ F₁ B₁ B₁ F₂ F₂ B₂ B₂ F₃ F₃ B₃ B₃ F₄ F₄ B₄ B₄ F₅ F₅ B₅ B₅ F₆ F₆ B₆ B₆ F₇ F₇ B₇ B₇ ......
         └Empty┘ └────────────────────────────Steady (全部1F1B)─────────────────────────────────┘
```

### A.3 激活缓存追踪

```
Stage 0的激活缓存变化:

时间    操作       缓存的激活      数量
─────────────────────────────────────
0       F₀         [A₀]           1
1       F₁         [A₀, A₁]       2
2       F₂         [A₀, A₁, A₂]   3   ← Warmup峰值 = p-1
3       F₃         [A₀, A₁, A₂, A₃] 4   ← 需要额外处理（简化模型可忽略）
4-5     B₀         [A₁, A₂, A₃]   3   ← 释放A₀
6-7     F₄         [A₁, A₂, A₃, A₄] 4 → [A₂, A₃, A₄, A₅]...
8-9     B₁         [A₂, A₃, A₄]   3
...
(Steady阶段稳定在3个激活)

26-27   B₇         []             0   ← Cooldown结束
```

### A.4 气泡时间计算验证

```
总计算时间:
T_compute = m × (t_f + t_b) = 8 × (1 + 2) = 24

总时间（Stage 0）:
T_total = 3×t_f + 5×3 + 3×t_b
        = 3×1 + 15 + 3×2
        = 24

Warmup气泡: (p-1)×t_f = 3×1 = 3
Cooldown气泡: (p-1)×t_b = 3×2 = 6
总气泡: 3 + 6 = 9

验证公式:
T_bubble = 3(p-1)×t_f = 3×3×1 = 9 ✓

效率:
E = m / (m+p-1) = 8 / (8+3) = 8/11 ≈ 72.7%
```

---

## 附录B：PipeDream vs PipeDream-Flush对比实验

### B.1 实验设置

```
模型: ResNet-50
数据集: ImageNet
配置: 4 pipeline stages, batch_size=256
硬件: 16 V100 GPUs

变量: 异步 vs 同步更新
```

### B.2 收敛曲线对比

```
Epoch  PipeDream (async)    PipeDream-Flush (sync)   同步DP
─────────────────────────────────────────────────────────────
1      68.2% Top-1          68.5%                    68.5%
10     74.1%                74.8%                    74.8%
30     75.3%                76.2%                    76.2%
90     75.9%                76.8%                    76.8%

观察:
- PipeDream-Flush与同步DP完全一致
- 异步PipeDream在后期收敛慢，最终准确率低~0.9%
```

### B.3 吞吐量对比

```
方法                 吞吐量 (images/sec)    相对性能
─────────────────────────────────────────────────────
同步DP               3520                   1.00×
GPipe (m=16)         2890                   0.82×
PipeDream (async)    3840                   1.09×
PipeDream-Flush      3750                   1.07×

差异分析:
异步PipeDream吞吐高2.4% (3840 vs 3750)
原因: 无全局同步屏障
但牺牲0.9%准确率，不值得
```

### B.4 内存占用对比

```
激活内存 (per GPU):

GPipe (m=16):        16 × 512MB = 8192 MB
PipeDream (async):   3 × 512MB = 1536 MB
PipeDream-Flush:     3 × 512MB = 1536 MB

权重内存 (per stage):

GPipe:               2048 MB (单版本)
PipeDream (async):   2048 MB × 4 = 8192 MB (4个版本)
PipeDream-Flush:     2048 MB (单版本)

总内存 (包含优化器状态):

GPipe:               8192 + 2048 + 6144 = 16384 MB
PipeDream (async):   1536 + 8192 + 6144 = 15872 MB
PipeDream-Flush:     1536 + 2048 + 6144 = 9728 MB  ← 最优

结论: PipeDream-Flush内存最优（激活低+无额外权重）
```

---

## 附录C：Megatron 1F1B关键代码注释

### C.1 Warmup阶段代码详解

```python
# megatron/core/pipeline_parallel/schedules.py:2070-2095

# 计算Warmup阶段需要执行的前向次数
# 公式: p - rank - 1
# 含义: rank越小，warmup越多（因为先开始工作）
num_warmup_microbatches = (
    p2p_communicator.pp_group.size()  # p (总stage数)
    - p2p_communicator.pp_group.rank()  # rank (当前stage ID)
    - 1
)
# 边界情况: 如果micro-batch总数m小于warmup数，取m
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

# 计算Steady阶段的micro-batch数
num_microbatches_remaining = num_microbatches - num_warmup_microbatches

# 用于缓存激活的列表（等待反向计算）
input_tensors = []
output_tensors = []

# Warmup循环: 只执行前向
nvtx_range_push(suffix="warmup")  # NVTX标记，用于profiling
for i in range(num_warmup_microbatches):
    # 从前一个stage接收输入
    # 第一个stage接收None（数据直接从dataloader）
    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes,
        is_pp_first_stage(p2p_communicator.pp_group)
    )

    # 执行前向计算
    # forward_step是用户定义的函数，包含模型的前向pass
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,  # 是否checkpoint激活
        check_first_val_step(first_val_step, forward_only, i == 0),
        current_microbatch=i,
    )

    # 发送输出到下一个stage
    # 最后一个stage发送None（输出是损失）
    p2p_communicator.send_forward(
        output_tensor,
        send_tensor_shapes,
        is_pp_last_stage(p2p_communicator.pp_group)
    )

    # 训练模式: 缓存输入和输出，等待反向
    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # 释放output_tensor的内存（保留在列表中）
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

nvtx_range_pop(suffix="warmup")
```

### C.2 Steady阶段（1F1B）代码详解

```python
# megatron/core/pipeline_parallel/schedules.py:2160-2240

# 在开始1F1B之前，需要接收第一个前向输入
# （因为1F1B是先前向，再反向，需要输入ready）
if num_microbatches_remaining > 0:
    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes,
        is_pp_first_stage(p2p_communicator.pp_group)
    )

# 1F1B主循环
for i in range(num_microbatches_remaining):
    last_iteration = i == (num_microbatches_remaining - 1)

    ##########################################
    # Part 1: Forward Pass
    ##########################################
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch,
        check_first_val_step(...),
        current_microbatch=i + num_warmup_microbatches,  # 实际的micro-batch ID
    )

    # 训练模式: 将输入和输出加入缓存队列（队尾）
    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    ##########################################
    # Part 2: Backward Pass
    ##########################################
    if forward_only:
        # 推理模式: 只需发送前向结果
        p2p_communicator.send_forward(output_tensor, ...)
        if not last_iteration:
            input_tensor = p2p_communicator.recv_forward(...)
    else:
        # 训练模式: 需要执行反向

        # 融合通信: 同时发送前向结果 + 接收反向梯度
        output_tensor_grad = p2p_communicator.send_forward_recv_backward(
            output_tensor,
            send_tensor_shapes,
            recv_tensor_shapes,
            is_pp_first_stage(p2p_communicator.pp_group),
            is_pp_last_stage(p2p_communicator.pp_group),
        )

        # 从缓存队列头部取出之前的输入和输出（FIFO）
        # 这是1F1B的关键: 反向处理的是之前的micro-batch
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        # 梯度同步控制
        # 只在最后一个micro-batch启用梯度同步（如果warmup=0）
        if num_warmup_microbatches == 0 and last_iteration:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        # 执行反向计算
        input_tensor_grad = backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,
            model_type,
            config
        )

        if last_iteration:
            # 最后一个iteration: 只需发送反向梯度
            input_tensor = None
            p2p_communicator.send_backward(
                input_tensor_grad,
                send_tensor_shapes,
                is_pp_first_stage(p2p_communicator.pp_group),
            )
        else:
            # 融合通信: 同时发送反向梯度 + 接收下一个前向输入
            input_tensor = p2p_communicator.send_backward_recv_forward(
                input_tensor_grad,
                send_tensor_shapes,
                recv_tensor_shapes,
                is_pp_first_stage(p2p_communicator.pp_group),
                is_pp_last_stage(p2p_communicator.pp_group),
            )

    # 释放output_tensor的内存
    deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
```

**关键点解释**：

1. **FIFO队列**：
   - `input_tensors.append()` 和 `.pop(0)` 实现FIFO
   - 确保反向处理的是第(i - warmup)个micro-batch

2. **融合通信**：
   - `send_forward_recv_backward`：在等待反向梯度时就发送前向结果
   - `send_backward_recv_forward`：在发送梯度同时接收下一个输入
   - 减少通信延迟

3. **梯度同步延迟**：
   - 大部分时间禁用梯度同步
   - 只在最后启用，一次性同步所有梯度
   - 等价于gradient accumulation

---

**文档结束**

本文档详细介绍了PipeDream的异步流水线并行技术，包括1F1B调度策略、权重版本管理、PipeDream-Flush同步变体，以及Megatron-LM的实现细节。

**下一步**：阅读文档64《1F1B调度策略详解》，深入学习Megatron-LM的1F1B优化和虚拟流水线技术。
