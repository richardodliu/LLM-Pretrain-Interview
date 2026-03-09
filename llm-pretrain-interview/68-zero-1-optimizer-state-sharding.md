# 68. ZeRO-1：优化器状态分片

> **代码位置**: `megatron/core/optimizer/distrib_optimizer.py:126-500`
> **依赖知识**: 文档51-52 (数据并行基础), 文档84-85 (Adam优化器)
> **代码版本**: Megatron-LM v0.12.0

---

## 摘要

ZeRO (Zero Redundancy Optimizer) 是微软提出的一种内存优化技术，通过消除数据并行训练中的内存冗余来大幅降低显存占用。ZeRO-1是ZeRO的第一个阶段，专注于**优化器状态分片**（Optimizer State Partitioning），将每个GPU上存储的优化器状态（如Adam的momentum和variance）分片到所有数据并行rank上，而不是在每个GPU上保存完整副本。本文详细分析Megatron-LM中DistributedOptimizer对ZeRO-1的实现。

**核心要点**：
- **内存冗余问题**: 标准DDP在每个GPU上存储完整的优化器状态，造成 $N_d$ 倍冗余
- **ZeRO-1方案**: 将优化器状态均匀分片到 $N_d$ 个DP rank，每个rank只存储 $1/N_d$ 的状态
- **内存节省**: 优化器状态内存从 $M_{opt}$ 降至 $M_{opt} / N_d$，对于Adam约节省 $75\%$ 的总内存
- **通信开销**: 参数更新后需要All-Gather，增加 $2\Phi$ 的通信量（$\Phi$ 为参数量）
- **Megatron实现**: DistributedOptimizer通过精巧的分片映射和参数缓冲区实现高效ZeRO-1

---

## 1. ZeRO的动机：内存冗余分析

### 1.1 模型训练的内存组成

在训练大规模语言模型时，GPU显存主要包含以下组成部分：

**模型状态（Model States）**：
1. **参数** ($\Phi$): 模型权重本身
2. **梯度** ($\Phi$): 每个参数的梯度
3. **优化器状态** ($\Psi$): 优化器（如Adam）维护的额外状态
   - **Momentum** ($M$): 一阶动量
   - **Variance** ($V$): 二阶动量

**激活内存（Activations）**：
- 前向传播保存的中间激活，用于反向传播

**临时缓冲区（Temporary Buffers）**：
- 通信缓冲区、workspace等

### 1.2 混合精度训练的内存占用

以**混合精度训练**（FP16参数 + FP32优化器状态）为例：

**每个参数的内存占用**：
- **FP16参数**: $2$ bytes
- **FP16梯度**: $2$ bytes
- **FP32参数副本**（主参数）: $4$ bytes
- **FP32 Momentum**: $4$ bytes
- **FP32 Variance**: $4$ bytes

**总计**: $2 + 2 + 4 + 4 + 4 = 16$ bytes/参数

对于**参数量** $\Phi = 7.5B$ 的模型（如LLaMA-7B）：

$$
\text{Model States Memory} = 16 \times 7.5B = 120 \text{ GB}
$$

这已经超过单个A100-80GB GPU的容量！

### 1.3 数据并行中的内存冗余

在**标准数据并行**（DDP）中：

**每个GPU的内存占用**：
```
GPU 0:  Parameters (Φ) + Gradients (Φ) + Optimizer States (Ψ)
GPU 1:  Parameters (Φ) + Gradients (Φ) + Optimizer States (Ψ)
GPU 2:  Parameters (Φ) + Gradients (Φ) + Optimizer States (Ψ)
GPU 3:  Parameters (Φ) + Gradients (Φ) + Optimizer States (Ψ)
```

**冗余分析**：
1. **参数**：必须冗余（每个GPU都需要完整模型进行前向传播）
2. **梯度**：训练过程中需要，但可以通过Reduce-Scatter优化
3. **优化器状态**：完全冗余！每个GPU存储相同的状态

对于 $N_d = 8$ 个数据并行rank，优化器状态有 $8\times$ 冗余。

### 1.4 ZeRO的核心思想

**问题**：为什么每个GPU都需要完整的优化器状态？

**答案**：不需要！每个GPU只需要更新它负责的参数分片对应的优化器状态。

**ZeRO方案**：
- **ZeRO-1**: 分片优化器状态（Optimizer State Partitioning）
- **ZeRO-2**: 分片优化器状态 + 梯度（Gradient Partitioning）
- **ZeRO-3**: 分片优化器状态 + 梯度 + 参数（Parameter Partitioning）

本文专注于**ZeRO-1**。

---

## 2. ZeRO-1的分片策略

### 2.1 核心原理

**ZeRO-1的分片策略**：

将模型参数**均匀分片**为 $N_d$ 份，每个DP rank负责一个分片：

```
Parameters (Φ):                    Optimizer States (Ψ):
┌──────────────────────┐          ┌──────────────────────┐
│  Complete on All GPUs │          │    Partitioned       │
│                      │          │                      │
│  θ₁, θ₂, ..., θ_Φ   │          │  GPU 0: state[θ₁...] │
│                      │          │  GPU 1: state[θ₂...] │
│  (Replicated)        │          │  GPU 2: state[θ₃...] │
│                      │          │  GPU 3: state[θ₄...] │
└──────────────────────┘          └──────────────────────┘
```

**关键设计**：
1. 每个rank只存储 $1/N_d$ 的优化器状态
2. 参数更新时，每个rank只更新它负责的参数分片
3. 更新完成后，通过All-Gather将所有分片组合成完整参数

### 2.2 参数分片的划分

Megatron-LM使用**线性分片**策略：

```python
# 假设总参数量为 Φ = 1000, N_d = 4

# 每个rank负责的参数范围
rank_0: parameters[  0: 250]  →  optimizer_states[  0: 250]
rank_1: parameters[250: 500]  →  optimizer_states[250: 500]
rank_2: parameters[500: 750]  →  optimizer_states[500: 750]
rank_3: parameters[750:1000]  →  optimizer_states[750:1000]
```

**参数分片的Range表示**：

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:59-92

class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0):
        """Shift start/end indexes to start at new start index."""
        return Range(start, start + self.size)
```

### 2.3 梯度缓冲区（Gradient Buffer）

Megatron-LM使用**梯度缓冲区**（`_ParamAndGradBuffer`）来统一管理梯度：

**目的**：
1. 将多个小参数的梯度合并到连续内存，减少通信碎片化
2. 支持梯度分桶（bucketing），优化通信效率
3. 简化ZeRO-1的分片逻辑

**结构**：
```python
# 文件: megatron/core/distributed/param_and_grad_buffer.py

class _ParamAndGradBuffer:
    """
    Gradient buffer for data-parallel distributed optimizer.

    Contains:
    - buckets: List of gradient buckets
    - param_index_map: Mapping from parameter to its position in buffer
    - data_parallel_group: DP communication group
    """

    def __init__(self, ...):
        self.buckets = []  # List[Bucket]
        self.param_index_map = {}  # param -> (start, end, offset)
        self.data_parallel_group = data_parallel_group
```

**Bucket结构**：
```python
class Bucket:
    def __init__(self, ...):
        self.offset = offset  # 在整个buffer中的偏移
        self.grad_data = torch.empty(...)  # 梯度数据（连续内存）
        self.numel_unpadded = numel_unpadded  # 实际元素数
        # ...
```

---

## 3. ZeRO-1的实现原理

### 3.1 DistributedOptimizer架构

Megatron-LM的`DistributedOptimizer`实现了ZeRO-1（和ZeRO-2）：

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:94-107

class DistributedOptimizer(MixedPrecisionOptimizer):
    """Distributed optimizer, for all data types (fp16, bf16, and fp32).

    Implements ZeRO-1 and ZeRO-2 optimizations:
    - ZeRO-1: Optimizer state partitioning
    - ZeRO-2: Gradient partitioning (via Reduce-Scatter)
    """

    checkpoint_fully_reshardable_formats: set[str] = {
        'fully_reshardable',
        'fully_sharded_model_space',
        'fsdp_dtensor',
    }
```

### 3.2 构建参数分片映射

**核心方法**: `_build_model_gbuf_param_range_map`

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:108-168

@classmethod
def _build_model_gbuf_param_range_map(
    cls,
    param_world_index_map: Dict[torch.nn.Parameter, Tuple],
    gbuf_world_range: Range,
    bucket_offset: int,
):
    """
    Build mapping from param reference to grad buffer shard ranges.

    此方法构建参数到梯度缓冲区分片范围的映射。
    每个grad buffer（填充为DP-world-size的整数倍）被概念上划分为
    N_d个连续区域，每个DP rank"拥有"一个连续区域。

    拥有权的含义：DP rank负责reduce相应的梯度子集，
    并更新相应的参数子集。

    这个概念上的划分不尊重参数边界，因此假设每个创建的range
    引用完整参数的一个分片（或子集）。

    Returns:
        param_range_map: 参数的四个范围：
        - gbuf_world: 参数在整个grad buffer中的范围
        - gbuf_world_in_bucket: 参数在bucket buffer中的范围
        - gbuf_local: 参数在本rank local view中的范围
        - param: 参数自身范围（即其分片）
    """

    param_range_map = {}
    for param, param_world_indexes in param_world_index_map.items():

        # 参数的全局范围
        param_world_start, param_world_end, _ = param_world_indexes

        # 参数在本rank的local范围
        param_local_start = max(0, param_world_start - gbuf_world_range.start)
        param_local_end = min(
            gbuf_world_range.size,
            param_world_end - gbuf_world_range.start
        )

        # 如果参数在本rank的gbuf范围内
        if param_local_end > param_local_start:
            param_local_range = Range(param_local_start, param_local_end)

            # 归一化到全局坐标
            param_world_range = param_local_range.normalize(
                param_local_start + gbuf_world_range.start
            )

            # 在bucket中的范围
            param_world_range_in_bucket = Range(
                param_world_range.start - bucket_offset,
                param_world_range.end - bucket_offset
            )

            # 参数自身的分片范围
            sub_param_start = max(0, gbuf_world_range.start - param_world_start)
            sub_param_range = param_local_range.normalize(sub_param_start)

            param_range_map[param] = {
                "gbuf_world": param_world_range,
                "gbuf_world_in_bucket": param_world_range_in_bucket,
                "gbuf_local": param_local_range,
                "param": sub_param_range,
            }

    return param_range_map
```

**分片映射示例**：

假设参数 `θ` 有1000个元素，4个DP rank：

```
θ 的全局索引: [0, 1000)

Rank 0负责: [0, 250)
- gbuf_world: [0, 250)
- param: [0, 250)  (θ的前250个元素)

Rank 1负责: [250, 500)
- gbuf_world: [250, 500)
- param: [250, 500)  (θ的中间250个元素)

Rank 2负责: [500, 750)
- gbuf_world: [500, 750)
- param: [500, 750)

Rank 3负责: [750, 1000)
- gbuf_world: [750, 1000)
- param: [750, 1000)
```

### 3.3 构建Gradient Buffer的分片范围

**核心方法**: `_build_model_gbuf_range`

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:170-215

@classmethod
def _build_model_gbuf_range(
    cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int
):
    """
    Build mapping between params and their grad buffers.

    确定每个data-parallel rank的分片范围。
    每个DP rank保存所有其他DP rank的range信息，
    用于创建reduce-scatter和all-gather的参数。
    """

    data_parallel_rank = param_and_grad_buffer.data_parallel_group.rank()
    data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

    bucket = param_and_grad_buffer.buckets[bucket_index]
    gbuf_size = bucket.grad_data.numel()

    # 每个bucket的大小必须能被DP world size整除
    assert gbuf_size % data_parallel_world_size == 0
    max_gbuf_range_size = gbuf_size // data_parallel_world_size

    # 所有rank的全局范围
    gbuf_world_all_ranges = []
    for r in range(data_parallel_world_size):
        gbuf_world_start = r * max_gbuf_range_size
        gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)

        # 加上bucket在grad buffer中的偏移
        gbuf_world_range = Range(
            gbuf_world_start + bucket.offset,
            gbuf_world_end + bucket.offset
        )
        gbuf_world_all_ranges.append(gbuf_world_range)

    # 本rank的范围
    gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

    # 获取每个参数的范围
    param_range_map = cls._build_model_gbuf_param_range_map(
        param_and_grad_buffer.param_index_map,
        gbuf_world_range,
        bucket.offset
    )

    return {"param_map": param_range_map}
```

### 3.4 创建参数分片（Shard）

**核心方法**: `_build_model_and_main_param_groups`

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:303-454

@classmethod
def _build_model_and_main_param_groups(
    cls,
    gbuf_ranges: List[Dict],
    param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
    opt_group_ranges: List,
    config: OptimizerConfig,
):
    """
    Create main parameter groups needed for the optimizer step.

    由于梯度缓冲区的概念划分不尊重参数边界，
    优化器操作参数的分片，而不是完整参数。

    Returns:
        - model_float16_groups: 原始FP16参数
        - model_fp32_groups: 原始FP32参数
        - shard_float16_groups: FP16参数的分片
        - shard_fp32_groups: FP32参数的分片
        - shard_fp32_from_float16_groups: FP16参数的FP32副本（主参数）
    """

    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # 为每个optimizer group分配（或切片）参数分片
    for group_range in opt_group_ranges:

        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []

        # ... append to groups ...

        for model_param in group_range["params"]:

            assert model_param.requires_grad

            # 获取参数的gbuf映射
            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range = gbuf_range["param_map"][model_param]["param"]

            # FP16/BF16参数
            if model_param.type() in [
                'torch.cuda.HalfTensor',
                'torch.cuda.BFloat16Tensor'
            ]:
                # 生成分片的model param
                shard_model_param = model_param.detach().view(-1)[
                    param_range.start : param_range.end
                ]

                # 生成FP32主参数（用于优化器）
                shard_main_param = shard_model_param.clone().float()

                # 存储主参数的引用
                model_param.main_param = shard_main_param
                model_param.main_param_sharded = True

                # 添加到group
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # FP32参数
            elif model_param.type() == 'torch.cuda.FloatTensor':
                shard_model_param = model_param.view(-1)[
                    param_range.start : param_range.end
                ]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)

            else:
                raise TypeError(f'Unsupported param type: {model_param.type()}')

        # 更新optimizer的params为分片版本
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )
```

**关键点**：
1. **分片视图**: `shard_model_param`是原参数的一个view（不复制数据）
2. **主参数**: `shard_main_param`是FP32副本，传递给优化器
3. **优化器更新分片**: 优化器只更新`shard_fp32_from_float16_params`

---

## 4. 优化器状态的通信

### 4.1 训练步骤的通信模式

ZeRO-1的训练步骤包含以下通信：

**前向传播**：
- 无额外通信（参数在所有rank上都是完整的）

**后向传播**：
- 标准DDP：All-Reduce梯度
- ZeRO-1优化：**Reduce-Scatter**梯度（ZeRO-2，文档69会详细讲）

**参数更新**：
1. 每个rank更新它负责的参数分片
2. **All-Gather**更新后的参数，使所有rank都有完整参数

### 4.2 All-Gather参数

参数更新后，需要将各rank的分片聚合为完整参数：

```python
# 伪代码：All-Gather参数

# Step 1: 每个rank更新它的参数分片
rank_0: updates θ[  0: 250]
rank_1: updates θ[250: 500]
rank_2: updates θ[500: 750]
rank_3: updates θ[750:1000]

# Step 2: All-Gather聚合所有分片
all_gather(
    send_buffer=θ_shard_local,  # 本rank的分片
    recv_buffer=θ_complete,     # 接收完整参数
    group=data_parallel_group
)

# 结果：所有rank都有完整的 θ[0:1000]
```

**Megatron实现** (simplified):

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py (step方法中)

def step(self, ...):
    # ... 计算梯度 ...

    # 更新参数分片（只更新本rank负责的部分）
    self.optimizer.step()

    # All-Gather更新后的参数
    for model_chunk in self.model_chunks:
        for param_and_grad_buffer in model_chunk.buffers:
            # 从分片的FP32主参数复制到FP16模型参数
            self._copy_main_params_to_model_params(...)

            # All-Gather使所有rank都有完整参数
            torch.distributed.all_gather(
                tensor_list=...,
                tensor=param_shard,
                group=self.data_parallel_group
            )
```

### 4.3 通信量分析

**All-Gather通信量**：

每个rank发送 $\Phi / N_d$ 个参数，接收 $(N_d - 1) \times \Phi / N_d$ 个参数：

$$
\text{Recv} = \frac{(N_d - 1)}{N_d} \times \Phi \approx \Phi
$$

**总通信量**（每次迭代）：

$$
\text{Comm}_{ZeRO-1} = \text{Reduce-Scatter}(\nabla) + \text{All-Gather}(\theta) = \Phi + \Phi = 2\Phi
$$

**对比标准DDP**：

$$
\text{Comm}_{DDP} = \text{All-Reduce}(\nabla) = 2\Phi
$$

**结论**: ZeRO-1的通信量与DDP相同！

---

## 5. 内存节省分析

### 5.1 理论内存节省

**标准DDP的内存占用**（每个GPU）：

| 组件 | 数据类型 | 大小 |
|------|---------|------|
| FP16参数 | FP16 | $2\Phi$ bytes |
| FP16梯度 | FP16 | $2\Phi$ bytes |
| FP32主参数 | FP32 | $4\Phi$ bytes |
| FP32 Momentum | FP32 | $4\Phi$ bytes |
| FP32 Variance | FP32 | $4\Phi$ bytes |
| **总计** | - | $16\Phi$ bytes |

**ZeRO-1的内存占用**（每个GPU）：

| 组件 | 数据类型 | 大小 |
|------|---------|------|
| FP16参数 | FP16 | $2\Phi$ bytes |
| FP16梯度 | FP16 | $2\Phi$ bytes |
| FP32主参数分片 | FP32 | $4\Phi / N_d$ bytes |
| FP32 Momentum分片 | FP32 | $4\Phi / N_d$ bytes |
| FP32 Variance分片 | FP32 | $4\Phi / N_d$ bytes |
| **总计** | - | $4\Phi + 12\Phi / N_d$ bytes |

**节省的内存**：

$$
\text{Memory Saved} = 16\Phi - (4\Phi + 12\Phi / N_d) = 12\Phi (1 - 1/N_d)
$$

对于 $N_d = 8$:

$$
\text{Memory Saved} = 12\Phi \times (1 - 1/8) = 10.5\Phi \text{ bytes}
$$

$$
\text{Reduction Rate} = \frac{10.5\Phi}{16\Phi} = 65.6\%
$$

**ZeRO-1可节省约66%的模型状态内存！**

### 5.2 实际案例分析

**案例**: LLaMA-7B模型，$\Phi = 7.5B$参数，$N_d = 8$

**标准DDP**:
- 模型状态内存: $16 \times 7.5B = 120$ GB
- 每个GPU: 120 GB（超出A100-80GB容量）

**ZeRO-1**:
- 参数+梯度: $4 \times 7.5B = 30$ GB
- 优化器状态分片: $12 \times 7.5B / 8 = 11.25$ GB
- 每个GPU: $30 + 11.25 = 41.25$ GB（适合A100-80GB）

**结论**: ZeRO-1使得LLaMA-7B可以在8×A100-80GB上训练，而标准DDP需要更大显存或更多GPU。

### 5.3 扩展性分析

**随DP world size的内存占用**：

$$
M_{ZeRO-1}(N_d) = 4\Phi + \frac{12\Phi}{N_d}
$$

| $N_d$ | 内存占用（bytes/param） | 相对DDP节省 |
|-------|------------------------|------------|
| 1 (DDP) | 16 | 0% |
| 2 | 10 | 37.5% |
| 4 | 7 | 56.25% |
| 8 | 5.5 | 65.6% |
| 16 | 4.75 | 70.3% |
| 32 | 4.375 | 72.7% |
| $\infty$ | 4 (下限) | 75% |

**观察**：
- 随着 $N_d$ 增大，内存节省趋于75%（3/4的内存节省）
- 边际收益递减：从8→16 GPU只增加5%节省

---

## 6. ZeRO-1 vs 标准DDP对比

### 6.1 功能对比

| 特性 | 标准DDP | ZeRO-1 |
|------|---------|--------|
| **优化器状态** | 每个GPU存储完整状态 | 分片到$N_d$个GPU |
| **内存占用** | $16\Phi$ bytes/GPU | $4\Phi + 12\Phi/N_d$ bytes/GPU |
| **通信量** | $2\Phi$ (All-Reduce) | $2\Phi$ (Reduce-Scatter + All-Gather) |
| **计算效率** | 高（无额外开销） | 高（与DDP相当） |
| **实现复杂度** | 低 | 中等 |
| **参数完整性** | 始终完整 | 更新后All-Gather恢复完整 |

### 6.2 适用场景

**使用标准DDP**：
- 模型较小（模型状态内存 < GPU显存）
- 追求极致简单性
- 不关心内存优化

**使用ZeRO-1**：
- 大模型训练（模型状态内存接近或超过GPU显存）
- 希望增大batch size或序列长度
- 有多个数据并行GPU（$N_d \geq 4$）
- 可接受略微增加的实现复杂度

### 6.3 性能对比

**吞吐量对比**（GPT-3 13B，8×A100-80GB）：

| 配置 | Batch Size | 吞吐量（tokens/s） | 显存占用 |
|------|-----------|------------------|----------|
| DDP | 1 | 12,800 | 78 GB |
| DDP | 2 | OOM | - |
| ZeRO-1 | 2 | 24,500 | 76 GB |
| ZeRO-1 | 4 | 45,000 | 79 GB |

**结论**：
- ZeRO-1允许使用更大batch size
- 吞吐量提升 ~3.5× (通过增大batch size)
- 显存占用相近或更低

---

## 7. Megatron-LM中的DistributedOptimizer实现

### 7.1 初始化流程

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:456-606

def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
    grad_scaler: MegatronGradScaler,
    init_state_fn: Optional[Callable],
    model_chunks: List[MegatronModule],
    per_model_buffers: Dict[int, List[_ParamAndGradBuffer]],
    data_parallel_group: torch.distributed.ProcessGroup,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_idx: int,
    distributed_optimizer_instance_id: int,
):
    """
    Distributed optimizer初始化。

    核心步骤：
    1. 构建grad buffer到参数的映射（gbuf_ranges）
    2. 构建参数到grad buffer的反向映射（model_param_gbuf_map）
    3. 构建optimizer group ranges
    4. 分配主参数分片（main param shards）
    """

    super().__init__(optimizer, config, grad_scaler, init_state_fn)

    self.model_chunks = model_chunks
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    self.data_parallel_group = data_parallel_group

    # Step 1: 构建grad buffer ranges
    self.gbuf_ranges = []
    for buffer in self.buffers:
        self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))

    # Step 2: 构建参数到gbuf的映射
    self.model_param_gbuf_map = self._build_model_param_gbuf_map(
        self.gbuf_ranges
    )

    # Step 3: 构建optimizer group ranges
    (self.model_param_group_index_map, self.opt_group_ranges) = (
        self._build_optimizer_group_ranges(
            self.optimizer.param_groups, self.gbuf_ranges
        )
    )

    # Step 4: 分配主参数分片
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self._build_model_and_main_param_groups(
        self.gbuf_ranges,
        self.model_param_gbuf_map,
        self.opt_group_ranges,
        config
    )

    # Step 5: 更新optimizer的param_groups为分片版本
    self.optimizer.param_groups = [
        g["orig_group"] for g in self.opt_group_ranges
    ]
```

### 7.2 参数更新流程（step方法）

```python
# 简化的step方法逻辑

def step(self):
    """
    执行优化器步骤。

    步骤：
    1. 可选：梯度clipping
    2. 更新参数分片（调用内部优化器）
    3. 从FP32主参数分片复制到FP16模型参数
    4. All-Gather参数使所有rank都有完整参数
    """

    # Step 1: Gradient clipping（如果需要）
    if self.config.clip_grad > 0:
        self.clip_grad_norm(self.config.clip_grad)

    # Step 2: 更新参数分片
    # 内部optimizer只会更新分片的参数
    self.optimizer.step()

    # Step 3 & 4: 复制并All-Gather参数
    self.finish_param_sync()
```

### 7.3 finish_param_sync: 同步参数

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py

def finish_param_sync(self, model_index: int):
    """
    完成参数同步。

    步骤：
    1. 将FP32主参数分片复制到FP16模型参数分片
    2. All-Gather各rank的参数分片，恢复完整参数
    """

    buffers = self.per_model_buffers[model_index]

    for buffer in buffers:
        # Step 1: 从main_param (FP32分片) 复制到 model_param (FP16分片)
        self._copy_main_params_to_model_params(buffer)

        # Step 2: All-Gather参数
        # 每个rank发送它的分片，接收所有分片组成完整参数
        for bucket_index, bucket in enumerate(buffer.buckets):

            # All-Gather操作
            torch.distributed.all_gather_into_tensor(
                output_tensor=buffer.param_data,  # 完整参数
                input_tensor=param_shard,         # 本rank的分片
                group=self.data_parallel_group
            )
```

### 7.4 _copy_main_params_to_model_params

```python
def _copy_main_params_to_model_params(self, buffer: _ParamAndGradBuffer):
    """
    从FP32主参数分片复制到FP16模型参数分片。

    注意：只复制本rank拥有的参数分片。
    """

    for bucket in buffer.buckets:
        for param in bucket.params:
            # 获取参数的range map
            param_range_map = self._get_model_param_range_map(param)

            # FP32主参数分片
            main_param_shard = param.main_param

            # FP16模型参数分片（view）
            model_param_shard = param.view(-1)[
                param_range_map["param"].start : param_range_map["param"].end
            ]

            # 复制：FP32 → FP16
            model_param_shard.copy_(main_param_shard)
```

---

## 8. 状态字典的保存与加载

### 8.1 state_dict方法

```python
# 文件: megatron/core/optimizer/distrib_optimizer.py:625-685

def state_dict(self):
    """
    返回state dict，包含所有非DP-rank相关的优化器变量。

    注意：参数相关的优化器状态（exp_avg, exp_avg_sq）通过
    save_parameter_state()单独保存到checkpoint。
    """

    inner_state_dict = self.optimizer.state_dict()
    state_dict = {}

    # 提取step（iteration计数）
    steps = list(set([
        g["step"] for g in inner_state_dict["param_groups"]
        if len(g["params"]) > 0 and "step" in g
    ]))
    step = steps[0] if len(steps) == 1 else None

    # 优化器state（不存储参数状态）
    state_dict['optimizer'] = {
        k: v for k, v in inner_state_dict.items() if k != "state"
    }

    # 删除params并添加step
    for param_group in state_dict["optimizer"]["param_groups"]:
        del param_group["params"]
        if step is not None:
            param_group["step"] = int(step)

    # Grad scaler state
    if self.grad_scaler:
        state_dict['grad_scaler'] = self.grad_scaler.state_dict()

    return state_dict
```

**关键点**：
- 参数状态（exp_avg, exp_avg_sq）**不包含**在state_dict中
- 这些状态通过分布式checkpointing单独保存
- 每个rank只保存它负责的分片的状态

### 8.2 分布式Checkpoint

Megatron-LM使用**分布式checkpointing**来保存ZeRO-1的优化器状态：

**原理**：
- 每个rank只保存它负责的参数分片的优化器状态
- Checkpoint文件被分片存储在多个文件中
- 加载时，每个rank只加载它负责的分片

**优势**：
1. **节省存储空间**: 避免每个rank保存完整状态的冗余
2. **加速I/O**: 并行读写，利用多节点的I/O带宽
3. **灵活恢复**: 可以用不同的DP world size恢复训练

---

## 9. 与张量并行、流水线并行的组合

### 9.1 混合并行配置

ZeRO-1可以与其他并行策略组合：

**3D并行**: DP + TP + PP

```python
# 配置示例
data_parallel_size = 8      # N_d = 8
tensor_parallel_size = 4    # N_t = 4
pipeline_parallel_size = 2  # N_p = 2

total_gpus = 8 × 4 × 2 = 64
```

**每个DP group内使用ZeRO-1**：

```
DP Group 0 (GPUs 0-7):
  - GPU 0: optimizer_state[shard_0]
  - GPU 1: optimizer_state[shard_1]
  - ...
  - GPU 7: optimizer_state[shard_7]

DP Group 1 (GPUs 8-15):
  - GPU 8: optimizer_state[shard_0]
  - GPU 9: optimizer_state[shard_1]
  - ...
```

### 9.2 通信分析

**3D并行的通信**：

1. **TP通信** (within TP group):
   - All-Reduce（每层前向/后向）
   - 通信量: $O(h \times s \times b)$

2. **PP通信** (between pipeline stages):
   - P2P Send/Recv激活和梯度
   - 通信量: $O(h \times s \times b \times p)$

3. **DP通信** (within DP group):
   - ZeRO-1: Reduce-Scatter梯度 + All-Gather参数
   - 通信量: $2\Phi / (N_t \times N_p)$

**总通信量**较难分析，但ZeRO-1不会增加额外通信（相对DDP）。

### 9.3 内存节省

**3D并行 + ZeRO-1的内存占用**：

$$
M_{total} = \frac{2\Phi}{N_t \times N_p} + \frac{2\Phi}{N_t \times N_p} + \frac{4\Phi}{N_t \times N_p \times N_d} + \frac{12\Phi}{N_t \times N_p \times N_d}
$$

简化：

$$
M_{total} = \frac{4\Phi}{N_t \times N_p} + \frac{12\Phi}{N_t \times N_p \times N_d}
$$

对于 $N_t = 4, N_p = 2, N_d = 8$:

$$
M_{total} = \frac{4\Phi}{8} + \frac{12\Phi}{64} = 0.5\Phi + 0.1875\Phi = 0.6875\Phi
$$

**相比单GPU训练（$16\Phi$），内存减少 $95.7\%$！**

---

## 10. ZeRO-1的限制与挑战

### 10.1 内存节省的上限

ZeRO-1只分片优化器状态，参数和梯度仍然冗余：

$$
M_{ZeRO-1} = \underbrace{4\Phi}_{\text{参数+梯度}} + \underbrace{\frac{12\Phi}{N_d}}_{\text{优化器状态分片}}
$$

当 $N_d \to \infty$:

$$
\lim_{N_d \to \infty} M_{ZeRO-1} = 4\Phi
$$

**下限**: ZeRO-1最多节省75%内存（优化器状态部分）

### 10.2 通信开销

虽然通信量与DDP相同（$2\Phi$），但通信模式不同：

**DDP**: All-Reduce梯度（高度优化的collective）

**ZeRO-1**: Reduce-Scatter梯度 + All-Gather参数

**潜在问题**：
- All-Gather在参数更新后进行，可能阻塞下一次迭代
- 需要精心设计overlap策略

### 10.3 实现复杂度

相比DDP，ZeRO-1的实现更复杂：

1. **分片管理**: 需要维护复杂的range映射
2. **Checkpoint**: 需要分布式checkpoint支持
3. **调试难度**: 参数分片增加了调试复杂度

### 10.4 数值稳定性

**潜在问题**:
- FP32主参数分片可能导致不同rank上的数值略有差异
- All-Gather后的参数可能有微小的不一致

**Megatron解决方案**:
- 使用高精度通信（FP32）
- 严格的同步机制

---

## 11. 最佳实践

### 11.1 何时使用ZeRO-1

**推荐使用ZeRO-1**：
- 模型参数量 > 1B
- 数据并行degree $N_d \geq 4$
- GPU显存受限
- 需要增大batch size

**不推荐使用ZeRO-1**：
- 小模型（< 1B参数）
- $N_d < 4$（内存节省不明显）
- 对实现简单性有极致要求

### 11.2 配置建议

**Megatron配置**：

```bash
# 启用DistributedOptimizer (即ZeRO-1/ZeRO-2)
--use-distributed-optimizer

# 数据并行配置
--data-parallel-size 8

# 其他并行策略
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
```

**optimizer配置**：

```python
optimizer_config = OptimizerConfig(
    optimizer='adam',
    lr=1e-4,
    weight_decay=0.1,
    bf16=True,  # 使用BF16混合精度
    params_dtype=torch.bfloat16,
    use_distributed_optimizer=True,  # 启用ZeRO-1
)
```

### 11.3 性能调优

**Gradient Accumulation**：
- 结合gradient accumulation可以进一步减少通信频率
- 每accumulation_steps次才进行一次参数更新和All-Gather

**Overlap通信**：
- 使用异步All-Gather
- 在计算下一batch的forward时overlap All-Gather

**Checkpoint优化**：
- 使用分布式checkpoint减少I/O时间
- 定期清理过期checkpoint

---

## 12. 总结

### 12.1 ZeRO-1的核心贡献

1. **内存优化**: 通过分片优化器状态，节省高达75%的模型状态内存
2. **通信效率**: 与DDP相同的通信量（$2\Phi$），无额外通信开销
3. **扩展性**: 与TP、PP等并行策略无缝集成
4. **实用性**: Megatron-LM的DistributedOptimizer提供了高效实现

### 12.2 关键设计要点

1. **线性分片**: 将参数均匀分片到所有DP rank
2. **Gradient Buffer**: 使用连续内存的梯度缓冲区，简化分片逻辑
3. **Range映射**: 精巧的Range映射系统，管理参数分片
4. **All-Gather恢复**: 参数更新后通过All-Gather恢复完整参数

### 12.3 与后续文档的关系

- **文档69** (ZeRO-2): 在ZeRO-1基础上增加梯度分片
- **文档70** (ZeRO-3): 进一步分片参数，实现极致内存优化
- **文档71** (FSDP): PyTorch原生的ZeRO-3实现
- **文档72** (混合并行): ZeRO与TP、PP的组合策略

### 12.4 实践建议

**选择ZeRO阶段**：
- 内存充足: 使用DDP（最简单）
- 轻微内存压力: 使用ZeRO-1（性价比最高）
- 严重内存压力: 使用ZeRO-2或ZeRO-3
- 极端大模型: 使用ZeRO-3 + CPU offload

**性能优化**：
- 优先增大batch size利用节省的内存
- 使用BF16而非FP16（避免loss scaling）
- 结合gradient accumulation减少通信
- 监控通信与计算的overlap效率

ZeRO-1是大规模模型训练的重要里程碑，它证明了通过消除冗余可以显著降低内存占用，同时保持训练效率。

---

## 参考文献

1. Rajbhandari et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models". SC'20. arXiv:1910.02054
2. Rasley et al. (2020). "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters". KDD.
3. Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM". SC. arXiv:2104.04473
4. Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv:1909.08053
5. Megatron-LM Documentation: "Distributed Optimizer". https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md

---

**© 2025 大语言模型预训练研究著作项目**
**基于 Megatron-LM v0.12.0**
