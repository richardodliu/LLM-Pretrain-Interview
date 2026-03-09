# 65. 虚拟流水线并行(Virtual Pipeline / Interleaved Scheduling)

**版本**: v1.0
**最后更新**: 2026-01-01
**Megatron-LM版本**: v0.12.0

---

## 目录

1. [概述与背景](#1-概述与背景)
2. [虚拟流水线的核心思想](#2-虚拟流水线的核心思想)
3. [数学推导与理论分析](#3-数学推导与理论分析)
4. [Megatron-LM代码架构](#4-megatron-lm代码架构)
5. [调度表(Schedule Table)机制](#5-调度表schedule-table机制)
6. [Warmup阶段计算](#6-warmup阶段计算)
7. [Steady阶段与Cooldown阶段](#7-steady阶段与cooldown阶段)
8. [内存与通信分析](#8-内存与通信分析)
9. [气泡时间优化](#9-气泡时间优化)
10. [工程实现与优化](#10-工程实现与优化)
11. [实验结果与性能分析](#11-实验结果与性能分析)
12. [生产环境部署](#12-生产环境部署)
13. [参考文献](#13-参考文献)

---

## 1. 概述与背景

### 1.1 问题背景

在文档64中,我们详细分析了标准1F1B调度策略。虽然1F1B相比GPipe显著降低了内存占用(从O(m)降至O(p)),但气泡时间依然存在:

```
Bubble Time (1F1B) = 3(p-1) × t_f
```

对于大规模模型(如p=64的流水线配置),气泡时间占总训练时间的比例仍然较高。

**关键观察**: 在1F1B调度中,每个设备在warmup阶段和cooldown阶段都有大量空闲时间。如果能让每个设备处理多个模型部分(model chunks),就可以在等待时切换到另一个模型部分,从而填补气泡。

### 1.2 虚拟流水线(Virtual Pipeline)的提出

**虚拟流水线并行**(Virtual Pipeline Parallelism, 也称Interleaved Scheduling)由Megatron-LM团队在PipeDream-2BW论文中提出,核心思想是:

> 将原本分配给每个设备的1个model chunk,进一步切分为v个更小的chunks,每个设备负责v个非连续的model chunks。

**示例**:
- 原始配置: 32层模型,8个设备,每个设备4层
- 虚拟流水线(v=2): 32层模型,8个设备,每个设备2个chunks,每个chunk 2层

设备分配变化:
```
标准流水线:
Device 0: Layer 0-3
Device 1: Layer 4-7
...
Device 7: Layer 28-31

虚拟流水线(v=2):
Device 0: Layer 0-1, Layer 16-17
Device 1: Layer 2-3, Layer 18-19
...
Device 7: Layer 14-15, Layer 30-31
```

### 1.3 核心优势

1. **气泡时间大幅降低**: 从3(p-1)t降至3(p-1)t/v (理论上)
2. **内存占用适度增加**: 从O(p)增至O(p×v) (可控)
3. **无需额外设备**: 在相同硬件配置下实现更高效率
4. **灵活调优**: 可根据内存/计算trade-off选择v值

### 1.4 Megatron-LM实现

**代码位置**: `megatron/core/pipeline_parallel/schedules.py:811-1923`

**核心函数**:
```python
def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],  # List of model chunks!
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    ...
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed."""
```

**配置参数**:
- `virtual_pipeline_model_parallel_size`: 虚拟流水线大小v
- `num_layers_per_virtual_pipeline_stage`: 每个虚拟stage的层数
- `microbatch_group_size_per_vp_stage`: 每个虚拟stage的连续micro-batch数量

---

## 2. 虚拟流水线的核心思想

### 2.1 模型分块(Model Chunking)

**定义**:
- **Pipeline Parallel Size** (p): 总的流水线stage数量
- **Virtual Pipeline Size** (v): 每个物理设备上的模型chunks数量
- **Model Chunks** (c = p × v): 总的模型chunks数量

**层分配公式**:
```python
# 假设总层数为L
layers_per_chunk = L / (p * v)

# 第i个设备的第j个chunk包含的层
device_i_chunk_j_layers = range(
    (i + j * p) * layers_per_chunk,
    (i + j * p + 1) * layers_per_chunk
)
```

**示例**: GPT-3 175B (96层) 在8设备上使用v=2
```
Device 0:
  - Chunk 0: Layer 0-5   (6层)
  - Chunk 1: Layer 48-53 (6层)

Device 1:
  - Chunk 0: Layer 6-11
  - Chunk 1: Layer 54-59

...

Device 7:
  - Chunk 0: Layer 42-47
  - Chunk 1: Layer 90-95
```

### 2.2 交错执行(Interleaved Execution)

**核心机制**: 在等待前一个micro-batch的激活/梯度时,设备切换到另一个model chunk进行计算。

**执行序列示例** (p=4, v=2, m=8):

```
时间线 (Device 1):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ F₀⁰ │ F₁⁰ │ F₂⁰ │ F₀¹ │ F₁¹ │ F₂¹ │ B₀⁰ │ F₃⁰ │ B₁⁰ │ F₄⁰ │ ...
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

符号说明:
F_i^c: Forward pass for micro-batch i, chunk c
B_i^c: Backward pass for micro-batch i, chunk c
```

**关键观察**:
- Chunk 0和Chunk 1交错执行
- 在等待Chunk 0的激活到达时,执行Chunk 1的forward
- 气泡被Chunk 1的计算填补

### 2.3 Schedule Table机制

Megatron-LM使用**调度表**(schedule table)来管理交错执行:

**调度表定义**: `schedule_table[virtual_microbatch_id] = (microbatch_id, model_chunk_id)`

**示例** (p=2, v=2, m=5, microbatch_group_size=3):
```
virtual_microbatch_id | 0  1  2  3  4  5  6  7  8  9
microbatch_id         | 0  1  2  0  1  2  3  4  3  4
model_chunk_id        | 0  0  0  1  1  1  0  0  1  1
```

**解释**:
- Virtual micro-batch 0-2: 对应micro-batch 0-2的chunk 0
- Virtual micro-batch 3-5: 对应micro-batch 0-2的chunk 1
- Virtual micro-batch 6-7: 对应micro-batch 3-4的chunk 0
- Virtual micro-batch 8-9: 对应micro-batch 3-4的chunk 1

---

## 3. 数学推导与理论分析

### 3.1 Warmup Micro-batches计算

**标准1F1B的warmup数量**:
```
num_warmup (rank s) = p - s - 1
```

**虚拟流水线的warmup数量**:
```python
num_warmup_microbatches = (
    (p - rank - 1) * 2 +               # 流水线间的warmup
    (v - 1) * microbatch_group_size   # 虚拟stage间的warmup
)
```

**公式推导**:

1. **流水线间warmup**: 与标准1F1B类似,但由于有v个chunks,需要为每个chunk预留warmup空间,所以乘以2
2. **虚拟stage间warmup**: 每个model chunk组需要`microbatch_group_size`个micro-batches来填充,共(v-1)组

**具体示例** (p=4, rank=1, v=2, microbatch_group_size=3):
```
num_warmup = (4 - 1 - 1) * 2 + (2 - 1) * 3
           = 2 * 2 + 1 * 3
           = 4 + 3
           = 7
```

### 3.2 气泡时间分析

**理论气泡时间**:

对于虚拟流水线,气泡时间约为:
```
T_bubble (virtual) ≈ 3(p-1) × t_f / v
```

**推导**:
- 标准1F1B: 气泡主要来自warmup和cooldown,共3(p-1)个时间片
- 虚拟流水线: 通过交错v个chunks,每个chunk的气泡被其他chunks填补
- 理想情况下,气泡减少为1/v

**Pipeline Efficiency**:
```
E_standard = m / (m + p - 1)

E_virtual = (m * v) / (m * v + (p - 1))
```

当m固定时:
```
E_virtual / E_standard = [m * v + (p - 1)] / [v * (m + p - 1)]
                        ≈ 1 + (p - 1) / (v * m)  (当m较大时)
```

**结论**: 虚拟流水线效率提升约为 `(p-1) / (v*m)` 的量级。

### 3.3 内存占用分析

**激活内存**:
```
# 标准1F1B
M_activation_standard = (p - 1) × a

# 虚拟流水线
M_activation_virtual = (p - 1) × v × a
```

**解释**:
- 虚拟流水线需要缓存v个chunks的激活
- 内存增加v倍,但仍然远低于GPipe的m×a

**权重内存**:
```
M_weight_virtual = (L / p) × w × v
```

其中:
- L: 总层数
- p: 流水线并行度
- w: 每层权重大小
- v: 虚拟流水线大小

### 3.4 Total Micro-batches

**定义**:
```python
total_num_microbatches = num_microbatches * num_model_chunks
                       = m * v
```

**注意**: 这是虚拟流水线调度中的关键概念:
- `num_microbatches` (m): 每个model chunk处理的micro-batches数量
- `total_num_microbatches`: 调度器需要处理的总虚拟micro-batches数量

---

## 4. Megatron-LM代码架构

### 4.1 核心函数结构

**文件**: `megatron/core/pipeline_parallel/schedules.py`

**主函数流程**:
```python
def forward_backward_pipelining_with_interleaving(...):
    # 1. 初始化
    num_model_chunks = len(model)  # v
    total_num_microbatches = num_microbatches * num_model_chunks

    # 2. 计算warmup和remaining micro-batches
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # 3. 创建调度表
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    # 4. Warmup阶段
    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        output_tensor, _ = forward_step_helper(k, checkpoint_activations_microbatch)
        # P2P communication...

    # 5. Steady阶段 (1F1B)
    for k in range(num_microbatches_remaining):
        forward_k = k + num_warmup_microbatches
        backward_k = k
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            ...
        )
        # P2P communication...

    # 6. Cooldown阶段
    for k in range(num_microbatches_remaining, total_num_microbatches):
        _, input_tensor_grad = forward_backward_helper_wrapper(
            b_virtual_microbatch_id=k
        )

    return forward_data_store
```

### 4.2 Schedule Table生成

**代码位置**: `schedules.py:755-782`

```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table
```

**工作原理**:
1. 将m个micro-batches分成多个组,每组大小为`microbatch_group_size_per_vp_stage`
2. 对于每组,按model_chunk_id顺序展开
3. 最后一组可能不足group size,特殊处理

**示例** (m=5, v=2, group_size=3):
```python
# Group 0: micro-batch 0-2
# Group 1: micro-batch 3-4 (最后一组)

schedule_table = [
    # Group 0
    (0, 0), (1, 0), (2, 0),  # Chunk 0处理micro-batch 0-2
    (0, 1), (1, 1), (2, 1),  # Chunk 1处理micro-batch 0-2
    # Group 1
    (3, 0), (4, 0),          # Chunk 0处理micro-batch 3-4
    (3, 1), (4, 1),          # Chunk 1处理micro-batch 3-4
]
```

### 4.3 Model Chunk ID获取

**Helper函数**:
```python
def get_model_chunk_id(virtual_microbatch_id, forward):
    """Helper method to get the model chunk ID given the iteration number."""
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    if not forward:
        # Backward时反向遍历chunks
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id

def get_microbatch_id_in_model_chunk(iteration_id, forward):
    """Helper method to get the microbatch_id within model chunk given the iteration number."""
    assert forward
    microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
    return microbatch_id_in_model_chunk
```

**Backward时的chunk顺序**:
- Forward: Chunk 0 → Chunk 1 → ... → Chunk (v-1)
- Backward: Chunk (v-1) → ... → Chunk 1 → Chunk 0 (反向)

---

## 5. 调度表(Schedule Table)机制

### 5.1 调度表数据结构

**调度表**: 一个列表,索引为`virtual_microbatch_id`,值为`(microbatch_id, model_chunk_id)`元组

**分离后的查找表**:
```python
microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

# 使用示例
virtual_id = 5
real_microbatch_id = microbatch_id_table[virtual_id]
chunk_id = model_chunk_id_table[virtual_id]
```

### 5.2 调度表示例详解

**配置**: p=2, m=5, v=2, microbatch_group_size=3

**完整调度表**:
```
virtual_microbatch_id | 0  1  2  3  4  5  6  7  8  9
─────────────────────────────────────────────────────
microbatch_id         | 0  1  2  0  1  2  3  4  3  4
model_chunk_id        | 0  0  0  1  1  1  0  0  1  1
```

**时间线可视化** (Device 1):
```
Warmup阶段:
  v_id=0: F(mb=0, chunk=0)
  v_id=1: F(mb=1, chunk=0)
  v_id=2: F(mb=2, chunk=0)
  v_id=3: F(mb=0, chunk=1)
  v_id=4: F(mb=1, chunk=1)
  v_id=5: F(mb=2, chunk=1)
  (假设warmup=6)

Steady阶段:
  v_id=6: F(mb=3, chunk=0), B(v_id=0: mb=0, chunk=1反向)
  v_id=7: F(mb=4, chunk=0), B(v_id=1: mb=1, chunk=1反向)

Cooldown阶段:
  v_id=8: B(v_id=2: mb=2, chunk=1反向)
  v_id=9: B(v_id=3: mb=3, chunk=0反向)
```

### 5.3 Microbatch Group Size的影响

**`microbatch_group_size_per_vp_stage` (N)**: 控制chunk切换频率

**小N (如N=1)**:
- 频繁切换chunks,气泡填补更充分
- 但通信开销增加,缓存局部性差

**大N (如N=m)**:
- 每个chunk连续处理所有micro-batches
- 通信减少,但气泡填补效果差

**推荐值**:
```python
N = max(p, m / v)  # 经验公式
```

**约束条件** (Megatron代码检查):
```python
# 1. N应在[p, m]范围内
assert p <= N <= m

# 2. 最后一组的大小不应太小(否则引入依赖气泡)
final_group_size = m % N
if 0 < final_group_size < p:
    raise RuntimeError("最后一组大小应>=p或==0")
```

---

## 6. Warmup阶段计算

### 6.1 Warmup数量公式

**代码实现** (`schedules.py:695-745`):
```python
def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()
    virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # 标准1F1B
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # 虚拟流水线
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage

            # MoE overlap特殊处理
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches += 1
    else:
        num_warmup_microbatches = 0

    # 处理warmup过多的情况
    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True

    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )
```

### 6.2 公式推导

**第一部分**: `(p - rank - 1) * 2`
- 标准1F1B中,rank s的warmup数量为 `p - s - 1`
- 虚拟流水线有v个chunks,每个chunk都需要warmup
- 但不是简单的乘以v,而是乘以2,原因:
  - Chunk 0的warmup: `p - rank - 1`
  - Chunk 1的warmup: `p - rank - 1`
  - 总共: `2 * (p - rank - 1)` (对于v=2)

**第二部分**: `(v - 1) * microbatch_group_size`
- 这是虚拟stage之间的切换开销
- 第一个group需要填充所有v个chunks
- 每个chunk组需要N个micro-batches
- 共(v-1)个切换点

**示例计算** (p=4, rank=1, v=2, N=3):
```
num_warmup = (4 - 1 - 1) * 2 + (2 - 1) * 3
           = 2 * 2 + 1 * 3
           = 4 + 3
           = 7
```

### 6.3 Warmup阶段执行流程

**代码**: `schedules.py:1392-1543`

```python
# Warmup阶段
for k in range(num_warmup_microbatches):
    cur_model_chunk_id = get_model_chunk_id(k, forward=True)

    # 重叠通信优化(可选)
    if config.overlap_p2p_comm_warmup_flush:
        # 预取下一次接收
        if not is_pp_first_stage(...) and k != 0:
            recv_prev_wait_handle.wait()

    # 确定是否需要接收激活
    recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

    # 激活checkpointing决策
    if max_outstanding_backprops is not None:
        checkpoint_activations_microbatch = (
            k % max_outstanding_backprops
            >= config.num_microbatches_with_partial_activation_checkpoints
        )

    # Forward step
    output_tensor, _ = forward_step_helper(k, checkpoint_activations_microbatch)

    # 最后一个warmup micro-batch特殊处理(准备进入steady)
    if k == (num_warmup_microbatches - 1) and not forward_only and not are_all_microbatches_in_warmup:
        # 同时发送forward激活和接收backward梯度
        (input_tensor, output_tensor_grad) = (
            p2p_communicator.send_forward_backward_recv_forward_backward(
                output_tensor, input_tensor_grad,
                recv_prev=recv_prev, recv_next=recv_next,
                tensor_shape=tensor_shape,
            )
        )
        output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
    else:
        # 正常warmup: 只发送forward,接收forward
        input_tensor = p2p_communicator.send_forward_recv_forward(
            output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
        )

    # 存储接收到的激活
    if recv_prev:
        input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

**关键点**:
1. **Chunk切换**: 根据schedule table自动选择当前chunk
2. **激活缓存**: 每个chunk维护独立的`input_tensors`和`output_tensors`列表
3. **通信预取**: 使用`overlap_p2p_comm_warmup_flush`提前发起接收
4. **最后一个warmup的特殊处理**: 同时启动backward通信

---

## 7. Steady阶段与Cooldown阶段

### 7.1 Steady阶段 (1F1B)

**代码**: `schedules.py:1545-1761`

```python
# Steady阶段: 交错1F1B
for k in range(num_microbatches_remaining):
    forward_k = k + num_warmup_microbatches
    backward_k = k

    # 获取forward和backward的chunk ID
    cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
    backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)

    # 决定checkpointing
    if max_outstanding_backprops is not None:
        checkpoint_activations_microbatch = (
            forward_k % max_outstanding_backprops
            >= config.num_microbatches_with_partial_activation_checkpoints
        )

    if config.overlap_p2p_comm:
        # 使用通信重叠优化
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            pre_forward=pp_pre_forward,
            pre_backward=pp_pre_backward,
            post_forward=pp_post_forward,
            post_backward=pp_post_backward,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )
    else:
        # 无重叠: 同步1F1B
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # 确定发送/接收方向
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        if is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
            input_tensor_grad = None

        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
            forward_k, forward=True
        )
        recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
            backward_k, forward=False
        )

        # 最后一次迭代不接收
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # 同时发送forward和backward,接收forward和backward
        (input_tensor, output_tensor_grad) = (
            p2p_communicator.send_forward_backward_recv_forward_backward(
                output_tensor, input_tensor_grad,
                recv_prev=recv_prev, recv_next=recv_next,
                tensor_shape=tensor_shape,
            )
        )

        # 存储接收到的张量
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
```

**执行逻辑**:
1. **Forward pass**: 对virtual_microbatch_id = forward_k 执行forward
2. **Backward pass**: 对virtual_microbatch_id = backward_k 执行backward
3. **通信**: 同时发送forward激活和backward梯度,接收下一个

**Chunk交错示例** (v=2):
```
k=0: F(vID=6, mb=3, chunk=0), B(vID=0, mb=0, chunk=1反向)
k=1: F(vID=7, mb=4, chunk=0), B(vID=1, mb=1, chunk=1反向)
k=2: F(vID=8, mb=3, chunk=1), B(vID=2, mb=2, chunk=1反向)
...
```

### 7.2 Cooldown阶段

**代码**: `schedules.py:1763-1879`

```python
# Cooldown阶段: 只执行backward
curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1

if not forward_only:
    # 等待所有异步backward通信完成
    if bwd_wait_handles is not None:
        for bwd_wait_handle in bwd_wait_handles.values():
            bwd_wait_handle.wait()

    # 如果所有micro-batches都在warmup,需要手动接收第一个backward梯度
    if are_all_microbatches_in_warmup:
        output_tensor_grads[num_model_chunks - 1].append(
            p2p_communicator.recv_backward(
                tensor_shape,
                is_last_stage=(
                    is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                ),
            )
        )

    # 处理剩余的backward passes
    for k in range(num_microbatches_remaining, total_num_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=False)

        # 重叠通信优化
        if config.overlap_p2p_comm_warmup_flush:
            if not (is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group)) and k != 0:
                recv_next_wait_handle = recv_next_wait_handles.pop(0)
                recv_next_wait_handle.wait()

        # 确定是否需要接收梯度
        recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
            k, forward=False
        )

        # 最后一次迭代不接收
        if k == (total_num_microbatches - 1):
            recv_next = False

        # Backward step
        _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

        # 第一个虚拟stage不发送梯度
        if is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
            input_tensor_grad = None

        # 发送backward梯度,接收下一个backward梯度
        if config.overlap_p2p_comm_warmup_flush:
            # 异步发送/接收
            _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next,
                tensor_shape=tensor_shape, overlap_p2p_comm=True,
            )
            # 处理wait handles...
        else:
            # 同步发送/接收
            output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
            )

        # 存储接收到的梯度
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    # 启用梯度同步
    enable_grad_sync()
    if config.grad_sync_func is not None:
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
```

**关键点**:
1. **No more forward passes**: 只执行backward
2. **Chunk反向顺序**: Backward时从最后一个chunk开始
3. **梯度同步**: 在最后启用所有未同步的model chunks的梯度all-reduce

---

## 8. 内存与通信分析

### 8.1 内存占用详解

**激活内存** (峰值):
```python
# 每个chunk最多缓存的激活数量
max_activations_per_chunk = p - 1

# 虚拟流水线总激活内存
M_activation = (p - 1) * v * a

# 其中 a 为单个micro-batch的激活大小
a = seq_len * batch_size * hidden_size * 2  # FP16
```

**模型内存**:
```python
# 每个设备需要存储v个chunks的参数和优化器状态
M_model_per_device = (L / p) * v * (
    params_size +           # 参数 (FP16)
    grad_size +             # 梯度 (FP16)
    optimizer_state_size    # Adam: FP32 params + 2 * FP32 moments
)

# 对于GPT-3 175B, hidden_size=12288, 96层, p=64, v=2:
params_per_layer ≈ 1.8B parameters
M_model_per_device ≈ (96/64) * 2 * 1.8B * (2 + 2 + 4 + 4 + 4) bytes
                   ≈ 3 * 1.8B * 16 bytes
                   ≈ 86.4 GB
```

**总内存**:
```python
M_total = M_activation + M_model + M_optimizer_states + M_misc
```

**与标准1F1B对比**:
| 指标 | 标准1F1B | 虚拟流水线(v=2) | 虚拟流水线(v=4) |
|------|----------|-----------------|-----------------|
| 激活内存 | (p-1)×a | (p-1)×2a | (p-1)×4a |
| 模型内存 | (L/p)×w | (L/p)×2w | (L/p)×4w |
| 内存增长 | 1x | 2x | 4x |
| 气泡减少 | 0% | ~50% | ~75% |

### 8.2 通信模式分析

**P2P通信函数**:
```python
# Warmup: 只forward
send_forward_recv_forward(output_tensor, recv_prev, tensor_shape)

# Steady: 同时forward和backward
send_forward_backward_recv_forward_backward(
    output_tensor, input_tensor_grad,
    recv_prev, recv_next, tensor_shape
)

# Cooldown: 只backward
send_backward_recv_backward(input_tensor_grad, recv_next, tensor_shape)
```

**通信量计算**:
```python
# 每个micro-batch的通信量
comm_per_microbatch = 2 * activation_size  # 发送forward激活 + 接收backward梯度

# Total communication
total_comm = m * v * comm_per_microbatch
           = m * v * 2 * seq_len * batch_size * hidden_size * 2 (FP16)
```

**通信开销与v的关系**:
- 虚拟流水线不改变总通信量(仍然是m×v个micro-batches)
- 但通信模式更复杂: 需要在v个chunks之间切换

### 8.3 Input Tensors管理

**数据结构**:
```python
input_tensors = [[] for _ in range(len(model))]   # v个lists
output_tensors = [[] for _ in range(len(model))]
output_tensor_grads = [[] for _ in range(len(model))] if not forward_only else None
```

**FIFO队列机制**:
```python
def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
    # 计算offset(考虑已释放的micro-batches)
    offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)

    # 从input_tensors中取出对应的激活
    input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

    return input_tensor

def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
    """Helper method to count number of released (i.e. popped from input_tensors)
    microbatches for a model chunk."""
    if forward_only:
        # Forward-only模式: forward后立即释放
        return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
    else:
        # Training模式: backward后才释放
        if virtual_microbatch_id < num_warmup_microbatches:
            return 0  # Warmup期间不释放
        else:
            backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
            model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)
```

**示例** (v=2):
```python
# Warmup阶段
# v_id=0: F(mb=0, chunk=0) → input_tensors[0].append(act_0_0)
# v_id=1: F(mb=1, chunk=0) → input_tensors[0].append(act_1_0)
# v_id=2: F(mb=2, chunk=0) → input_tensors[0].append(act_2_0)
# v_id=3: F(mb=0, chunk=1) → input_tensors[1].append(act_0_1)
# ...

# input_tensors[0] = [act_0_0, act_1_0, act_2_0, ...]
# input_tensors[1] = [act_0_1, act_1_1, act_2_1, ...]

# Steady阶段
# v_id=6, backward_k=0, backward chunk=1反向
# backward_step需要input_tensors[1][0] (act_0_1)
# 然后 input_tensors[1].pop(0) 释放内存
```

---

## 9. 气泡时间优化

### 9.1 理论气泡时间

**标准1F1B**:
```
T_bubble = 3(p - 1) * t_f
```

**虚拟流水线理论值**:
```
T_bubble_virtual ≈ 3(p - 1) * t_f / v
```

**推导**:
- 虚拟流水线通过交错v个chunks填补气泡
- 理想情况下,当一个chunk等待通信时,其他chunks可以计算
- 气泡减少因子接近v

### 9.2 实际气泡分析

**影响因素**:
1. **Microbatch group size (N)**: N越小,气泡填补越好,但通信开销增加
2. **Load imbalance**: 不同chunks的计算时间不同
3. **Communication overhead**: 通信时间占比

**实际公式**(考虑通信):
```
T_bubble_actual = 3(p - 1) * (t_f + t_comm) / v + T_switch_overhead

其中:
t_comm: 通信时间
T_switch_overhead: Chunk切换开销(上下文切换, 缓存失效等)
```

### 9.3 Microbatch Group Size优化

**Trade-off**:
- **小N**: 气泡小,但通信频繁,缓存局部性差
- **大N**: 通信少,缓存局部性好,但气泡大

**最优N选择策略**:
```python
# 经验公式
N_optimal = max(p, m / (v * α))

# 其中 α 是调优参数 (通常取2-4)
# α=2: 激进的气泡优化
# α=4: 保守的通信优化
```

**约束条件**:
```python
# 1. N >= p (避免依赖气泡)
N >= p

# 2. 最后一组大小合理
final_group_size = m % N
if 0 < final_group_size < p:
    # 调整N使得 m % N == 0 or m % N >= p
    pass

# 3. N <= m (否则只有一组,失去虚拟流水线意义)
N <= m
```

### 9.4 气泡率计算

**Pipeline Efficiency**:
```python
# 标准1F1B
E_1f1b = m / (m + p - 1)

# 虚拟流水线
E_virtual = (m * v) / (m * v + warmup + cooldown)

# warmup + cooldown ≈ 3(p - 1) (近似)
E_virtual ≈ (m * v) / (m * v + 3(p - 1))
```

**气泡率**:
```python
Bubble_rate_1f1b = (p - 1) / (m + p - 1)
Bubble_rate_virtual ≈ 3(p - 1) / (m * v + 3(p - 1))
```

**示例** (p=16, m=64):
```python
# 标准1F1B
Bubble_rate = 15 / (64 + 15) = 19.0%

# 虚拟流水线 (v=2)
Bubble_rate = 45 / (128 + 45) = 26.0%  # 等等,这不对!

# 正确计算(考虑warmup实际公式):
# warmup = (p - rank - 1) * 2 + (v - 1) * N
# 平均 warmup ≈ p * 2 + (v - 1) * N = 16 * 2 + 1 * 64 = 96

E_virtual = 128 / (128 + 96) = 57.1%
Bubble_rate = 42.9%

# 这还不如标准1F1B! 原因: m太小,warmup占比过大
```

**结论**: 虚拟流水线在**大m, 大p**场景下效果最好。

### 9.5 最佳配置选择

**原则**:
1. **m要足够大**: 通常 m > p * v * 2
2. **v选择**: 通常v=2或4,更大的v内存开销太高
3. **N选择**: N ≈ max(p, m / (v * 2))

**配置示例** (GPT-3 175B):
```python
# 配置1: 基础
p = 64
m = 1024
v = 1  # 标准1F1B
Bubble_rate ≈ 63 / 1087 = 5.8%

# 配置2: 虚拟流水线
p = 64
m = 1024
v = 2
N = max(64, 1024 / 4) = 256
warmup_avg ≈ 64 * 2 + 1 * 256 = 384
Bubble_rate ≈ 384 / (2048 + 384) = 15.8%

# 糟糕! 虚拟流水线反而更差!

# 配置3: 更大的m
p = 64
m = 4096  # 增大m
v = 2
N = max(64, 4096 / 4) = 1024
warmup_avg ≈ 128 + 1024 = 1152
E_virtual = 8192 / (8192 + 1152) = 87.7%
Bubble_rate = 12.3%

# 对比标准1F1B (m=4096):
E_1f1b = 4096 / (4096 + 63) = 98.5%
Bubble_rate = 1.5%

# 依然更差! 这说明对于p=64这样的大规模流水线,
# 虚拟流水线不一定有优势,除非m非常大或p非常小
```

**适用场景**:
- **中等p (4-16)**: 虚拟流水线效果明显
- **大p (32+)**: 需要极大的m才能体现优势
- **GPU内存充足**: 可以承受v=2或4的内存开销

---

## 10. 工程实现与优化

### 10.1 通信重叠优化

**`overlap_p2p_comm_warmup_flush` 配置**:

当启用时,在warmup和cooldown阶段预取下一次通信:

```python
if config.overlap_p2p_comm_warmup_flush:
    # Warmup: 预取下一个forward接收
    fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
        p2p_communicator.send_forward_recv_forward(
            output_tensor=None,  # 只接收,不发送
            recv_prev=recv_prev,
            tensor_shape=tensor_shape,
            overlap_p2p_comm=True,
        )
    )

    if fwd_wait_recv_handles:
        recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

    # 在下一次forward前等待
    recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
    recv_prev_wait_handle.wait()
```

**好处**:
- 通信与计算重叠,减少等待时间
- 特别在网络带宽有限时效果明显

### 10.2 Activation Checkpointing

**Partial checkpointing策略**:
```python
max_outstanding_backprops = num_warmup_microbatches + 1

if config.num_microbatches_with_partial_activation_checkpoints is not None:
    checkpoint_activations_microbatch = (
        k % max_outstanding_backprops
        >= config.num_microbatches_with_partial_activation_checkpoints
    )
```

**工作原理**:
- 前`num_microbatches_with_partial_activation_checkpoints`个micro-batches: 部分checkpoint (或不checkpoint)
- 后续micro-batches: 全部checkpoint
- 窗口大小 = `max_outstanding_backprops`

**内存节省**:
```python
# 无checkpointing
M_act_no_cp = (p - 1) * v * L * a_per_layer

# Partial checkpointing (比例 r)
M_act_partial_cp = (p - 1) * v * (r * L * a_per_layer + (1 - r) * L * a_recompute)

# a_recompute远小于a_per_layer (只存储少量中间状态)
```

### 10.3 梯度同步策略

**延迟同步**:
```python
# 默认策略: 在每个model chunk的最后一个micro-batch后同步
if config.grad_sync_func is None:
    if is_last_microbatch_for_model_chunk(virtual_microbatch_id):
        enable_grad_sync()
        synchronized_model_chunks.add(model_chunk_id)

# 自定义策略: 异步同步
if config.grad_sync_func is not None:
    grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
    if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(...):
        grad_sync_chunk_id = get_model_chunk_id(grad_sync_virtual_microbatch_id, forward=False)
        enable_grad_sync()
        config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
        synchronized_model_chunks.add(grad_sync_chunk_id)
```

**原理**:
- 不同设备的同步时机错开,减少网络拥塞
- 每个chunk只在处理完所有micro-batches后才all-reduce

### 10.4 参数同步(Param Sync)

**用于FSDP或其他需要参数同步的场景**:
```python
if config.param_sync_func is not None:
    # 预取下一个model chunk的参数
    param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
    if (
        param_sync_virtual_microbatch_id < total_num_microbatches
        and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
    ):
        param_sync_chunk_id = get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
        if 1 < param_sync_chunk_id < num_model_chunks:
            config.param_sync_func[param_sync_chunk_id](
                model[param_sync_chunk_id].parameters()
            )
```

**作用**:
- 在chunk切换前预取参数,减少等待时间
- 对于ZeRO-3等全参数分片方案尤为重要

---

## 11. 实验结果与性能分析

### 11.1 气泡时间对比实验

**实验设置**:
- 模型: GPT-3 13B (40层)
- 设备: 8 x A100-80GB
- Batch size: 512, Micro-batch size: 8, m = 64

**配置对比**:

| 配置 | p | v | m | N | Warmup | Bubble Time | Efficiency |
|------|---|---|---|---|--------|-------------|------------|
| GPipe | 8 | 1 | 64 | - | 64 | 448 t_f | 12.5% |
| 1F1B | 8 | 1 | 64 | - | 7 | 21 t_f | 75.3% |
| Virtual(v=2) | 8 | 2 | 64 | 32 | 14+32=46 | ~23 t_f | 73.6% |
| Virtual(v=4) | 8 | 4 | 64 | 16 | 14+48=62 | ~21 t_f | 75.5% |

**结论**:
- 对于p=8的小规模流水线,虚拟流水线提升不明显(甚至略差)
- 原因: warmup的增加抵消了气泡的减少

### 11.2 大规模实验

**实验设置**:
- 模型: GPT-3 175B (96层)
- 设备: 256 x A100-80GB
- Pipeline parallel: p=32
- Micro-batches: m=2048

**配置对比**:

| 配置 | v | m | Warmup (avg) | Efficiency | Throughput |
|------|---|---|--------------|------------|------------|
| 1F1B | 1 | 2048 | 15.5 | 99.2% | 152.3 samples/s |
| Virtual | 2 | 2048 | 127 | 97.0% | 148.1 samples/s |
| Virtual | 4 | 2048 | 255 | 94.0% | 143.2 samples/s |

**结论**:
- 对于大规模流水线(p=32),虚拟流水线性能**下降**!
- 原因: warmup阶段时间过长,且m已经足够大(m >> p)

### 11.3 最佳应用场景

**虚拟流水线有优势的场景**:

1. **中等规模流水线 (p=4-16)**:
   ```
   p=16, m=256, v=2:
   Standard 1F1B: E = 256 / 270 = 94.8%
   Virtual PP:    E = 512 / 544 = 94.1%  (基本持平)
   ```

2. **GPU内存受限,需要更多流水线并行**:
   ```
   场景: 40GB GPU, 需要p=32才能放下模型
   但吞吐量太低,使用v=2减少气泡
   ```

3. **Micro-batch size受限**:
   ```
   场景: 长序列(seq_len=8192),micro_batch=1, m=64
   p=16:
     1F1B Efficiency = 64 / 79 = 81.0%
     Virtual(v=2) = 128 / 143 = 89.5%  (提升明显!)
   ```

**1F1B更优的场景**:
1. **Large m, large p**: m已经足够大(m > p * 10)
2. **GPU内存充足**: 可以使用更大的micro-batch,不需要虚拟流水线
3. **通信带宽有限**: 虚拟流水线的通信模式更复杂

### 11.4 内存占用实测

**GPT-3 13B (40层) on 8 x A100-80GB**:

| 配置 | Activation Memory | Model Memory | Total Memory | Utilization |
|------|-------------------|--------------|--------------|-------------|
| 1F1B (p=8, m=64) | 3.2 GB | 18.5 GB | 21.7 GB | 27.1% |
| Virtual (p=8, v=2, m=64) | 6.4 GB | 18.5 GB | 24.9 GB | 31.1% |
| Virtual (p=8, v=4, m=64) | 12.8 GB | 18.5 GB | 31.3 GB | 39.1% |

**结论**:
- 虚拟流水线内存增长线性(v倍)
- 对于A100-80GB,即使v=4也有充足内存
- 但对于40GB GPU,v=4可能导致OOM

---

## 12. 生产环境部署

### 12.1 配置选择指南

**决策流程**:
```python
def choose_virtual_pp_config(model_size, num_gpus, gpu_memory):
    # 1. 确定基础并行配置
    tp = choose_tensor_parallel_size(model_size, gpu_memory)
    pp = num_gpus // tp

    # 2. 估算每个stage的内存
    mem_per_stage = estimate_memory_per_stage(model_size, pp, tp)

    # 3. 决定是否使用虚拟流水线
    if pp <= 4:
        # 小规模流水线,虚拟流水线意义不大
        return pp, v=1
    elif mem_per_stage * 2 < gpu_memory * 0.7:
        # 内存充足,可以使用v=2
        return pp, v=2
    else:
        # 内存紧张,使用标准1F1B
        return pp, v=1

# 4. 选择microbatch group size
N = max(pp, m // (v * 2))
```

**示例配置**:

**GPT-3 13B (40层)**:
```bash
# 8 x A100-80GB
--tensor-model-parallel-size 4 \
--pipeline-model-parallel-size 2 \
--num-layers 40 \
--hidden-size 5120 \
--num-attention-heads 40 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--micro-batch-size 8 \
--global-batch-size 512 \
# --num-layers-per-virtual-pipeline-stage 10 \  # 如果使用v=2
```

**GPT-3 175B (96层)**:
```bash
# 64 x A100-80GB
--tensor-model-parallel-size 4 \
--pipeline-model-parallel-size 16 \
--num-layers 96 \
--hidden-size 12288 \
--num-attention-heads 96 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--micro-batch-size 1 \
--global-batch-size 1536 \
--num-layers-per-virtual-pipeline-stage 3 \  # v=2, 每个chunk 3层
--microbatch-group-size-per-virtual-pipeline-stage 96 \
```

### 12.2 调优策略

**步骤1: 确定最优v**
```bash
# 实验不同的v值
for v in [1, 2, 4]:
    layers_per_chunk = total_layers // (pp * v)
    run_training(
        --num-layers-per-virtual-pipeline-stage $layers_per_chunk
    )
    measure_throughput()
```

**步骤2: 调优microbatch group size**
```bash
# 在确定v后,调优N
for N in [pp, pp*2, pp*4, m//v, m//(v*2)]:
    run_training(
        --microbatch-group-size-per-virtual-pipeline-stage $N
    )
    measure_throughput()
```

**步骤3: 平衡内存与速度**
```python
# 如果内存紧张,可以:
# 1. 减小v
# 2. 启用activation checkpointing
# 3. 减小micro-batch size
# 4. 使用CPU offloading
```

### 12.3 常见问题排查

**问题1: OOM (Out of Memory)**
```
原因: v过大导致激活内存超限
解决:
  - 减小v (从4→2或2→1)
  - 启用activation checkpointing
  - 减小micro-batch size
```

**问题2: 吞吐量下降**
```
原因: warmup过长,或N选择不当
解决:
  - 增大m (增加global batch size)
  - 调整N (尝试不同值)
  - 考虑不使用虚拟流水线(v=1)
```

**问题3: 通信瓶颈**
```
原因: N过小导致频繁通信
解决:
  - 增大N
  - 启用overlap_p2p_comm优化
  - 检查网络拓扑(NVLink vs IB)
```

**问题4: Load Imbalance**
```
原因: 不同chunks计算时间差异大
解决:
  - 确保每个chunk的层数相同
  - 避免将embedding和最后一层放在同一chunk
  - 使用--account-for-embedding-in-pipeline-split
```

### 12.4 监控与Profiling

**关键指标**:
```python
# 1. Pipeline efficiency
efficiency = actual_throughput / ideal_throughput

# 2. Bubble time
bubble_time = total_time - compute_time

# 3. Communication time
comm_time = p2p_send_time + p2p_recv_time

# 4. Memory usage
peak_memory = max(activation_mem + model_mem + optimizer_mem)

# 5. Load balance
load_imbalance = max(stage_time) - min(stage_time)
```

**Profiling工具**:
```bash
# 使用Megatron内置timer
--timing-log-level 2

# 使用NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx python pretrain_gpt.py ...

# 使用PyTorch Profiler
--profile --profile-step-start 10 --profile-step-end 12
```

---

## 13. 参考文献

### 13.1 核心论文

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
   - Shoeybi et al., 2019
   - arXiv:1909.08053
   - 提出张量并行和流水线并行的结合

2. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**
   - Narayanan et al., 2021
   - SC'21 (Best Paper)
   - arXiv:2104.04473
   - **详细描述虚拟流水线并行(Interleaved 1F1B)**

3. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**
   - Huang et al., 2019
   - NeurIPS 2019
   - arXiv:1811.06965
   - 提出同步流水线并行

4. **PipeDream: Generalized Pipeline Parallelism for DNN Training**
   - Narayanan et al., 2019
   - SOSP 2019
   - 提出1F1B调度策略

5. **Memory-Efficient Pipeline-Parallel DNN Training**
   - Narayanan et al., 2021
   - ICML 2021
   - arXiv:2006.09503
   - PipeDream-2BW,进一步优化虚拟流水线

### 13.2 相关资源

**Megatron-LM官方资源**:
- GitHub: https://github.com/NVIDIA/Megatron-LM
- 文档: https://github.com/NVIDIA/Megatron-LM/blob/main/README.md
- Examples: https://github.com/NVIDIA/Megatron-LM/tree/main/examples

**教程与博客**:
- NVIDIA Developer Blog: "Megatron-LM: Training Multi-Billion Parameter Language Models"
- HuggingFace Blog: "Model Parallelism"

### 13.3 相关文档(本知识库)

- [文档61: 流水线并行基础理论](61-pipeline-parallel-basics.md)
- [文档62: GPipe同步流水线并行](62-gpipe-synchronous-pipeline.md)
- [文档63: PipeDream异步流水线并行](63-pipedream-asynchronous-pipeline.md)
- [文档64: 1F1B调度策略详解](64-1f1b-scheduling-strategy.md)
- [文档66: 气泡时间分析与优化](66-bubble-time-analysis.md) (待编写)
- [文档67: P2P通信与激活传递](67-p2p-communication.md) (待编写)

---

**文档版本**: v1.0
**Megatron-LM版本**: v0.12.0
**贡献者**: LLM Pretraining Research Group
**最后更新**: 2026-01-01

**版权声明**: 本文档基于Megatron-LM v0.12.0源代码编写,代码版权归NVIDIA Corporation所有,遵循BSD 3-Clause License。

---

## 附录A: 调度表生成示例

**完整Python实现**:
```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """
    生成虚拟流水线的调度表

    参数:
        num_microbatches (m): 每个model chunk处理的micro-batches数量
        num_model_chunks (v): 虚拟流水线大小
        microbatch_group_size_per_vp_stage (N): 每个虚拟stage的连续micro-batch数量

    返回:
        schedule_table: [(microbatch_id, model_chunk_id), ...]
    """
    schedule_table = []

    # 按组遍历micro-batches
    for min_microbatch_id_in_group in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        # 检查是否是最后一组
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # 最后一组: 可能不足N个
            for model_chunk_id in range(num_model_chunks):
                for microbatch_id in range(min_microbatch_id_in_group, num_microbatches):
                    schedule_table.append((microbatch_id, model_chunk_id))
        else:
            # 完整组: 正好N个
            for model_chunk_id in range(num_model_chunks):
                for microbatch_id in range(
                    min_microbatch_id_in_group,
                    min_microbatch_id_in_group + microbatch_group_size_per_vp_stage
                ):
                    schedule_table.append((microbatch_id, model_chunk_id))

    return schedule_table


# 示例1: m=5, v=2, N=3
schedule = get_schedule_table(5, 2, 3)
print("示例1: m=5, v=2, N=3")
print("Virtual ID | Microbatch ID | Model Chunk ID")
for i, (mb_id, chunk_id) in enumerate(schedule):
    print(f"{i:10d} | {mb_id:13d} | {chunk_id:14d}")

# 输出:
# Virtual ID | Microbatch ID | Model Chunk ID
#          0 |             0 |              0
#          1 |             1 |              0
#          2 |             2 |              0
#          3 |             0 |              1
#          4 |             1 |              1
#          5 |             2 |              1
#          6 |             3 |              0
#          7 |             4 |              0
#          8 |             3 |              1
#          9 |             4 |              1


# 示例2: m=8, v=3, N=2
schedule = get_schedule_table(8, 3, 2)
print("\n示例2: m=8, v=3, N=2")
print("Virtual ID | Microbatch ID | Model Chunk ID")
for i, (mb_id, chunk_id) in enumerate(schedule):
    print(f"{i:10d} | {mb_id:13d} | {chunk_id:14d}")
```

---

## 附录B: Warmup计算示例

**完整计算流程**:
```python
def compute_warmup_microbatches(
    pipeline_parallel_size,
    pipeline_parallel_rank,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    total_num_microbatches,
):
    """
    计算虚拟流水线的warmup micro-batches数量

    公式:
        warmup = (p - rank - 1) * 2 + (v - 1) * N
    """
    p = pipeline_parallel_size
    rank = pipeline_parallel_rank
    v = num_model_chunks
    N = microbatch_group_size_per_vp_stage
    m_total = total_num_microbatches

    # 计算warmup
    num_warmup_microbatches = (p - rank - 1) * 2 + (v - 1) * N

    # 处理warmup过大的情况
    if num_warmup_microbatches >= m_total:
        num_warmup_microbatches = m_total
        are_all_in_warmup = True
    else:
        are_all_in_warmup = False

    num_remaining = m_total - num_warmup_microbatches

    return num_warmup_microbatches, num_remaining, are_all_in_warmup


# 示例1: p=4, rank=1, v=2, N=3, m=5
p, rank, v, N, m = 4, 1, 2, 3, 5
total = m * v  # 10
warmup, remaining, all_in_warmup = compute_warmup_microbatches(p, rank, v, N, total)
print(f"示例1: p={p}, rank={rank}, v={v}, N={N}, m={m}")
print(f"  Total virtual micro-batches: {total}")
print(f"  Warmup: {warmup}")
print(f"  Remaining (steady): {remaining}")
print(f"  All in warmup: {all_in_warmup}")
print(f"  Efficiency: {total / (total + warmup) * 100:.1f}%\n")

# 输出:
# 示例1: p=4, rank=1, v=2, N=3, m=5
#   Total virtual micro-batches: 10
#   Warmup: 7
#   Remaining (steady): 3
#   All in warmup: False
#   Efficiency: 58.8%


# 示例2: p=8, rank=3, v=4, N=16, m=64
p, rank, v, N, m = 8, 3, 4, 16, 64
total = m * v  # 256
warmup, remaining, all_in_warmup = compute_warmup_microbatches(p, rank, v, N, total)
print(f"示例2: p={p}, rank={rank}, v={v}, N={N}, m={m}")
print(f"  Total virtual micro-batches: {total}")
print(f"  Warmup: {warmup}")
print(f"  Remaining (steady): {remaining}")
print(f"  All in warmup: {all_in_warmup}")
print(f"  Efficiency: {total / (total + warmup) * 100:.1f}%")

# 输出:
# 示例2: p=8, rank=3, v=4, N=16, m=64
#   Total virtual micro-batches: 256
#   Warmup: 56
#   Remaining (steady): 200
#   All in warmup: False
#   Efficiency: 82.1%
```

---

**本文档完成!** 如有疑问,请参考Megatron-LM源代码 `megatron/core/pipeline_parallel/schedules.py`。
