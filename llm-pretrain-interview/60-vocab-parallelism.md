# 60. 词汇表并行(Vocab Parallelism)

> **版本**: 1.0
> **作者**: Megatron-LM 研究团队
> **日期**: 2025-12-31
> **Megatron版本**: v0.12.0
> **代码位置**: `megatron/core/tensor_parallel/layers.py:188-316`, `megatron/core/tensor_parallel/cross_entropy.py`

---

## 📋 目录

1. [概述](#1-概述)
2. [核心概念](#2-核心概念)
3. [数学原理](#3-数学原理)
4. [Embedding层的张量并行](#4-embedding层的张量并行)
5. [输出层(LM Head)的张量并行](#5-输出层lm-head的张量并行)
6. [词汇表并行的交叉熵损失](#6-词汇表并行的交叉熵损失)
7. [Megatron-LM代码实现](#7-megatron-lm代码实现)
8. [性能分析与优化](#8-性能分析与优化)
9. [最佳实践与调试技巧](#9-最佳实践与调试技巧)
10. [面试常见问题](#10-面试常见问题)
11. [总结](#11-总结)
12. [参考文献](#12-参考文献)
13. [附录](#13-附录)

---

## 1. 概述

### 1.1 什么是词汇表并行

**词汇表并行(Vocab Parallelism)** 是张量并行的一种特殊形式，专门用于处理模型中的两个关键层：
- **Embedding层**: 将token ID映射为稠密向量
- **输出层(LM Head)**: 将隐藏状态映射回词汇表，生成预测logits

在大规模语言模型中，词汇表大小通常在10K到250K之间：
- GPT-3: 50,257
- LLaMA: 32,000
- Qwen: 151,643
- GPT-4: 约100,000 (估计)

当词汇表较大时，Embedding和输出层的参数会占用大量内存：
```
参数量 = 词汇表大小 × 隐藏维度 × 2
```

例如，对于Qwen-72B (vocab_size=151,643, hidden_size=8,192):
```
词汇表相关参数 = 151,643 × 8,192 × 2 × 4 bytes (FP32)
                = 9.9 GB
```

**词汇表并行**通过将词汇表切分到多个GPU上，显著降低单GPU的内存占用。

---

### 1.2 为什么需要词汇表并行

#### 问题背景

在Transformer模型中，Embedding层和输出层的参数量随词汇表大小线性增长：

| 模型 | 词汇表大小 | 隐藏维度 | 词汇表参数量 (FP16) |
|------|------------|----------|----------------------|
| GPT-2 | 50,257 | 1,024 | 195 MB |
| LLaMA-7B | 32,000 | 4,096 | 500 MB |
| LLaMA-70B | 32,000 | 8,192 | 1.0 GB |
| Qwen-14B | 151,643 | 5,120 | 2.96 GB |
| Qwen-72B | 151,643 | 8,192 | 4.74 GB |

对于超大词汇表（如多语言模型），单个Embedding层就可能占用数GB显存。

#### 核心优势

1. **内存降低**: 参数量按GPU数线性降低
   ```
   单GPU参数量 = 原始参数量 / p
   ```

2. **激活降低**: 输出logits内存降低
   ```
   原始: [s, b, V]  →  并行: [s, b, V/p]
   ```

3. **通信高效**: 仅需要少量AllReduce通信
   - Embedding前向: 1次AllReduce
   - 输出层前向: 无需通信
   - 损失计算: 2-3次AllReduce

4. **扩展性强**: 对超大词汇表(>100K)尤其有效

---

### 1.3 与其他并行策略的关系

词汇表并行是**张量并行的子集**，但具有独特性质：

#### 对比其他张量并行策略

| 策略 | 切分对象 | 切分维度 | 通信位置 |
|------|----------|----------|----------|
| **列并行(FC1)** | 权重矩阵输出维度 | $W \in \mathbb{R}^{m \times n} \to W_i \in \mathbb{R}^{m \times n/p}$ | 无(前向) |
| **行并行(FC2)** | 权重矩阵输入维度 | $W \in \mathbb{R}^{m \times n} \to W_i \in \mathbb{R}^{m/p \times n}$ | AllReduce(前向) |
| **词汇表并行** | Embedding/输出层 | $W \in \mathbb{R}^{V \times d} \to W_i \in \mathbb{R}^{V/p \times d}$ | AllReduce(Embedding前向) |

#### 组合使用

在实际训练中，词汇表并行通常与其他张量并行策略结合：

```python
# 典型的张量并行配置
tp_size = 8

# Embedding层: 词汇表并行
vocab_parallel_embedding = VocabParallelEmbedding(
    num_embeddings=151643,  # Qwen词汇表
    embedding_dim=8192,
    ...
)  # 每GPU: 151643/8 ≈ 18,955 embeddings

# Transformer层: MHA + MLP张量并行
# (如文档56-59所述)

# 输出层: 词汇表并行
output_layer = ColumnParallelLinear(
    input_size=8192,
    output_size=151643,  # 词汇表大小
    gather_output=False,  # 不收集，保持切分状态
    ...
)  # 每GPU: 151643/8 ≈ 18,955 logits
```

---

### 1.4 本文档结构

本文档将全面介绍词汇表并行的实现原理与代码细节：

1. **第2章**: 核心概念（词汇表切分、VocabUtility等）
2. **第3章**: 数学原理（前向/反向传播推导）
3. **第4章**: Embedding层并行实现
4. **第5章**: 输出层(LM Head)并行实现
5. **第6章**: 并行交叉熵损失计算（**核心难点**）
6. **第7章**: Megatron-LM完整代码解析
7. **第8章**: 性能分析与通信优化
8. **第9章**: 最佳实践与调试技巧
9. **第10章**: 5个深度面试问题

---

## 2. 核心概念

### 2.1 词汇表切分策略

#### 2.1.1 均匀切分原则

Megatron-LM采用**沿词汇表维度均匀切分**的策略：

```
全局词汇表: [0, V)
GPU 0: [0, V/p)
GPU 1: [V/p, 2V/p)
...
GPU p-1: [(p-1)V/p, V)
```

**示例**: 词汇表大小V=32,000，4个GPU

| GPU编号 | 词汇范围 | 大小 |
|---------|----------|------|
| GPU 0 | [0, 8000) | 8,000 |
| GPU 1 | [8000, 16000) | 8,000 |
| GPU 2 | [16000, 24000) | 8,000 |
| GPU 3 | [24000, 32000) | 8,000 |

#### 2.1.2 VocabUtility工具类

**代码位置**: `megatron/core/tensor_parallel/utils.py:97-121`

Megatron-LM提供`VocabUtility`类来计算词汇表分片范围：

```python
class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last)
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        """根据每个分片的词汇表大小计算范围"""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """根据全局词汇表大小计算范围"""
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
```

**使用示例**:
```python
vocab_size = 32000
rank = 1  # GPU 1
world_size = 4

start, end = VocabUtility.vocab_range_from_global_vocab_size(
    vocab_size, rank, world_size
)
# 返回: (8000, 16000)
```

---

### 2.2 Embedding层的并行化

#### 2.2.1 标准Embedding层

PyTorch标准的Embedding层实现：

```python
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=32000,  # 词汇表大小
    embedding_dim=4096      # 隐藏维度
)
# 权重形状: [32000, 4096]
# 参数量: 32000 × 4096 × 2 bytes (FP16) = 250 MB

input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [b, s] = [2, 3]
output = embedding(input_ids)  # [2, 3, 4096]
```

#### 2.2.2 词汇表并行Embedding

将词汇表切分到多个GPU：

```python
# GPU 0: 词汇范围 [0, 8000)
embedding_0 = nn.Embedding(
    num_embeddings=8000,   # 1/4 词汇表
    embedding_dim=4096
)
# 权重形状: [8000, 4096]
# 参数量: 62.5 MB  ← 减少到原来的1/4

# GPU 1: 词汇范围 [8000, 16000)
embedding_1 = nn.Embedding(num_embeddings=8000, ...)

# ... 类似地GPU 2和GPU 3
```

**关键问题**: 如何处理输入token ID不在本GPU词汇范围内的情况？

**解决方案**: 掩码(Masking) + AllReduce

---

### 2.3 输出层的并行化

#### 2.3.1 标准输出层(LM Head)

语言模型的输出层是一个线性变换，将隐藏状态映射回词汇表：

```python
lm_head = nn.Linear(
    in_features=4096,      # 隐藏维度
    out_features=32000,    # 词汇表大小
    bias=False
)
# 权重形状: [32000, 4096]  (PyTorch存储为转置形式)
# 参数量: 32000 × 4096 × 2 bytes (FP16) = 250 MB

hidden_states = torch.randn(2, 3, 4096)  # [b, s, d]
logits = lm_head(hidden_states)  # [2, 3, 32000]
```

#### 2.3.2 词汇表并行输出层

输出层可以看作是**列并行线性层**（输出维度被切分）：

```python
# GPU 0: 输出logits的范围 [0, 8000)
lm_head_0 = ColumnParallelLinear(
    input_size=4096,
    output_size=32000,
    gather_output=False,  # 🔑 不收集输出，保持切分
    ...
)
# 每GPU输出: [b, s, 8000]

hidden_states = torch.randn(2, 3, 4096)  # [b, s, d] - 完整的
logits_partial = lm_head_0(hidden_states)  # [2, 3, 8000] - 部分的
```

**关键区别**:
- 普通列并行(FC1): `gather_output=False`，后续还有激活函数和行并行层
- 输出层并行: `gather_output=False`，输出保持切分状态，直接送入并行交叉熵

---

### 2.4 并行交叉熵损失计算

#### 2.4.1 标准交叉熵损失

对于语言模型，交叉熵损失的标准计算流程：

```python
import torch.nn.functional as F

logits = model(input_ids)  # [b, s, V] - V=32000
# 重塑为 [b*s, V]
logits_2d = logits.view(-1, 32000)
targets_1d = targets.view(-1)  # [b*s]

# 计算交叉熵
loss = F.cross_entropy(logits_2d, targets_1d)
```

**交叉熵公式**:
$$
\mathcal{L}(\mathbf{z}, y) = -\log \frac{e^{z_y}}{\sum_{j=1}^{V} e^{z_j}} = \log \left( \sum_{j=1}^{V} e^{z_j} \right) - z_y
$$

其中：
- $\mathbf{z} \in \mathbb{R}^V$: logits向量
- $y$: 目标类别
- $z_y$: 目标类别的logit值

#### 2.4.2 词汇表并行的挑战

当logits被切分到多个GPU时：

```python
# GPU 0: logits_0 ∈ ℝ^[b,s,V/p]  对应词汇 [0, V/p)
# GPU 1: logits_1 ∈ ℝ^[b,s,V/p]  对应词汇 [V/p, 2V/p)
# ...
```

**问题1**: 如何计算全局的 $\sum_{j=1}^{V} e^{z_j}$？
- 每个GPU只有部分logits，需要**AllReduce**聚合

**问题2**: 如何获取目标类别的logit $z_y$？
- 目标token可能在任意GPU上，需要**掩码+AllReduce**

**问题3**: 如何计算梯度？
- Softmax梯度依赖全局归一化项

这些问题的详细解决方案见**第6章**。

---

## 3. 数学原理

### 3.1 Embedding层的前向传播

#### 3.1.1 标准Embedding前向

给定输入token IDs $\mathbf{x} \in \mathbb{N}^{s \times b}$，Embedding层查表得到：

$$
\mathbf{E}(\mathbf{x}) = \left[ \mathbf{W}_E[x_{1,1}], \mathbf{W}_E[x_{1,2}], \ldots, \mathbf{W}_E[x_{s,b}] \right] \in \mathbb{R}^{s \times b \times d}
$$

其中：
- $\mathbf{W}_E \in \mathbb{R}^{V \times d}$: Embedding权重矩阵
- $\mathbf{W}_E[i] \in \mathbb{R}^d$: 第 $i$ 个token的embedding向量
- $s$: 序列长度
- $b$: batch大小
- $d$: 隐藏维度
- $V$: 词汇表大小

#### 3.1.2 词汇表并行Embedding前向

将Embedding权重矩阵沿词汇维度切分到 $p$ 个GPU：

$$
\mathbf{W}_E \in \mathbb{R}^{V \times d} = \begin{bmatrix}
\mathbf{W}_{E,0} \\
\mathbf{W}_{E,1} \\
\vdots \\
\mathbf{W}_{E,p-1}
\end{bmatrix}, \quad \mathbf{W}_{E,i} \in \mathbb{R}^{(V/p) \times d}
$$

**GPU $i$ 的词汇范围**: $[\text{vocab\_start}_i, \text{vocab\_end}_i)$

其中：
$$
\begin{aligned}
\text{vocab\_start}_i &= i \cdot (V / p) \\
\text{vocab\_end}_i &= (i + 1) \cdot (V / p)
\end{aligned}
$$

**前向计算步骤**:

1. **创建掩码**: GPU $i$ 标记不在其词汇范围内的token
   $$
   \text{mask}_i[j, k] = \begin{cases}
   1, & \text{if } x_{j,k} < \text{vocab\_start}_i \text{ or } x_{j,k} \geq \text{vocab\_end}_i \\
   0, & \text{otherwise}
   \end{cases}
   $$

2. **偏移输入**: 将输入token ID调整到本地索引
   $$
   x'_{j,k} = \begin{cases}
   x_{j,k} - \text{vocab\_start}_i, & \text{if mask}_i[j, k] = 0 \\
   0, & \text{otherwise}
   \end{cases}
   $$

3. **本地查表**: 使用本地Embedding矩阵
   $$
   \mathbf{H}_i = \mathbf{W}_{E,i}[x'] \in \mathbb{R}^{s \times b \times d}
   $$

4. **应用掩码**: 将不属于本GPU的token的embedding置零
   $$
   \mathbf{H}_i[j, k, :] = \begin{cases}
   \mathbf{H}_i[j, k, :], & \text{if mask}_i[j, k] = 0 \\
   \mathbf{0}, & \text{otherwise}
   \end{cases}
   $$

5. **AllReduce聚合**: 所有GPU的结果求和，得到完整的embedding
   $$
   \mathbf{E}(\mathbf{x}) = \sum_{i=0}^{p-1} \mathbf{H}_i \in \mathbb{R}^{s \times b \times d}
   $$

**定理3.1** (Embedding并行正确性):
$$
\sum_{i=0}^{p-1} \mathbf{H}_i = \mathbf{E}(\mathbf{x})
$$

**证明**:
对于任意token位置 $(j, k)$，其token ID为 $x_{j,k}$。

设 $x_{j,k}$ 属于GPU $i^*$ 的词汇范围，即：
$$
\text{vocab\_start}_{i^*} \leq x_{j,k} < \text{vocab\_end}_{i^*}
$$

则：
- 对于GPU $i^* $: $\mathbf{H}_{i^*}[j, k, :] = \mathbf{W}_{E,i^*}[x_{j,k} - \text{vocab\_start}_{i^*}] = \mathbf{W}_E[x_{j,k}]$
- 对于其他GPU $i \neq i^*$: $\mathbf{H}_i[j, k, :] = \mathbf{0}$ (被掩码)

因此：
$$
\left( \sum_{i=0}^{p-1} \mathbf{H}_i \right)[j, k, :] = \mathbf{H}_{i^*}[j, k, :] = \mathbf{W}_E[x_{j,k}] = \mathbf{E}(\mathbf{x})[j, k, :]
$$

对所有位置 $(j, k)$ 成立，故原命题成立。 □

---

### 3.2 Embedding层的反向传播

#### 3.2.1 反向传播输入

前向传播的最后一步是AllReduce：
$$
\mathbf{E}(\mathbf{x}) = \sum_{i=0}^{p-1} \mathbf{H}_i
$$

反向传播时，梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{E}(\mathbf{x})}$ 通过AllReduce的反向（恒等复制）分发到所有GPU：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{E}(\mathbf{x})} \quad \forall i
$$

#### 3.2.2 掩码的反向

掩码操作在前向时将某些位置置零：
$$
\mathbf{H}_i[j, k, :] = \begin{cases}
\mathbf{W}_{E,i}[x'_{j,k}], & \text{if mask}_i[j, k] = 0 \\
\mathbf{0}, & \text{otherwise}
\end{cases}
$$

反向时，梯度在被掩码的位置也被置零：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{E,i}[x'_{j,k}]} = \begin{cases}
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_i[j, k, :]}, & \text{if mask}_i[j, k] = 0 \\
\mathbf{0}, & \text{otherwise}
\end{cases}
$$

#### 3.2.3 Embedding查表的反向

Embedding查表操作的反向传播是**梯度累加**：

对于权重 $\mathbf{W}_{E,i}[j]$，其梯度是所有查找该token的位置的梯度之和：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{E,i}[j]} = \sum_{\substack{(m,n): \\ x'_{m,n} = j \\ \text{mask}_i[m,n]=0}} \frac{\partial \mathcal{L}}{\partial \mathbf{H}_i[m, n, :]}
$$

**无需额外通信**: 每个GPU的Embedding权重梯度是独立的，因为每个token只属于一个GPU。

---

### 3.3 输出层(LM Head)的前向传播

#### 3.3.1 标准输出层前向

给定Transformer的最后一层输出 $\mathbf{H} \in \mathbb{R}^{s \times b \times d}$，输出层计算logits：

$$
\mathbf{Z} = \mathbf{H} \mathbf{W}_O^T \in \mathbb{R}^{s \times b \times V}
$$

其中：
- $\mathbf{W}_O \in \mathbb{R}^{V \times d}$: 输出层权重矩阵
- $\mathbf{Z}[i, j, k]$: 位置 $(i, j)$ 对token $k$ 的logit

#### 3.3.2 词汇表并行输出层前向

将输出层权重沿词汇维度（输出维度）切分：

$$
\mathbf{W}_O = \begin{bmatrix}
\mathbf{W}_{O,0} \\
\mathbf{W}_{O,1} \\
\vdots \\
\mathbf{W}_{O,p-1}
\end{bmatrix}, \quad \mathbf{W}_{O,i} \in \mathbb{R}^{(V/p) \times d}
$$

GPU $i$ 计算部分logits：
$$
\mathbf{Z}_i = \mathbf{H} \mathbf{W}_{O,i}^T \in \mathbb{R}^{s \times b \times (V/p)}
$$

**关键**: 输入 $\mathbf{H}$ 在所有GPU上是**完整的**（来自AllReduce），因此这是标准的列并行操作。

**无需通信**: 输出保持切分状态，直接送入并行交叉熵损失。

---

### 3.4 输出层的反向传播

#### 3.4.1 梯度来源

输出层的梯度来自交叉熵损失的反向传播（详见第6章）：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \in \mathbb{R}^{s \times b \times (V/p)}
$$

#### 3.4.2 权重梯度

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{O,i}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \right)^T \mathbf{H} \in \mathbb{R}^{(V/p) \times d}
$$

**无需通信**: 每个GPU独立计算自己的权重梯度。

#### 3.4.3 输入梯度

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \mathbf{W}_{O,i} \in \mathbb{R}^{s \times b \times d}
$$

**AllReduce通信**: 需要聚合所有GPU的输入梯度：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}} = \sum_{i=0}^{p-1} \frac{\partial \mathcal{L}}{\partial \mathbf{H}_i}
$$

**定理3.2** (输出层反向传播正确性):
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}} = \sum_{i=0}^{p-1} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \mathbf{W}_{O,i}
$$

**证明**:
标准的链式法则：
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{H}}
&= \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \frac{\partial \mathbf{Z}}{\partial \mathbf{H}} \\
&= \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \mathbf{W}_O \\
&= \frac{\partial \mathcal{L}}{\partial \begin{bmatrix} \mathbf{Z}_0 & \mathbf{Z}_1 & \cdots & \mathbf{Z}_{p-1} \end{bmatrix}} \begin{bmatrix} \mathbf{W}_{O,0} \\ \mathbf{W}_{O,1} \\ \vdots \\ \mathbf{W}_{O,p-1} \end{bmatrix} \\
&= \sum_{i=0}^{p-1} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \mathbf{W}_{O,i}
\end{aligned}
$$

□

---

## 4. Embedding层的张量并行

### 4.1 设计思想

Embedding层的张量并行遵循**词汇表切分+掩码+AllReduce**的设计模式：

```
输入token IDs (完整) → [查表+掩码] (本地) → AllReduce → Embedding输出 (完整)
```

#### 设计原理

1. **词汇表切分**: 每个GPU只存储 $V/p$ 个token的embedding
2. **掩码机制**: 每个GPU将不属于自己词汇范围的token的embedding置零
3. **AllReduce聚合**: 所有GPU的部分结果相加，得到完整的embedding

#### 为什么这样设计有效？

关键观察：对于任意token，它**只属于一个GPU的词汇范围**。

- GPU $i$ 持有token $x$ 的embedding → 输出非零向量
- 其他GPU持有token $x$ 的掩码 → 输出零向量

AllReduce后，每个token位置恰好有一个非零贡献。

---

### 4.2 VocabParallelEmbedding类

**代码位置**: `megatron/core/tensor_parallel/layers.py:188-316`

#### 4.2.1 类定义与初始化

```python
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(VocabParallelEmbedding, self).__init__()

        # 保存配置
        self.num_embeddings = num_embeddings             # 全局词汇表大小 V
        self.embedding_dim = embedding_dim                # 隐藏维度 d
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tp_group = tp_group

        # 获取张量并行组
        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)

        # 🔑 计算本GPU的词汇范围
        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_pg_rank(self.tp_group),
                get_pg_size(self.tp_group)
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )  # V/p

        self.deterministic_mode = config.deterministic_mode

        # ✅ 分配权重并初始化
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,              # 全局词汇表大小
                    self.embedding_dim,
                    self.num_embeddings_per_partition,  # 本GPU的词汇数
                    0,                                  # partition_dim=0 (沿词汇维度切分)
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=get_pg_rank(self.tp_group),
                    world_size=get_pg_size(self.tp_group),
                )
        else:
            # GPU初始化
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,  # 沿词汇维度切分
                    stride=1
                )
```

**关键点**:
1. `vocab_start_index` 和 `vocab_end_index` 定义了本GPU的词汇范围 $[start, end)$
2. 权重形状: `[num_embeddings_per_partition, embedding_dim]` = $[V/p, d]$
3. `partition_dim=0` 表示沿第0维（词汇维度）切分

---

#### 4.2.2 前向传播实现

```python
def forward(self, input_):
    """Forward.

    Args:
        input_ (torch.Tensor): Input token IDs, shape [batch, seq_len]
    """

    # 🔑 步骤1: 创建掩码（仅多GPU时需要）
    if self.tp_group.size() > 1:
        # 标记不在本GPU词汇范围内的token
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)

        # 步骤2: 调整输入索引到本地范围
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0  # 不在范围内的token设为0索引
    else:
        # 单GPU时无需掩码
        masked_input = input_

    # 🔑 步骤3: 本地Embedding查表
    if self.deterministic_mode:
        # 确定性模式: 直接索引（可复现）
        output_parallel = self.weight[masked_input]
    else:
        # 非确定性模式: 使用PyTorch的F.embedding（更快）
        output_parallel = F.embedding(masked_input, self.weight)

    # 🔑 步骤4: 应用掩码，将不属于本GPU的token的embedding置零
    if self.tp_group.size() > 1:
        output_parallel[input_mask, :] = 0.0

    # 🔑 步骤5: AllReduce或ReduceScatter聚合
    if self.reduce_scatter_embeddings:
        # 使用ReduceScatter（配合序列并行）
        # 数据格式转换: [b, s, h] → [s, b, h]
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )
    else:
        # 标准AllReduce: 所有GPU持有完整的embedding
        output = reduce_from_tensor_model_parallel_region(
            output_parallel, group=self.tp_group
        )

    return output
```

**代码解析**:

1. **掩码创建** (line 268-270):
   ```python
   input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
   ```
   - `input_ < vocab_start_index`: token在本GPU之前的词汇范围
   - `input_ >= vocab_end_index`: token在本GPU之后的词汇范围
   - `|`: 逻辑或，合并两个条件

2. **索引调整** (line 272-273):
   ```python
   masked_input = input_.clone() - self.vocab_start_index
   masked_input[input_mask] = 0
   ```
   - 将全局token ID转换为本地索引（减去起始偏移）
   - 不在范围内的token设为0（避免索引越界）

3. **Embedding查表** (line 277-281):
   - **确定性模式**: 直接索引 `self.weight[masked_input]`，保证可复现性
   - **非确定性模式**: 使用 `F.embedding`，性能更好

4. **掩码应用** (line 284):
   ```python
   output_parallel[input_mask, :] = 0.0
   ```
   - 将不属于本GPU的token的embedding向量全部置零
   - 这样AllReduce后，每个token只有一个GPU贡献非零值

5. **AllReduce聚合** (line 293-294):
   ```python
   output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
   ```
   - 定义在 `megatron/core/tensor_parallel/mappings.py`
   - 前向: AllReduce (SUM)
   - 反向: Identity (每个GPU都收到完整梯度)

---

### 4.3 前向传播示例

#### 示例设置

- 词汇表大小: $V = 16$
- 隐藏维度: $d = 4$
- 张量并行度: $p = 4$
- 输入: `input_ids = [[2, 5, 10], [7, 14, 3]]` (shape: [2, 3])

#### GPU词汇分配

| GPU | 词汇范围 | token数量 |
|-----|----------|-----------|
| GPU 0 | [0, 4) | 4 |
| GPU 1 | [4, 8) | 4 |
| GPU 2 | [8, 12) | 4 |
| GPU 3 | [12, 16) | 4 |

#### 逐步计算

**GPU 0**:
```python
vocab_start = 0, vocab_end = 4

# 步骤1: 创建掩码
input_ids = [[2, 5, 10], [7, 14, 3]]
input_mask = [[False, True, True], [True, True, False]]
#  2在范围内   5,10,7,14不在   3在范围内

# 步骤2: 调整索引
masked_input = [[2-0=2, 0, 0], [0, 0, 3-0=3]]
                = [[2, 0, 0], [0, 0, 3]]

# 步骤3: Embedding查表
# weight_0.shape = [4, 4]，对应token 0,1,2,3的embedding
output_parallel = weight_0[[2, 0, 0], [0, 0, 3]]
# 形状: [2, 3, 4]

# 步骤4: 应用掩码
output_parallel[0, 1, :] = 0  # token 5不在范围内
output_parallel[0, 2, :] = 0  # token 10不在范围内
output_parallel[1, 0, :] = 0  # token 7不在范围内
output_parallel[1, 1, :] = 0  # token 14不在范围内
# 保留: output_parallel[0,0,:] (token 2), output_parallel[1,2,:] (token 3)
```

**GPU 1**:
```python
vocab_start = 4, vocab_end = 8

input_mask = [[True, False, True], [False, True, True]]
#  5在范围内   7在范围内

masked_input = [[0, 5-4=1, 0], [7-4=3, 0, 0]]
                = [[0, 1, 0], [3, 0, 0]]

output_parallel = weight_1[[0, 1, 0], [3, 0, 0]]

# 应用掩码，保留token 5和token 7的embedding
```

**GPU 2**:
```python
vocab_start = 8, vocab_end = 12

input_mask = [[True, True, False], [True, True, True]]
#  10在范围内

masked_input = [[0, 0, 10-8=2], [0, 0, 0]]

# 保留token 10的embedding
```

**GPU 3**:
```python
vocab_start = 12, vocab_end = 16

input_mask = [[True, True, True], [True, False, True]]
#  14在范围内

# 保留token 14的embedding
```

**步骤5: AllReduce**

所有GPU的 `output_parallel` 求和，每个位置只有一个GPU贡献非零值：

```python
output = (
    GPU0的output_parallel +  # token 2和3的embedding
    GPU1的output_parallel +  # token 5和7的embedding
    GPU2的output_parallel +  # token 10的embedding
    GPU3的output_parallel    # token 14的embedding
)
# 最终output形状: [2, 3, 4]，每个位置都是对应token的正确embedding
```

---

### 4.4 反向传播分析

#### 4.4.1 AllReduce的反向

AllReduce的定义（参考mappings.py）：
```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)  # AllReduce SUM

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Identity
```

**前向**: AllReduce (SUM)
$$
\text{output} = \sum_{i=0}^{p-1} \text{output\_parallel}_i
$$

**反向**: 恒等复制
$$
\frac{\partial \mathcal{L}}{\partial \text{output\_parallel}_i} = \frac{\partial \mathcal{L}}{\partial \text{output}}
$$

所有GPU收到**完整的梯度**。

---

#### 4.4.2 掩码的反向

掩码操作：
```python
output_parallel[input_mask, :] = 0.0
```

**前向**: 掩码位置置零
**反向**: 掩码位置的梯度也被置零

```python
grad_output_parallel = grad_output.clone()
grad_output_parallel[input_mask, :] = 0.0
```

这确保了每个GPU只计算属于自己词汇范围的token的梯度。

---

#### 4.4.3 Embedding查表的反向

Embedding查表的反向传播是**梯度累加**到权重：

```python
# 前向
output_parallel = self.weight[masked_input]  # shape: [b, s, d]

# 反向
for i in range(batch_size):
    for j in range(seq_len):
        if not input_mask[i, j]:  # 仅对未掩码的token
            token_id = masked_input[i, j]
            # 累加梯度
            grad_weight[token_id] += grad_output_parallel[i, j, :]
```

**无需额外通信**: 因为每个token只属于一个GPU，所以权重梯度是独立的。

---

### 4.5 通信分析

#### 4.5.1 通信量

**前向传播**:
- 1次AllReduce: `output_parallel` 的shape为 $[s, b, d]$
- 通信量: $s \times b \times d \times \text{sizeof(dtype)} \times \frac{p-1}{p}$ bytes

**反向传播**:
- 0次通信（权重梯度独立）

**示例**: $s=2048, b=4, d=8192, p=8$, FP16
$$
\begin{aligned}
\text{通信量}
&= 2048 \times 4 \times 8192 \times 2 \times \frac{7}{8} \\
&= 117 \text{ MB}
\end{aligned}
$$

#### 4.5.2 与序列并行的结合

当启用序列并行时，可以使用**ReduceScatter**代替AllReduce：

```python
if self.reduce_scatter_embeddings:
    output = reduce_scatter_to_sequence_parallel_region(
        output_parallel, group=self.tp_group
    )
```

**ReduceScatter**:
- 前向: 每个GPU只保留 $1/p$ 的序列长度
- 通信量: 与AllReduce相同，但内存占用更低

**输出形状**:
- AllReduce: $[s, b, d]$ (所有GPU)
- ReduceScatter: $[s/p, b, d]$ (每个GPU)

---

### 4.6 内存分析

#### 4.6.1 参数内存

| 项目 | 标准Embedding | 词汇表并行Embedding |
|------|---------------|----------------------|
| 权重形状 | $[V, d]$ | $[V/p, d]$ |
| 参数量 | $V \times d$ | $V \times d / p$ |
| 内存占用 (FP16) | $2Vd$ bytes | $2Vd/p$ bytes |

**示例**: Qwen-72B, $V=151,643, d=8,192, p=8$
- 标准: $151,643 \times 8,192 \times 2 = 2.37$ GB
- 并行: $2.37 / 8 = 0.30$ GB  ← **减少87.5%**

#### 4.6.2 激活内存

**前向传播中间激活**:

| 变量 | 形状 | 内存 (FP16) |
|------|------|-------------|
| `input_` | $[b, s]$ | $2bs$ bytes |
| `masked_input` | $[b, s]$ | $2bs$ bytes (临时) |
| `input_mask` | $[b, s]$ | $bs/8$ bytes (bool) |
| `output_parallel` | $[b, s, d]$ | $2bsd$ bytes |
| `output` | $[b, s, d]$ | $2bsd$ bytes |

**关键观察**: 激活内存**不受并行度影响**（AllReduce后所有GPU持有完整激活）。

---

## 5. 输出层(LM Head)的张量并行

### 5.1 输出层的作用

在语言模型中，输出层（也称为LM Head）负责将Transformer最后一层的隐藏状态映射回词汇表空间：

$$
\mathbf{Z} = \mathbf{H}_{final} \mathbf{W}_O^T
$$

其中：
- $\mathbf{H}_{final} \in \mathbb{R}^{s \times b \times d}$: 最后一层的隐藏状态
- $\mathbf{W}_O \in \mathbb{R}^{V \times d}$: 输出层权重矩阵
- $\mathbf{Z} \in \mathbb{R}^{s \times b \times V}$: 输出logits

**输出层 vs Embedding层的关系**:

在许多LLM中（如GPT系列），输出层权重与Embedding层权重**共享**（weight tying）：
```python
# 权重共享
lm_head.weight = embedding.weight
```

优势：
1. 减少参数量（节省 $V \times d$ 个参数）
2. 更好的泛化性能

在词汇表并行中，无论是否共享权重，都采用相同的并行策略。

---

### 5.2 输出层并行实现

输出层的并行化本质上是**列并行线性层**（参考文档57-59）：

```python
# 输出层：列并行
output_layer = ColumnParallelLinear(
    input_size=hidden_size,      # d
    output_size=vocab_size,      # V
    gather_output=False,         # 🔑 不收集输出，保持切分
    bias=False,
    ...
)
```

**关键参数**: `gather_output=False`

这确保输出logits保持切分状态：
- 每个GPU输出: $\mathbf{Z}_i \in \mathbb{R}^{s \times b \times (V/p)}$
- 切分维度: 词汇维度

---

### 5.3 输出层前向传播

#### 5.3.1 数学表达

GPU $i$ 计算部分logits：

$$
\mathbf{Z}_i = \mathbf{H}_{final} \mathbf{W}_{O,i}^T \in \mathbb{R}^{s \times b \times (V/p)}
$$

其中 $\mathbf{W}_{O,i}$ 对应词汇范围 $[i \cdot V/p, (i+1) \cdot V/p)$。

**无需通信**:
- 输入 $\mathbf{H}_{final}$ 在所有GPU上是完整的（来自前面层的AllReduce）
- 输出保持切分状态，直接送入并行交叉熵

---

#### 5.3.2 代码实现

```python
class ColumnParallelLinear(torch.nn.Module):
    def forward(self, input_):
        """
        Args:
            input_: [s, b, d] - 完整的隐藏状态
        Returns:
            output: [s, b, V/p] - 切分的logits
            bias: None or [V/p]
        """
        # 如果启用了异步通信（对于后续层），这里是Identity
        # 但对于输出层，input_已经是完整的，无需通信
        input_parallel = input_

        # 🔑 本地矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        # output_parallel.shape = [s, b, V/p]

        # 🔑 gather_output=False，不收集输出
        if self.gather_output:
            # 其他层：AllGather收集完整输出
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            # 输出层：保持切分状态
            output = output_parallel

        return output, None  # bias已加入output
```

**关键区别**:

| 层 | `gather_output` | 输出形状 | 通信 |
|------|-----------------|----------|------|
| **FC1** (列并行) | False | $[s, b, 4d/p]$ | 无 |
| **输出层** | False | $[s, b, V/p]$ | 无 |
| **普通列并行** | True | $[s, b, \text{output\_size}]$ | AllGather |

---

### 5.4 输出层反向传播

#### 5.4.1 梯度来源

输出层的梯度来自交叉熵损失（详见第6章）：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} = \text{softmax}(\mathbf{Z})_i - \mathbf{1}_{y_i} \in \mathbb{R}^{s \times b \times (V/p)}
$$

其中：
- $\text{softmax}(\mathbf{Z})_i$: GPU $i$ 的词汇范围对应的softmax值
- $\mathbf{1}_{y_i}$: one-hot向量（仅目标token位置为1）

---

#### 5.4.2 权重梯度

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{O,i}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \right)^T \mathbf{H}_{final} \in \mathbb{R}^{(V/p) \times d}
$$

**无需通信**: 每个GPU独立计算自己的权重梯度。

---

#### 5.4.3 输入梯度（反向传播到Transformer）

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_{final,i}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i} \mathbf{W}_{O,i} \in \mathbb{R}^{s \times b \times d}
$$

**AllReduce通信**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{H}_{final}} = \sum_{i=0}^{p-1} \frac{\partial \mathcal{L}}{\partial \mathbf{H}_{final,i}}
$$

这是**列并行层的标准反向传播模式**（参考文档57）。

---

### 5.5 权重共享(Weight Tying)的并行

当输出层与Embedding层共享权重时：

```python
# 标准权重共享
lm_head.weight = embedding.weight

# 词汇表并行下的权重共享
lm_head_parallel.weight = vocab_parallel_embedding.weight
# 两者都是 [V/p, d] 形状，对应相同的词汇范围
```

**关键要点**:
1. **词汇范围必须一致**: 两者的 `vocab_start_index` 和 `vocab_end_index` 必须相同
2. **梯度累积**: Embedding和输出层的梯度会累加到同一个权重上
3. **内存节省**: 共享权重时，参数量从 $2 \times V \times d / p$ 降至 $V \times d / p$

---

## 6. 词汇表并行的交叉熵损失

**本章是词汇表并行的核心难点**，详细介绍如何在logits被切分的情况下正确计算交叉熵损失。

### 6.1 标准交叉熵损失回顾

对于单个样本，交叉熵损失定义为：

$$
\mathcal{L}(\mathbf{z}, y) = -\log \frac{e^{z_y}}{\sum_{j=1}^{V} e^{z_j}} = \log \left( \sum_{j=1}^{V} e^{z_j} \right) - z_y
$$

其中：
- $\mathbf{z} \in \mathbb{R}^V$: logits向量
- $y \in \{1, \ldots, V\}$: 目标类别
- $z_y$: 目标类别的logit值

#### 数值稳定性技巧

直接计算 $\sum_{j=1}^{V} e^{z_j}$ 容易溢出，标准做法是**减去最大值**：

$$
\begin{aligned}
\mathcal{L}(\mathbf{z}, y)
&= \log \left( \sum_{j=1}^{V} e^{z_j} \right) - z_y \\
&= \log \left( e^{z_{max}} \sum_{j=1}^{V} e^{z_j - z_{max}} \right) - z_y \\
&= z_{max} + \log \left( \sum_{j=1}^{V} e^{z_j - z_{max}} \right) - z_y
\end{aligned}
$$

其中 $z_{max} = \max_{j=1}^{V} z_j$。

---

### 6.2 并行交叉熵的挑战

当logits被切分到 $p$ 个GPU时：

```python
# GPU 0: z_0 ∈ ℝ^(V/p)  对应词汇 [0, V/p)
# GPU 1: z_1 ∈ ℝ^(V/p)  对应词汇 [V/p, 2V/p)
# ...
# GPU p-1: z_{p-1} ∈ ℝ^(V/p)  对应词汇 [(p-1)V/p, V)
```

**三大挑战**:

1. **挑战1**: 如何计算全局最大值 $z_{max}$？
   - 每个GPU只能看到部分logits
   - 需要 **AllReduce(MAX)** 获取全局最大值

2. **挑战2**: 如何计算全局指数和 $\sum_{j=1}^{V} e^{z_j - z_{max}}$？
   - 每个GPU只能计算部分和 $\sum_{j \in \text{partition}_i} e^{z_j - z_{max}}$
   - 需要 **AllReduce(SUM)** 聚合部分和

3. **挑战3**: 如何获取目标logit $z_y$？
   - 目标token可能在任意GPU上
   - 需要 **掩码 + AllReduce(SUM)** 提取

---

### 6.3 并行交叉熵前向传播

**代码位置**: `megatron/core/tensor_parallel/cross_entropy.py:122-189`

#### 6.3.1 算法步骤

```python
def vocab_parallel_cross_entropy_forward(
    vocab_parallel_logits,  # [s*b, V/p] on each GPU
    target,                  # [s*b] (完整的目标)
    label_smoothing=0.0
):
    """
    并行交叉熵前向传播

    Args:
        vocab_parallel_logits: 切分的logits，每个GPU持有 [s*b, V/p]
        target: 完整的目标token IDs，shape [s*b]
        label_smoothing: 标签平滑系数

    Returns:
        loss: 交叉熵损失，shape [s*b]
    """
    # 🔑 步骤1: 计算全局最大值
    # 1.1 局部最大值
    logits_max_local = torch.max(vocab_parallel_logits, dim=-1)[0]  # [s*b]

    # 1.2 全局最大值（AllReduce MAX）
    logits_max = logits_max_local.clone()
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=get_tensor_model_parallel_group()
    )  # [s*b] - 所有GPU持有相同的全局最大值

    # 🔑 步骤2: 数值稳定化
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
    # 所有logits <= 0，避免exp溢出

    # 🔑 步骤3: 提取目标logit（带掩码）
    # 3.1 计算本GPU的词汇范围
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    partition_vocab_size = vocab_parallel_logits.size()[-1]  # V/p
    vocab_start_index = rank * partition_vocab_size
    vocab_end_index = (rank + 1) * partition_vocab_size

    # 3.2 创建掩码：标记目标token是否在本GPU的词汇范围内
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    # target_mask[i] = True 表示 target[i] 不在本GPU范围内

    # 3.3 调整目标索引到本地范围
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0  # 不在范围内的设为0（避免越界）

    # 3.4 提取目标logit（仅对本GPU范围内的目标）
    arange_1d = torch.arange(start=0, end=vocab_parallel_logits.size()[0],
                              device=vocab_parallel_logits.device)
    predicted_logits = vocab_parallel_logits[arange_1d, masked_target]  # [s*b]
    predicted_logits = predicted_logits.clone().contiguous()

    # 3.5 应用掩码：不在本GPU范围内的目标logit置零
    predicted_logits[target_mask] = 0.0

    # 3.6 AllReduce SUM：每个样本的目标logit只有一个GPU贡献非零值
    torch.distributed.all_reduce(
        predicted_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )  # [s*b] - 所有GPU持有完整的目标logit

    # 🔑 步骤4: 计算指数和
    # 4.1 计算局部指数和
    exp_logits = torch.exp(vocab_parallel_logits)  # [s*b, V/p]
    sum_exp_logits_local = exp_logits.sum(dim=-1)  # [s*b]

    # 4.2 全局指数和（AllReduce SUM）
    sum_exp_logits = sum_exp_logits_local.clone()
    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )  # [s*b] - 所有GPU持有相同的全局指数和

    # 🔑 步骤5: 计算损失
    loss = torch.log(sum_exp_logits) - predicted_logits  # [s*b]

    # 🔑 步骤6（可选）: 标签平滑
    if label_smoothing > 0:
        # 归一化exp_logits为概率
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # 计算log概率
        log_probs = torch.log(exp_logits)
        mean_log_probs = log_probs.mean(dim=-1)

        # 平滑损失
        smoothing = label_smoothing * V / (V - 1)
        loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    return loss, exp_logits, target_mask, masked_target
```

---

#### 6.3.2 通信模式总结

| 步骤 | 通信类型 | 通信数据 | 大小 |
|------|----------|----------|------|
| 步骤1 | AllReduce(MAX) | `logits_max` | $[s \times b]$ |
| 步骤3 | AllReduce(SUM) | `predicted_logits` | $[s \times b]$ |
| 步骤4 | AllReduce(SUM) | `sum_exp_logits` | $[s \times b]$ |
| **总计** | **3次AllReduce** | - | $3 \times s \times b \times \text{sizeof(dtype)}$ |

**示例**: $s=2048, b=4, p=8$, FP16
$$
\text{总通信量} = 3 \times 2048 \times 4 \times 2 \times \frac{7}{8} = 43 \text{ KB}
$$

非常小！（相比于logits的大小 $s \times b \times V \times 2$ bytes）

---

### 6.4 并行交叉熵反向传播

#### 6.4.1 标准交叉熵梯度

对于标准交叉熵，logits的梯度是：

$$
\frac{\partial \mathcal{L}}{\partial z_j} = \begin{cases}
p_j - 1, & \text{if } j = y \\
p_j, & \text{otherwise}
\end{cases}
$$

其中 $p_j = \frac{e^{z_j}}{\sum_{k=1}^{V} e^{z_k}}$ 是softmax概率。

向量形式：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{e}_y
$$

其中 $\mathbf{e}_y$ 是one-hot向量。

---

#### 6.4.2 并行梯度计算

**代码位置**: `megatron/core/tensor_parallel/cross_entropy.py:191-216`

```python
def vocab_parallel_cross_entropy_backward(
    grad_output,      # [s*b] - 来自后续层的梯度
    exp_logits,       # [s*b, V/p] - 前向保存的exp(logits)，已归一化为softmax
    target_mask,      # [s*b] - 前向保存的掩码
    masked_target,    # [s*b] - 前向保存的本地目标索引
    label_smoothing=0.0
):
    """
    并行交叉熵反向传播

    Returns:
        grad_input: 对logits的梯度，shape [s*b, V/p]
    """
    # 🔑 步骤1: 初始化梯度为softmax概率
    grad_input = exp_logits  # [s*b, V/p]
    # 注意: exp_logits在前向已经被归一化为softmax概率

    # 🔑 步骤2: 减去one-hot向量（仅对目标类别）
    # 将grad_input展平为2D
    partition_vocab_size = grad_input.size()[-1]
    grad_2d = grad_input.view(-1, partition_vocab_size)  # [s*b, V/p]

    # 🔑 步骤3: 处理one-hot向量
    arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                              device=grad_2d.device)

    if label_smoothing > 0:
        # 标签平滑情况
        V = partition_vocab_size * get_tensor_model_parallel_world_size()
        smoothing = label_smoothing * V / (V - 1)

        # 对目标类别: grad -= (1 - smoothing)
        softmax_update = 1.0 - target_mask.view(-1).float()  # 仅未掩码的为1
        grad_2d[arange_1d, masked_target] -= (1.0 - smoothing) * softmax_update

        # 对所有类别: grad -= smoothing / V
        grad_2d -= smoothing / V
    else:
        # 标准情况：对目标类别 grad -= 1
        softmax_update = 1.0 - target_mask.view(-1).float()
        grad_2d[arange_1d, masked_target] -= softmax_update

    # 🔑 步骤4: 乘以来自后续层的梯度
    grad_input.mul_(grad_output.unsqueeze(dim=-1))

    return grad_input  # [s*b, V/p]
```

**关键点**:
1. **softmax已在前向计算**: `exp_logits` 已被归一化，直接用作梯度初值
2. **掩码处理**: 仅对本GPU词汇范围内的目标token减1
3. **无需通信**: 梯度计算完全本地化

---

### 6.5 完整的VocabParallelCrossEntropy类

**代码位置**: `megatron/core/tensor_parallel/cross_entropy.py:16-217`

```python
class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel ranks.
    """

    @staticmethod
    def calculate_logits_max(vocab_parallel_logits):
        """计算全局最大值（带AllReduce）"""
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        # 这里省略AllReduce，由调用者处理
        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits, target, logits_max,
        vocab_start_index, vocab_end_index
    ):
        """提取目标logit（带掩码）"""
        # 数值稳定化
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # 创建掩码
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # 提取目标logit
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                  device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # 计算指数和
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(exp_logits, predicted_logits, sum_exp_logits):
        """计算最终损失"""
        loss = torch.log(sum_exp_logits) - predicted_logits

        # 归一化exp_logits为softmax概率（用于反向传播）
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def calculate_gradients(grad_2d, arange_1d, masked_target_1d,
                             softmax_update, grad_input, grad_output):
        """计算梯度"""
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """自定义autograd函数"""

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        # 步骤1: 计算全局最大值
        vocab_parallel_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(vocab_parallel_logits)
        )
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group()
        )

        # 步骤2: 获取词汇范围
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index = rank * partition_vocab_size
        vocab_end_index = (rank + 1) * partition_vocab_size

        # 步骤3: 计算预测logit和指数和
        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max,
                vocab_start_index, vocab_end_index
            )
        )

        # 步骤4: AllReduce聚合
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )

        # 步骤5: 计算损失
        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        # 步骤6: 标签平滑（如果需要）
        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        # 保存用于反向传播
        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # 恢复保存的张量
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # 初始化梯度
        grad_input = softmax
        partition_vocab_size = grad_input.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                  device=grad_2d.device)

        # 计算softmax_update
        softmax_update = 1.0 - target_mask.view(-1).float()

        # 应用标签平滑（如果有）
        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            grad_2d -= smoothing / vocab_size
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d, arange_1d, masked_target_1d, softmax_update,
                grad_input, grad_output
            )

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    并行交叉熵损失的公开接口

    Args:
        vocab_parallel_logits: 切分的logits，shape [s, b, V/p]
        target: 完整的目标，shape [s, b]
        label_smoothing: 标签平滑系数，范围 [0.0, 1.0)

    Returns:
        loss: 交叉熵损失，shape [s*b]
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
```

---

### 6.6 并行交叉熵的正确性证明

**定理6.1** (并行交叉熵等价性):

设标准交叉熵损失为：
$$
\mathcal{L}_{\text{std}}(\mathbf{z}, y) = \log \left( \sum_{j=1}^{V} e^{z_j} \right) - z_y
$$

并行交叉熵损失为：
$$
\mathcal{L}_{\text{parallel}}(\{\mathbf{z}_0, \ldots, \mathbf{z}_{p-1}\}, y) = \log \left( \sum_{i=0}^{p-1} \sum_{j \in \text{partition}_i} e^{z_j} \right) - z_y
$$

则：
$$
\mathcal{L}_{\text{parallel}} = \mathcal{L}_{\text{std}}
$$

**证明**:

1. **指数和的等价性**:
   $$
   \sum_{i=0}^{p-1} \sum_{j \in \text{partition}_i} e^{z_j} = \sum_{j=1}^{V} e^{z_j}
   $$
   这是因为词汇表被无重叠地切分到 $p$ 个GPU。

2. **目标logit的等价性**:
   设目标token $y$ 属于GPU $i^*$ 的词汇范围。
   - GPU $i^*$ 的 `predicted_logits` = $z_y$（未被掩码）
   - 其他GPU $i \neq i^*$ 的 `predicted_logits` = $0$（被掩码）

   AllReduce(SUM)后：
   $$
   \text{predicted\_logits} = z_y
   $$

3. **损失相等**:
   $$
   \begin{aligned}
   \mathcal{L}_{\text{parallel}}
   &= \log \left( \text{AllReduce\_SUM}\left( \sum_{j \in \text{partition}_i} e^{z_j} \right) \right) - \text{AllReduce\_SUM}(\text{predicted\_logits}_i) \\
   &= \log \left( \sum_{j=1}^{V} e^{z_j} \right) - z_y \\
   &= \mathcal{L}_{\text{std}}
   \end{aligned}
   $$

□

---

## 7. Megatron-LM代码实现

本章分析Megatron-LM中词汇表并行的完整实现，包括Embedding层、输出层和交叉熵损失的集成。

### 7.1 GPT模型中的词汇表并行

**代码位置**: `megatron/core/models/gpt/gpt_model.py`

#### 7.1.1 Embedding层初始化

```python
class GPTModel(MegatronModule):
    def __init__(self, config, ...):
        # ...

        # 🔑 词汇表并行的Embedding层
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.config.position_embedding_type,
            ...
        )
```

**LanguageModelEmbedding内部** (`megatron/core/models/common/embeddings/language_model_embedding.py`):

```python
class LanguageModelEmbedding(MegatronModule):
    def __init__(self, config, vocab_size, ...):
        super().__init__(config)

        # 🔑 VocabParallelEmbedding
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
            init_method=config.init_method,
            reduce_scatter_embeddings=config.sequence_parallel,  # 序列并行时使用ReduceScatter
            config=config,
        )

        # 位置编码（如果需要）
        if self.position_embedding_type == 'learned_absolute':
            self.position_embeddings = nn.Embedding(
                max_sequence_length,
                config.hidden_size
            )
```

---

#### 7.1.2 输出层初始化

```python
class GPTModel(MegatronModule):
    def __init__(self, config, ...):
        # ...

        # 🔑 输出层（列并行）
        self.output_layer = ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,  # 🔑 不收集输出，保持切分
            skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
        )

        # 🔑 权重共享（如果启用）
        if self.share_embeddings_and_output_weights:
            self.initialize_last_stage_with_word_embeddings()
```

**权重共享实现**:

```python
def initialize_last_stage_with_word_embeddings(self):
    """
    共享Embedding和输出层权重
    """
    # 仅在pipeline的最后一阶段执行
    if not self._output_layer_weight_initialized:
        assert self.output_layer.weight.shape == self.shared_embedding_or_output_weight().shape
        # 共享权重
        self.output_layer.weight = self.shared_embedding_or_output_weight()
        self._output_layer_weight_initialized = True
```

---

### 7.2 前向传播流程

```python
def forward(self, input_ids, position_ids, attention_mask, ...):
    """
    Args:
        input_ids: [b, s] - 输入token IDs
        ...

    Returns:
        output: [s, b, V/p] - 切分的logits（如果return_embeddings=False）
    """
    # 🔑 步骤1: Embedding层（词汇表并行）
    embeddings = self.embedding(input_ids, position_ids)
    # embeddings.shape = [s, b, d] - 完整的（AllReduce后）

    # 步骤2: Transformer层
    hidden_states = self.decoder(
        embeddings,
        attention_mask,
        ...
    )
    # hidden_states.shape = [s, b, d] - 完整的

    # 🔑 步骤3: 输出层（列并行，词汇表切分）
    if not return_embeddings:
        logits, _ = self.output_layer(hidden_states)
        # logits.shape = [s, b, V/p] - 切分的
        return logits
    else:
        return hidden_states
```

---

### 7.3 训练中的损失计算

**代码位置**: `megatron/core/models/gpt/gpt_model.py` (loss function)

```python
def gpt_loss_function(labels, output_tensor):
    """
    计算GPT的交叉熵损失

    Args:
        labels: [b, s] - 目标token IDs
        output_tensor: [s, b, V/p] - 模型输出的切分logits

    Returns:
        loss: 标量损失
        loss_dict: 包含详细信息的字典
    """
    # 🔑 使用并行交叉熵
    losses = vocab_parallel_cross_entropy(
        output_tensor.contiguous().float(),  # [s, b, V/p]
        labels.contiguous()                   # [b, s]
    )
    # losses.shape = [s*b]

    # 计算平均损失
    loss = torch.mean(losses)

    # 返回
    return loss, {'lm loss': loss}
```

---

### 7.4 配置参数

**在训练脚本中设置词汇表并行**:

```bash
# 词汇表大小
--vocab-size 32000 \

# 张量并行度（自动应用到Embedding和输出层）
--tensor-model-parallel-size 8 \

# 权重共享（可选）
--share-embeddings-and-output-weights \

# 序列并行（与词汇表并行结合）
--sequence-parallel \
```

**在代码中**:

```python
from megatron.core import ModelParallelConfig

config = ModelParallelConfig(
    tensor_model_parallel_size=8,  # TP=8
    sequence_parallel=True,         # 启用序列并行
    ...
)

model = GPTModel(
    config=config,
    vocab_size=32000,
    share_embeddings_and_output_weights=True,
    ...
)
```

---

## 8. 性能分析与优化

### 8.1 内存节省分析

#### 8.1.1 参数内存

**Qwen-72B示例** ($V=151,643, d=8,192, p=8$):

| 组件 | 标准 (GB) | 词汇表并行 (GB) | 节省 |
|------|-----------|-----------------|------|
| Embedding权重 | 2.37 | 0.30 | 87.5% |
| 输出层权重 | 2.37 | 0.30 | 87.5% |
| **总计（无权重共享）** | **4.74** | **0.59** | **87.5%** |
| **总计（权重共享）** | **2.37** | **0.30** | **87.5%** |

#### 8.1.2 激活内存

**前向传播** ($s=2048, b=4, d=8192, p=8$):

| 变量 | 标准 | 词汇表并行 | 节省 |
|------|------|------------|------|
| Embedding输出 | 512 MB | 512 MB | 0% |
| 输出logits | 2.0 GB | 250 MB | 87.5% |

**关键观察**:
- **Embedding输出**: AllReduce后所有GPU持有完整副本，无内存节省
- **输出logits**: 保持切分状态直到损失计算，**显著节省**

---

### 8.2 通信开销分析

#### 8.2.1 通信量统计

**单次迭代** ($s=2048, b=4, d=8192, V=151643, p=8$, FP16):

| 操作 | 通信类型 | 数据形状 | 通信量 (MB) |
|------|----------|----------|-------------|
| Embedding前向 | AllReduce | $[s, b, d]$ | 117 |
| 交叉熵-logits_max | AllReduce(MAX) | $[s, b]$ | 0.014 |
| 交叉熵-predicted_logits | AllReduce(SUM) | $[s, b]$ | 0.014 |
| 交叉熵-sum_exp_logits | AllReduce(SUM) | $[s, b]$ | 0.014 |
| 输出层反向 | AllReduce | $[s, b, d]$ | 117 |
| **总计** | - | - | **234** |

**对比Transformer层通信** (每层):
- MHA: ~117 MB (Attention输出AllReduce)
- MLP: ~117 MB (FC2输出AllReduce)

**结论**: 词汇表并行的通信开销与Transformer层相当，**不是瓶颈**。

---

#### 8.2.2 通信优化

1. **使用ReduceScatter代替AllReduce** (Embedding层):
   ```python
   vocab_parallel_embedding = VocabParallelEmbedding(
       ...,
       reduce_scatter_embeddings=True,  # 启用ReduceScatter
   )
   ```
   - 内存节省: Embedding输出从 $[s, b, d]$ 降至 $[s/p, b, d]$
   - 通信量: 不变

2. **异步通信**:
   Megatron已在后台实现，无需手动优化。

3. **梯度累积**:
   交叉熵的3次小AllReduce可以批量处理（Megatron自动优化）。

---

### 8.3 计算开销

#### 8.3.1 FLOPs分析

**Embedding层**:
- 查表操作: O(1)，几乎无计算开销
- AllReduce: 纯通信，无计算

**输出层**:
- 矩阵乘法: $s \times b \times d \times (V/p)$ FLOPs
- 与MLP的FC1相当

**交叉熵损失**:
- Exp/Log: $s \times b \times V$ ops（各GPU计算$V/p$）
- 计算量很小

**结论**: 词汇表并行的计算开销**可忽略**。

---

### 8.4 扩展性分析

#### 8.4.1 强扩展性

固定模型大小，增加GPU数量：

| TP | Embedding参数/GPU | 输出logits/GPU | 通信量 |
|-----|-------------------|----------------|--------|
| 1 | 2.37 GB | 2.0 GB | 0 |
| 2 | 1.19 GB | 1.0 GB | ~117 MB |
| 4 | 0.59 GB | 0.5 GB | ~117 MB |
| 8 | 0.30 GB | 0.25 GB | ~117 MB |

**扩展效率**: > 95%（通信量几乎不随TP增加）

---

#### 8.4.2 对超大词汇表的支持

**多语言模型** (如$V=250,000$):

| TP | 参数/GPU (FP16) | 是否可行 |
|----|-----------------|----------|
| 1 | 4.0 GB | ✅ |
| 4 | 1.0 GB | ✅ (推荐) |
| 8 | 0.5 GB | ✅ (高效) |
| 16 | 0.25 GB | ✅ (过度) |

**建议**: 根据词汇表大小选择合适的TP：
- $V < 50K$: TP=1-2（无需词汇表并行）
- $50K \leq V < 100K$: TP=2-4
- $V \geq 100K$: TP=4-8

---

## 9. 最佳实践与调试技巧

### 9.1 配置建议

#### 9.1.1 何时启用词汇表并行

✅ **应该启用**:
- 词汇表大小 > 50,000
- 多语言模型（词汇表通常 > 100,000）
- 内存受限的环境

❌ **可以不启用**:
- 词汇表大小 < 30,000
- 单GPU训练
- 推理阶段（如果内存充足）

---

#### 9.1.2 与其他并行策略的结合

**推荐配置**:

```python
# GPT-3 175B级别模型
config = ModelParallelConfig(
    tensor_model_parallel_size=8,      # 张量并行
    pipeline_model_parallel_size=8,    # 流水线并行
    sequence_parallel=True,             # 序列并行（与词汇表并行结合）
    ...
)
```

**组合效果**:
- **TP + 词汇表并行**: 自动应用，无需额外配置
- **TP + 序列并行**: Embedding层使用ReduceScatter，进一步节省内存
- **PP + 词汇表并行**: Embedding和输出层通常在pipeline的不同阶段

---

### 9.2 常见错误与解决

#### 9.2.1 词汇范围不一致

**错误现象**:
```
RuntimeError: target token ID 12000 out of range [0, 4000)
```

**原因**: Embedding和输出层的词汇范围不一致（可能由于错误的初始化）

**解决方案**:
```python
# 确保两者使用相同的tp_group
embedding = VocabParallelEmbedding(..., tp_group=tp_group)
output_layer = ColumnParallelLinear(..., tp_group=tp_group)
```

---

#### 9.2.2 权重共享失败

**错误现象**:
```
AssertionError: Embedding and output layer shapes mismatch
```

**原因**: 权重形状不匹配（可能词汇表大小设置错误）

**解决方案**:
```python
# 检查词汇表大小
assert embedding.num_embeddings == output_layer.output_size
assert embedding.num_embeddings_per_partition == output_layer.output_size_per_partition
```

---

#### 9.2.3 交叉熵NaN损失

**错误现象**:
```
loss = NaN
```

**原因**:
1. logits数值过大导致exp溢出（数值稳定性问题）
2. 学习率过大
3. 梯度爆炸

**解决方案**:
```python
# 1. 检查logits范围
print(f"Logits min: {logits.min()}, max: {logits.max()}")
# 应该在[-100, 100]范围内

# 2. 使用梯度裁剪
optimizer = MegatronOptimizer(..., clip_grad=1.0)

# 3. 降低学习率
lr_scheduler = WarmupDecayLR(..., max_lr=1e-4)
```

---

### 9.3 性能调优

#### 9.3.1 Profile通信时间

```python
import torch.distributed as dist
import time

# Profile AllReduce时间
start = time.time()
dist.all_reduce(tensor, group=tp_group)
torch.cuda.synchronize()
end = time.time()

print(f"AllReduce time: {(end - start) * 1000:.2f} ms")
```

**预期时间** (A100, NVLink, TP=8):
- Embedding AllReduce (117 MB): ~2-3 ms
- 交叉熵AllReduce (0.014 MB × 3): < 0.1 ms

如果时间显著超过预期，检查：
- 网络配置（确保使用NVLink/NVSwitch）
- NCCL版本（建议 >= 2.15）

---

#### 9.3.2 减少内存碎片

```python
# 使用连续张量
logits = logits.contiguous()
labels = labels.contiguous()

# 及时释放中间变量
del intermediate_logits
torch.cuda.empty_cache()
```

---

### 9.4 调试技巧

#### 9.4.1 验证并行正确性

**单步测试**:

```python
# 步骤1: 验证Embedding并行
input_ids = torch.tensor([[1, 100, 1000]], device='cuda')
embedding_parallel_output = vocab_parallel_embedding(input_ids)

# 与标准Embedding对比
embedding_std = nn.Embedding(vocab_size, hidden_size).cuda()
embedding_std_output = embedding_std(input_ids)

# 应该一致（数值误差< 1e-5）
assert torch.allclose(embedding_parallel_output, embedding_std_output, atol=1e-5)
```

---

#### 9.4.2 检查梯度

```python
# 检查Embedding梯度
print(f"Embedding grad shape: {vocab_parallel_embedding.weight.grad.shape}")
print(f"Embedding grad norm: {vocab_parallel_embedding.weight.grad.norm()}")

# 检查输出层梯度
print(f"Output layer grad shape: {output_layer.weight.grad.shape}")
print(f"Output layer grad norm: {output_layer.weight.grad.norm()}")

# 梯度不应该为None或NaN
assert vocab_parallel_embedding.weight.grad is not None
assert not torch.isnan(vocab_parallel_embedding.weight.grad).any()
```

---

## 10. 面试常见问题

### 问题1: 词汇表并行与其他张量并行策略的区别是什么？

**答案**:

词汇表并行是张量并行的一种特殊形式，主要区别在于：

1. **切分对象不同**:
   - 列并行(FC1): 切分权重矩阵的输出维度
   - 行并行(FC2): 切分权重矩阵的输入维度
   - **词汇表并行**: 切分词汇表维度（Embedding的第0维，输出层的输出维度）

2. **通信模式不同**:
   - 列并行: 前向无通信，反向AllReduce
   - 行并行: 前向AllReduce，反向无通信
   - **词汇表并行**:
     - Embedding前向AllReduce，反向无通信
     - 输出层前向无通信，反向AllReduce
     - 交叉熵需要额外2-3次小AllReduce

3. **处理方式不同**:
   - 列/行并行: 直接矩阵切分
   - **词汇表并行**: 需要掩码机制处理不同GPU的词汇范围

**数学表达**:

列并行: $Y = XW^T, \quad W = [W_0 | W_1 | \cdots | W_{p-1}]$

词汇表并行: $E = \begin{bmatrix} E_0 \\ E_1 \\ \vdots \\ E_{p-1} \end{bmatrix}$

---

### 问题2: 为什么并行交叉熵需要3次AllReduce？能否优化到更少？

**答案**:

**需要3次AllReduce的原因**:

1. **AllReduce(MAX)**: 计算全局最大值 $z_{max} = \max_j z_j$
   - 目的: 数值稳定性（防止exp溢出）
   - 必须: 每个GPU只知道部分logits的最大值

2. **AllReduce(SUM)**: 聚合目标logit $z_y$
   - 目的: 获取损失中的 $-z_y$ 项
   - 必须: 目标token可能在任意GPU上

3. **AllReduce(SUM)**: 聚合指数和 $\sum_j e^{z_j - z_{max}}$
   - 目的: 计算softmax分母
   - 必须: 每个GPU只能计算部分和

**能否优化？**

理论上可以合并为**1次AllReduce**，但需要：
- 自定义CUDA kernel
- 同时计算MAX和SUM
- 通信量增加（传输更多中间结果）

实践中**不值得优化**，因为：
- 通信数据量极小（每次AllReduce仅传输 $s \times b$ 个标量）
- 3次AllReduce总时间 < 0.1 ms（相比于Transformer层的~2 ms可忽略）

Megatron选择清晰的实现而非过度优化。

---

### 问题3: 词汇表并行如何处理词汇表大小不能被TP整除的情况？

**答案**:

Megatron使用**padding策略**：

```python
def divide(numerator, denominator):
    """确保能整除，不能则向上取整"""
    assert numerator % denominator == 0, \
        f'{numerator} is not divisible by {denominator}'
    return numerator // denominator
```

**如果无法整除**，有两种方案：

**方案1: Padding词汇表**（推荐）
```python
# 原始词汇表大小: 32001
# TP=8时，32001 / 8 = 4000.125 (不能整除)

# Padding到32008
padded_vocab_size = ((vocab_size + tp_size - 1) // tp_size) * tp_size
# 32008 / 8 = 4001 (能整除)

vocab_parallel_embedding = VocabParallelEmbedding(
    num_embeddings=padded_vocab_size,
    ...
)
```

padding的token永远不会被使用，内存浪费可忽略。

**方案2: 不均匀切分**（复杂，不推荐）
```python
# GPU 0-6: 4001个token
# GPU 7: 3995个token (32001 - 7*4001)
```

需要特殊处理，Megatron不支持。

**最佳实践**: 选择词汇表大小时考虑TP，例如：
- TP=8: 选择32000 (8的倍数)
- TP=4: 选择32000, 64000等

---

### 问题4: 词汇表并行对推理延迟有什么影响？

**答案**:

**推理单个样本** (batch_size=1):

| 操作 | 延迟 (ms) | 是否并行受益 |
|------|-----------|--------------|
| Embedding查表 | < 0.01 | 否 |
| Embedding AllReduce | ~0.5 | **增加延迟** |
| Transformer层 | 50-100 | 是 (TP加速) |
| 输出层矩阵乘法 | 5-10 | 是 (TP加速) |
| 交叉熵AllReduce | < 0.1 | **增加延迟** |

**结论**:
1. **训练**: 词汇表并行是**必要的**（节省内存）
2. **推理（batch_size=1）**: 词汇表并行可能**增加延迟**（AllReduce开销）
3. **推理（batch_size > 32）**: 词汇表并行**有益**（计算并行化抵消通信）

**推荐策略**:
- 训练: 始终使用词汇表并行（TP > 1时）
- 推理:
  - 如果内存充足且batch_size小 → 考虑TP=1
  - 如果内存受限或batch_size大 → 使用词汇表并行

---

### 问题5: 如何处理Embedding权重与输出层权重共享时的梯度累积？

**答案**:

**权重共享**意味着两层使用**同一个参数**：

```python
output_layer.weight = embedding.weight  # 指向同一块内存
```

**梯度累积机制**:

PyTorch的autograd会**自动累加**梯度：

1. **Embedding反向传播**:
   ```python
   embedding.weight.grad += grad_from_embedding  # [V/p, d]
   ```

2. **输出层反向传播**:
   ```python
   output_layer.weight.grad += grad_from_output  # [V/p, d]
   ```

由于 `output_layer.weight is embedding.weight`，两个梯度自动累加到**同一个.grad**。

**最终梯度**:
$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{emb}} + \frac{\partial \mathcal{L}}{\partial W_{out}}
$$

**注意事项**:
1. **优化器更新**: 只需调用一次 `optimizer.step()`（因为是同一个参数）
2. **梯度裁剪**: 裁剪值需要考虑双倍梯度（或在Embedding和输出层分别裁剪）
3. **内存**: `.grad` 张量只有一份（节省内存）

**Megatron实现**:

```python
# 权重共享在初始化时设置
if self.share_embeddings_and_output_weights:
    self.output_layer.weight = self.embedding.word_embeddings.weight

# 优化器会自动处理梯度累积，无需特殊代码
```

---

## 11. 总结

### 11.1 核心要点

词汇表并行是张量并行的重要组成部分，专门用于Embedding和输出层的并行化：

1. **设计模式**: 词汇表切分 + 掩码 + AllReduce
2. **内存节省**: 参数和激活内存按TP线性降低
3. **通信开销**: 极小（< 1%训练时间）
4. **关键难点**: 并行交叉熵损失的正确实现

---

### 11.2 实现要点

| 组件 | 关键实现 | 代码位置 |
|------|----------|----------|
| **VocabUtility** | 词汇范围计算 | `tensor_parallel/utils.py:97-121` |
| **VocabParallelEmbedding** | Embedding并行 | `tensor_parallel/layers.py:188-316` |
| **ColumnParallelLinear** | 输出层并行 | `tensor_parallel/layers.py:724-1062` |
| **vocab_parallel_cross_entropy** | 并行交叉熵 | `tensor_parallel/cross_entropy.py:16-233` |

---

### 11.3 通信模式总结

| 阶段 | 通信次数 | 通信类型 | 数据大小 |
|------|----------|----------|----------|
| Embedding前向 | 1 | AllReduce(SUM) | $[s, b, d]$ |
| 输出层前向 | 0 | - | - |
| 交叉熵前向 | 3 | AllReduce(MAX/SUM) | $[s, b]$ × 3 |
| 输出层反向 | 1 | AllReduce(SUM) | $[s, b, d]$ |
| Embedding反向 | 0 | - | - |
| **总计** | **5** | - | ~**234 MB** |

---

### 11.4 性能特点

✅ **优势**:
- 参数内存降低87.5% (TP=8)
- 激活内存降低87.5% (输出logits)
- 通信开销可忽略
- 扩展性优秀 (> 95%)

⚠️ **限制**:
- 需要AllReduce通信（对单样本推理有影响）
- Embedding输出仍需完整副本（AllReduce后）
- 词汇表大小应能被TP整除

---

### 11.5 适用场景

**应该使用词汇表并行**:
- 词汇表大小 > 50,000
- 多语言模型
- 训练大规模LLM

**可以不使用**:
- 词汇表大小 < 30,000
- 单GPU训练
- 低延迟推理（batch_size=1）

---

### 11.6 与本系列其他文档的关系

词汇表并行是张量并行(56-60)的最后一块拼图：

```
文档56: 张量并行数学原理      ← 理论基础
文档57: 列并行与行并行        ← 输出层使用列并行
文档58: 注意力层张量并行      ← 上下文
文档59: MLP层张量并行         ← 上下文
文档60: 词汇表并行 (本文档)   ← 特殊的列并行 + 并行交叉熵
```

**下一步**: 文档61-67将介绍流水线并行，它与张量并行结合可实现超大模型训练。

---

## 12. 参考文献

1. **Megatron-LM 核心论文**:
   - Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv:1909.08053.

2. **张量并行理论**:
   - Dean et al. (2012). "Large Scale Distributed Deep Networks". NIPS.

3. **交叉熵损失**:
   - Goodfellow et al. (2016). "Deep Learning". MIT Press. (Chapter 6: Deep Feedforward Networks)

4. **Megatron-LM代码**:
   - GitHub: https://github.com/NVIDIA/Megatron-LM
   - 版本: v0.12.0
   - 许可证: Apache 2.0

5. **相关文档**:
   - 文档56: 张量并行的数学原理
   - 文档57: 列并行与行并行详解
   - 文档58: 注意力层的张量并行
   - 文档59: MLP层的张量并行

---

## 13. 附录

### 附录A: 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $V$ | 词汇表大小 | 标量 |
| $d$ | 隐藏维度 | 标量 |
| $p$ | 张量并行度 (TP size) | 标量 |
| $s$ | 序列长度 | 标量 |
| $b$ | batch大小 | 标量 |
| $\mathbf{W}_E$ | Embedding权重矩阵 | $\mathbb{R}^{V \times d}$ |
| $\mathbf{W}_{E,i}$ | GPU $i$ 的Embedding权重 | $\mathbb{R}^{(V/p) \times d}$ |
| $\mathbf{W}_O$ | 输出层权重矩阵 | $\mathbb{R}^{V \times d}$ |
| $\mathbf{W}_{O,i}$ | GPU $i$ 的输出层权重 | $\mathbb{R}^{(V/p) \times d}$ |
| $\mathbf{z}$ | Logits向量 | $\mathbb{R}^V$ |
| $\mathbf{Z}$ | Logits矩阵 | $\mathbb{R}^{s \times b \times V}$ |
| $\mathbf{Z}_i$ | GPU $i$ 的logits | $\mathbb{R}^{s \times b \times (V/p)}$ |
| $y$ | 目标token ID | $\{1, \ldots, V\}$ |
| $\mathcal{L}$ | 交叉熵损失 | 标量 |

---

### 附录B: VocabParallelEmbedding完整API

```python
class VocabParallelEmbedding(torch.nn.Module):
    """
    Embedding层的词汇表并行实现

    参数:
        num_embeddings (int): 全局词汇表大小 V
        embedding_dim (int): Embedding维度 d
        init_method (Callable): 权重初始化方法
        reduce_scatter_embeddings (bool): 是否使用ReduceScatter（序列并行）
        config (ModelParallelConfig): 并行配置
        tp_group (ProcessGroup, optional): 张量并行组

    属性:
        vocab_start_index (int): 本GPU的词汇起始索引
        vocab_end_index (int): 本GPU的词汇结束索引（不包含）
        num_embeddings_per_partition (int): 本GPU的词汇数量 V/p
        weight (Parameter): Embedding权重，shape [V/p, d]

    方法:
        forward(input_):
            输入: input_ (Tensor) - Token IDs, shape [b, s]
            输出: embeddings (Tensor) - Embeddings, shape [b, s, d] or [s/p, b, d]

        sharded_state_dict(prefix, sharded_offsets, metadata):
            返回分片的state dict（用于checkpoint）
    """
```

**使用示例**:

```python
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core import ModelParallelConfig

config = ModelParallelConfig(tensor_model_parallel_size=8)

embedding = VocabParallelEmbedding(
    num_embeddings=32000,
    embedding_dim=4096,
    init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
    reduce_scatter_embeddings=False,
    config=config,
)

input_ids = torch.randint(0, 32000, (4, 512), device='cuda')  # [b, s]
embeddings = embedding(input_ids)  # [4, 512, 4096]
```

---

### 附录C: vocab_parallel_cross_entropy完整API

```python
def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    词汇表并行的交叉熵损失

    参数:
        vocab_parallel_logits (Tensor): 切分的logits
            - shape: [sequence_length, batch_size, vocab_size_per_partition]
            - dtype: float32 or float16
            - 每个GPU持有 vocab_size/tp_size 个词汇的logits

        target (Tensor): 完整的目标token IDs
            - shape: [batch_size, sequence_length] or [sequence_length, batch_size]
            - dtype: int64
            - 值范围: [0, vocab_size)
            - 所有GPU持有相同的target

        label_smoothing (float, optional): 标签平滑系数
            - 范围: [0.0, 1.0)
            - 默认: 0.0 (无平滑)
            - 推荐: 0.1 for LLM预训练

    返回:
        loss (Tensor): 交叉熵损失
            - shape: [sequence_length * batch_size]
            - dtype: float32
            - 需要手动调用 .mean() 获取标量损失

    通信:
        - 3次AllReduce (MAX, SUM, SUM)
        - 通信量: 3 * sequence_length * batch_size * sizeof(float)

    示例:
        >>> logits = model(input_ids)  # [s, b, V/p] on each GPU
        >>> target = labels            # [b, s] on all GPUs
        >>> losses = vocab_parallel_cross_entropy(logits, target)
        >>> loss = losses.mean()
        >>> loss.backward()
    """
```

---

### 附录D: 训练脚本配置示例

**Qwen-72B训练配置**:

```bash
#!/bin/bash

# 词汇表相关
VOCAB_SIZE=151643        # Qwen词汇表
TOKENIZER_TYPE=Qwen2Tokenizer

# 并行配置
TP=8                     # 张量并行度
PP=8                     # 流水线并行度
DP=16                    # 数据并行度 (自动计算: 总GPU数 / TP / PP)

# 模型配置
HIDDEN_SIZE=8192
FFN_SIZE=24576           # 3 * HIDDEN_SIZE
NUM_LAYERS=80
NUM_ATTENTION_HEADS=64
NUM_KV_HEADS=8           # GQA

# 训练配置
SEQ_LEN=2048
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024   # DP * BATCH_SIZE * gradient_accumulation_steps

# 启动训练
torchrun --nproc_per_node=8 --nnodes=$NUM_NODES \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --vocab-size $VOCAB_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_SIZE \
    --num-layers $NUM_LAYERS \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    --seq-length $SEQ_LEN \
    --micro-batch-size $BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --share-embeddings-and-output-weights \  # 🔑 权重共享
    --tokenizer-type $TOKENIZER_TYPE \
    --lr 1.5e-4 \
    --train-iters 500000 \
    --save-interval 1000 \
    --eval-interval 100 \
    --data-path $DATA_PATH \
    --split 99,1,0 \
    --distributed-backend nccl \
    --fp16  # or --bf16
```

---

### 附录E: 调试检查清单

**词汇表并行调试步骤**:

- [ ] **配置检查**
  - [ ] 词汇表大小能被TP整除
  - [ ] Embedding和输出层使用相同的 `vocab_size`
  - [ ] 权重共享配置正确（如果启用）

- [ ] **前向传播检查**
  - [ ] Embedding输出形状: `[s, b, d]` (AllReduce后) 或 `[s/p, b, d]` (ReduceScatter)
  - [ ] 输出logits形状: `[s, b, V/p]`
  - [ ] 所有GPU的 `vocab_start_index` 和 `vocab_end_index` 不重叠

- [ ] **损失计算检查**
  - [ ] 交叉熵损失不是NaN
  - [ ] 损失值在合理范围 (0.1-10.0)
  - [ ] 所有GPU的损失值相同（应该完全一致）

- [ ] **反向传播检查**
  - [ ] Embedding权重梯度形状: `[V/p, d]`
  - [ ] 输出层权重梯度形状: `[V/p, d]`
  - [ ] 梯度不是None或NaN
  - [ ] 梯度范数在合理范围 (< 100)

- [ ] **性能检查**
  - [ ] AllReduce通信时间 < 5 ms
  - [ ] 总训练时间与无词汇表并行相比差异 < 5%
  - [ ] 内存使用量降低约 $2Vd/p$ bytes

**调试命令**:

```bash
# 检查Embedding层
python -c "
from megatron.core.tensor_parallel import VocabParallelEmbedding
import torch
torch.distributed.init_process_group(backend='nccl')
emb = VocabParallelEmbedding(32000, 4096, ...)
print(f'Vocab range: [{emb.vocab_start_index}, {emb.vocab_end_index})')
print(f'Weight shape: {emb.weight.shape}')
"

# Profile通信
python -m torch.distributed.run --nproc_per_node=8 \
    profile_vocab_parallel.py
```

---

**文档结束** 🎉

本文档共约**2,560行**，全面介绍了词汇表并行，涵盖：

- ✅ Embedding层并行（掩码+AllReduce）
- ✅ 输出层并行（列并行）
- ✅ **并行交叉熵损失**（核心难点）
- ✅ Megatron-LM完整代码解析（500+行分析）
- ✅ 性能分析与优化策略
- ✅ 5个深度面试问题
- ✅ 调试技巧与最佳实践

**下一步**：
- 编写文档61（流水线并行基础）
- 更新TODO.md标记文档60为已完成 ✅

**项目进度**：60/100 (60%)
**张量并行系列**：56-60 **全部完成** ✅

---

**© 2025 大语言模型预训练研究著作项目**
**基于 Megatron-LM v0.12.0**

