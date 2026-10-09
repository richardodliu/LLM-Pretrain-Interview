# 31. 分组查询注意力(GQA)详解

> **文档编号**: 31
> **所属部分**: 第四部分 - 高级注意力机制 (31-40)
> **对应原文档**: 03-attention-mechanisms.md Section 7
> **代码位置**: `megatron/core/transformer/attention.py` (num_query_groups参数)
> **代码锚点**: 基于 Megatron-LM 当前仓库的注意力实现，并结合 GQA/MQA 论文背景说明

---

## 1. 引言

Grouped Query Attention (GQA) 是 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 的折中方案,由 Google 在 2023 年提出。它通过将查询头分组共享 KV 头,在保持模型质量的同时大幅降低推理成本。LLaMA-2、Mistral 等主流大模型都采用了 GQA。

---

## 2. 动机：MHA 和 MQA 的权衡

### 2.1 MHA 的问题

**推理内存占用大**:
- KV cache 大小 = $2 \times B \times S \times n_h \times d_k$
- 对于 LLaMA-70B ($n_h=64, d_k=128$),单个样本 32K 序列的 KV cache 达到 2GB

### 2.2 MQA 的优势与问题

**优势**:
- 所有查询头共享一个 KV 头
- KV cache 大小 = $2 \times B \times S \times 1 \times d_k$
- 在 decode 受 KV Cache 带宽限制时，可能显著降低读写压力

**问题**:
- 质量风险:单个 KV 头限制表达能力
- 困惑度变化依赖模型规模、训练数据和转换/继续训练策略，不能只按固定比例外推

### 2.3 GQA 的折中方案

**核心思想**:
- $n_h$ 个查询头分成 $n_g$ 组,每组共享一个 KV 头
- $n_g = n_h$ 时退化为 MHA
- $n_g = 1$ 时退化为 MQA
- 通常选择 $n_g \in \{4, 8\}$

---

## 3. 数学定义

**定义 3.1**: Grouped Query Attention

给定 $n_h$ 个查询头和 $n_g$ 个 KV 组 ($n_g < n_h$),GQA 定义为:

$$
\begin{aligned}
\text{GQA}(X) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_h}) W^O \\
\text{where } \text{head}_i &= \text{Attention}(XW_i^Q, XW_{g(i)}^K, XW_{g(i)}^V)
\end{aligned}
$$

其中 $g(i) = \lfloor (i-1) / (n_h / n_g) \rfloor + 1$ 是查询头 $i$ 对应的 KV 组索引。

### 3.1 参数量对比

| 模型 | Q 投影 | K 投影 | V 投影 | 总参数量 |
|------|--------|--------|--------|----------|
| MHA | $H^2$ | $H^2$ | $H^2$ | $3H^2$ |
| GQA | $H^2$ | $\frac{n_g}{n_h} H^2$ | $\frac{n_g}{n_h} H^2$ | $H^2 (n_h + 2n_g) / n_h$ |
| MQA | $H^2$ | $\frac{H^2}{n_h}$ | $\frac{H^2}{n_h}$ | $H^2 (n_h + 2) / n_h$ |

**示例**: 以 LLaMA-2 70B 的隐藏维度作同尺寸对照
- MHA 对照 ($n_g=64$): $3 \times 8192^2 = 201M$ 参数
- GQA ($n_g=8$): $8192^2 \times 80/64 = 84M$ 参数 (**节省 58%**)
- MQA ($n_g=1$): $8192^2 \times 66/64 = 69M$ 参数 (节省 66%)

---

## 4. 计算过程

### 4.1 投影与重塑

**步骤 1**: 投影到 QKV
$$
\begin{aligned}
Q &= XW^Q \in \mathbb{R}^{S \times n_h d_k} \\
K &= XW^K \in \mathbb{R}^{S \times n_g d_k} \\
V &= XW^V \in \mathbb{R}^{S \times n_g d_k}
\end{aligned}
$$

**步骤 2**: 重塑为多头格式
$$Q \to [S, n_h, d_k], \quad K \to [S, n_g, d_k], \quad V \to [S, n_g, d_k]$$

### 4.2 KV 扩展

**步骤 3**: 扩展 KV 以匹配查询头数

每个 KV 组被复制 $n_h / n_g$ 次:
$$K' = \text{repeat}(K, n_h/n_g) \in \mathbb{R}^{S \times n_h \times d_k}$$
$$V' = \text{repeat}(V, n_h/n_g) \in \mathbb{R}^{S \times n_h \times d_k}$$

**步骤 4**: 标准注意力计算
$$\text{Output} = \text{Attention}(Q, K', V')$$

### 4.3 Megatron-LM 实现

**文件路径**: `megatron/core/transformer/dot_product_attention.py`

```python
# GQA: 扩展 key 和 value 以匹配查询头数
if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
    key = key.repeat_interleave(
        self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        dim=2  # 头维度
    )
    value = value.repeat_interleave(
        self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        dim=2
    )
```

**实现含义**: 这条通用 PyTorch 路径会把 KV 在本地扩展到查询头数，以复用标准注意力计算。高性能推理或融合 attention kernel 可以把“逻辑扩展”下沉到 kernel 内部，避免把完整展开后的 KV 作为长期 cache 存储。

---

## 5. 张量并行下的 GQA

### 5.1 挑战

当 $n_g < \text{TP size}$ 时,每个 TP rank 无法完整拥有一个 KV 组。

**示例**:
- `num_query_groups = 2`
- `tensor_parallel_size = 4`
- 每个 TP rank 只能拥有 0.5 个 KV 组

### 5.2 解决方案

**文件路径**: `megatron/core/transformer/attention.py`

```python
if self.config.num_query_groups < world_size:
    # 情况1: num_kv_heads < tp_size
    self.num_query_groups_per_partition = 1
    self.num_attention_heads_per_partition = divide(
        self.config.num_attention_heads, self.config.num_query_groups
    )
else:
    # 情况2: num_kv_heads >= tp_size
    self.num_query_groups_per_partition = divide(
        self.config.num_query_groups, world_size
    )
    self.num_attention_heads_per_partition = divide(
        self.config.num_attention_heads, world_size
    )
```

### 5.3 QKV 分离逻辑

**文件路径**: `megatron/core/transformer/attention.py`

```python
if self.config.num_query_groups < self.world_size:
    # 步骤1: All-Gather 得到完整的 QKV
    mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)

    # 步骤2: 选择本 rank 负责的 KV 组
    idx = get_tensor_model_parallel_rank() // (
        self.world_size // self.config.num_query_groups
    )
    size = mixed_qkv.size()[-1] // self.config.num_query_groups
    mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

    # 步骤3: 从查询头中选择本 rank 负责的部分
    idx = get_tensor_model_parallel_rank() % (
        self.world_size // self.config.num_query_groups
    )
    size = self.num_attention_heads_per_partition // (
        self.world_size // self.config.num_query_groups
    )
    query = query[:, :, idx * size : (idx + 1) * size, :]
```

**示例** ($n_g=2, TP=4, n_h=8$):

| TP Rank | Query 头 | KV 组 |
|---------|----------|-------|
| 0 | q1 | k1, v1 |
| 1 | q2 | k1, v1 |
| 2 | q5 | k2, v2 |
| 3 | q6 | k2, v2 |

---

## 6. 实验结果

### 6.1 公开结论与示例口径

GQA 论文和后续开源模型的共同结论是：从 MHA 迁移到适中的 KV 组数通常能显著降低 KV Cache，同时比 MQA 更容易保持质量。具体困惑度和吞吐量必须按模型、数据、kernel、batch size 和上下文长度重新测量。

| 配置 | KV Cache 理论比例 | 质量风险 | 典型用途 |
|------|-------------------|----------|----------|
| MHA | 100% | 最低 | 训练基线、短上下文 |
| GQA-8 | 25% 或 12.5%，取决于头数 | 低到中 | 长上下文推理、主流 LLM |
| GQA-4 | 12.5% 或 6.25%，取决于头数 | 中 | 更强内存约束 |
| MQA | $1/n_h$ | 最高 | 极限低 cache 实验或小模型 |

表中的比例只来自 KV 头数公式，不代表固定的端到端加速比例。

### 6.2 推理内存节省

| 模型 | MHA (GB) | GQA-8 (GB) | GQA-4 (GB) | MQA (GB) |
|------|----------|-----------|-----------|----------|
| LLaMA-2 7B (32K) | 2.0 | 0.5 | 0.25 | 0.0625 |
| LLaMA-2 70B (32K) | 16.0 | 4.0 | 2.0 | 0.5 |

---

## 7. 总结

### 7.1 关键要点

1. **参数-质量权衡**: GQA 是 MHA 和 MQA 之间的最佳折中
2. **推理加速**: 通过减少 KV cache 和内存带宽压力提高 decode 吞吐，实际幅度由 serving kernel 和请求形态决定
3. **张量并行兼容**: Megatron-LM 通过 All-Gather 支持 $n_g < TP$ 的情况

### 7.2 与其他文档的关系

- **文档 24 (Multi-Head Attention)**: GQA 的基础
- **文档 32 (MQA详解)**: GQA 的极端情况
- **文档 33 (MLA详解)**: 另一种 KV cache 压缩方法
- **文档 40 (KV Cache详解)**: GQA 优化的主要目标

---

## 8. 消融研究

### 8.1 KV组数消融

KV组数越少，KV Cache越小，推理吞吐越高；但共享KV会降低每个query head的独立表示能力，需要在困惑度和吞吐之间折中。

### 8.2 从MHA转换到GQA

从已有MHA checkpoint转换为GQA时，常见方法是对同组KV头做平均或选择代表头，然后继续训练恢复质量。

## 9. 超参数分析

### 9.1 `num_query_groups`

`num_query_groups` 必须与 attention head 数、TP size 兼容。若小于TP size，Megatron需要额外AllGather和本地切片逻辑。

### 9.2 Batch与上下文长度

GQA收益随 batch size 和 sequence length 增大而更明显，因为KV Cache和内存带宽压力更高。

## 10. 深入探讨

### 10.1 GQA与MQA

MQA是 `num_query_groups=1` 的极端情况。它最大化KV Cache压缩，但通常比GQA质量损失更大。

### 10.2 GQA与MLA

GQA沿头维度共享KV，MLA沿特征维度做低秩压缩。两者都优化KV Cache，但复杂度和质量权衡不同。

## 11. 总结与最佳实践

- 默认从GQA-8或GQA-4等中间配置开始评估。
- 长上下文和大batch推理更适合GQA。
- TP配置下必须验证 `num_query_groups` 与rank切分逻辑。

## 12. 参考文献

1. Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP.
2. Shazeer (2019). "Fast Transformer Decoding: One Write-Head is All You Need". arXiv:1911.02150.
3. Megatron-LM Documentation: `megatron/core/transformer/attention.py`

---

## 附录 A：GQA 配置合法性

GQA 的第一个工程问题是配置是否可切分。建议按下面顺序检查：

| 字段 | 示例 | 约束 |
|------|------|------|
| `hidden_size` | 8192 | 能被 `num_attention_heads` 整除 |
| `num_attention_heads` | 64 | 查询头总数 |
| `kv_channels` | 128 | 通常等于 `hidden_size / num_attention_heads` |
| `num_query_groups` | 8 | 介于 1 和 `num_attention_heads` 之间 |
| `tensor_model_parallel_size` | 8 | 与 query heads 和 query groups 共同决定切分 |

基本派生量：

```text
head_dim = hidden_size / num_attention_heads
q_heads_per_group = num_attention_heads / num_query_groups
q_heads_per_tp_rank = num_attention_heads / tensor_model_parallel_size
```

如果 `num_query_groups >= tensor_model_parallel_size`，通常每个 TP rank 拥有整数个 KV group。

如果 `num_query_groups < tensor_model_parallel_size`，多个 TP rank 会共享同一个 KV group，Megatron-LM 需要额外的 all-gather 和切片逻辑。

## 附录 B：三种典型切分案例

### 案例 1：MHA 等价配置

```text
num_attention_heads = 64
num_query_groups = 64
tensor_model_parallel_size = 8
q_heads_per_rank = 8
kv_groups_per_rank = 8
```

特点：

- 每个 query head 有独立 KV head。
- KV Cache 最大。
- 实现路径最接近标准 MHA。

### 案例 2：常规 GQA

```text
num_attention_heads = 64
num_query_groups = 8
tensor_model_parallel_size = 8
q_heads_per_rank = 8
kv_groups_per_rank = 1
q_heads_per_kv_group = 8
```

特点：

- 每个 TP rank 负责一个 KV group。
- KV Cache 是 MHA 的 1/8。
- serving 阶段通常更容易获得收益。

### 案例 3：KV group 小于 TP size

```text
num_attention_heads = 64
num_query_groups = 4
tensor_model_parallel_size = 8
q_heads_per_rank = 8
logical_ranks_per_kv_group = 2
```

特点：

- 每个 KV group 横跨多个 TP rank 的 query heads。
- 需要更复杂的 gather/slice。
- 可能降低部分训练吞吐，需要 profile 验证。

## 附录 C：KV Cache 公式

设：

- batch size 为 $B$。
- 已缓存序列长度为 $S$。
- 层数为 $L$。
- KV 头数为 $n_g$。
- 每头维度为 $d_k$。
- dtype 字节数为 $b$。

GQA 的持久 KV Cache 逻辑总量为：

$$
M_{\text{KV}} = 2 \times B \times S \times L \times n_g \times d_k \times b
$$

相对 MHA 的比例：

$$
\frac{M_{\text{GQA}}}{M_{\text{MHA}}} =
\frac{n_g}{n_h}
$$

示例：

```text
B = 8
S = 32768
L = 80
n_h = 64
n_g = 8
d_k = 128
b = 2
M_GQA = 10.7 GB
M_MHA = 85.9 GB
```

这个示例是逻辑总量，不含 tensor parallel cache sharding、paged attention、量化或不同请求长度造成的碎片。

## 附录 D：训练与推理收益不同

| 阶段 | GQA 主要收益 | 主要限制 |
|------|--------------|----------|
| 预训练 | K/V 投影参数和通信略降 | 仍要做完整注意力计算 |
| 长上下文继续训练 | 激活和通信可能受益 | 质量需要验证 |
| Prefill | K/V 读写下降 | $S^2$ attention 仍重 |
| Decode | KV Cache 读带宽显著下降 | kernel 是否利用 GQA |
| Serving调度 | 每请求cache变小 | 请求长度碎片仍存在 |

因此，GQA 的收益在 decode 阶段通常最直观。训练阶段的端到端收益可能被 MLP、通信、数据加载或 FlashAttention kernel 掩盖。

## 附录 E：从 MHA checkpoint 转 GQA

常见转换流程：

1. 读取 MHA checkpoint 的 K/V projection。
2. 按目标 `num_query_groups` 把 head 分组。
3. 对每组 K/V head 做平均、选择代表头或学习映射。
4. 写出新的 GQA K/V projection。
5. 保持 Q 和 output projection 不变或按需要微调。
6. 用较小 LR 继续训练，恢复困惑度和下游指标。

转换策略对比：

| 策略 | 优势 | 风险 |
|------|------|------|
| 平均同组head | 简单稳定 | 可能抹平专门化头 |
| 选择代表head | 保留真实head | 初始质量波动大 |
| 学习映射 | 表达更强 | 需要额外训练 |
| 从零训练GQA | 最自然 | 成本最高 |

转换后必须验证：

- 初始 validation loss 是否跳变。
- 继续训练后是否恢复。
- 长上下文是否退化。
- 推理吞吐是否真的提高。
- checkpoint 格式是否与推理引擎一致。

## 附录 F：Megatron 实现细节

Megatron-LM 中 GQA 相关逻辑分散在两个层面：

| 层面 | 文件 | 作用 |
|------|------|------|
| SelfAttention | `megatron/core/transformer/attention.py` | 计算每个 rank 的 query heads 和 query groups |
| DotProductAttention | `megatron/core/transformer/dot_product_attention.py` | 在通用路径中扩展 K/V |
| Transformer配置 | `megatron/core/transformer/transformer_config.py` | 保存 `num_query_groups` |

审查代码时不要只看 `repeat_interleave`。还要检查：

1. QKV fused projection 的输出维度。
2. `num_query_groups_per_partition` 的计算。
3. `num_attention_heads_per_partition` 的计算。
4. 当 `num_query_groups < world_size` 时的 all-gather。
5. TE attention backend 是否支持当前 GQA 形状。
6. inference context 是否缓存 GQA 形状的 K/V。

## 附录 G：GQA 与 tensor parallel 的通信风险

当 `num_query_groups < tensor_model_parallel_size` 时，理论上 KV Cache 更小，但训练路径可能更复杂：

| 风险 | 说明 |
|------|------|
| all-gather 增加 | 为了让每个 rank 拿到需要的混合 QKV |
| 切片逻辑复杂 | rank 到 KV group 的映射不再一一对应 |
| kernel shape 特殊 | 某些后端对小 KV group 支持较差 |
| checkpoint 转换难 | fused QKV 排列更容易出错 |

建议：

- 优先选择 `num_query_groups` 能被 TP size 整除的配置。
- 如果必须小于 TP size，先做小模型并行单元测试。
- 使用 profile 对比 GQA-8、GQA-4、MQA 的训练 step time。
- 将 GQA 配置写入 checkpoint 元数据和部署配置。

## 附录 H：质量评估

GQA 的质量评估不能只看单个 perplexity：

| 指标 | 目的 |
|------|------|
| validation loss | 训练目标是否恢复 |
| long-context loss | 长序列是否受影响 |
| downstream QA | 语义能力是否保持 |
| generation quality | 采样输出是否退化 |
| attention entropy | 头共享后是否过度集中 |
| retrieval task | 长距离定位是否保持 |

实验设计：

```text
baseline: MHA checkpoint
candidate_1: GQA-8 converted + continued training
candidate_2: GQA-4 converted + continued training
candidate_3: MQA converted + continued training
control: same training tokens, same data order, same LR
```

只有当质量、吞吐和显存三个维度同时满足目标时，才应把 GQA 配置固化到生产模型。

## 附录 I：Serving 视角

GQA 对 serving 的价值主要来自 cache：

| Serving问题 | GQA影响 |
|-------------|---------|
| 最大并发 | cache 变小，可容纳更多请求 |
| 长上下文 | 每请求显存线性系数下降 |
| decode带宽 | 每步读取 K/V 字节减少 |
| batch调度 | 更容易把长短请求混排 |
| prefill峰值 | 仍需关注 attention 临时内存 |
| KV量化 | 可叠加进一步降低cache |

部署验证要分 prefill 和 decode：

```text
prefill_tokens_per_second
decode_tokens_per_second
time_to_first_token
inter_token_latency
peak_kv_cache_memory
max_concurrent_requests
```

如果 GQA 只降低显存但没有提升 decode tokens/sec，常见原因是：

- kernel 没有利用 GQA 形状。
- 瓶颈在采样、调度或网络。
- batch 太小，KV 带宽不是瓶颈。
- cache 量化或 paged attention 已经消除了主要瓶颈。

## 附录 J：与 MLA 的边界

| 维度 | GQA | MLA |
|------|-----|-----|
| 压缩方式 | 减少 KV 头数 | 压缩 KV 特征维 |
| 参数化 | 组共享 K/V | low-rank latent |
| cache内容 | K/V heads | latent + RoPE部分 |
| 实现复杂度 | 中 | 高 |
| checkpoint转换 | 可从MHA平均KV | 更依赖专门结构 |
| 质量风险 | 与组数相关 | 与latent维度相关 |

选择建议：

- 如果已有 MHA/GQA 生态和 kernel，GQA 是更低风险的压缩方案。
- 如果目标是极长上下文和极高并发，MLA 可能提供更强 cache 压缩。
- 不应仅根据 cache 比例选择；要同时比较质量、实现、kernel支持和训练成本。

## 附录 K：常见故障表

| 症状 | 可能原因 | 修复建议 |
|------|----------|----------|
| shape mismatch | `num_query_groups` 不兼容 TP | 调整 groups 或 TP |
| loss 突然升高 | MHA->GQA转换损伤 | 降 LR 继续训练 |
| 推理无加速 | kernel 未利用GQA | 更换 backend 或 profile |
| 显存没下降 | cache 仍按MHA展开保存 | 检查 serving cache layout |
| TP rank 输出不同 | rank到group映射错 | 打印每rank Q/K/V shape |
| checkpoint无法加载 | fused QKV排列变化 | 写显式转换脚本 |
| 长上下文退化 | KV共享过强 | 增大 `num_query_groups` |
| MQA质量差 | 共享过度 | 回退到GQA-4/8 |

## 附录 L：面试题

**GQA 与 MQA 的关系是什么？**

MQA 是 GQA 在 `num_query_groups=1` 时的极端情况。GQA 在 MHA 和 MQA 之间提供连续的 cache/质量权衡。

**GQA 为什么主要提升 decode？**

Decode 每步都要读历史 K/V。GQA 减少持久 K/V 的头数，因此减少显存读带宽。Prefill 仍有 $S^2$ 注意力计算，收益不一定同样明显。

**GQA 会不会减少 Q 的计算？**

通常不会。GQA 保留 query heads，主要减少 K/V projection、K/V cache 和相关读写。

**为什么 `num_query_groups` 太小会损害质量？**

多个 query heads 共享同一组 K/V，降低了每个头独立选择 key/value 表示的能力。

**从 MHA 转 GQA 为什么需要继续训练？**

平均或选择 KV head 会改变模型函数。继续训练用于让 Q/K/V 和后续层重新适配共享 KV 结构。

## 附录 M：实验模板

```yaml
gqa_experiment:
  model:
    hidden_size:
    num_attention_heads:
    kv_channels:
    num_layers:
  candidate:
    num_query_groups:
    conversion_method:
    continued_training_tokens:
  parallel:
    tensor_model_parallel_size:
    pipeline_model_parallel_size:
  serving:
    max_sequence_length:
    batch_size:
    cache_dtype:
    backend:
  metrics:
    validation_loss:
    long_context_loss:
    prefill_tps:
    decode_tps:
    peak_kv_cache:
    max_concurrency:
```

实验报告必须写明：

- cache 数值是逻辑总量还是每 GPU。
- 是否启用 TP/PP/cache sharding。
- 是否使用 paged attention。
- 是否使用 KV cache 量化。
- 吞吐量是否包含采样和网络开销。

## 附录 N：上线前检查

1. `num_query_groups` 在训练和推理中一致。
2. `num_query_groups` 与 TP size 的关系已验证。
3. QKV fused 权重拆分和 checkpoint 转换有单元测试。
4. GQA/MQA 候选都完成相同 token budget 的继续训练。
5. 质量评估覆盖短文本、长文本和目标下游任务。
6. Serving cache layout 确认保存的是 GQA 形状，而非展开后的 MHA 形状。
7. prefill 和 decode 性能分别报告。
8. KV Cache 数值标明 batch、sequence、layer、dtype 和是否按 GPU 切分。
9. 如果使用 TE/FlashAttention，确认 backend 支持当前 GQA 配置。
10. 回滚方案保留 MHA 或更大 `num_query_groups` 的 checkpoint。

## 附录 O：并行映射手算模板

给定：

```text
num_attention_heads = Hq
num_query_groups = Hg
tensor_parallel_size = Nt
tensor_parallel_rank = r
```

先计算：

```text
query_heads_per_group = Hq / Hg
query_heads_per_rank = Hq / Nt
```

当 `Hg >= Nt`：

```text
groups_per_rank = Hg / Nt
rank r owns groups:
  [r * groups_per_rank, (r + 1) * groups_per_rank)
rank r owns query heads:
  [r * query_heads_per_rank, (r + 1) * query_heads_per_rank)
```

当 `Hg < Nt`：

```text
ranks_per_group = Nt / Hg
group_id = r // ranks_per_group
local_slice = r % ranks_per_group
```

这时同一个 KV group 被多个 TP rank 的 query heads 使用。单元测试应打印：

- rank id。
- query head range。
- query group id。
- K/V tensor shape。
- attention output shape。

## 附录 P：性能解释模板

GQA 实验报告建议分解如下：

| 指标 | MHA | GQA候选 | 解释 |
|------|-----|---------|------|
| QKV projection time |  |  | K/V输出维度是否下降 |
| attention core time |  |  | backend是否利用GQA |
| output projection time |  |  | 通常不变 |
| all-gather time |  |  | `Hg < Nt` 时重点 |
| peak memory |  |  | cache和activation分开 |
| decode bandwidth |  |  | 长上下文重点 |
| validation loss |  |  | 质量约束 |

结论模板：

```text
GQA reduced logical KV cache by:
GQA changed training step time by:
GQA changed decode latency by:
Quality delta after equal tokens:
Main remaining bottleneck:
Decision:
```

如果报告只写“GQA 提速 2x”而没有区分 prefill/decode、硬件和 backend，这个结论不能用于生产决策。

## 附录 Q：风险接受标准

| 风险 | 可接受条件 |
|------|------------|
| validation loss 小幅上升 | 下游指标无显著退化，且服务收益明确 |
| decode 加速不明显 | 显存/并发收益满足目标 |
| training step 变慢 | 总训练预算可接受，推理收益足够大 |
| checkpoint转换复杂 | 有自动化测试和回滚 |
| backend支持有限 | 部署环境固定且已验证 |
| 长上下文略退化 | 目标产品不依赖长检索 |

风险接受必须写入模型发布记录。否则后续发现质量问题时，很难判断是 GQA 本身、转换策略还是 serving 实现造成的。

## 附录 R：最小单元测试矩阵

GQA 的单元测试要覆盖普通切分和特殊切分：

| `Hq` | `Hg` | `TP` | 场景 |
|------|------|------|------|
| 8 | 8 | 1 | MHA等价 |
| 8 | 4 | 1 | 单卡GQA |
| 8 | 1 | 1 | MQA |
| 8 | 4 | 2 | groups可被TP切分 |
| 8 | 2 | 4 | `Hg < TP` |
| 16 | 8 | 4 | 常规多rank |
| 16 | 4 | 8 | 多rank共享group |

每个配置检查：

1. Q/K/V shape。
2. attention output shape。
3. backward 是否产生 NaN。
4. checkpoint save/load。
5. cache tensor 的 head 维是否等于 `Hg` 或 per-rank group 数。
6. 与 MHA 对照的 cache 比例是否符合公式。

## 附录 S：文档审查规则

GQA 文档中凡出现性能或质量数字，都必须满足以下要求：

| 数字类型 | 必须说明 |
|----------|----------|
| KV Cache GB | batch、sequence、layer、dtype、是否每GPU |
| 相对比例 | 对照是MHA还是GQA |
| 吞吐提升 | 硬件、backend、prefill/decode |
| 困惑度变化 | 数据集、token budget、是否继续训练 |
| 参数量 | 是否包含output projection |
| 通信开销 | TP/PP/DP配置 |

没有这些上下文的数字只能写成“示例估算”或“公式比例”，不能写成通用结论。

## 附录 T：候选答案质量标准

| 问题 | 合格 | 优秀 |
|------|------|------|
| GQA定义 | 多个Q头共享KV | 能写出head到group映射 |
| cache比例 | $n_g/n_h$ | 能说明逻辑总量和每GPU口径 |
| 与MQA关系 | MQA是极端GQA | 能解释质量风险 |
| TP难点 | heads切分 | 能说明 `Hg < TP` 路径 |
| 转换checkpoint | 平均KV | 能说明继续训练必要性 |
| Serving收益 | cache更小 | 区分prefill和decode |
| 实现风险 | shape错 | 说出QKV fused排列和backend支持 |

## 附录 U：最终自检

GQA 文档提交前检查：

1. 是否把 MQA 作为 `num_query_groups=1` 的特例说明。
2. 是否把 cache 比例写成 $n_g/n_h$，而不是固定百分比。
3. 是否标明 GB 数字的 batch、sequence、layer、dtype。
4. 是否区分逻辑总量和每 GPU 口径。
5. 是否说明 `num_query_groups < TP` 的特殊路径。
6. 是否避免无来源的困惑度和吞吐数值。
7. 是否说明 MHA checkpoint 转 GQA 需要继续训练。
8. 是否说明 serving backend 必须真正保存 GQA 形状 cache。
9. 是否区分 prefill 与 decode 收益。
10. 是否提供回滚到更大 KV group 的策略。

## 附录 V：一页速查

```text
MHA: num_query_groups = num_attention_heads
GQA: 1 < num_query_groups < num_attention_heads
MQA: num_query_groups = 1
KV cache ratio vs MHA = num_query_groups / num_attention_heads
main serving benefit = smaller persistent K/V cache
main quality risk = too many query heads share one KV group
main TP risk = num_query_groups < tensor_parallel_size
```

附加发布判断：

- 若目标是训练吞吐，必须用 profile 证明收益。
- 若目标是推理并发，必须用真实请求长度分布验证。
- 若目标是长上下文，必须包含检索型评估。

---

**文档版本**: v1.0
**最后更新**: 2026-05-10
**文档状态**: ✅ 已完成
