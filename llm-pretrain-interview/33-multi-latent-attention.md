# 33. Multi-Latent Attention (MLA) 详解

> **文档编号**: 33
> **所属部分**: 第四部分 - 高级注意力机制 (31-40)
> **对应原文档**: 03-attention-mechanisms.md Section 8
> **代码位置**: `megatron/core/transformer/multi_latent_attention.py`
> **代码锚点**: 基于 Megatron-LM 当前仓库的 MLA 相关实现，并结合 DeepSeek-V2 论文背景说明

---

## 1. 引言

Multi-Latent Attention (MLA) 是 DeepSeek-V2 (2024) 提出的一种压缩 KV Cache 的注意力机制。按 DeepSeek-V2 公开配置的“每 token 缓存元素数”口径，MLA 可把持久 KV Cache 降低到传统 Multi-Head Attention 的一小部分，从而把长上下文推理的主要瓶颈从显存容量转向 kernel、调度和带宽效率。

本文档将详细讲解 MLA 的设计动机、数学原理、Absorption 优化技术,以及 Megatron-LM 中的具体实现。

---

## 2. 动机: 进一步压缩 KV Cache

### 2.1 问题陈述

即使使用 Grouped Query Attention (GQA),大模型的 KV Cache 仍然是推理的主要瓶颈。

### 2.2 同宽度 GQA-8 对照的 KV Cache 分析

**模型配置**:
- 隐藏维度 $H = 5120$
- 查询头数 $n_h = 128$
- 每头维度 $d_k = 128$
- GQA 组数 $n_g = 8$

以下计算是为了说明“若同宽度模型使用 GQA-8”，KV Cache 会达到什么量级；DeepSeek-V2 本身使用的是 MLA。

**KV Cache 大小** (每个 token):
$$\text{KV}_{\text{per\_token}} = 2 \times n_g \times d_k = 2 \times 8 \times 128 = 2048 \text{ floats}$$

**总 KV Cache** (序列长度 32K, batch size 32):
$$\text{KV}_{\text{total}} = 32 \times 32K \times 2048 \times 2 = 4.3 \text{ GB (BF16)}$$

**问题**:
- 4.3 GB 仅是单层的 KV Cache
- 完整模型 (80 层) 需要约 344 GB
- 严重限制了推理的吞吐量和上下文长度

### 2.3 MLA 的核心思想

通过以下三个步骤压缩 KV Cache:

1. **低秩投影**: 将 $H$ 维隐藏状态投影到低维 latent 空间 ($d_r \ll H$)
2. **缓存 latent**: 只缓存低维 latent,而不是完整的 KV
3. **解压缩**: 需要时从 latent 重构 KV

---

## 3. MLA 的数学定义

### 3.1 符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $X$ | 输入隐藏状态 | $\mathbb{R}^{S \times H}$ |
| $H$ | 隐藏维度 | 5120 |
| $d_r^Q$ | Query latent 维度 | 1536 |
| $d_r^{KV}$ | KV latent 维度 | 512 |
| $d_{\text{qk}}$ | QK 内容维度 | 192 |
| $d_v$ | Value 维度 | 128 |
| $d_{\text{rope}}$ | RoPE 维度 | 64 |
| $n_h$ | 查询头数 | 128 |

### 3.2 Query 分支 (可选低秩)

$$
\begin{aligned}
C^Q &= X W_{\text{down}}^Q \in \mathbb{R}^{S \times d_r^Q} \quad &\text{(Down projection)} \\
C^Q &= \text{RMSNorm}(C^Q) \\
Q &= C^Q W_{\text{up}}^Q \in \mathbb{R}^{S \times (n_h \cdot d_{\text{qk}})} \quad &\text{(Up projection)}
\end{aligned}
$$

**参数量**:
$$\text{Params}^Q = H \times d_r^Q + d_r^Q \times n_h \times d_{\text{qk}}$$

### 3.3 KV 分支 (强制低秩)

$$
\begin{aligned}
C^{KV} &= X W_{\text{down}}^{KV} \in \mathbb{R}^{S \times (d_r^{KV} + d_{\text{rope}})} \quad &\text{(Down projection)} \\
C^{KV}_{\text{latent}} &= C^{KV}[:, :d_r^{KV}] \in \mathbb{R}^{S \times d_r^{KV}} \\
C^{KV}_{\text{rope}} &= C^{KV}[:, d_r^{KV}:] \in \mathbb{R}^{S \times d_{\text{rope}}} \\
C^{KV}_{\text{latent}} &= \text{RMSNorm}(C^{KV}_{\text{latent}}) \\
KV &= C^{KV}_{\text{latent}} W_{\text{up}}^{KV} \in \mathbb{R}^{S \times (n_h \cdot (d_{\text{qk}} + d_v))} \quad &\text{(Up projection)}
\end{aligned}
$$

**关键点**:
- $C^{KV}_{\text{rope}}$ 不经过 LayerNorm,直接用于 RoPE
- 只有 $C^{KV}_{\text{latent}}$ 被缓存

**参数量**:
$$\text{Params}^{KV} = H \times (d_r^{KV} + d_{\text{rope}}) + d_r^{KV} \times n_h \times (d_{\text{qk}} + d_v)$$

### 3.4 分离并应用 RoPE

$$
\begin{aligned}
Q &= Q.\text{view}(S, n_h, d_{\text{qk}} + d_{\text{rope}}) \\
Q_{\text{content}}, Q_{\text{rope}} &= Q.\text{split}([d_{\text{qk}}, d_{\text{rope}}], \text{dim}=-1) \\
Q_{\text{rope}} &= \text{RoPE}(Q_{\text{rope}}, \text{pos\_ids}) \\
Q &= [Q_{\text{content}}, Q_{\text{rope}}] \in \mathbb{R}^{S \times n_h \times (d_{\text{qk}} + d_{\text{rope}})}
\end{aligned}
$$

类似地处理 $K, V$:
$$
\begin{aligned}
K_{\text{content}}, V &= KV.\text{split}([d_{\text{qk}}, d_v], \text{dim}=-1) \\
K_{\text{rope}} &= \text{RoPE}(C^{KV}_{\text{rope}}, \text{pos\_ids}).\text{expand}(n_h, d_{\text{rope}}) \\
K &= [K_{\text{content}}, K_{\text{rope}}] \in \mathbb{R}^{S \times n_h \times (d_{\text{qk}} + d_{\text{rope}})}
\end{aligned}
$$

**特殊处理**: $K_{\text{rope}}$ 从 latent 的 $C^{KV}_{\text{rope}}$ 扩展而来,所有头共享相同的 RoPE 部分。

### 3.5 注意力计算

$$\text{Output} = \text{Attention}(Q, K, V)$$

标准的 Scaled Dot-Product Attention,与 Multi-Head Attention 相同。

---

## 4. 参数量和 KV Cache 对比

### 4.1 参数量对比

**DeepSeek-V2 配置**: $H = 5120$, $n_h = 128$, $d_{\text{qk}} = 192$, $d_v = 128$, $d_{\text{rope}} = 64$, $d_r^Q = 1536$, $d_r^{KV} = 512$

| 模块 | MHA | GQA-8 | MLA |
|------|-----|-------|-----|
| Q 投影 | $H \times n_h d_k$ | $H \times n_h d_k$ | $H \times d_r^Q + d_r^Q \times n_h (d_{\text{qk}} + d_{\text{rope}})$ |
| KV 投影 | $2 H \times n_h d_k$ | $2 H \times n_g d_k$ | $H \times (d_r^{KV} + d_{\text{rope}}) + d_r^{KV} \times n_h (d_{\text{qk}} + d_v)$ |
| **总计** | $3 H^2$ | $H^2 (n_h + 2n_g) / n_h$ | 取决于低秩维度 |

**具体数值**:
- **MHA**: $3 \times 5120^2 = 78.6M$ 参数
- **GQA-8**: $5120^2 \times (128 + 16) / 128 = 29.5M$ 参数
- **MLA**: $(5120 \times 1536 + 1536 \times 128 \times 256) + (5120 \times 576 + 512 \times 128 \times 320) \approx 82.1M$ 投影参数

在这组维度口径下，MLA 的投影参数并不比 MHA/GQA 更少。MLA 的主要优势是持久 KV Cache 变小，而不是单层投影参数一定下降。

### 4.2 KV Cache 大小对比

**每个 token 的 KV Cache**:

| 模型 | KV Cache 大小 | 相对 MHA |
|------|--------------|----------|
| MHA | $2 \times n_h \times d_k = 2 \times 128 \times 128 = 32768$ | 100% |
| GQA-8 | $2 \times n_g \times d_k = 2 \times 8 \times 128 = 2048$ | 6.25% |
| MLA | $d_r^{KV} + d_{\text{rope}} = 512 + 64 = 576$ | **1.76%** |

**压缩比**:
- MLA 相比 MHA 节省 **98.24%** 的 KV Cache
- MLA 相比 GQA-8 节省 **71.9%** 的 KV Cache

### 4.3 实际内存节省

**场景**: 同宽度 80 层模型, 序列长度 32K, batch size 32

| 模型 | KV Cache (单层) | KV Cache (80 层) |
|------|----------------|------------------|
| MHA | 68.7 GB | 5.5 TB |
| GQA-8 | 4.3 GB | 343.6 GB |
| MLA | 1.2 GB | 96.6 GB |

这个表按 BF16、batch size 32、完整 80 层持久 cache 估算，不考虑 tensor parallel/cache sharding、paged cache、量化和请求调度。MLA 的收益是把 cache 从 TB/数百 GB 量级压低，但是否能单 GPU 承载还取决于 batch、上下文、模型参数和并行切分。

---

## 5. MLA 的 Absorption 优化

### 5.1 问题陈述

推理时仍需解压缩 KV,引入额外计算开销:

**正常路径**:
$$
\begin{aligned}
K &= C^{KV}_{\text{latent}} W_{\text{up}}^K \in \mathbb{R}^{S \times n_h \times d_{\text{qk}}} \\
\text{scores} &= Q K^{\top} = Q (C^{KV}_{\text{latent}} W_{\text{up}}^K)^{\top}
\end{aligned}
$$

每次 decode 都需要计算 $C^{KV}_{\text{latent}} W_{\text{up}}^K$,复杂度 $O(S \times d_r^{KV} \times n_h \times d_{\text{qk}})$。

### 5.2 Absorption 技术

**核心思想**: 预先将 $W_{\text{up}}^K$ 吸收到 Query 中。

**数学推导**:
$$
\begin{aligned}
\text{scores} &= Q (C^{KV}_{\text{latent}} W_{\text{up}}^K)^{\top} \\
&= Q (W_{\text{up}}^K)^{\top} (C^{KV}_{\text{latent}})^{\top} \\
&= \tilde{Q} (C^{KV}_{\text{latent}})^{\top}
\end{aligned}
$$

其中:
$$\tilde{Q} = Q (W_{\text{up}}^K)^{\top} \in \mathbb{R}^{S \times n_h \times d_r^{KV}}$$

**优势**:
- 直接用缓存的 $C^{KV}_{\text{latent}}$ 计算注意力分数
- 避免解压缩 $W_{\text{up}}^K$
- Decode 阶段 (单 token 生成) 复杂度从 $O(S \times d_r^{KV} \times n_h \times d_{\text{qk}})$ 降到 $O(n_h \times d_{\text{qk}} \times d_r^{KV})$

### 5.3 Value 的解压缩

注意力权重计算后,仍需解压缩 Value:

$$
\begin{aligned}
\text{attn\_weights} &= \text{softmax}(\tilde{Q} (C^{KV}_{\text{latent}})^{\top}) \\
V &= C^{KV}_{\text{latent}} W_{\text{up}}^V \in \mathbb{R}^{S \times n_h \times d_v} \\
\text{output} &= \text{attn\_weights} \cdot V
\end{aligned}
$$

但可以进一步优化为:
$$
\text{output} = (\text{attn\_weights} \cdot C^{KV}_{\text{latent}}) W_{\text{up}}^V
$$

先计算加权和 (复杂度 $O(n_h \times S \times d_r^{KV})$),再解压缩 (复杂度 $O(n_h \times d_r^{KV} \times d_v)$)。

---

## 6. Megatron-LM 中的代码实现

### 6.1 Absorption 准备

**文件路径**: `megatron/core/transformer/multi_latent_attention.py:835-895`

```python
def prepare_for_absorption(self):
    """准备 Absorption 优化

    功能:
        1. 分离融合的 LayerNorm + Linear 层
        2. 提取 K 和 V 的 up projection 权重
        3. 在 decode 阶段用于 absorption
    """
    if not hasattr(self, "up_k_weight"):
        with torch.no_grad():
            # 分离融合层
            linear_kv_up_proj_norm, linear_kv_up_proj_linear = (
                split_te_layernorm_column_parallel_linear(
                    self.linear_kv_up_proj,
                    self.config,
                    None,
                    self.linear_kv_up_proj.tp_group
                )
            )

            # 更新 kv_layernorm (原来是 identity)
            self.kv_layernorm = linear_kv_up_proj_norm

            # 保存 linear 层用于 prefill 阶段的解压缩
            self.linear_kv_up_proj_linear = linear_kv_up_proj_linear

            # 提取 KV up projection 权重
            kv_up_weight = self.linear_kv_up_proj.weight  # [n_h * (d_qk + d_v), d_r_kv]
            kv_up_weight = kv_up_weight.view(
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
                self.config.kv_lora_rank,
            )

            # 分离 K 和 V 权重
            self.up_k_weight = kv_up_weight[:, :self.config.qk_head_dim, :]
            # [n_h, d_qk, d_r_kv]
            self.up_v_weight = kv_up_weight[:, self.config.qk_head_dim:, :]
            # [n_h, d_v, d_r_kv]

            # 删除原始融合层
            del self.linear_kv_up_proj
```

**代码解析**:
1. **分离融合层**: `split_te_layernorm_column_parallel_linear` 将 TransformerEngine 的融合层分离为 LayerNorm 和 Linear
2. **提取权重**: 从 `linear_kv_up_proj` 提取 $W_{\text{up}}^K$ 和 $W_{\text{up}}^V$
3. **内存优化**: 删除原始融合层,避免重复存储

### 6.2 Decode 阶段的 Absorption

**文件路径**: `megatron/core/transformer/multi_latent_attention.py:639-658`

```python
# 计算 absorbed query
q_content = torch.einsum("sbhd,hdk->sbhk", q_no_pe, self.up_k_weight)
# q_no_pe: [s, b, n_h, d_qk] - Query 的内容部分 (无 RoPE)
# up_k_weight: [n_h, d_qk, d_r_kv] - K 的 up projection 权重
# q_content: [s, b, n_h, d_r_kv] - Absorbed query

query = torch.cat([q_content, q_pos_emb], dim=-1)
# query: [s, b, n_h, d_r_kv + d_rope]

# KV cache 直接是 compressed latent
key = kv_cached  # [s, b, d_r_kv + d_rope]
value = None  # V 在 attention 后再解压缩
```

**代码解析**:
1. **Einsum 优化**: `sbhd,hdk->sbhk` 等价于 $Q (W_{\text{up}}^K)^{\top}$
2. **拼接 RoPE**: Query 包含 absorbed 内容部分和 RoPE 部分
3. **延迟解压**: Value 设为 `None`,在注意力后再解压

### 6.3 Attention 后解压缩 V

**文件路径**: `megatron/core/transformer/multi_latent_attention.py:305-311`

```python
if self.cache_mla_latents and inference_context.is_decode_only():
    # core_attn_out: [s, b, n_h, d_r_kv] - 注意力加权和 (未解压)
    # up_v_weight: [n_h, d_v, d_r_kv] - V 的 up projection 权重
    core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
    # 输出: [s, b, n_h, d_v]

    core_attn_out = core_attn_out.contiguous()
    core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)
    # 展平: [s, b, n_h * d_v]
```

**代码解析**:
1. **条件判断**: 只在 decode 模式且启用 latent 缓存时执行
2. **Einsum 解压**: `sbhc,hdc->sbhd` 等价于 $\text{weighted\_latent} \times W_{\text{up}}^V$
3. **展平输出**: 将多头输出合并为单个向量

---

## 7. Prefill vs Decode 的不同路径

### 7.1 Prefill 阶段 (首次输入)

**特点**: 输入长度 $S \gg 1$,没有 KV Cache

**计算路径**:
1. 正常解压缩 KV: $K = C^{KV}_{\text{latent}} W_{\text{up}}^K$
2. 标准注意力计算
3. 缓存 $C^{KV}_{\text{latent}}$ 和 $C^{KV}_{\text{rope}}$

**复杂度**:
- 解压缩: $O(S \times d_r^{KV} \times n_h \times d_{\text{qk}})$
- 注意力: $O(S^2 \times n_h \times d_{\text{qk}})$

### 7.2 Decode 阶段 (自回归生成)

**特点**: 每次输入 1 个 token,有 KV Cache

**计算路径**:
1. Absorb K 权重到 Query: $\tilde{Q} = Q (W_{\text{up}}^K)^{\top}$
2. 直接用缓存的 latent 计算注意力
3. 解压缩 Value: $\text{output} = \text{attn\_weights} \times C^{KV}_{\text{latent}} \times W_{\text{up}}^V$

**复杂度**:
- Absorption: $O(n_h \times d_{\text{qk}} \times d_r^{KV})$ (仅新 token)
- 注意力: $O(S \times n_h \times d_r^{KV})$ (用 latent 而非完整 KV)
- 解压 V: $O(n_h \times d_r^{KV} \times d_v)$ (仅新 token 的输出)

### 7.3 对比表

| 阶段 | KV 表示 | Q 处理 | 注意力复杂度 | 内存占用 |
|------|---------|--------|--------------|----------|
| **Prefill** | 完整 KV | 标准 | $O(S^2 n_h d_{\text{qk}})$ | 高 (临时) |
| **Decode** | Latent | Absorbed | $O(S n_h d_r^{KV})$ | 低 (持久) |

---

## 8. MLA 的优缺点

### 8.1 优点

1. **极致的 KV Cache 压缩**: 相比 MHA 节省 >98% 内存
2. **推理吞吐量提升**: 更大的 batch size,更长的上下文
3. **Absorption 优化**: Decode 阶段无解压缩开销 (仅对 K)
4. **参数量不是主要卖点**: 按具体维度配置可能增加或减少投影参数，MLA 的核心收益是持久 KV Cache 变小

### 8.2 缺点

1. **预训练质量略降**: 低秩约束可能限制表达能力
   - DeepSeek-V2 通过增加模型宽度和 MoE 补偿
2. **Prefill 阶段额外计算**: 需要解压缩 KV
   - 但 Prefill 通常是一次性的,影响较小
3. **实现复杂度高**: 需要特殊的 kernel 支持
   - Megatron-LM 通过 TransformerEngine 优化
4. **内存碎片化**: Latent 和 RoPE 部分需要分别管理

### 8.3 适用场景

| 场景 | 推荐度 | 原因 |
|------|--------|------|
| **长上下文推理 (128K+)** | ⭐⭐⭐⭐⭐ | KV Cache 是主要瓶颈 |
| **大 batch size 推理服务** | ⭐⭐⭐⭐⭐ | 内存节省允许更大 batch |
| **内存受限的推理部署** | ⭐⭐⭐⭐ | 可在更小 GPU 上运行 |
| **短上下文推理 (<2K)** | ⭐⭐ | KV Cache 不是瓶颈,额外复杂度不值得 |
| **训练** | ⭐⭐⭐ | 参数量减少,但需要更大模型补偿质量损失 |

---

## 9. MLA 与 GQA 的对比

### 9.1 压缩策略

| 方法 | 压缩维度 | 压缩方式 | 压缩比 (DeepSeek-V2) |
|------|----------|----------|---------------------|
| **GQA** | 头数 | 组共享 KV | 6.25% (相比 MHA) |
| **MLA** | 特征维度 | 低秩投影 | 1.76% (相比 MHA) |

### 9.2 质量 vs 内存权衡

**理论分析**:
- **GQA**: 仅减少头数,每头的表达能力不变
- **MLA**: 强制低秩约束,可能损失信息

**论文结论** (DeepSeek-V2 报告):
- DeepSeek-V2 报告的重点是 MLA 在大幅降低 KV Cache 的同时保持有竞争力的训练质量。
- 具体困惑度差值依赖模型宽度、MoE配置、训练数据和token budget；不应把单一数值当作通用常数。

### 9.3 工程复杂度

| 方面 | GQA | MLA |
|------|-----|-----|
| **实现复杂度** | 简单 (只需改 `repeat_interleave`) | 复杂 (需要 Absorption, 两阶段处理) |
| **Kernel 支持** | 标准 kernel 即可 | 需要融合 Einsum kernel |
| **调试难度** | 低 | 中等 |

---

## 10. 总结

### 10.1 关键要点

1. **低秩压缩**: MLA 通过 down-up projection 将 KV Cache 压缩到原来的 1.76%
2. **Absorption 优化**: Decode 阶段预先吸收 K 的 up projection,避免解压缩开销
3. **两阶段处理**: Prefill 使用完整 KV,Decode 使用 latent
4. **RoPE 特殊处理**: RoPE 部分不经过 LayerNorm,直接从 down projection 提取

### 10.2 与其他文档的关系

- **文档 23 (缩放点积注意力)**: MLA 的注意力计算与标准方法相同
- **文档 24 (多头注意力)**: MLA 是多头注意力的变种
- **文档 28 (RoPE详解)**: MLA 需要特殊的 RoPE 维度排列
- **文档 31 (GQA详解)**: MLA 和 GQA 是两种不同的 KV Cache 压缩策略
- **文档 40 (KV Cache详解)**: MLA 改变了 KV Cache 的存储内容

### 10.3 进一步学习

- **低秩分解理论**: 理解为什么 KV 可以被低秩近似
- **DeepSeek-V2 论文**: 完整的实验结果和消融研究
- **Einsum 优化**: 如何编写高效的 Einsum kernel

---

## 11. 总结与最佳实践

### 11.1 工程要点

- MLA的核心收益来自KV Cache低秩压缩。
- Decode阶段应尽量使用absorption减少重复解压缩。
- RoPE部分和latent部分的维度管理必须清晰，否则容易出现shape和position错误。

### 11.2 调优建议

先验证标准MHA/GQA baseline，再引入MLA；每次改变latent维度、RoPE维度或absorption路径，都需要重新检查困惑度、吞吐和KV Cache占用。

## 12. 参考文献

1. DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model". arXiv:2405.04434.
2. Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP.
3. Shazeer (2019). "Fast Transformer Decoding: One Write-Head is All You Need". arXiv:1911.02150.
4. Megatron-LM Documentation: `megatron/core/transformer/multi_latent_attention.py`

---

## 附录 A：MLA 维度审查

MLA 的实现比 GQA 更容易出 shape 错，因为它同时拆分 latent、RoPE 和 value 维度。

| 符号 | 含义 | 典型审查 |
|------|------|----------|
| $H$ | hidden size | 输入输出主维度 |
| $n_h$ | query heads | 与 Q 输出维度相关 |
| $d_r^Q$ | query latent dim | 影响 Q down/up projection |
| $d_r^{KV}$ | KV latent dim | 影响持久 cache |
| $d_{\text{qk}}$ | content QK dim | 参与 content score |
| $d_{\text{rope}}$ | RoPE dim | 参与位置 score |
| $d_v$ | value dim | 影响 attention 输出 |

MLA 的 Q/K score 可以拆成两部分：

```text
score = q_content @ k_content^T + q_rope @ k_rope^T
```

因此必须保证：

- content 部分维度一致。
- RoPE 部分维度一致。
- RoPE 只应用于对应维度。
- cache 保存 latent 和 RoPE key 所需的信息。
- output projection 能接收解压后的 value 表示。

## 附录 B：参数量与 cache 分开看

MLA 的核心收益是 cache，不是所有配置下的参数量下降。建议把两类指标分开记录：

| 指标 | 公式口径 | 解释 |
|------|----------|------|
| Q projection params | $H d_r^Q + d_r^Q n_h(d_{\text{qk}}+d_{\text{rope}})$ | Query低秩是否省参数 |
| KV projection params | $H(d_r^{KV}+d_{\text{rope}})+d_r^{KV}n_h(d_{\text{qk}}+d_v)$ | KV低秩投影成本 |
| cache elems/token | $d_r^{KV}+d_{\text{rope}}$ | 持久cache核心指标 |
| MHA cache elems/token | $2n_hd_k$ | 对照基准 |
| GQA cache elems/token | $2n_gd_k$ | GQA对照 |

调参时不要用“参数量下降”解释所有收益。一个 MLA 配置可能投影参数更多，但长上下文 decode 仍然更省显存。

## 附录 C：Absorption 的直觉

普通 decode 路径需要从 latent 还原 K：

$$
K = C^{KV} W^K_{\text{up}}
$$

注意力 score 为：

$$
QK^\top = Q (C^{KV} W^K_{\text{up}})^\top
$$

可以重排为：

$$
QK^\top = (Q (W^K_{\text{up}})^\top) (C^{KV})^\top
$$

这就是 absorption 的核心：把 K 的 up projection “吸收”到 query 侧，decode 时直接让 query 与 cached latent 做点积。

收益：

- 避免每步为所有历史 token 解压 K。
- 持久 cache 保持 latent 表示。
- decode 更接近带低秩 key 的注意力。

限制：

- RoPE 部分不能简单吸收到 content latent 中。
- Value 解压仍要在 attention 后处理。
- Prefill 阶段为了高效训练/计算，可能仍使用不同路径。

## 附录 D：Prefill 与 Decode 路径

| 项目 | Prefill | Decode |
|------|---------|--------|
| query length | 长 | 通常为1 |
| key length | 长 | 历史长度 |
| 主要瓶颈 | attention计算和临时显存 | cache读带宽 |
| K处理 | 可临时解压完整K | 尽量使用absorption |
| V处理 | 可临时解压 | attention后解压或融合 |
| RoPE | 批量位置 | 当前position offset |

测试时必须分别测：

```text
prefill latency
decode latency
time to first token
inter-token latency
peak temporary memory
persistent kv cache memory
```

只报告总 tokens/sec 会掩盖 MLA 在不同阶段的收益和开销。

## 附录 E：Megatron 审查路径

| 目标 | 文件 |
|------|------|
| MLA 模块 | `megatron/core/transformer/multi_latent_attention.py` |
| SelfAttention/RoPE路径 | `megatron/core/transformer/attention.py` |
| DotProductAttention对照 | `megatron/core/transformer/dot_product_attention.py` |
| Transformer配置 | `megatron/core/transformer/transformer_config.py` |

审查问题：

1. `d_r^{KV}` 与 cache tensor 最后一维是否一致。
2. RoPE 维度是否从 latent 中正确分离。
3. absorption 权重是否与 checkpoint 中 up projection 对齐。
4. prefill 与 decode 是否使用同一数学语义。
5. inference context 是否保存 compressed latent，而不是完整 K/V。
6. dtype cast 是否在 latent、RoPE、value 路径中一致。

## 附录 F：MLA 与 GQA 选择

| 场景 | 更倾向 GQA | 更倾向 MLA |
|------|------------|------------|
| 需要低风险落地 | 是 | 否 |
| 已有 MHA checkpoint | 是 | 否 |
| 极长上下文 serving | 可能 | 是 |
| kernel生态成熟 | 是 | 取决于实现 |
| 能重新预训练 | 可选 | 更可行 |
| 只做短上下文 | 是 | 通常不值得 |
| cache 显存是第一瓶颈 | 可选 | 是 |

如果团队没有成熟 MLA kernel 和 checkpoint 工具，先使用 GQA 往往更稳。MLA 更适合从模型设计阶段就纳入，而不是训练后临时替换。

## 附录 G：质量风险

MLA 的低秩约束可能影响表达能力。需要重点观察：

| 指标 | 风险信号 |
|------|----------|
| training loss | 相同token下持续高于baseline |
| validation loss | 低秩导致欠拟合 |
| long-context eval | 位置和cache交互异常 |
| retrieval task | 长距离key信息损失 |
| generation diversity | 输出变窄或重复 |
| expert load | MoE场景下专家补偿异常 |

消融建议：

1. 固定模型宽度，改变 $d_r^{KV}$。
2. 固定 $d_r^{KV}$，改变 $d_{\text{rope}}$。
3. 比较 absorption on/off 的 decode 速度和数值一致性。
4. 比较 GQA baseline 与 MLA candidate。
5. 分别报告 prefill 和 decode。

## 附录 H：cache 估算模板

```text
B = batch size
S = cached sequence length
L = number of layers
dtype_bytes = 2 for BF16/FP16
mha_elems = 2 * num_heads * head_dim
gqa_elems = 2 * num_query_groups * head_dim
mla_elems = kv_lora_rank + qk_rope_head_dim

cache_bytes = B * S * L * elems_per_token * dtype_bytes
```

报告时要写清楚：

- 是逻辑总量还是每 GPU。
- 是否按 TP 切分。
- 是否使用 paged cache。
- 是否使用 cache 量化。
- batch size 是最大并发还是当前请求数。
- sequence length 是 prompt 长度、生成后总长度还是上限。

## 附录 I：常见故障

| 症状 | 可能原因 | 修复建议 |
|------|----------|----------|
| shape mismatch | latent/RoPE split 错 | 打印每一步维度 |
| cache 没变小 | 保存了完整 K/V | 检查 inference context |
| decode 慢 | absorption 未生效 | profile K解压路径 |
| prefill 慢 | 额外 projection 开销 | 检查融合 kernel |
| loss 高 | latent rank 太低 | 增大 $d_r^{KV}$ |
| 长上下文错 | RoPE offset 错 | 对比 cache on/off |
| checkpoint错 | up/down权重排列错 | 写转换单测 |
| dtype NaN | scale/cast 不一致 | 检查混合精度边界 |

## 附录 J：面试题

**MLA 和 low-rank attention 是一回事吗？**

MLA 使用低秩 latent 压缩 KV 表示，但最终注意力仍要表达 query 与 key/value 的交互。它不是简单把 attention matrix 做低秩近似。

**为什么 cache 可以存 latent？**

因为 K/V 可以由 latent 通过 up projection 重构，decode 时还能通过 absorption 避免显式重构全部历史 K。

**RoPE 部分为什么单独处理？**

位置相位需要直接参与 QK 点积。如果把 RoPE 信息完全压进 latent，可能破坏相对位置结构。

**MLA 是否总比 GQA 好？**

不是。MLA cache 更小，但实现更复杂，质量和 kernel 支持都需要验证。

**为什么说参数量不是 MLA 的核心收益？**

低秩 down/up projection 的参数量取决于 rank 和头维度。某些配置下投影参数可能高于 MHA/GQA，但持久 cache 仍明显更小。

## 附录 K：上线前检查

1. cache 中保存的是 latent + RoPE 所需部分。
2. prefill 与 decode logits 在短序列上可对齐。
3. absorption on/off 的数值差异在容忍范围。
4. RoPE offset 在 chunked prefill 中正确。
5. cache 估算标明 batch、sequence、layer、dtype。
6. 投影参数量与 cache 元素数分开报告。
7. MLA 与 GQA baseline 使用相同 token budget。
8. kernel 支持当前 dtype 和并行配置。
9. checkpoint 保存 down/up projection 和 absorption 所需权重。
10. 文档中所有数值都标明是公式估算还是论文报告。

## 附录 L：实现验证顺序

MLA 不适合直接从全规模训练开始验证。建议顺序：

| 阶段 | 配置 | 目标 |
|------|------|------|
| 单层 CPU/FP32 | 极小 shape | 验证公式和shape |
| 单层 GPU/BF16 | 小 batch | 验证 dtype 和 kernel |
| 多层小模型 | 短序列 | 验证训练 loss |
| cache on/off | decode短文本 | 验证推理一致性 |
| absorption on/off | 同一输入 | 验证重排正确 |
| 长上下文 | 目标长度 | 验证 cache 和 RoPE |
| serving profile | 真实batch | 验证吞吐和显存 |

每个阶段失败时都不要继续扩大规模。MLA 的错误常常不会在 shape 层面暴露，而是在长上下文或 decode cache 中表现为质量退化。

## 附录 M：数值对齐测试

短序列下可以做三种对齐：

1. 标准解压 K/V 路径 vs absorption 路径。
2. prefill 全量计算 vs prefill 后逐 token decode。
3. cache disabled vs cache enabled。

记录字段：

```text
max_abs_logit_diff:
mean_abs_logit_diff:
relative_output_diff:
dtype:
sequence_length:
batch_size:
use_absorption:
use_cache:
```

如果 FP32 下差异已经很大，优先查公式和权重排列；如果只有 BF16/FP16 下差异大，再查 cast、scale 和 kernel。

## 附录 N：发布说明必须包含

1. MLA cache 元素数口径。
2. latent rank 和 RoPE dim。
3. 与 GQA baseline 的质量对比。
4. prefill/decode 分阶段性能。
5. 是否依赖特定 TransformerEngine 或自定义 kernel。
6. checkpoint 是否能被非 MLA 推理框架加载。
7. 长上下文任务的评估集。
8. 已知失败模式和回滚配置。

## 附录 O：文档数值审查规则

MLA 文档中最容易混淆三类数字：

| 数字 | 含义 | 容易误写成 |
|------|------|------------|
| cache元素数 | 每token每层持久缓存 | 参数量收益 |
| projection参数 | down/up projection权重 | cache收益 |
| 端到端显存 | 参数+cache+临时激活 | 单独KV Cache |

审查规则：

1. 写 cache 比例时，明确对照是 MHA、GQA 还是 MQA。
2. 写 GB 时，明确 batch、sequence、layer、dtype。
3. 写“单 GPU可承载”时，必须同时列出模型参数和并行切分。
4. 写“质量持平”时，必须说明数据集和 token budget。
5. 写“decode更快”时，必须说明是否启用 absorption 和对应 kernel。

## 附录 P：候选答案质量标准

| 问题 | 合格 | 优秀 |
|------|------|------|
| MLA核心 | 低秩压缩KV | 区分latent和RoPE部分 |
| absorption | 避免解压K | 写出矩阵重排 |
| 与GQA区别 | 压缩维度不同 | 比较cache、质量、实现 |
| 参数量 | 可能变化 | 明确不是核心收益 |
| cache估算 | 用元素数公式 | 标明逻辑总量/每GPU |
| 风险 | 实现复杂 | 能列出prefill/decode差异 |

## 附录 Q：最终自检

MLA 文档提交前检查：

1. 是否把 cache 元素数和投影参数分开。
2. 是否说明 DeepSeek-V2 本身使用 MLA，而 GQA 数值只是对照。
3. 是否标明所有 GB 数字的 batch、sequence、layer、dtype。
4. 是否说明 absorption 主要优化 decode。
5. 是否说明 RoPE 部分需要单独处理。
6. 是否避免“单 GPU可承载”这类无并行口径结论。
7. 是否保留 GQA baseline 作为比较对象。
8. 是否说明 MLA 不是所有场景都优于 GQA。

---

**文档版本**: v1.0
**最后更新**: 2026-05-10
**文档状态**: ✅ 已完成
