# 28. RoPE位置编码

> **文档编号**: 28
> **所属部分**: 第三部分 - Transformer基础架构 (21-30)
> **对应原文档**: 03-attention-mechanisms.md Section 9
> **代码位置**: `megatron/core/models/common/embeddings/rotary_pos_embedding.py`
> **代码锚点**: 基于 Megatron-LM 当前仓库的 RoPE 实现，并结合位置编码论文背景说明

---

## 1. 引言

Rotary Position Embedding (RoPE) 是一种基于旋转矩阵的位置编码方法,由 Su et al. (2021) 提出。它通过在查询和键向量上应用旋转变换来编码位置信息,同时保持相对位置的特性。RoPE 已被广泛应用于现代大语言模型中,包括 LLaMA、Qwen、Mistral 等主流模型。

本文档将详细讲解 RoPE 的数学原理、与传统位置编码的对比、高效实现方法,以及 Megatron-LM 中的具体代码实现。

---

## 2. 位置编码的必要性

### 2.1 问题陈述

原始的点积注意力对位置不敏感:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$$

如果交换序列中两个位置,注意力输出不变(置换不变性)。

### 2.2 示例

**输入序列**:
- 原始: "I love NLP"
- 交换后: "NLP love I"

**问题**: 注意力权重相同,因为只依赖内容,不依赖位置。

### 2.3 现有解决方案

| 方法 | 提出者 | 核心思想 | 优缺点 |
|------|--------|----------|--------|
| **绝对位置编码** | Vaswani et al. (2017) | 将位置信息加到输入嵌入 | 简单,但外推能力差 |
| **相对位置编码** | Shaw et al. (2018) | 在注意力分数中引入相对位置偏置 | 外推能力好,但需要额外参数 |
| **RoPE** | Su et al. (2021) | 通过旋转矩阵编码位置信息 | 外推能力强,无参数,理论优雅 |

---

## 3. RoPE 的数学推导

### 3.1 核心思想

使用旋转矩阵编码绝对位置,同时保持相对位置信息。

### 3.2 旋转矩阵的定义

**定理 3.1**: RoPE 的构造

对于位置 $m$ 的向量 $\mathbf{x}$,定义旋转函数:
$$f_{\text{RoPE}}(\mathbf{x}, m) = \mathbf{R}_m \mathbf{x}$$

其中 $\mathbf{R}_m$ 是旋转矩阵,满足:
1. $\mathbf{R}_m^{\top} \mathbf{R}_m = I$ (正交性)
2. $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$ (可加性)

### 3.3 二维旋转矩阵

在二维子空间中,旋转矩阵为:
$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

对于位置 $m$,设 $\theta_m = m \cdot \theta_{\text{base}}$,则:
$$\mathbf{R}_m = \begin{bmatrix} \cos(m\theta_{\text{base}}) & -\sin(m\theta_{\text{base}}) \\ \sin(m\theta_{\text{base}}) & \cos(m\theta_{\text{base}}) \end{bmatrix}$$

### 3.4 可加性验证

**证明** $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$:

$$
\begin{aligned}
\mathbf{R}_m \mathbf{R}_n &= \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{bmatrix} \\
&= \begin{bmatrix} \cos((m+n)\theta) & -\sin((m+n)\theta) \\ \sin((m+n)\theta) & \cos((m+n)\theta) \end{bmatrix} = \mathbf{R}_{m+n}
\end{aligned}
$$

这利用了三角函数的和角公式:
- $\cos(m\theta)\cos(n\theta) - \sin(m\theta)\sin(n\theta) = \cos((m+n)\theta)$
- $\sin(m\theta)\cos(n\theta) + \cos(m\theta)\sin(n\theta) = \sin((m+n)\theta)$

### 3.5 扩展到高维

对于 $d$ 维向量 ($d$ 为偶数),将其分为 $d/2$ 对,每对应用不同频率的旋转:

$$\mathbf{R}_m^{(d)} = \begin{bmatrix}
\mathbf{R}_m(\theta_1) & & & \\
& \mathbf{R}_m(\theta_2) & & \\
& & \ddots & \\
& & & \mathbf{R}_m(\theta_{d/2})
\end{bmatrix}$$

其中频率定义为:
$$\theta_i = \theta_{\text{base}}^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1$$

通常 $\theta_{\text{base}} = 10000$ (沿用 Transformer 原始论文的设置)。

---

## 4. RoPE 的相对位置性质

### 4.1 定理陈述

**定理 4.1**: RoPE 保持相对位置信息

对于位置 $m$ 的查询 $\mathbf{q}_m$ 和位置 $n$ 的键 $\mathbf{k}_n$,注意力分数为:
$$\text{score}(m, n) = (\mathbf{R}_m \mathbf{q}_m)^{\top} (\mathbf{R}_n \mathbf{k}_n) = \mathbf{q}_m^{\top} \mathbf{R}_m^{\top} \mathbf{R}_n \mathbf{k}_n = \mathbf{q}_m^{\top} \mathbf{R}_{n-m} \mathbf{k}_n$$

### 4.2 证明

$$
\begin{aligned}
\mathbf{R}_m^{\top} \mathbf{R}_n &= \mathbf{R}_{-m} \mathbf{R}_n \quad \text{(因为 } \mathbf{R}_{-m} = \mathbf{R}_m^{\top}\text{)} \\
&= \mathbf{R}_{n-m} \quad \text{(可加性)}
\end{aligned}
$$

### 4.3 含义

注意力分数只依赖相对位置 $n - m$,不依赖绝对位置 $m, n$。这与语言的局部性特性一致:
- 单词之间的关系主要由相对距离决定
- "今天"和"明天"的关系与它们的绝对位置无关,只与相对距离(1)有关

---

## 5. RoPE 的高效实现

### 5.1 问题陈述

直接矩阵乘法 $\mathbf{R}_m \mathbf{x}$ 需要 $O(d^2)$ 时间,对于大模型来说开销过大。

### 5.2 向量化形式

利用块对角结构,每个 $2 \times 2$ 块独立计算。

对于 $\mathbf{x} = [x_0, x_1, x_2, x_3, \ldots, x_{d-1}]^{\top}$,将其重组为:
$$\mathbf{x}_{\text{pairs}} = [(x_0, x_1), (x_2, x_3), \ldots, (x_{d-2}, x_{d-1})]$$

每对应用旋转:
$$
\begin{bmatrix} x_i' \\ x_{i+1}' \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_i \\ x_{i+1} \end{bmatrix}
$$

展开为:
$$
\begin{aligned}
x_i' &= x_i \cos(m\theta_i) - x_{i+1} \sin(m\theta_i) \\
x_{i+1}' &= x_i \sin(m\theta_i) + x_{i+1} \cos(m\theta_i)
\end{aligned}
$$

### 5.3 向量化实现公式

$$\mathbf{x}' = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})$$

其中:
- $\odot$ 是逐元素乘法
- $\cos(m\boldsymbol{\theta}) = [\cos(m\theta_0), \cos(m\theta_0), \cos(m\theta_1), \cos(m\theta_1), \ldots]$
- $\sin(m\boldsymbol{\theta}) = [\sin(m\theta_0), \sin(m\theta_0), \sin(m\theta_1), \sin(m\theta_1), \ldots]$
- $\text{rotate\_half}(\mathbf{x}) = [-x_1, x_0, -x_3, x_2, \ldots]$

### 5.4 复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 直接矩阵乘法 | $O(d^2)$ | 计算 $\mathbf{R}_m \mathbf{x}$ |
| 向量化实现 | $O(d)$ | 逐元素乘法和加法 |

向量化实现将复杂度从 $O(d^2)$ 降到 $O(d)$,对于 $d = 128$ 的情况,理论加速 128 倍。

---

## 6. Megatron-LM 中的代码实现

### 6.1 旋转辅助函数

**文件路径**: `megatron/core/models/common/embeddings/rope_utils.py:73-89`

```python
def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """旋转一半维度的符号

    Args:
        x: 输入张量
        rotary_interleaved: 是否使用交错模式

    Returns:
        符号旋转后的张量
    """
    if not rotary_interleaved:
        # 连续模式: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        # 交错模式: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        x1 = x[:, :, :, ::2]   # 偶数索引
        x2 = x[:, :, :, 1::2]  # 奇数索引
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)
```

### 6.2 应用 RoPE

**文件路径**: `megatron/core/models/common/embeddings/rope_utils.py:92-126`

```python
def _apply_rotary_pos_emb_bshd(
    t: Tensor,  # [S, B, n_h, d_k]
    freqs: Tensor,  # [S, 1, 1, d_k]
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """应用 RoPE 到输入张量

    数学形式:
        output = t * cos(freqs) + rotate_half(t) * sin(freqs)
    """
    rot_dim = freqs.shape[-1]

    # 只对前 rot_dim 维应用 RoPE
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # MLA 特殊处理: 重新排列维度
    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # 计算 cos 和 sin
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    # 应用旋转: t' = t * cos + rotate_half(t) * sin
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    # 拼接未旋转部分
    return torch.cat((t, t_pass), dim=-1)
```

**代码解析**:
1. **部分 RoPE**: 只对前 `rot_dim` 维应用旋转,其余维度保持不变
2. **MLA 兼容性**: Multi-Latent Attention 需要特殊的维度排列
3. **mscale**: 用于长度外推的缩放因子 (NTK-aware scaling)

### 6.3 频率计算

**文件路径**: `megatron/core/models/common/embeddings/rotary_pos_embedding.py:50-80`

```python
class RotaryEmbedding(nn.Module):
    """RoPE 位置编码模块"""

    def __init__(
        self,
        dim: int,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_percent: float = 1.0,
    ):
        """初始化 RoPE

        Args:
            dim: 旋转维度
            rotary_base: 频率基数 (默认 10000)
            seq_len_interpolation_factor: 序列长度插值因子
            rotary_percent: 应用 RoPE 的维度百分比
        """
        super().__init__()

        # 计算实际旋转维度
        self.rotary_dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor

        # 计算频率: θ_i = base^{-2i/dim}
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> Tensor:
        """前向传播: 生成位置编码

        Returns:
            freqs: [seq_len, 1, 1, rotary_dim] - RoPE 频率
        """
        # 位置索引: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # 序列长度插值 (用于外推)
        if self.seq_len_interpolation_factor is not None:
            t = t / self.seq_len_interpolation_factor

        # 外积: [seq_len, rotary_dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # 扩展维度: [seq_len, 1, 1, rotary_dim/2]
        freqs = freqs.unsqueeze(1).unsqueeze(2)

        # 复制以匹配输入维度: [seq_len, 1, 1, rotary_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb
```

**代码解析**:
1. **频率计算**: $\theta_i = \text{base}^{-2i/d}$,使用 `torch.outer` 一次性计算所有位置的频率
2. **序列长度插值**: 通过缩放位置索引来支持更长的序列 (Linear Scaling)
3. **部分 RoPE**: `rotary_percent` 控制应用 RoPE 的维度比例 (默认 100%)

---

## 7. RoPE 的优势

### 7.1 相比绝对位置编码

| 特性 | 绝对位置编码 | RoPE |
|------|-------------|------|
| **外推能力** | 差 (训练长度固定) | 强 (可推广到更长序列) |
| **相对位置** | 无 | 有 (注意力分数只依赖相对位置) |
| **参数量** | 需要学习位置嵌入 | 无参数 |
| **实现复杂度** | 简单 | 中等 |

### 7.2 相比其他相对位置编码

| 特性 | T5 相对位置偏置 | ALiBi | RoPE |
|------|-----------------|-------|------|
| **简单高效** | 需要修改注意力计算 | 简单 (加法) | 简单 (乘法) |
| **理论优雅** | 启发式 | 启发式 | 基于旋转群 |
| **参数量** | 需要学习偏置 | 无参数 | 无参数 |
| **外推能力** | 好 | 好 | **最好** |

### 7.3 实验验证

Su et al. (2021) 在 WikiText-103 上的实验结果:

| 方法 | 困惑度 (Perplexity) | 外推能力 (512→1024) |
|------|---------------------|---------------------|
| 绝对位置编码 | 18.3 | 爆炸 |
| T5 相对位置 | 17.9 | 18.5 |
| **RoPE** | **17.6** | **17.8** |

**结论**:
- 在该论文设置中，RoPE 的训练困惑度和长度外推表现较强。
- RoPE 的相对位置信息来自旋转相位差，因此比绝对位置表更容易外推到更长位置。
- 外推质量仍受训练长度、频率基底、插值策略和目标任务影响，不能只按表中数值直接迁移。

---

## 8. RoPE 的变种和扩展

### 8.1 线性插值 (Linear Scaling)

**问题**: 直接外推到更长序列时,注意力模式可能不稳定。

**解决方案**: 缩放位置索引
$$t' = \frac{t}{s}, \quad s = \frac{L_{\text{new}}}{L_{\text{train}}}$$

**代码**:
```python
if self.seq_len_interpolation_factor is not None:
    t = t / self.seq_len_interpolation_factor
```

### 8.2 NTK-aware Scaling

**问题**: 线性插值在高频分量上效果不好。

**解决方案**: 调整频率基数
$$\theta_{\text{base}}' = \theta_{\text{base}} \cdot s^{d/(d-2)}$$

这保持了低频分量不变,只缩放高频分量。

### 8.3 YaRN (Yet another RoPE extensioN)

**核心思想**: 结合 NTK-aware scaling 和注意力熵正则化。

**优势**: 可以外推到 128K+ 的序列长度。

### 8.4 Megatron-LM 支持

| 扩展方法 | Megatron-LM 支持 | 配置参数 |
|----------|------------------|----------|
| 线性插值 | ✅ | `seq_len_interpolation_factor` |
| NTK-aware | ✅ | 通过调整 `rotary_base` |
| YaRN | ⚠️ (部分) | 需要自定义 |

---

## 9. 与注意力机制的集成

### 9.1 应用时机

RoPE 在注意力计算前应用到查询和键:

```python
# 1. 计算 Q, K, V
Q = X @ W_Q  # [S, B, n_h, d_k]
K = X @ W_K  # [S, B, n_h, d_k]
V = X @ W_V  # [S, B, n_h, d_k]

# 2. 应用 RoPE
freqs = rope_module(seq_len)  # [S, 1, 1, d_k]
Q = apply_rope(Q, freqs)
K = apply_rope(K, freqs)

# 3. 注意力计算
attn_output = scaled_dot_product_attention(Q, K, V)
```

### 9.2 为什么不对 V 应用 RoPE?

**数学原因**: 注意力分数只依赖 $Q$ 和 $K$ 的点积:
$$\text{score}(m, n) = \mathbf{q}_m^{\top} \mathbf{k}_n$$

应用 RoPE 到 $Q, K$ 后:
$$\text{score}(m, n) = (\mathbf{R}_m \mathbf{q}_m)^{\top} (\mathbf{R}_n \mathbf{k}_n) = \mathbf{q}_m^{\top} \mathbf{R}_{n-m} \mathbf{k}_n$$

已经实现了相对位置编码,对 $V$ 应用 RoPE 不会改变注意力权重。

**直觉解释**: $V$ 是被聚合的内容,位置信息只需要影响"怎么聚合"(注意力权重),不需要影响"聚合什么"(值向量)。

---

## 10. 总结

### 10.1 关键要点

1. **数学优雅**: RoPE 基于旋转群的性质,理论基础扎实
2. **相对位置**: 注意力分数只依赖相对位置,符合语言局部性
3. **高效实现**: 通过向量化将复杂度从 $O(d^2)$ 降到 $O(d)$
4. **外推能力**: 可以推广到训练时未见过的序列长度
5. **无参数**: 不需要学习额外的位置嵌入参数

### 10.2 与其他文档的关系

- **文档 22 (自注意力机制)**: RoPE 是自注意力机制中位置编码的一种实现
- **文档 23 (缩放点积注意力)**: RoPE 在注意力计算前应用到 Q 和 K
- **文档 33 (MLA详解)**: MLA 需要特殊的 RoPE 维度排列
- **文档 40 (KV Cache详解)**: 推理时需要为缓存的 K 应用 RoPE

### 10.3 进一步学习

- **长度外推技术**: Linear Scaling, NTK-aware Scaling, YaRN
- **ALiBi**: 另一种流行的相对位置编码方法
- **Flash Attention**: 如何在融合 kernel 中高效实现 RoPE

---

## 11. 总结与最佳实践

### 11.1 工程要点

- RoPE只作用于查询和键，不作用于值向量。
- 推理KV Cache中缓存的K必须与对应position一致。
- 长度外推时优先记录 `rotary_base`、interpolation factor、YaRN/NTK配置，避免恢复或推理阶段不一致。

### 11.2 常见错误

- position id 与实际token位置错位会导致长上下文质量明显下降。
- TP/CP切分下RoPE维度排列错误会造成不同rank结果不一致。

## 12. 参考文献

1. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv:2104.09864.
2. Vaswani et al. (2017). "Attention is All You Need". NeurIPS.
3. Press et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation". ICLR.
4. Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models". arXiv:2309.00071.
5. Megatron-LM Documentation: `megatron/core/models/common/embeddings/rotary_pos_embedding.py`

---

## 附录 A：RoPE 维度约束

RoPE 的实现需要保证被旋转的维度可以成对处理：

| 字段 | 作用 | 约束 |
|------|------|------|
| `kv_channels` | 每头 Q/K 维度 | RoPE维度不能超过它 |
| `rotary_percent` | 使用RoPE的维度比例 | 结果通常需要为偶数 |
| `rotary_interleaved` | 维度排列方式 | checkpoint与推理必须一致 |
| `rotary_base` | 频率基底 | 影响长位置相位 |
| sequence offset | 推理位置偏移 | 必须与KV Cache位置一致 |

典型配置：

```text
kv_channels = 128
rotary_percent = 1.0
rotary_dim = 128
rotary_base = 10000
rotary_interleaved = false
```

部分 RoPE 配置：

```text
kv_channels = 128
rotary_percent = 0.5
rotary_dim = 64
pass_through_dim = 64
```

部分 RoPE 下，只有前 `rotary_dim` 维带有位置信息，其余维度保持内容表示。训练和推理必须使用同一切分方式。

## 附录 B：实现路径审查

| 目标 | 文件 | 审查点 |
|------|------|--------|
| RoPE频率构造 | `megatron/core/models/common/embeddings/rotary_pos_embedding.py` | base、dim、dtype、device |
| RoPE应用 | `megatron/core/transformer/attention.py` | Q/K应用时机 |
| inference offset | `megatron/core/transformer/attention.py` | decode只取当前位置 |
| YaRN扩展 | `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py` | scaling参数 |
| Transformer配置 | `megatron/core/transformer/transformer_config.py` | rotary相关字段 |

审查时要问：

1. 旋转表长度是否覆盖训练或推理最大位置。
2. context parallel 下 rotary 序列长度是否乘上 CP size。
3. decode 时 query 用当前位置，key 写入 cache 前是否已旋转。
4. checkpoint 中是否记录了 `rotary_base` 和 scaling 配置。
5. 推理服务是否在 prompt chunking 时维护正确 offset。

## 附录 C：位置错位故障

RoPE 最隐蔽的问题是 position id 错位。常见症状：

| 症状 | 可能原因 |
|------|----------|
| 短文本正常，长文本退化 | 推理使用的 base/scaling 与训练不同 |
| batch size 变化后输出不同 | padding position 未正确处理 |
| chunked prefill 后质量下降 | sequence offset 未累加 |
| KV Cache 命中但输出异常 | cache 中 K 的位置与查询位置不一致 |
| TP/CP 下结果不一致 | rotary embedding 切片不同 |
| checkpoint 迁移后困惑度升高 | interleaved布局不一致 |

排查顺序：

1. 打印每个 token 的 position id。
2. 对比一次性 prefill 与分块 prefill 的 logits。
3. 关闭 KV Cache，对比 decode 输出。
4. 固定 batch 中 padding，检查有效 token 的 position 是否不变。
5. 检查训练脚本和推理脚本中的 base、percent、interleaved。

## 附录 D：长上下文扩展策略

| 策略 | 核心思想 | 优势 | 风险 |
|------|----------|------|------|
| 直接外推 | 使用训练时RoPE到更长位置 | 简单 | 高频相位可能失配 |
| Linear scaling | position按比例缩小 | 易实现 | 可能牺牲短距离分辨率 |
| NTK-aware | 调整频率基底 | 保持部分频率结构 | 参数选择敏感 |
| YaRN | 插值和温度修正组合 | 实证表现强 | 配置更多 |
| 继续训练 | 在长序列上适配 | 最可靠 | 成本高 |

选择建议：

- 只是轻微超过训练长度，先测试直接外推和简单 scaling。
- 需要 4x 以上长度扩展，应做长上下文继续训练或高质量指令微调验证。
- 如果服务中同时有短文本和长文本，要确认扩展策略不会显著损害短文本质量。
- 所有长度扩展实验都应报告训练长度、目标长度和 evaluation 长度。

## 附录 E：RoPE 与 KV Cache

在自回归推理中，K 通常在写入 cache 前应用 RoPE：

```text
for each decode step t:
    q_t = project_query(x_t)
    k_t = project_key(x_t)
    q_t = rope(q_t, position=t)
    k_t = rope(k_t, position=t)
    append k_t to kv_cache
    attend q_t to cached keys
```

这样做的好处：

- cache 中的 K 已经携带对应位置。
- decode 时不必反复对历史 K 应用 RoPE。
- chunked prefill 和 single prefill 更容易对齐。

风险：

- 如果 position offset 错，错误会被写入 cache 并持续影响后续 token。
- 如果换了 RoPE scaling，旧 cache 不能继续复用。
- 如果 prompt 被截断或滑窗移动，要重新定义位置口径。

## 附录 F：数学不变量

RoPE 的核心不变量是相对位置性质：

$$
\langle R_m q, R_n k \rangle
= q^\top R_{n-m} k
$$

工程含义：

1. 注意力分数能感知相对距离 $n-m$。
2. Q 和 K 必须使用同一组旋转频率。
3. V 不需要旋转，因为位置关系已经进入 attention weights。
4. 如果 Q/K 维度切分不同，该不变量会被破坏。
5. 旋转矩阵应保持范数，理论上不改变向量长度。

可用于单元测试的不变量：

| 测试 | 期望 |
|------|------|
| norm preservation | `norm(rope(x)) ~= norm(x)` |
| position zero | `position=0` 时接近恒等变换 |
| relative shift | 同时平移 Q/K 位置时分数结构一致 |
| dtype consistency | BF16/FP32误差可解释 |
| interleaved consistency | 同一布局训练推理一致 |

## 附录 G：配置记录模板

RoPE 配置必须写入实验记录：

```yaml
position_embedding:
  type: rope
  rotary_base:
  rotary_percent:
  rotary_interleaved:
  seq_length_train:
  seq_length_target:
  scaling:
    type:
    factor:
    original_max_position:
    yarn_beta_fast:
    yarn_beta_slow:
```

缺少这些字段会导致 checkpoint 复现困难。尤其是 long-context 继续训练后，推理脚本必须知道原始训练长度和缩放策略。

## 附录 H：实验设计

RoPE 实验建议至少包括：

| 实验 | 固定项 | 变量 | 指标 |
|------|--------|------|------|
| base对比 | 模型、数据、长度 | `rotary_base` | validation loss |
| percent对比 | head dim、base | `rotary_percent` | loss、速度 |
| 外推对比 | 训练checkpoint | 目标长度 | long-context loss |
| chunked prefill | 输入文本 | chunk size | logits一致性 |
| cache一致性 | prompt | cache on/off | decode logits |
| dtype对比 | 配置 | BF16/FP32 | 数值误差 |

外推实验不能只用困惑度。还应测试：

- needle-in-a-haystack 或长程检索。
- 多文档问答。
- 长代码补全。
- 长对话位置一致性。
- 短文本回归，确认没有短上下文退化。

## 附录 I：面试题

**RoPE 为什么能表达相对位置？**

因为位置 $m$ 和 $n$ 的旋转点积可以化简为只依赖 $n-m$ 的相对旋转。注意力 logits 因此包含相对距离信息。

**为什么不对 V 应用 RoPE？**

V 被 attention weights 加权求和。位置信息通过 QK 分数决定权重后已经进入输出；旋转 V 会改变内容向量本身，通常没有必要。

**RoPE 和 ALiBi 的主要差别是什么？**

RoPE 改变 Q/K 表示，通过旋转相位进入点积；ALiBi 直接给 attention logits 加线性距离偏置。

**为什么长上下文需要 scaling？**

训练时没有见过的位置可能对应过高频或相位别名。scaling 试图把更长位置映射到训练可接受的频率范围。

**interleaved 布局为什么重要？**

它决定哪些维度成对旋转。训练和推理布局不一致会让同一权重看到不同的几何结构。

## 附录 J：上线前检查

1. 训练和推理的 `rotary_base` 一致。
2. 训练和推理的 `rotary_percent` 一致。
3. `rotary_interleaved` 与 checkpoint 匹配。
4. 推理 position offset 在 prefill/decode/chunked prefill 中一致。
5. KV Cache 中的 K 已按正确位置旋转。
6. 长上下文 scaling 配置写入 checkpoint 或部署配置。
7. CP/TP 下 rotary embedding 切片正确。
8. 外推实验同时覆盖短文本和长文本。
9. cache on/off 的短序列 logits 差异在容忍范围内。
10. 文档中的外推结论标明论文或实验口径。

## 附录 K：Checkpoint 迁移风险

RoPE 本身没有可学习参数，但 checkpoint 迁移仍可能失败，因为权重是在某种位置编码语义下训练出来的。

| 迁移项 | 是否可直接改 | 风险 |
|--------|--------------|------|
| `rotary_base` | 不建议 | 长短距离相位都变 |
| `rotary_percent` | 不建议 | Q/K 部分维度语义变 |
| `rotary_interleaved` | 不可随意改 | 维度配对完全不同 |
| 最大长度 | 可扩展但需验证 | 外推退化 |
| YaRN/NTK scaling | 需继续训练或评估 | 频率结构改变 |

如果必须迁移：

1. 先在短序列上对齐 logits。
2. 再在训练长度附近测 validation loss。
3. 最后测目标长上下文任务。
4. 如果短序列已退化，不要继续解释为“外推问题”。

## 附录 L：RoPE 与数据格式

位置 id 不只是模型内部问题，也受数据管线影响：

| 数据形态 | position 处理 |
|----------|---------------|
| packed sequence | 每个 segment 是否重置位置要与mask一致 |
| multi-document batch | 文档边界是否可见 |
| chat template | system/user/assistant token 都会消耗位置 |
| padding left | position id 需要跳过pad或保持一致策略 |
| padding right | causal mask通常更简单 |
| sliding window | 窗口移动后位置是绝对还是相对 |

训练和推理 position 策略不一致，会表现为“离线验证正常，在线长对话异常”。

## 附录 M：源码阅读问答

**为什么 `get_rotary_seq_len` 要考虑 inference context？**

推理时需要为最大缓存长度或当前上下文生成足够的 RoPE 表，decode 阶段还要根据 sequence offset 选择当前位置。

**为什么 context parallel 会影响 rotary 序列长度？**

CP 会把序列维切分到不同 rank。为了保证全局位置一致，RoPE长度和切片必须按全局序列口径处理。

**为什么 cache 中的 K 通常已经旋转？**

这样 decode 时只需旋转当前 query/key，不必每步重算所有历史 key 的旋转。

**如何判断 interleaved 配置错了？**

短序列 logits、validation loss 或复制 checkpoint 后的质量会明显不一致；这是布局错误，不是普通随机波动。

## 附录 N：最小一致性实验

RoPE 的最小一致性实验应覆盖三条路径：

```text
path_1: full prefill
path_2: chunked prefill
path_3: prefill + decode with kv cache
```

同一输入下记录：

```text
max_abs_logit_diff(path_1, path_2)
max_abs_logit_diff(path_1, path_3)
position_ids
sequence_offset
rotary_base
rotary_percent
rotary_interleaved
```

通过标准：

- FP32 下差异应接近数值舍入。
- BF16/FP16 下差异应可解释。
- 改变 chunk size 不应改变有效 token 的 position id。
- padding 变化不应改变非 padding token 的相对位置语义。

如果 cache 路径与 full prefill 不一致，优先检查 K 写入 cache 前是否已应用正确位置的 RoPE。

## 附录 O：最终自检

提交前检查：

1. RoPE 结论是否标明论文或实验口径。
2. 长度外推是否避免写成无条件保证。
3. `rotary_base`、`rotary_percent`、`rotary_interleaved` 是否都被提及。
4. KV Cache 场景是否说明 position offset。
5. 是否说明 V 不应用 RoPE 的原因。
6. 是否覆盖 chunked prefill。
7. 是否覆盖 checkpoint 迁移风险。
8. 是否说明训练和推理配置必须一致。
9. 是否说明 position id 与 padding 策略的关系。
10. 是否提供 cache on/off 对齐测试。

---

**文档版本**: v1.0
**最后更新**: 2026-05-10
**文档状态**: ✅ 已完成
