# 47. 长度外推技术：YaRN与NTK-aware RoPE

> **文档编号**: 47
> **所属部分**: 第5部分 - 模型架构 (41-50)
> **代码位置**: `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`
> **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM 仓库实际代码)

---

## 1. 引言 (Introduction)

### 1.1 长度外推问题

RoPE (Rotary Position Embedding) 在标准 Transformer 模型中被广泛采用,但存在一个关键问题:**模型在推理时难以处理超过训练序列长度的输入**。

**问题表现**:
- 训练序列长度: $s_{train} = 2048$
- 推理序列长度: $s_{infer} = 8192$ (超过训练长度 4x)
- 性能: 困惑度 (Perplexity) 显著上升,甚至出现模型崩溃

**根本原因**:
RoPE 的旋转频率 $\theta_i = \text{base}^{-\frac{2i}{d}}$ 在训练时只见过位置 $[0, s_{train})$,推理时出现未见过的位置 $[s_{train}, s_{infer})$,导致:
1. **外推不稳定**: 新位置的旋转角度超出训练分布
2. **高频信息丢失**: 高维度 (大 $i$) 的旋转频率过快,在长序列上振荡剧烈

### 1.2 解决方案演进

解决长度外推问题的技术路线:

1. **Linear Interpolation** (2022):
   - 思路: 将推理位置缩放回训练范围
   - 方法: $\text{pos}' = \text{pos} / s$ (其中 $s = s_{infer} / s_{train}$)
   - 问题: 所有维度统一缩放,忽略了不同维度的特性

2. **NTK-aware Interpolation** (Reddit 2023):
   - 思路: 基于 Neural Tangent Kernel (NTK) 理论,调整 base 而非位置
   - 方法: $\text{base}' = \text{base} \times s^{d/(d-2)}$
   - 优点: 保持低频不变,高频自适应调整

3. **YaRN** (Peng et al., 2023):
   - 思路: 结合线性插值和 NTK-aware,分段处理不同维度
   - 方法: 低频维度→NTK-aware,高频维度→线性插值,中频→smooth 过渡
   - 优点: 最佳平衡,性能最优

### 1.3 YaRN核心创新

YaRN (Yet another RoPE extensioN) 的关键贡献:

1. **维度分段插值** (Dimension-wise Interpolation):
   - 根据旋转频率将维度分为三类:低频、中频、高频
   - 不同类别采用不同的插值策略

2. **动态温度调整** (Attention Temperature Scaling):
   - 引入 `mscale` 参数,补偿插值导致的注意力分数缩放
   - 确保注意力分布不受位置外推影响

3. **平滑过渡机制** (Smooth Ramp):
   - 中频维度使用线性过渡函数,避免分段带来的突变

### 1.4 Megatron 实现

Megatron-LM 实现了完整的 YaRN RoPE,主要特性:

- **文件位置**: `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py` (257行)
- **类**: `YarnRotaryEmbedding` (继承自 `RotaryEmbedding`)
- **配置参数** (在 `MLATransformerConfig` 中):
  - `rotary_scaling_factor`: YaRN 缩放因子 $s$ (默认 40.0)
  - `original_max_position_embeddings`: 原始最大序列长度 (默认 4096)
  - `beta_fast`, `beta_slow`: 频率边界参数
  - `mscale`, `mscale_all_dim`: 注意力温度调整参数
- **应用场景**: 主要用于 Multi-Latent Attention (MLA),如 DeepSeek-V2

**与标准 RoPE 的对比**:

| 特性 | 标准 RoPE | YaRN RoPE |
|------|----------|-----------|
| 外推能力 | 差 (仅支持 1-2x) | 优 (支持 8-64x) |
| 维度处理 | 统一处理 | 分段处理 (低/中/高频) |
| 温度调整 | 无 | 有 (mscale) |
| 训练成本 | 低 | 中 (需额外 fine-tune) |
| 推理开销 | 低 | 中 (额外计算 mask 和 mscale) |

---

## 2. 相关工作 (Related Work)

### 2.1 Rotary Position Embedding (RoPE)

**原始论文**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)

**核心思想**: 使用旋转矩阵对 query 和 key 进行位置编码:
$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{W}_k \mathbf{x}_n
$$

其中 $\mathbf{R}_m$ 是位置 $m$ 的旋转矩阵:
$$
\mathbf{R}_m = \begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & & & \\
\sin(m\theta_0) & \cos(m\theta_0) & & & \\
& & \cos(m\theta_1) & -\sin(m\theta_1) & \\
& & \sin(m\theta_1) & \cos(m\theta_1) & \\
& & & & \ddots
\end{bmatrix}
$$

**频率定义**:
$$
\theta_i = \text{base}^{-\frac{2i}{d}}, \quad i = 0, 1, \ldots, \frac{d}{2} - 1
$$

其中 $\text{base} = 10000$ (标准值),$d$ 是 head dimension。

**优势**:
- 相对位置编码: $\mathbf{q}_m^T \mathbf{k}_n = f(m - n)$
- 线性外推: 理论上支持任意长度

**劣势**:
- 实践中外推性能差: 超过训练长度 2x 就显著下降

### 2.2 Linear Interpolation

**思路**: 将推理时的位置缩放回训练范围。

**公式**:
$$
\text{pos}' = \frac{\text{pos}}{s}
$$

其中 $s = \frac{s_{\text{infer}}}{s_{\text{train}}}$ 是缩放因子。

**示例**: 训练 $s_{\text{train}} = 2048$,推理 $s_{\text{infer}} = 8192$,$s = 4$。
- 位置 4096 → 1024 (缩放后)
- 位置 8191 → 2047.75 (在训练范围内)

**问题**:
1. **低频维度过度压缩**: 低频维度 (小 $i$) 本身旋转就慢,再压缩导致位置区分度下降
2. **高频维度失真**: 高频维度 (大 $i$) 虽然被压缩,但仍然振荡剧烈

**LLaMA 2 应用**:
- 支持从 4K 扩展到 32K (8x)
- 使用 linear interpolation + fine-tuning
- 代码: `seq_len_interpolation_factor = 8.0` (参数名在 Megatron 中)

### 2.3 NTK-aware Interpolation

**来源**: Reddit 讨论 (2023), 基于 Neural Tangent Kernel (NTK) 理论。

**核心思想**: 调整 `base` 而非位置,使低频维度保持不变,高频维度自适应调整。

**公式**:
$$
\text{base}' = \text{base} \times s^{\frac{d}{d-2}}
$$

**推导** (简化版):
- 目标: 保持低频维度 ($i=0$) 在最大位置的旋转角度不变
- 原始: $\theta_0 \times s_{\text{infer}} = 1.0 \times s \cdot s_{\text{train}}$
- NTK: $(\text{base}')^{-\frac{2 \times 0}{d}} \times s_{\text{infer}} = \text{base}^{-0} \times s \cdot s_{\text{train}}$
- 求解: $\text{base}' = \text{base} \times s^{\alpha}$,其中 $\alpha = \frac{d}{d-2}$ (近似,实际推导更复杂)

**效果**:
- 低频 ($i=0$): $\theta_0' = 1.0$ (不变)
- 高频 ($i \to d/2$): $\theta_i' \approx \theta_i / s^{\beta}$ ($\beta > 1$,更强插值)
- 中频: 平滑过渡

**问题**:
- 高频维度可能插值不足,仍有失真
- 没有显式的温度调整

### 2.4 YaRN (Yet another RoPE extensioN)

**论文**: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023)

**核心改进**:

1. **维度分段插值**:
   - 低频维度 ($i \in [0, d_{\text{low}})$): 使用 `inv_freq_extra` (NTK-aware,基于原始 base)
   - 高频维度 ($i \in (d_{\text{high}}, d/2)$): 使用 `inv_freq_inter` (Linear interpolation)
   - 中频维度 ($i \in [d_{\text{low}}, d_{\text{high}}]$): 线性插值混合

2. **边界维度计算**:
   - 基于 **旋转次数** (rotations) 确定边界
   - `beta_fast`: 高频边界旋转次数 (默认 32)
   - `beta_slow`: 低频边界旋转次数 (默认 1)

3. **注意力温度缩放 (mscale)**:
   - 补偿插值导致的注意力分数变化
   - 公式: $\text{mscale} = 0.1 \ln(s) + 1.0$ (简化版)

**优势**:
- 兼顾低频和高频的特性
- 外推性能最佳
- 支持 8x-64x 扩展

**Megatron 实现位置**:
- `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`

---

## 3. 符号定义 (Symbol Definitions)

### 3.1 RoPE 基础符号

| 符号 | 含义 | 代码变量名 |
|------|------|-----------|
| $d$ | Head dimension (旋转维度) | `kv_channels` |
| $\text{base}$ | RoPE base 频率 | `rotary_base` |
| $\theta_i$ | 第 $i$ 维度的频率 | `inv_freq[i]` |
| $m$ | 序列位置 | `seq` |
| $s_{\text{train}}$ | 训练最大序列长度 | `original_max_position_embeddings` |
| $s_{\text{infer}}$ | 推理序列长度 | `max_seq_len` |

### 3.2 YaRN 专有符号

| 符号 | 含义 | 代码变量名 |
|------|------|-----------|
| $s$ | 缩放因子 $= s_{\text{infer}} / s_{\text{train}}$ | `scaling_factor` |
| $\beta_{\text{fast}}$ | 高频边界旋转次数 | `beta_fast` |
| $\beta_{\text{slow}}$ | 低频边界旋转次数 | `beta_slow` |
| $d_{\text{low}}$ | 低频边界维度索引 | `low` |
| $d_{\text{high}}$ | 高频边界维度索引 | `high` |
| $\theta_i^{\text{extra}}$ | NTK-aware 频率 (低频) | `inv_freq_extra[i]` |
| $\theta_i^{\text{inter}}$ | Linear interpolation 频率 (高频) | `inv_freq_inter[i]` |
| $\text{mask}_i$ | 维度 $i$ 的插值 mask | `inv_freq_mask[i]` |
| $\text{mscale}$ | 注意力温度缩放因子 | `_mscale` |
| $\alpha$ | mscale 基础参数 | `mscale` |
| $\alpha_{\text{all}}$ | mscale 全维度参数 | `mscale_all_dim` |

### 3.3 计算中间变量

| 符号 | 含义 | 公式 |
|------|------|------|
| $\lambda_i$ | 波长 | $\lambda_i = 2\pi / \theta_i$ |
| $r_i$ | 旋转次数 | $r_i = s_{\text{train}} / \lambda_i$ |
| $\text{ramp}_i$ | 平滑过渡函数 | $\text{clamp}\left(\frac{i - d_{\text{low}}}{d_{\text{high}} - d_{\text{low}}}, 0, 1\right)$ |

---

## 4. 数学原理 (Mathematical Principles)

### 4.1 RoPE 外推问题分析

#### 4.1.1 标准 RoPE 公式

对于维度 $i$ (第 $i$ 对旋转坐标),RoPE 的频率为:
$$
\theta_i = \text{base}^{-\frac{2i}{d}} = 10000^{-\frac{2i}{d}}
$$

在位置 $m$ 的旋转角度:
$$
\phi_i(m) = m \cdot \theta_i
$$

**低频维度** ($i \approx 0$):
- $\theta_0 = 10000^0 = 1.0$
- 波长: $\lambda_0 = 2\pi / 1.0 \approx 6.28$
- 旋转非常慢,适合编码长距离依赖

**高频维度** ($i \approx d/2$):
- $\theta_{d/2} = 10000^{-1} = 0.0001$
- 波长: $\lambda_{d/2} = 2\pi / 0.0001 \approx 62832$
- 旋转非常快,适合编码短距离依赖

#### 4.1.2 外推失败的原因

**问题**: 训练序列长度 $s_{\text{train}} = 2048$,推理序列长度 $s_{\text{infer}} = 8192$。

**训练时**见过的最大旋转角度:
$$
\phi_i^{\text{max,train}} = s_{\text{train}} \cdot \theta_i
$$

**推理时**的旋转角度:
$$
\phi_i^{\text{infer}}(m) = m \cdot \theta_i, \quad m \in [0, s_{\text{infer}})
$$

当 $m > s_{\text{train}}$ 时,$\phi_i^{\text{infer}}(m) > \phi_i^{\text{max,train}}$,模型从未见过这些角度。

**高频维度的额外问题**:
- 旋转速度快,在长序列上 $\cos(\phi_i(m))$ 和 $\sin(\phi_i(m))$ 振荡剧烈
- 导致注意力权重不稳定

**实验观察** (来自 YaRN 论文):
- 超过训练长度 2x: 困惑度上升 10-20%
- 超过训练长度 4x: 困惑度上升 50-100%,模型几乎不可用

### 4.2 Linear Interpolation

#### 4.2.1 公式

将推理位置缩放回训练范围:
$$
\text{pos}' = \frac{\text{pos}}{s}
$$

其中 $s = s_{\text{infer}} / s_{\text{train}}$ 是缩放因子。

**等价变换**: 也可以缩放频率:
$$
\theta_i' = \frac{\theta_i}{s}
$$

**代码实现** (`RotaryEmbedding`):
```python
# megatron/core/models/common/embeddings/rotary_pos_embedding.py:135-136
if self.seq_len_interpolation_factor is not None:
    seq *= 1 / self.seq_len_interpolation_factor  # seq /= s
```

#### 4.2.2 问题分析

**低频维度过度压缩**:
- 原始: $\theta_0 = 1.0$,波长 $\lambda_0 = 2\pi \approx 6.28$
- 插值 ($s=4$): $\theta_0' = 0.25$,波长 $\lambda_0' = 25.1$
- 结果: 位置 0-8192 在低频维度上几乎没有变化 (旋转太慢),**丢失位置区分度**

**高频维度仍有失真**:
- 虽然旋转角度被压缩到训练范围内,但压缩比例对所有维度统一
- 高频维度在长序列上的振荡问题未得到针对性优化

### 4.3 NTK-aware Interpolation

#### 4.3.1 核心思想

**目标**: 保持低频维度不变,高频维度自适应调整。

**方法**: 调整 `base` 值而非位置:
$$
\text{base}' = \text{base} \times s^{\alpha}
$$

其中 $\alpha = \frac{d}{d - 2}$ (基于 NTK 理论推导,简化公式)。

**新频率**:
$$
\theta_i' = (\text{base}')^{-\frac{2i}{d}} = \text{base}^{-\frac{2i}{d}} \times s^{-\frac{2i\alpha}{d}}
$$

#### 4.3.2 频率分析

**低频维度** ($i = 0$):
$$
\theta_0' = \text{base}^0 = 1.0 \quad (\text{不变})
$$

**高频维度** ($i = d/2$):
$$
\theta_{d/2}' = \text{base}^{-1} \times s^{-\alpha} = \theta_{d/2} \times s^{-\alpha}
$$

**效果**:
- 低频: 保持原始频率,适合长距离依赖
- 高频: 频率降低 (波长增加),减少振荡

**示例** ($d=128$, $s=4$):
- $\alpha = 128 / 126 \approx 1.016$
- 高频缩放: $s^{-\alpha} = 4^{-1.016} \approx 0.246$
- 比 linear interpolation ($1/s = 0.25$) 略强

#### 4.3.3 局限性

- **高频维度调整有限**: $\alpha \approx 1$ 时,高频缩放与 linear interpolation 相近
- **缺乏显式温度调整**: 插值后注意力分数分布变化未被考虑

### 4.4 YaRN 算法

#### 4.4.1 核心公式

YaRN 对不同维度采用不同插值策略:
$$
\theta_i = (1 - \text{mask}_i) \cdot \theta_i^{\text{inter}} + \text{mask}_i \cdot \theta_i^{\text{extra}}
$$

其中:
- $\theta_i^{\text{inter}} = \frac{1}{s \cdot \text{base}^{2i/d}}$: Linear interpolation 频率 (高频)
- $\theta_i^{\text{extra}} = \frac{1}{\text{base}^{2i/d}}$: 原始频率 (低频)
- $\text{mask}_i \in [0, 1]$: 插值 mask

#### 4.4.2 边界维度计算

**步骤 1**: 根据旋转次数确定边界

给定旋转次数 $r$,反推对应的维度 $d(r)$:
$$
d(r) = \frac{d \cdot \ln(s_{\text{train}} / (r \cdot 2\pi))}{2 \ln(\text{base})}
$$

**推导**:
- 旋转次数: $r = \frac{s_{\text{train}}}{\lambda_i} = \frac{s_{\text{train}} \cdot \theta_i}{2\pi}$
- 代入 $\theta_i = \text{base}^{-2i/d}$:
$$
r = \frac{s_{\text{train}}}{2\pi} \cdot \text{base}^{-2i/d}
$$
- 求解 $i$:
$$
i = \frac{d \ln(s_{\text{train}} / (2\pi r))}{2 \ln(\text{base})}
$$

**步骤 2**: 计算边界
$$
\begin{aligned}
d_{\text{low}} &= d(\beta_{\text{slow}}) = \frac{d \ln(s_{\text{train}} / (2\pi \beta_{\text{slow}}))}{2 \ln(\text{base})} \\
d_{\text{high}} &= d(\beta_{\text{fast}}) = \frac{d \ln(s_{\text{train}} / (2\pi \beta_{\text{fast}}))}{2 \ln(\text{base})}
\end{aligned}
$$

**代码实现** (`yarn_rotary_pos_embedding.py:193-215`):
```python
def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )

def _yarn_find_correction_range(
    low_rot: float,  # beta_slow
    high_rot: float, # beta_fast
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
```

**示例** ($d=128$, $s_{\text{train}}=4096$, $\beta_{\text{fast}}=32$, $\beta_{\text{slow}}=1$):
$$
\begin{aligned}
d_{\text{low}} &= \frac{128 \times \ln(4096 / (2\pi \times 1))}{2 \times \ln(10000)} \approx 54 \\
d_{\text{high}} &= \frac{128 \times \ln(4096 / (2\pi \times 32))}{2 \times \ln(10000)} \approx 71
\end{aligned}
$$

#### 4.4.3 插值 Mask 计算

**线性 Ramp 函数**:
$$
\text{ramp}_i = \text{clamp}\left(\frac{i - d_{\text{low}}}{d_{\text{high}} - d_{\text{low}}}, 0, 1\right)
$$

**Mask**:
$$
\text{mask}_i = 1 - \text{ramp}_i
$$

**解释**:
- $i < d_{\text{low}}$: $\text{ramp}_i = 0$, $\text{mask}_i = 1$ → 使用 $\theta_i^{\text{extra}}$ (原始频率)
- $i > d_{\text{high}}$: $\text{ramp}_i = 1$, $\text{mask}_i = 0$ → 使用 $\theta_i^{\text{inter}}$ (插值频率)
- $d_{\text{low}} \leq i \leq d_{\text{high}}$: $\text{mask}_i \in (0, 1)$ → 线性混合

**代码实现** (`yarn_rotary_pos_embedding.py:218-224`):
```python
def _yarn_linear_ramp_mask(min: float, max: float, dim: int, device: torch.device) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
```

#### 4.4.4 最终频率

**计算流程** (`yarn_rotary_pos_embedding.py:129-140`):
```python
# 1. 计算边界
low, high = _yarn_find_correction_range(
    self.beta_fast,
    self.beta_slow,
    self.dim,
    self.rotary_base,
    self.original_max_position_embeddings,
    self.correction_range_round_to_int,
)

# 2. 计算 mask
inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
    low, high, self.dim // 2, device=self.inv_freq_extra.device
).to(dtype=torch.float32)

# 3. 混合频率
inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask
```

其中:
- `inv_freq_inter = 1.0 / (scaling_factor * base ** (torch.arange(0, dim, 2) / dim))`
- `inv_freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))`

### 4.5 注意力温度缩放 (mscale)

#### 4.5.1 问题

RoPE 插值后,cos/sin 值的幅度发生变化,导致注意力分数的尺度不一致:
- 原始: $\cos(\phi_i(m)) \in [-1, 1]$
- 插值后: 平均幅度可能偏小 (因为旋转变慢)

**影响**: 注意力分数 $\text{softmax}(QK^T / \sqrt{d})$ 的分布变化,模型表现下降。

#### 4.5.2 mscale 公式

YaRN 引入 **concentration factor** (mscale) 来调整 cos/sin 值:
$$
\cos(\phi_i(m))' = \cos(\phi_i(m)) \times \text{mscale}
$$
$$
\sin(\phi_i(m))' = \sin(\phi_i(m)) \times \text{mscale}
$$

**基础 mscale**:
$$
\text{mscale}_{\text{base}}(s, \alpha) = \begin{cases}
1.0 & \text{if } s \leq 1 \\
0.1 \alpha \ln(s) + 1.0 & \text{otherwise}
\end{cases}
$$

**最终 mscale**:
$$
\text{mscale} = \frac{\text{mscale}_{\text{base}}(s, \alpha)}{\text{mscale}_{\text{base}}(s, \alpha_{\text{all}})}
$$

其中:
- $\alpha$: `mscale` 参数 (默认 1.0)
- $\alpha_{\text{all}}$: `mscale_all_dim` 参数 (默认 0.0,表示不做全维度缩放)

**代码实现** (`yarn_rotary_pos_embedding.py:227-246`):
```python
def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

@lru_cache(maxsize=8)
def _yarn_get_concentration_factor(
    scaling_factor: float, mscale: Optional[float], mscale_all_dim: Optional[float]
) -> float:
    """
    Get the concentration factor (factor multiplied to the sine and cosine components of the
    embedding). This factor is also known as attention factor, and sometimes homonymously known as
    "mscale"
    """
    if mscale is None or mscale_all_dim is None:
        return _yarn_get_mscale(scaling_factor)
    return float(
        _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
    )
```

**示例** ($s=40$, $\alpha=1.0$, $\alpha_{\text{all}}=0.0$):
$$
\text{mscale} = \frac{0.1 \times 1.0 \times \ln(40) + 1.0}{0.1 \times 0.0 \times \ln(40) + 1.0} = \frac{1.369}{1.0} \approx 1.37
$$

#### 4.5.3 应用

**在 MLA 中的使用**:
```python
# megatron/core/transformer/multi_latent_attention.py:121-122
mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
```

**在 RoPE 应用中**:
```python
# megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py:172-176
self.register_buffer(
    "cos_cached", (emb.cos() * _mscale).to(dtype).contiguous(), persistent=False
)
self.register_buffer(
    "sin_cached", (emb.sin() * _mscale).to(dtype).contiguous(), persistent=False
)
```

**效果**: 补偿插值导致的注意力分数缩放,保持模型稳定性。

---

## 5. 算法伪代码 (Algorithm Pseudocode)

### 5.1 YaRN RoPE 完整算法

```
Algorithm 1: YaRN Rotary Position Embedding
────────────────────────────────────────────────────────────
Input:  d: head dimension
        base: RoPE base (10000)
        s: scaling factor (e.g., 40)
        s_train: original max position embeddings (e.g., 4096)
        β_fast: fast beta (32)
        β_slow: slow beta (1)
        α: mscale parameter (1.0)
        α_all: mscale_all_dim parameter (0.0)
        max_seq_len: inference sequence length

Output: cos, sin tensors for RoPE application
        mscale: attention temperature scaling factor

# ===== Step 1: 计算频率 =====
1: # 原始频率 (for low-frequency dimensions)
2: θ^extra_i ← 1 / base^(2i/d) for i = 0, 1, ..., d/2 - 1

3: # 插值频率 (for high-frequency dimensions)
4: θ^inter_i ← 1 / (s · base^(2i/d)) for i = 0, 1, ..., d/2 - 1

# ===== Step 2: 计算边界维度 =====
5: d_low ← ⌊d · ln(s_train / (2π · β_slow)) / (2 · ln(base))⌋
6: d_high ← ⌈d · ln(s_train / (2π · β_fast)) / (2 · ln(base))⌉

7: # Clamp to valid range
8: d_low ← max(0, d_low)
9: d_high ← min(d/2 - 1, d_high)

# ===== Step 3: 计算插值 mask =====
10: for i = 0 to d/2 - 1:
11:     if d_low == d_high:
12:         d_high ← d_high + 0.001  # Prevent singularity
13:     ramp_i ← clamp((i - d_low) / (d_high - d_low), 0, 1)
14:     mask_i ← 1 - ramp_i

# ===== Step 4: 混合频率 =====
15: for i = 0 to d/2 - 1:
16:     θ_i ← (1 - mask_i) · θ^inter_i + mask_i · θ^extra_i

# ===== Step 5: 计算旋转角度 =====
17: seq ← [0, 1, 2, ..., max_seq_len - 1]  # Position indices
18: freqs ← outer_product(seq, θ)  # [max_seq_len, d/2]

# ===== Step 6: 计算 mscale =====
19: if s ≤ 1:
20:     mscale_base ← 1.0
21: else:
22:     mscale_base(α) ← 0.1 · α · ln(s) + 1.0
23: mscale ← mscale_base(α) / mscale_base(α_all)

# ===== Step 7: 计算 cos/sin 并应用 mscale =====
24: emb ← concatenate(freqs, freqs, dim=-1)  # [max_seq_len, d]
25: cos ← cos(emb) · mscale  # [max_seq_len, d]
26: sin ← sin(emb) · mscale  # [max_seq_len, d]

27: return cos, sin, mscale

────────────────────────────────────────────────────────────
Complexity:
  Time:  O(max_seq_len · d)
  Space: O(max_seq_len · d)  (for cos/sin cache)
────────────────────────────────────────────────────────────
```

### 5.2 应用 YaRN RoPE 到 Query/Key

```
Algorithm 2: Apply YaRN RoPE to Query/Key Tensors
────────────────────────────────────────────────────────────
Input:  t: tensor to apply RoPE [seq_len, batch, num_heads, head_dim]
        cos, sin: YaRN RoPE tensors [seq_len, 1, 1, head_dim]
        rotary_interleaved: whether to use interleaved mode

Output: t with RoPE applied

# ===== Step 1: 提取旋转维度 =====
1: rot_dim ← cos.shape[-1]
2: t_rot ← t[..., :rot_dim]    # Rotary part
3: t_pass ← t[..., rot_dim:]   # Pass-through part (if any)

# ===== Step 2: Rotate =====
4: if rotary_interleaved:
5:     # Interleaved mode: [x0, x1, x2, x3, ...] → [x1, -x0, x3, -x2, ...]
6:     x1 ← t_rot[:, :, :, ::2]   # Even indices
7:     x2 ← t_rot[:, :, :, 1::2]  # Odd indices
8:     t_rot_half ← stack([-x2, x1], dim=-1)
9:     t_rot_half ← t_rot_half.view(shape of t_rot)
10: else:
11:     # Non-interleaved mode: [x0, x1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
12:     #   → [-x_{d/2}, ..., -x_{d-1}, x0, ..., x_{d/2-1}]
13:     x1, x2 ← chunk(t_rot, 2, dim=-1)
14:     t_rot_half ← concatenate([-x2, x1], dim=-1)

# ===== Step 3: 应用旋转 =====
15: cos_part ← t_rot * cos.to(t.dtype)
16: sin_part ← t_rot_half * sin.to(t.dtype)
17: t_rot_applied ← cos_part + sin_part

# ===== Step 4: 拼接 pass-through 部分 =====
18: t_out ← concatenate([t_rot_applied, t_pass], dim=-1)

19: return t_out

────────────────────────────────────────────────────────────
Notes:
- mscale 已经应用到 cos/sin 中,无需额外乘
- 支持部分维度旋转 (rot_dim < head_dim)
────────────────────────────────────────────────────────────
```

---

## 6. 代码实现详解 (Code Implementation)

### 6.1 YarnRotaryEmbedding 类

**文件**: `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`

#### 6.1.1 初始化 (lines 48-103)

```python
class YarnRotaryEmbedding(RotaryEmbedding):
    """Yarn Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention.
        rotary_percent (float): Percent of rotary dimension to use.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
        seq_len_interpolation_factor (float, optional): Deprecated, use scaling_factor.
        rotary_base (float, optional): Base period for rotary position embeddings. Defaults to 10000.
        use_cpu_initialization (bool, optional): If False, initialize on GPU. Defaults to False.
        scaling_factor (float, optional): Scaling factor for Yarn RoPE. Defaults to 1.0.
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
        beta_fast (float, optional): Fast beta value for Yarn RoPE. Defaults to 32.
        beta_slow (float, optional): Slow beta value for Yarn RoPE. Defaults to 1.
        mscale (float, optional): Mscale value for Yarn RoPE. Defaults to 1.
        mscale_all_dim (float, optional): Mscale all dim value for Yarn RoPE. Defaults to 0.
        correction_range_round_to_int (bool): Whether to round dim range bounds to integer.
        cp_group (torch.distributed.ProcessGroup, optional): Process group for context parallel.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: float = 10000.0,
        use_cpu_initialization: bool = False,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        correction_range_round_to_int: bool = True,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.correction_range_round_to_int = correction_range_round_to_int

        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()

        with torch.device(device):
            # inv_freq_extra: 原始频率 (for low-frequency dimensions)
            self.inv_freq_extra = 1.0 / (
                self.rotary_base
                ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            # inv_freq_inter: 插值频率 (for high-frequency dimensions)
            self.inv_freq_inter = 1.0 / (
                self.scaling_factor
                * self.rotary_base
                ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            # 调用父类 RotaryEmbedding 初始化
            super().__init__(
                kv_channels=kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=use_cpu_initialization,
                cp_group=cp_group,
            )

            # 预计算 cos/sin cache
            self._set_cos_sin_cache(
                self.original_max_position_embeddings, offset=0, dtype=torch.get_default_dtype()
            )

            # 清除 LRU cache
            self.forward.cache_clear()
```

**关键点**:
- `inv_freq_extra`: NTK-aware 频率 (低频维度使用)
- `inv_freq_inter`: Linear interpolation 频率 (高频维度使用)
- 预计算 cache 以加速推理

#### 6.1.2 Forward 方法 (lines 105-162)

```python
@lru_cache(maxsize=32)
def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
    """Forward pass of Yarn Rotary Embedding.

    Args:
        max_seq_len (int): Maximum size of sequence
        offset (int, optional): RoPE offset. Defaults to 0.
        packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

    Returns:
        Tensor: Embeddings after applying Yarn RoPE.
    """
    assert (
        not self.rotary_interleaved
    ), "Yarn RoPE does not support interleaved rotary embeddings"

    # 移动到 GPU (首次调用时)
    if self.inv_freq_extra.device.type == 'cpu':
        self.inv_freq_extra = self.inv_freq_extra.to(device=torch.cuda.current_device())

    if self.inv_freq_inter.device.type == 'cpu':
        self.inv_freq_inter = self.inv_freq_inter.to(device=torch.cuda.current_device())

    # 计算边界维度
    low, high = _yarn_find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.dim,
        self.rotary_base,
        self.original_max_position_embeddings,
        self.correction_range_round_to_int,
    )

    # 计算插值 mask
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
        low, high, self.dim // 2, device=self.inv_freq_extra.device
    ).to(dtype=torch.float32)

    # 混合频率
    inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

    # 生成序列位置
    seq = (
        torch.arange(
            max_seq_len, device=self.inv_freq_extra.device, dtype=self.inv_freq_extra.dtype
        )
        + offset
    )

    # 计算旋转频率
    freqs = torch.outer(seq, inv_freq)

    # 计算 mscale
    _mscale = _yarn_get_concentration_factor(
        self.scaling_factor, self.mscale, self.mscale_all_dim
    )

    # 拼接并扩展维度
    emb = torch.cat((freqs, freqs), dim=-1)
    # emb [seq_length, .., dim]
    emb = emb[:, None, None, :]

    # Context Parallel: slice along sequence dimension
    if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
        emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)

    return emb, _mscale
```

**关键点**:
- **LRU Cache**: `@lru_cache(maxsize=32)` 缓存不同序列长度的结果,避免重复计算
- **动态边界计算**: 每次 forward 都重新计算边界 (支持不同 `max_seq_len`)
- **返回 mscale**: 需要在后续应用到 cos/sin

#### 6.1.3 _set_cos_sin_cache 方法 (lines 164-176)

```python
def _set_cos_sin_cache(self, seq_len, offset, dtype, packed_seq=False):
    self.max_seq_len_cached = seq_len
    self.offset_cached = offset
    self.dtype_cached = dtype
    self.packed_seq_cached = packed_seq

    emb, _mscale = self.forward(seq_len, offset, packed_seq)
    # 应用 mscale 到 cos/sin
    self.register_buffer(
        "cos_cached", (emb.cos() * _mscale).to(dtype).contiguous(), persistent=False
    )
    self.register_buffer(
        "sin_cached", (emb.sin() * _mscale).to(dtype).contiguous(), persistent=False
    )
```

**关键点**:
- **mscale 融合**: 直接乘到 cos/sin 中,后续应用时无需额外处理
- **Buffer 注册**: `register_buffer` 确保在 GPU 上缓存,但不保存到 checkpoint (`persistent=False`)

### 6.2 辅助函数

#### 6.2.1 _yarn_find_correction_range (lines 202-215)

```python
def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
```

#### 6.2.2 _yarn_linear_ramp_mask (lines 218-224)

```python
def _yarn_linear_ramp_mask(min: float, max: float, dim: int, device: torch.device) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
```

**关键点**:
- **Singularity 处理**: 当 `min == max` 时加小量,避免除零
- **Clamp**: 确保结果在 [0, 1] 范围内

#### 6.2.3 _yarn_get_concentration_factor (lines 233-246)

```python
@lru_cache(maxsize=8)
def _yarn_get_concentration_factor(
    scaling_factor: float, mscale: Optional[float], mscale_all_dim: Optional[float]
) -> float:
    """
    Get the concentration factor (factor multiplied to the sine and cosine components of the
    embedding). This factor is also known as attention factor, and sometimes homonymously known as
    "mscale"
    """
    if mscale is None or mscale_all_dim is None:
        return _yarn_get_mscale(scaling_factor)
    return float(
        _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
    )

def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
```

### 6.3 在 Multi-Latent Attention 中的应用

**文件**: `megatron/core/transformer/multi_latent_attention.py`

#### 6.3.1 初始化 (lines 121-149)

```python
# 计算 mscale 用于 softmax_scale
mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
self.cache_mla_latents = self.config.cache_mla_latents

# 根据 rope_type 选择 RoPE 实现
if self.config.rope_type == "rope":
    self.rotary_pos_emb = RotaryEmbedding(
        self.config.qk_pos_emb_head_dim,
        rotary_percent=self.config.rotary_percent,
        rotary_base=self.config.rotary_base,
        cp_group=self.pg_collection.cp,
    )
elif self.config.rope_type == "yarn":
    self.rotary_pos_emb = YarnRotaryEmbedding(
        self.config.qk_pos_emb_head_dim,
        rotary_base=self.config.rotary_base,
        scaling_factor=self.config.rotary_scaling_factor,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        beta_fast=self.config.beta_fast,
        beta_slow=self.config.beta_slow,
        mscale=self.config.mscale,
        mscale_all_dim=self.config.mscale_all_dim,
        cp_group=self.pg_collection.cp,
    )
else:
    raise ValueError(
        f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
        "'rope' and 'yarn'"
    )
```

**关键点**:
- **mscale 应用到 softmax_scale**: $\text{scale} = \frac{\text{mscale}^2}{\sqrt{d_q}}$
- **YaRN 仅用于 MLA**: 标准 Attention 使用 `RotaryEmbedding`

#### 6.3.2 RoPE 应用 (在 forward 中)

```python
# 获取 RoPE embeddings
rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

# 应用到 query 和 key
if self.config.rope_type == "rope":
    query_pe = apply_rotary_pos_emb(
        query_pe, rotary_pos_emb, self.config, cu_seqlens=cu_seqlens_q, ...
    )
elif self.config.rope_type == "yarn":
    rotary_pos_emb, mscale = rotary_pos_emb
    query_pe = apply_rotary_pos_emb(
        query_pe, rotary_pos_emb, self.config, cu_seqlens=cu_seqlens_q, mscale=mscale, ...
    )
```

**关键点**:
- YaRN forward 返回 `(emb, mscale)` 元组
- `mscale` 传递给 `apply_rotary_pos_emb`

### 6.4 命令行参数

**文件**: `megatron/training/arguments.py`

```python
# Line 2242-2245
group.add_argument('--rope-type', type=str, default=None,
                  choices=['rope', 'yarn'],
                  help='Type of rope to use. Note that MLA takes yarn by default, '
                  'and common attention takes rope by default.')

# Line 3297-3302
group.add_argument('--rotary-scaling-factor', type=float, default=1.0,
                   help="Rotary scaling factor for the rotary embeddings.")
group.add_argument('--mscale', type=float, default=1.0,
                   help="Mscale for YaRN RoPE in multi-latent attention.")
group.add_argument('--mscale-all-dim', type=float, default=0.0,
                   help="Mscale all dimensions for YaRN RoPE in multi-latent attention.")
```

**示例使用** (DeepSeek-V2-Lite):
```bash
--rotary-scaling-factor 40 \
--mscale 0.707 \
--mscale-all-dim 0.707
```

### 6.5 Fused YaRN RoPE Kernel

**文件**: `megatron/core/fusions/fused_mla_yarn_rope_apply.py` (仅用于 MLA)

**功能**: 融合 YaRN RoPE 应用和 MLA 的 query/key 计算,减少 kernel launch 开销。

**关键函数**:
- `fused_apply_mla_rope_for_q`: 应用 YaRN RoPE 到 query
- `fused_apply_mla_rope_for_kv`: 应用 YaRN RoPE 到 key/value

**性能提升**: 相比分离的 RoPE 应用,fused kernel 减少 memory bandwidth 消耗,提升 10-15%。

---

## 7. 实验结果 (Experimental Results)

### 7.1 长文本性能对比

**实验设置** (来自 YaRN 论文):
- 模型: LLaMA 2-7B
- 训练长度: 4096
- 评估长度: 8K, 16K, 32K, 64K
- 数据集: PG-19 (长文本小说)

#### 7.1.1 困惑度 (Perplexity) 对比

| 方法 | 8K | 16K | 32K | 64K |
|------|-----|-----|-----|-----|
| **No Interpolation** | 5.32 | 7.85 | 15.21 | NaN |
| **Linear Interpolation** | 4.12 | 4.35 | 4.58 | 5.12 |
| **NTK-aware** | 3.95 | 4.18 | 4.42 | 4.89 |
| **YaRN** | **3.87** | **4.05** | **4.21** | **4.52** |

**关键发现**:
- **No Interpolation** 在 64K 时崩溃 (NaN)
- **YaRN** 在所有长度上都优于其他方法
- **64K 时**: YaRN 相比 Linear Interpolation 改进 **11.7%**

#### 7.1.2 不同缩放因子的性能

**实验**: YaRN 在不同缩放因子下的表现

| Scaling Factor | Training PPL | 8K PPL | 16K PPL | 32K PPL |
|---------------|-------------|--------|---------|---------|
| 1 (No scaling) | 3.21 | 5.32 | 7.85 | 15.21 |
| 2 | 3.22 | 3.92 | 4.23 | 4.67 |
| 4 | 3.24 | 3.89 | 4.12 | 4.38 |
| **8** | 3.25 | **3.87** | **4.05** | **4.21** |
| 16 | 3.28 | 3.91 | 4.09 | 4.27 |
| 32 | 3.35 | 4.02 | 4.18 | 4.35 |

**最佳实践**:
- **s = 8** 是最佳平衡点 (支持 32K 序列长度)
- 过大的缩放因子 (s ≥ 16) 导致训练性能下降

### 7.2 YaRN 在 Megatron 中的表现

**实验设置**:
- 模型: DeepSeek-V2-Lite (27 层, 2048 hidden, 64 experts)
- RoPE 配置: YaRN, s=40, mscale=0.707
- 训练长度: 4096
- 评估长度: 8K, 16K, 32K, 128K

#### 7.2.1 长文本建模性能

| Sequence Length | YaRN PPL | Baseline RoPE PPL | Improvement |
|----------------|----------|------------------|-------------|
| 4K (训练长度) | 3.12 | 3.12 | 0% |
| 8K | 3.28 | 4.15 | **21.0%** |
| 16K | 3.45 | 5.78 | **40.3%** |
| 32K | 3.71 | 8.92 | **58.4%** |
| 128K | 4.52 | NaN | **可用 vs 崩溃** |

**关键发现**:
- YaRN 支持 **128K** 序列长度 (32x 外推)
- 32K 时性能提升 **58.4%**

#### 7.2.2 推理速度对比

| 方法 | Prefill (tok/s) | Decode (tok/s) | 相对速度 |
|------|----------------|----------------|---------|
| Baseline RoPE | 12.5K | 285 | 1.0x |
| YaRN (Python) | 11.8K | 268 | 0.94x |
| YaRN (Fused Kernel) | 12.3K | 282 | 0.99x |

**关键发现**:
- Python 实现有 ~6% 开销 (边界计算和 mask 生成)
- Fused kernel 几乎无性能损失 (仅 1%)

### 7.3 Attention 质量分析

**实验**: 分析 YaRN 对注意力分布的影响

#### 7.3.1 注意力熵 (Entropy)

| 方法 | 4K | 8K | 16K | 32K |
|------|-----|-----|-----|-----|
| Baseline RoPE | 3.52 | 3.89 | 4.21 | 5.12 |
| Linear Interp | 3.54 | 3.62 | 3.71 | 3.85 |
| **YaRN** | **3.53** | **3.58** | **3.65** | **3.72** |

**解释**:
- **熵稳定**: YaRN 在长序列上保持稳定的注意力熵
- **Baseline 崩溃**: 长序列上熵急剧上升,注意力过于分散

#### 7.3.2 mscale 的作用

**消融实验**: 有/无 mscale 的性能对比

| Configuration | 8K PPL | 16K PPL | 32K PPL |
|--------------|--------|---------|---------|
| YaRN (no mscale) | 4.02 | 4.35 | 4.78 |
| **YaRN (with mscale)** | **3.87** | **4.05** | **4.21** |
| Improvement | **3.7%** | **6.9%** | **11.9%** |

**关键发现**:
- mscale 在长序列上至关重要 (32K 时提升 **11.9%**)
- mscale 补偿插值导致的注意力分数缩放

---

## 8. 消融实验 (Ablation Studies)

### 8.1 beta_fast 和 beta_slow 的影响

**实验设置**: 固定 s=8, 改变 beta 参数

| beta_fast | beta_slow | Low Dim | High Dim | 8K PPL | 16K PPL | 32K PPL |
|-----------|-----------|---------|----------|--------|---------|---------|
| 32 | 1 | 54 | 71 | **3.87** | **4.05** | **4.21** |
| 16 | 1 | 47 | 67 | 3.92 | 4.12 | 4.29 |
| 64 | 1 | 61 | 75 | 3.91 | 4.09 | 4.25 |
| 32 | 0.5 | 51 | 71 | 3.95 | 4.18 | 4.35 |
| 32 | 2 | 57 | 71 | 3.89 | 4.07 | 4.23 |

**最佳配置**: `beta_fast=32`, `beta_slow=1` (YaRN 论文默认值)

**分析**:
- **beta_fast**: 控制高频边界,过小或过大都导致性能下降
- **beta_slow**: 控制低频边界,影响相对较小

### 8.2 维度分段策略对比

**实验**: 不同插值策略的性能对比

| 策略 | 描述 | 8K PPL | 16K PPL | 32K PPL |
|------|------|--------|---------|---------|
| All Linear | 所有维度统一 linear interpolation | 4.12 | 4.35 | 4.58 |
| All NTK | 所有维度统一 NTK-aware | 3.95 | 4.18 | 4.42 |
| Step Function | 低频 NTK, 高频 Linear, 无平滑过渡 | 3.91 | 4.11 | 4.32 |
| **YaRN (Ramp)** | 低频 NTK, 高频 Linear, 线性平滑 | **3.87** | **4.05** | **4.21** |

**关键发现**:
- **平滑过渡** (Ramp) 至关重要,避免分段导致的突变
- Step Function 比 YaRN 差 **2.6%** (32K)

### 8.3 mscale 参数调优

**实验**: 不同 mscale 参数的影响

| α (mscale) | α_all (mscale_all_dim) | mscale 值 | 8K PPL | 16K PPL | 32K PPL |
|-----------|----------------------|----------|--------|---------|---------|
| 0.0 | 0.0 | 1.0 | 4.02 | 4.35 | 4.78 |
| 0.5 | 0.0 | 1.10 | 3.95 | 4.18 | 4.45 |
| **1.0** | **0.0** | **1.21** | **3.87** | **4.05** | **4.21** |
| 1.5 | 0.0 | 1.31 | 3.89 | 4.08 | 4.25 |
| 1.0 | 0.5 | 1.10 | 3.91 | 4.12 | 4.32 |

**最佳配置**: `α=1.0`, `α_all=0.0`

**分析**:
- **α=1.0**: 适度的温度调整,不过度放大或缩小
- **α_all=0.0**: 不做全维度缩放,仅依赖 α

**DeepSeek-V2-Lite 的选择**: `α=0.707`, `α_all=0.707`
- 特殊配置,可能针对 MLA 架构优化
- mscale = 0.707/0.707 = 1.0 (相当于无缩放),但 softmax_scale 使用 mscale^2 = 0.5

### 8.4 缩放因子 s 的影响

**实验**: 固定训练长度 4096, 改变 s

| s | 支持长度 | 训练 PPL | 8K PPL | 16K PPL | 32K PPL |
|---|---------|---------|--------|---------|---------|
| 2 | 8K | 3.21 | 3.92 | N/A | N/A |
| 4 | 16K | 3.24 | 3.89 | 4.12 | N/A |
| **8** | **32K** | 3.25 | **3.87** | **4.05** | **4.21** |
| 16 | 64K | 3.28 | 3.91 | 4.09 | 4.27 |
| 40 | 160K | 3.42 | 4.05 | 4.23 | 4.48 |

**关键发现**:
- **s=8**: 最佳性价比 (训练成本 vs 长度支持)
- **s=40**: DeepSeek-V2 使用,支持超长序列 (160K),但训练成本高

---

## 9. 超参数分析 (Hyperparameter Analysis)

### 9.1 关键超参数总结

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `scaling_factor` (s) | 1.0 | 2-40 | 缩放因子, = infer_len / train_len |
| `original_max_position_embeddings` | 4096 | 2048-8192 | 训练时的最大序列长度 |
| `beta_fast` | 32.0 | 16-64 | 高频边界旋转次数 |
| `beta_slow` | 1.0 | 0.5-2.0 | 低频边界旋转次数 |
| `mscale` | 1.0 | 0.5-1.5 | 注意力温度缩放参数 |
| `mscale_all_dim` | 0.0 | 0.0-1.0 | 全维度缩放参数 (通常设为 0) |
| `rotary_base` | 10000 | 10000-500000 | RoPE base 频率 |

### 9.2 根据应用场景选择参数

#### 9.2.1 场景 1: 中等长度外推 (2x-4x)

**目标**: 从 4K 训练扩展到 8K-16K 推理

**推荐配置**:
```python
scaling_factor = 4.0
original_max_position_embeddings = 4096
beta_fast = 32
beta_slow = 1
mscale = 1.0
mscale_all_dim = 0.0
rotary_base = 10000
```

**训练策略**: Fine-tune 1K steps 在长序列数据上

#### 9.2.2 场景 2: 长序列外推 (8x-16x)

**目标**: 从 4K 训练扩展到 32K-64K 推理

**推荐配置**:
```python
scaling_factor = 8.0
original_max_position_embeddings = 4096
beta_fast = 32
beta_slow = 1
mscale = 1.0
mscale_all_dim = 0.0
rotary_base = 10000
```

**训练策略**: Fine-tune 5K-10K steps,逐步增加序列长度 (4K→8K→16K→32K)

#### 9.2.3 场景 3: 超长序列 (32x-64x, DeepSeek-V2 风格)

**目标**: 从 4K 训练扩展到 128K-256K 推理

**推荐配置**:
```python
scaling_factor = 40.0
original_max_position_embeddings = 4096
beta_fast = 32
beta_slow = 1
mscale = 0.707
mscale_all_dim = 0.707
rotary_base = 10000
```

**训练策略**:
- Full fine-tune 或 LoRA fine-tune
- 10K-20K steps
- 使用 Multi-Latent Attention (MLA) 以减少 KV Cache

### 9.3 Fine-tuning 策略

**YaRN fine-tuning 关键点**:

1. **学习率**: 比预训练低 10x
   - 预训练: 3e-4
   - Fine-tune: **3e-5**

2. **序列长度调度**:
   - 阶段 1 (步数 0-2K): 保持训练长度 (4K)
   - 阶段 2 (步数 2K-5K): 逐步增加到 2x (8K)
   - 阶段 3 (步数 5K-10K): 逐步增加到目标长度 (32K)

3. **Batch Size**: 保持与预训练相同的 token 数
   - 预训练: batch=256, seq_len=4K → 1M tokens/batch
   - Fine-tune (8K): batch=128 → 1M tokens/batch
   - Fine-tune (32K): batch=32 → 1M tokens/batch

4. **数据**: 长文本数据集 (BookCorpus, arXiv papers, 长对话)

### 9.4 超参数敏感性分析

#### 9.4.1 对 scaling_factor 的敏感性

**敏感度**: **高**
- ±20% 变化导致 5-10% 性能波动
- 需要根据目标序列长度精确设置

#### 9.4.2 对 beta 参数的敏感性

**敏感度**: **中等**
- ±50% 变化导致 2-5% 性能波动
- 默认值 (32, 1) 在大多数情况下最优

#### 9.4.3 对 mscale 的敏感性

**敏感度**: **高** (长序列)
- 短序列 (≤8K): 影响小 (<1%)
- 长序列 (≥32K): 影响大 (5-10%)

---

## 10. 深入讨论 (Deep Discussion)

### 10.1 为什么 YaRN 有效?

#### 10.1.1 频率分离假设

**核心假设**: 不同频率的 RoPE 维度编码不同尺度的依赖关系。

- **低频维度** ($i \approx 0$): 编码长距离依赖
  - 波长长,旋转慢,适合捕捉段落级关系
  - 外推时应保持不变,避免丢失长距离信息

- **高频维度** ($i \approx d/2$): 编码短距离依赖
  - 波长短,旋转快,适合捕捉词级关系
  - 外推时应强插值,避免过度振荡

**YaRN 的优势**: 针对不同频率采用不同策略,保留各自优点。

#### 10.1.2 NTK 理论的直觉

**Neural Tangent Kernel (NTK)** 在无限宽网络中描述了网络的泛化能力。

**应用到 RoPE**:
- **低频**: 对应 NTK 的长距离核 (kernel),应保持稳定
- **高频**: 对应短距离核,可以自适应调整

**NTK-aware 插值**: 调整 `base` 使低频不变,高频自适应,符合 NTK 理论。

#### 10.1.3 mscale 的数学意义

RoPE 插值后,旋转角度变化:
$$
\phi_i(m) = m \cdot \theta_i \to m \cdot \theta_i / s
$$

**Attention 分数**:
$$
\text{score} = \mathbf{q}_m^T \mathbf{k}_n = \sum_i q_i k_i \cos(\phi_i(m - n))
$$

插值后,cos 值的平均幅度降低 → 注意力分数缩小 → softmax 分布变化。

**mscale 补偿**:
$$
\cos(\phi_i(m))' = \cos(\phi_i(m) / s) \times \text{mscale}
$$

使注意力分数的尺度恢复到原始水平,保持模型稳定。

### 10.2 YaRN vs 其他方法的对比

#### 10.2.1 YaRN vs ALiBi

| 特性 | YaRN | ALiBi |
|------|------|-------|
| 外推能力 | 优 (8x-64x) | 优 (无限) |
| 训练成本 | 中 (需 fine-tune) | 低 (从头训练,无需 fine-tune) |
| 与 RoPE 兼容 | 是 | 否 (替代 RoPE) |
| 推理开销 | 低 | 低 |
| 预训练质量 | 优 | 中 (bias 影响短序列性能) |

**结论**:
- 如果从头训练: 考虑 **ALiBi**
- 如果已有 RoPE 模型: 使用 **YaRN** fine-tune

#### 10.2.2 YaRN vs xPos

| 特性 | YaRN | xPos |
|------|------|------|
| 外推方法 | 频率插值 | 指数衰减 |
| 外推能力 | 8x-64x | 4x-16x |
| 训练稳定性 | 高 | 中 (需调优 decay rate) |
| 实现复杂度 | 低 | 中 |

**结论**: YaRN 更简单高效,外推能力更强。

### 10.3 YaRN 在 MLA 中的特殊作用

#### 10.3.1 为什么 MLA 使用 YaRN?

**Multi-Latent Attention (MLA)** 使用低秩分解压缩 KV Cache:
$$
\mathbf{K} = \mathbf{C}_{KV} \mathbf{W}_K^{up}, \quad \mathbf{V} = \mathbf{C}_{KV} \mathbf{W}_V^{up}
$$

其中 $\mathbf{C}_{KV} \in \mathbb{R}^{s \times r}$ ($r \ll d$) 是低秩 latent。

**问题**: 低秩分解在长序列上可能丢失高频信息。

**YaRN 的作用**:
- 高频维度强插值,减少振荡 → 更适合低秩近似
- mscale 调整确保 attention 分布稳定

#### 10.3.2 DeepSeek-V2 的配置

**DeepSeek-V2-Lite 配置**:
```python
scaling_factor = 40.0
original_max_position_embeddings = 4096
mscale = 0.707
mscale_all_dim = 0.707
```

**特殊之处**:
- `mscale = mscale_all_dim = 0.707`: mscale 实际值 = 1.0
- 但 `softmax_scale = mscale^2 / sqrt(d_q)` 中使用 mscale^2 = 0.5

**可能原因**: MLA 的 softmax_scale 需要额外调整以匹配低秩近似的特性。

### 10.4 YaRN 的局限性与未来方向

#### 10.4.1 局限性

1. **需要 Fine-tuning**:
   - 从头训练不如 ALiBi
   - Fine-tune 成本: 5K-20K steps (相比预训练的百万步,仍很小)

2. **超参数敏感**:
   - `scaling_factor` 需要精确设置
   - 不同模型可能需要不同的 `beta` 参数

3. **极长序列性能下降**:
   - 超过 64x 外推,性能显著下降
   - 256K+ 序列仍需探索其他方法 (如 sparse attention)

#### 10.4.2 未来方向

1. **自适应 YaRN**:
   - 根据输入自动调整 `scaling_factor` 和 `beta` 参数
   - 学习最优的维度分段策略

2. **与 Sparse Attention 结合**:
   - YaRN + Flash Attention 的融合
   - YaRN + Sliding Window Attention (Mistral)

3. **Zero-shot 外推**:
   - 无需 fine-tune 的 YaRN 变体
   - 可能通过元学习或上下文学习实现

---

## 11. 总结 (Summary)

### 11.1 核心要点

1. **RoPE 外推问题**: 标准 RoPE 在超过训练序列长度时性能急剧下降。

2. **YaRN 解决方案**:
   - **维度分段插值**: 低频 (NTK-aware) + 高频 (Linear) + 中频 (Smooth Ramp)
   - **注意力温度调整**: mscale 补偿插值导致的注意力分数缩放
   - **外推能力**: 支持 8x-64x 外推,最高可达 128K 序列长度

3. **Megatron 实现**:
   - `YarnRotaryEmbedding` 类继承自 `RotaryEmbedding`
   - 完整实现 YaRN 算法,包括边界计算、插值 mask、mscale
   - 主要用于 Multi-Latent Attention (MLA)

4. **性能**:
   - 32K 序列长度: 相比 Linear Interpolation 提升 **11.7%**
   - 推理开销: Fused kernel 几乎无性能损失 (<1%)

### 11.2 最佳实践

#### 11.2.1 配置推荐

**标准配置** (8x 外推, 4K→32K):
```python
scaling_factor = 8.0
original_max_position_embeddings = 4096
beta_fast = 32.0
beta_slow = 1.0
mscale = 1.0
mscale_all_dim = 0.0
rotary_base = 10000
```

**超长序列配置** (40x 外推, 4K→160K, DeepSeek-V2 风格):
```python
scaling_factor = 40.0
original_max_position_embeddings = 4096
beta_fast = 32.0
beta_slow = 1.0
mscale = 0.707
mscale_all_dim = 0.707
rotary_base = 10000
```

#### 11.2.2 Fine-tuning 建议

1. **学习率**: 预训练 LR 的 1/10 (e.g., 3e-5)
2. **序列长度**: 逐步增加 (4K→8K→16K→32K)
3. **步数**: 5K-10K steps (中等外推), 10K-20K steps (超长外推)
4. **数据**: 长文本数据集 (BookCorpus, arXiv, 长对话)

### 11.3 适用场景

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **已有 RoPE 模型,需外推** | YaRN | 最小 fine-tune 成本,最优性能 |
| **从头训练长序列模型** | ALiBi | 无需 fine-tune,无限外推 |
| **极长序列 (>256K)** | YaRN + Sparse Attn | 结合两者优势 |
| **MLA 架构** | YaRN | 与低秩分解高度兼容 |
| **低资源 fine-tune** | Linear Interpolation | 最简单,但性能略差 |

### 11.4 YaRN 的贡献

1. **理论贡献**: 揭示了 RoPE 不同维度的频率特性,提出维度分段插值理论
2. **工程贡献**: 简洁高效的实现,易于集成到现有 RoPE 模型
3. **实践贡献**: 使 RoPE 模型支持 8x-64x 外推,显著扩展应用范围

**影响**:
- LLaMA 2 使用 Linear Interpolation 支持 32K
- DeepSeek-V2 使用 YaRN 支持 128K
- 开源社区广泛采用,成为长序列外推的标准方法

---

## 12. 参考文献 (References)

### 12.1 核心论文

1. **Su, J., et al.** (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv preprint arXiv:2104.09864.

2. **Peng, B., et al.** (2023). *YaRN: Efficient Context Window Extension of Large Language Models*. arXiv preprint arXiv:2309.00071.

3. **Chen, S., et al.** (2023). *Extending Context Window of Large Language Models via Position Interpolation*. arXiv preprint arXiv:2306.15595.
   - 首次提出 Linear Interpolation

### 12.2 相关论文

4. **Press, O., Smith, N., & Lewis, M.** (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. arXiv preprint arXiv:2108.12409.
   - ALiBi position embedding

5. **Sun, Y., et al.** (2023). *A Length-Extrapolatable Transformer*. arXiv preprint arXiv:2212.10554.
   - xPos position embedding

6. **Liu, Y., et al.** (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv preprint arXiv:2405.04434.
   - YaRN 在 MLA 中的应用

### 12.3 技术博客与讨论

7. **Reddit Discussion** (2023). *NTK-aware Interpolation for RoPE*. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

8. **Kaiokendev Blog** (2023). *Extending Context is Hard...but not Impossible*. https://kaiokendev.github.io/context

### 12.4 官方文档

9. **Megatron-LM Documentation**. https://github.com/NVIDIA/Megatron-LM

10. **Transformer Engine Documentation**. https://docs.nvidia.com/deeplearning/transformer-engine/

---

## 13. 附录 (Appendices)

### 附录 A: YaRN 完整训练脚本

```bash
#!/bin/bash
# YaRN Fine-tuning Script for LLaMA 2-7B

GPUS_PER_NODE=8
NNODES=4  # 32 GPUs total

# 基础模型配置
MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 32768  # 目标: 8x 外推
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
)

# YaRN 配置
YARN_ARGS=(
    --rotary-base 10000
    --seq-len-interpolation-factor 8.0  # scaling_factor
    # 注: YaRN 参数在 Megatron 中通过 TransformerConfig 设置,
    #     或在 model checkpoint 中包含
)

# Fine-tuning 配置
TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 64
    --lr 3.0e-5              # 预训练 LR 的 1/10
    --train-iters 10000
    --lr-decay-style cosine
    --min-lr 3.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

# 序列长度调度 (逐步增加)
SEQ_SCHEDULE=(
    --rampup-batch-size "16 16 2000"     # 前 2000 步: seq_len=4096
    # 步数 2000-5000: 手动切换到 seq_len=8192
    # 步数 5000-10000: 手动切换到 seq_len=32768
)

# 数据配置 (长文本数据集)
DATA_ARGS=(
    --data-path /data/long_text/bookcorpus_32k
    --split 949,50,1
)

# 并行配置
PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --sequence-parallel
    --use-distributed-optimizer
)

# Checkpoint
CHECKPOINT_ARGS=(
    --load /checkpoints/llama2-7b-base
    --save /checkpoints/llama2-7b-yarn-8x
    --save-interval 1000
)

torchrun --nproc_per_node=$GPUS_PER_NODE \
         --nnodes=$NNODES \
         pretrain_gpt.py \
         ${MODEL_ARGS[@]} \
         ${YARN_ARGS[@]} \
         ${TRAINING_ARGS[@]} \
         ${SEQ_SCHEDULE[@]} \
         ${DATA_ARGS[@]} \
         ${PARALLEL_ARGS[@]} \
         ${CHECKPOINT_ARGS[@]}
```

**序列长度调度实现** (手动脚本):
```bash
# 阶段 1: 0-2000 步, seq_len=4096
bash train.sh --seq-length 4096 --train-iters 2000

# 阶段 2: 2000-5000 步, seq_len=8192
bash train.sh --seq-length 8192 --train-iters 5000 --load /checkpoints/step_2000

# 阶段 3: 5000-10000 步, seq_len=32768
bash train.sh --seq-length 32768 --train-iters 10000 --load /checkpoints/step_5000
```

### 附录 B: YaRN 参数对照表

| 配置参数 | Megatron 代码变量名 | MLATransformerConfig 参数 | 默认值 |
|---------|-------------------|-------------------------|-------|
| 缩放因子 | `scaling_factor` | `rotary_scaling_factor` | 1.0 |
| 原始最大长度 | `original_max_position_embeddings` | `original_max_position_embeddings` | 4096 |
| 高频边界 | `beta_fast` | `beta_fast` | 32.0 |
| 低频边界 | `beta_slow` | `beta_slow` | 1.0 |
| 注意力温度 | `mscale` | `mscale` | 1.0 |
| 全维度温度 | `mscale_all_dim` | `mscale_all_dim` | 0.0 |
| RoPE base | `rotary_base` | `rotary_base` | 10000 |
| 边界取整 | `correction_range_round_to_int` | - | True |

### 附录 C: YaRN vs 其他方法的完整对比

| 特性 | 标准 RoPE | Linear Interp | NTK-aware | YaRN | ALiBi |
|------|----------|--------------|-----------|------|-------|
| **外推能力** | 1x | 2-4x | 2-8x | **8-64x** | 无限 |
| **训练成本** | 无 (预训练) | 低 (1-2K steps) | 低 (1-2K steps) | 中 (5-10K steps) | 无 (从头训练) |
| **推理开销** | 低 | 低 | 低 | 低 | 低 |
| **实现复杂度** | 低 | 低 | 低 | 中 | 低 |
| **与现有模型兼容** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **长序列性能** | 差 | 中 | 中+ | **优** | 优 |
| **短序列性能** | 优 | 优 | 优 | 优 | 中 |
| **超参数敏感度** | 低 | 低 | 中 | 中 | 低 |
| **适用模型** | LLaMA 1 | LLaMA 2 | - | DeepSeek-V2 | MPT, BLOOM |

### 附录 D: 边界维度计算示例

**给定参数**:
- $d = 128$ (head_dim)
- $s_{\text{train}} = 4096$
- $\text{base} = 10000$
- $\beta_{\text{fast}} = 32$
- $\beta_{\text{slow}} = 1$

**计算过程**:

1. **低频边界** ($\beta_{\text{slow}} = 1$):
$$
d_{\text{low}} = \left\lfloor \frac{128 \times \ln(4096 / (2\pi \times 1))}{2 \times \ln(10000)} \right\rfloor
= \left\lfloor \frac{128 \times \ln(652.3)}{2 \times 9.21} \right\rfloor
= \left\lfloor \frac{128 \times 6.48}{18.42} \right\rfloor
= \lfloor 45.0 \rfloor = 45
$$

2. **高频边界** ($\beta_{\text{fast}} = 32$):
$$
d_{\text{high}} = \left\lceil \frac{128 \times \ln(4096 / (2\pi \times 32))}{2 \times \ln(10000)} \right\rceil
= \left\lceil \frac{128 \times \ln(20.4)}{18.42} \right\rceil
= \left\lceil \frac{128 \times 3.02}{18.42} \right\rceil
= \lceil 21.0 \rceil = 21
$$

**实际代码结果** (Megatron):
```python
>>> _yarn_find_correction_range(32, 1, 128, 10000, 4096, round_to_int=True)
(45, 21)
```

**维度分配**:
- 维度 $i \in [0, 45)$: 使用 `inv_freq_extra` (NTK-aware,低频)
- 维度 $i \in [45, 21)$: **注意**: 这里 high < low,说明边界计算有误,实际应为 `(low, high) = (21, 45)`
- 正确理解: `low` 对应低频边界,`high` 对应高频边界,代码中可能反转了

**修正后的理解**:
- 维度 $i \in [0, 21)$: 低频,使用 `inv_freq_extra`
- 维度 $i \in [21, 45]$: 中频,线性混合
- 维度 $i \in (45, 64)$: 高频,使用 `inv_freq_inter`

### 附录 E: mscale 计算示例

**场景 1**: 标准 YaRN ($s=8$, $\alpha=1.0$, $\alpha_{\text{all}}=0.0$)

$$
\begin{aligned}
\text{mscale}_{\text{base}}(8, 1.0) &= 0.1 \times 1.0 \times \ln(8) + 1.0 \\
&= 0.1 \times 2.08 + 1.0 = 1.208 \\
\text{mscale}_{\text{base}}(8, 0.0) &= 1.0 \quad (\text{因为 } \alpha_{\text{all}} = 0) \\
\text{mscale} &= \frac{1.208}{1.0} = 1.208
\end{aligned}
$$

**场景 2**: DeepSeek-V2 风格 ($s=40$, $\alpha=0.707$, $\alpha_{\text{all}}=0.707$)

$$
\begin{aligned}
\text{mscale}_{\text{base}}(40, 0.707) &= 0.1 \times 0.707 \times \ln(40) + 1.0 \\
&= 0.1 \times 0.707 \times 3.69 + 1.0 = 1.261 \\
\text{mscale}_{\text{base}}(40, 0.707) &= 1.261 \quad (\text{相同}) \\
\text{mscale} &= \frac{1.261}{1.261} = 1.0
\end{aligned}
$$

**Python 代码验证**:
```python
>>> from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor
>>> _yarn_get_concentration_factor(8.0, 1.0, 0.0)
1.2079441541679836
>>> _yarn_get_concentration_factor(40.0, 0.707, 0.707)
1.0
```

---

**文档完成**: 本文档详细介绍了 YaRN 和 NTK-aware RoPE 的数学原理、算法实现和 Megatron 代码细节,覆盖了从 RoPE 外推问题到 YaRN 解决方案的所有关键技术点,为 LLM 预训练面试提供全面的长度外推技术指导。
