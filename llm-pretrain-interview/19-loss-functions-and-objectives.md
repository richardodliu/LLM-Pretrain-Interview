# 19. 损失函数设计与优化目标

> **文档编号**: 19
> **所属部分**: 第二部分 - 深度学习基础 (11-20)
> **代码位置**: `megatron/core/tensor_parallel/cross_entropy.py`
> **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM v0.12.0实际代码)

---

## 1. 引言

### 1.1 概述

损失函数(Loss Function)是监督学习的核心,定义了模型预测与真实目标之间的差异程度。对于大语言模型预训练,损失函数设计直接影响训练稳定性、收敛速度和最终性能。本文档系统讲解大语言模型常用的损失函数,包括交叉熵损失、困惑度、标签平滑等技术,以及 Megatron-LM 中分布式损失计算的高效实现。

### 1.2 前置知识

**必备数学基础**:
- 概率论: 概率分布、似然函数、最大似然估计
- 信息论: 信息熵、KL散度、交叉熵
- 微积分: 梯度、Hessian矩阵、数值稳定性

**必备深度学习知识**:
- 神经网络基础与反向传播
- Softmax函数与数值稳定性
- 浮点数表示(FP32/FP16/BF16)

**相关概念**:
- 自动微分与计算图
- Softmax的梯度推导
- 分布式训练中的通信模式

### 1.3 文档组织结构

本文档首先介绍信息论基础和损失函数的数学原理(第2-3节),然后详细讲解分类损失函数、语言建模损失等常见损失设计(第4-7节),接着讨论数值稳定性问题和高效实现(第8-9节),最后深入探讨损失函数设计原则、正则化策略和超参数调优(第10-12节)。

---

## 2. 相关工作

### 2.1 历史发展

**早期深度学习**（2011-2015）:
- **交叉熵损失** (Kullback-Leibler, 1951): 信息论基础
- **Softmax分类器** (Bridle, 1989): 多分类标准方法
- **MSE损失** (Legendre, 1805): 传统回归方法

**神经网络时代**（2012-2018）:
- **图像分类** (AlexNet, VGG): 交叉熵 + SGD
- **Focal Loss** (Lin et al., 2017): 处理类别不平衡
- **Triplet Loss** (Hinton et al., 2015): 相似性学习

**深度学习规模化**（2018-2021）:
- **Transformer时代** (Vaswani et al., 2017): 语言建模损失
- **BERT** (Devlin et al., 2018): 掩码语言建模 (MLM)
- **GPT-2/3** (Radford et al., 2019/2020): 因果语言建模 (CLM)

**大模型预训练**（2021-至今）:
- **T5** (Raffel et al., 2019): 多任务学习损失设计
- **PALM** (Chowdhery et al., 2022): 混合任务预训练
- **Megatron-LM** (NVIDIA, 2023): 分布式损失计算

### 2.2 Megatron-LM 的创新

**并行交叉熵** (`VocabParallelCrossEntropy`):
1. **词汇并行分片**: 将词汇表分割到多个GPU,每个GPU只计算分片内的logits
2. **分布式Softmax**: 通过all-reduce操作得到完整的分母项
3. **数值稳定性**: 使用LogSumExp技巧防止浮点数溢出
4. **标签平滑支持**: 内置标签平滑正则化

**设计特点**:
- 支持FP16/BF16混合精度训练
- 融合算子减少kernel启动开销
- 与张量并行完全集成
- 支持梯度缩放和损失缩放

### 2.3 与其他框架的对比

| 特性 | PyTorch原生 | DeepSpeed | Megatron-LM |
|-----|-----------|-----------|-------------|
| 并行交叉熵 | ✗ | ✗ | ✅ |
| 标签平滑 | ✅ | ✅ | ✅ |
| 词汇并行 | ✗ | ✅ | ✅ |
| LogSumExp稳定性 | ✅ | ✅ | ✅ |
| 梯度累积支持 | ✅ | ✅ | ✅ |

---

## 3. 符号定义

### 3.1 数学符号表

| 符号 | 含义 | 维度 |
|-----|------|------|
| $V$ | 词汇表大小 | 标量 |
| $\mathcal{V}$ | 词汇表集合 | 有限集 |
| $\mathbf{x}$ | 输入序列 | $[S]$ (序列长度) |
| $\mathbf{y}$ | 目标标签序列 | $[S]$ |
| $y_t$ | 位置$t$的目标词汇ID | 整数 |
| $\mathbf{z}$ | logits (原始输出) | $[S, V]$ |
| $z_{t,v}$ | 位置$t$对词汇ID $v$的logit | 浮点数 |
| $\mathbf{p}$ | 概率分布 | $[S, V]$ |
| $p_{t,v}$ | $P(v \mid \mathbf{x}_{<t})$ | $[0, 1]$ |
| $L$ | 损失 | 标量 |
| $L_t$ | 位置$t$的交叉熵损失 | 标量 |
| $\text{PPL}$ | 困惑度 | 标量 |
| $\alpha$ | 标签平滑系数 | $[0, 1)$ |
| $H$ | 信息熵 | 标量 |
| $D_{\text{KL}}$ | KL散度 | 标量 |
| $\theta$ | 模型参数 | 高维向量 |

### 3.2 代码变量约定

| 变量 | 含义 | 类型 |
|------|------|------|
| `vocab_parallel_logits` | 分片的logits | `Tensor[*, partition_vocab_size]` |
| `target` | 目标词汇ID序列 | `Tensor[*]` (long) |
| `logits_max` | logits的最大值(数值稳定) | `Tensor[*]` |
| `exp_logits` | $\exp(\text{logits} - \text{max})$ | `Tensor[*, V]` |
| `sum_exp_logits` | 分母项 $\sum_v \exp(z_v - \max)$ | `Tensor[*]` |
| `softmax` | 注意力权重 | `Tensor[*, V]` |
| `loss` | 每样本的损失 | `Tensor[*]` |
| `label_smoothing` | 标签平滑系数 | `float` |

---

## 4. 信息论基础

### 4.1 信息熵

**定义 4.1**: 信息熵 (Information Entropy)

对于离散概率分布 $p(x)$,信息熵定义为:
$$H(p) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

通常使用自然对数,则熵的单位为"纳特"(nat)。

**直觉理解**:
- 熵度量分布的"不确定性"或"混乱程度"
- 均匀分布(最不确定): $H = \log |\mathcal{X}|$ (最大)
- Dirac分布(完全确定): $H = 0$ (最小)

**示例**:
```
分布 p = [1, 0, 0]        # Dirac分布
H = -(1*log(1) + 0 + 0) = 0

分布 p = [0.33, 0.33, 0.34] # 均匀分布
H ≈ -(3 * 0.33 * log(0.33)) ≈ 1.099

分布 p = [0.7, 0.2, 0.1]   # 非均匀分布
H = -(0.7*log(0.7) + 0.2*log(0.2) + 0.1*log(0.1)) ≈ 0.802
```

### 4.2 KL散度与相对熵

**定义 4.2**: KL散度 (Kullback-Leibler Divergence)

给定两个概率分布 $p$ 和 $q$,KL散度定义为:
$$D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

**性质**:
1. **非负性**: $D_{\text{KL}}(p \| q) \geq 0$,当且仅当 $p = q$ 时等号成立
2. **非对称性**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ (不是真正的距离)
3. **方向性**: $D_{\text{KL}}(p \| q)$ 度量用 $q$ 逼近 $p$ 的信息损失

**展开推导**:
$$
\begin{aligned}
D_{\text{KL}}(p \| q) &= \sum_{x} p(x) \log \frac{p(x)}{q(x)} \\
&= \sum_{x} p(x) [\log p(x) - \log q(x)] \\
&= -\sum_{x} p(x) \log q(x) + \sum_{x} p(x) \log p(x) \\
&= -\sum_{x} p(x) \log q(x) - H(p) \\
&= H(p, q) - H(p)
\end{aligned}
$$

其中 $H(p, q) = -\sum_x p(x) \log q(x)$ 称为"交叉熵"。

### 4.3 交叉熵

**定义 4.3**: 交叉熵 (Cross-Entropy)

给定真实分布 $p$ 和预测分布 $q$,交叉熵定义为:
$$H(p, q) = -\sum_{x} p(x) \log q(x) = -\mathbb{E}_{x \sim p}[\log q(x)]$$

**与KL散度的关系**:
$$D_{\text{KL}}(p \| q) = H(p, q) - H(p)$$

当真实分布 $p$ 固定时,最小化交叉熵等价于最小化KL散度。

**分类任务的交叉熵**:

对于单个样本,真实标签是 one-hot 向量 $\mathbf{y} \in \{0, 1\}^V$,则:
$$H(\mathbf{y}, \mathbf{\hat{p}}) = -\sum_{v=1}^{V} y_v \log \hat{p}_v$$

假设 $y_c = 1$ (真实类别是 $c$),其他位置为 0,则:
$$H(\mathbf{y}, \mathbf{\hat{p}}) = -\log \hat{p}_c$$

这就是标准的**交叉熵损失**。

### 4.4 最大似然估计与交叉熵的关系

**定理 4.1**: 最大似然估计等价于最小化交叉熵

给定样本 $\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^N$,最大化似然函数:
$$\ell(\theta) = \log \prod_{i=1}^N p(y^{(i)} \mid \mathbf{x}^{(i)}; \theta) = \sum_{i=1}^N \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \theta)$$

等价于最小化平均交叉熵:
$$L(\theta) = -\frac{1}{N} \sum_{i=1}^N \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \theta)$$

---

## 5. 分类损失函数

### 5.1 标准交叉熵损失

**定义 5.1**: 交叉熵损失 (Cross-Entropy Loss)

给定 logits $\mathbf{z} \in \mathbb{R}^V$ 和目标标签 $y \in \{1, 2, \ldots, V\}$,交叉熵损失定义为:
$$L_{\text{CE}} = -\log p(y \mid \mathbf{z}) = -\log \frac{\exp(z_y)}{\sum_{v=1}^{V} \exp(z_v)}$$

**步骤分解**:

**步骤1: Softmax计算**
$$p_v = \text{softmax}(z_v) = \frac{\exp(z_v)}{\sum_{v'=1}^{V} \exp(z_{v'})}$$

**步骤2: 交叉熵损失**
$$L = -\log p_y$$

**步骤3: 数值稳定的实现**

直接计算 $\exp(z_v)$ 会导致数值溢出或下溢。使用LogSumExp技巧:
$$\log \sum_{v} \exp(z_v) = z_{\max} + \log \sum_{v} \exp(z_v - z_{\max})$$

其中 $z_{\max} = \max_v z_v$。

因此:
$$L = \log \sum_{v} \exp(z_v) - z_y = z_{\max} + \log \sum_{v} \exp(z_v - z_{\max}) - z_y$$

### 5.2 梯度推导

**定理 5.1**: 交叉熵损失的梯度

对于 logit $z_v$ 的梯度:
$$\frac{\partial L}{\partial z_v} = p_v - \mathbb{1}(v = y)$$

**证明**:
$$
\begin{aligned}
\frac{\partial L}{\partial z_v} &= \frac{\partial}{\partial z_v} \left(-\log p_y\right) \\
&= -\frac{\partial}{\partial z_v} \log p_y \\
&= -\frac{1}{p_y} \frac{\partial p_y}{\partial z_v}
\end{aligned}
$$

对于 Softmax 的梯度:
$$\frac{\partial p_y}{\partial z_v} = p_y(p_v - \mathbb{1}(v = y)) = \begin{cases}
p_y(p_v - 1) = -p_y(1 - p_v) & \text{if } v = y \\
p_y \cdot p_v & \text{if } v \neq y
\end{cases}$$

因此:
$$\frac{\partial L}{\partial z_v} = -\frac{1}{p_y} \cdot p_y(p_v - \mathbb{1}(v = y)) = p_v - \mathbb{1}(v = y)$$

**直觉理解**: 梯度等于模型预测的概率减去真实标签指示。这导致:
- 当 $p_v$ 接近 1(正确预测): 梯度接近 0
- 当 $p_v$ 接近 0(错误预测): 梯度接近 1 或 -1

### 5.3 类别不平衡问题与Focal Loss

**问题陈述**: 在高度不平衡的数据集上(如目标检测),简单的交叉熵倾向于被主导类别的样本驱动。

**Focal Loss** (Lin et al., 2017):
$$L_{\text{focal}} = -\alpha_t (1 - p_t)^{\gamma} \log p_t$$

其中:
- $p_t = p$ if $y = 1$ else $1 - p$: 真实类别的模型预测概率
- $\gamma \geq 0$: 焦点参数(通常 $\gamma = 2$),控制困难样本的加权
- $\alpha_t$: 类别权重(平衡因子)

**机制分析**:
1. **$(1-p_t)^{\gamma}$ 项**: 困难样本(小 $p_t$)的损失权重更大
   - 当 $p_t = 0.9$: $(1-0.9)^2 = 0.01$,损失被缩小100倍
   - 当 $p_t = 0.5$: $(1-0.5)^2 = 0.25$,损失被缩小4倍

2. **效果**: 自动关注困难样本,下调简单样本的权重

**大模型预训练中的适用性**: Focal Loss在LLM预训练中较少使用,原因是:
- LLM预训练通常使用均衡的数据混合
- 困难样本的自动加权可能导致训练不稳定
- 更简单的标签平滑和权重衰减足以处理不平衡

---

## 6. 语言建模的损失函数

### 6.1 因果语言建模(CLM)

**定义 6.1**: 因果语言模型的目标函数

给定输入序列 $\mathbf{x} = [x_1, x_2, \ldots, x_S]$,因果语言模型(CLM)的损失为:
$$L_{\text{CLM}} = -\sum_{t=1}^{S} \log p(x_t \mid x_{<t}; \theta) = \sum_{t=1}^{S} L_t$$

其中 $L_t = -\log p(x_t \mid x_1, \ldots, x_{t-1}; \theta)$ 是位置 $t$ 的交叉熵损失。

**展开计算**:

对于每个位置 $t$:
1. 计算logits: $\mathbf{z}_t = f_\theta(x_{<t}) \in \mathbb{R}^V$
2. 应用Softmax: $p_t = \text{softmax}(\mathbf{z}_t)$
3. 计算损失: $L_t = -\log p_t[x_t]$

**总损失**:
$$L = \frac{1}{S} \sum_{t=1}^{S} L_t$$

通常会使用梯度累积,在多个batch上累积损失,然后归一化。

**实现示例** (PyTorch):
```python
def compute_clm_loss(logits, labels, vocab_size):
    """
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] (目标词汇ID)

    Returns:
        loss: 标量
    """
    # Reshape为2D便于计算
    logits_2d = logits.view(-1, vocab_size)  # [B*S, V]
    labels_1d = labels.view(-1)               # [B*S]

    # 交叉熵损失
    loss = F.cross_entropy(logits_2d, labels_1d, reduction='mean')
    return loss
```

### 6.2 掩码语言建模(MLM)

**定义 6.2**: 掩码语言模型的目标函数

在BERT等模型中,掩码语言建模(MLM)随机选择15%的token进行掩码,目标是预测这些被掩码的token。

$$L_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log p(x_t \mid \mathbf{x}_{\text{masked}}; \theta)$$

其中 $\mathcal{M}$ 是被掩码的位置集合。

**掩码策略** (BERT):
- 80%: 替换为 [MASK] token
- 10%: 替换为随机token
- 10%: 保持原token不变

**目的**: 迫使模型学习双向上下文表示,而不仅仅是单向的因果关系。

**实现示例**:
```python
def compute_mlm_loss(logits, labels, attention_mask=None):
    """
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] (被掩码位置为token_id, 未掩码为-100)
        attention_mask: [batch_size, seq_len]

    Returns:
        loss: 标量
    """
    # 只计算被掩码位置的损失
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,  # 忽略未掩码位置
        reduction='mean'
    )
    return loss
```

### 6.3 困惑度(Perplexity)

**定义 6.3**: 困惑度 (Perplexity)

困惑度是语言建模中的标准评估指标,定义为:
$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(x_i)\right) = \exp(\bar{L})$$

其中 $\bar{L} = \frac{1}{N} \sum_{i=1}^{N} L_i$ 是平均损失。

**直觉理解**:
- PPL = 1: 模型完美预测(损失 = 0)
- PPL = 词汇表大小: 等价于均匀分布(无信息)
- PPL 越小,模型表现越好

**示例计算**:
```
如果平均损失 L̄ = 0.5:
PPL = exp(0.5) ≈ 1.65

如果平均损失 L̄ = 2.0:
PPL = exp(2.0) ≈ 7.39

如果平均损失 L̄ = 5.0:
PPL = exp(5.0) ≈ 148.41
```

**与损失的关系**:
- 损失下降 → PPL下降(指数级)
- 损失为0 → PPL为1(完美预测)
- 损失为ln(V) → PPL为V(随机预测)

### 6.4 序列级别的损失计算

**问题**: 不同样本长度不同,如何公平地计算损失?

**解决方案**:

**方案1: 序列级平均** (最常用)
$$L = \frac{1}{S} \sum_{t=1}^{S} L_t$$

优点: 独立于序列长度,便于比较不同数据
缺点: 长序列中的错误被平均化

**方案2: 总体求和**
$$L = \sum_{t=1}^{S} L_t$$

优点: 保留长序列的信息
缺点: 长序列的损失自然更大,难以比较

**Megatron-LM中的做法**:
```python
# 标准方法: 在一个batch内的所有有效token上平均
loss = loss.sum() / num_tokens
```

---

## 7. 序列到序列的损失函数

### 7.1 Encoder-Decoder架构

**定义 7.1**: Seq2Seq的目标函数

给定输入序列 $\mathbf{x} = [x_1, \ldots, x_S_x]$ 和目标序列 $\mathbf{y} = [y_1, \ldots, y_{S_y}]$,Seq2Seq模型的损失为:
$$L_{\text{Seq2Seq}} = -\sum_{t=1}^{S_y} \log p(y_t \mid y_{<t}, \mathbf{x}; \theta)$$

**两个阶段**:
1. **编码器**: 将输入序列编码为上下文向量 $\mathbf{c} = \text{Encoder}(\mathbf{x})$
2. **解码器**: 在给定编码器输出和前面的解码器输出条件下,预测下一个token

**与语言建模的区别**:
- CLM: 只依赖于前面的token $y_{<t}$
- Seq2Seq: 同时依赖前面的token和整个输入序列

### 7.2 多任务学习中的损失组合

**定义 7.2**: 加权多任务损失

当训练多个任务时,总损失为:
$$L_{\text{total}} = \sum_{k=1}^{K} \lambda_k L_k$$

其中 $\lambda_k$ 是第 $k$ 个任务的权重。

**权重选择策略**:

**策略1: 固定权重**
```python
lambda_k = {0.8, 0.2} # 主任务 80%, 辅助任务 20%
loss = 0.8 * loss_task1 + 0.2 * loss_task2
```

**策略2: 动态权重** (Uncertainty Weighting, Kendall et al., 2018)
$$\lambda_k = \frac{1}{2\sigma_k^2}$$

其中 $\sigma_k^2$ 是第 $k$ 个任务的方差(可学习参数)。

直觉: 高不确定性的任务自动获得更低的权重。

**示例**:
```python
log_vars = nn.Parameter(torch.zeros(num_tasks))

loss = 0
for k in range(num_tasks):
    task_loss = compute_task_loss(outputs[k], targets[k])
    precision = torch.exp(-log_vars[k])
    weighted_loss = precision * task_loss + log_vars[k]
    loss += weighted_loss
```

---

## 8. 数值稳定性与LogSumExp技巧

### 8.1 数值稳定性问题

**问题**: 直接计算Softmax会导致数值溢出或下溢。

**例子**: 假设 $\mathbf{z} = [100, 101, 102]$

```
直接计算:
exp(100) = 在FP32中溢出 (>= 1e38)
exp(101) = 溢出
exp(102) = 溢出

导致: softmax 为 NaN
```

### 8.2 LogSumExp技巧

**定理 8.1**: LogSumExp的数值稳定实现

$$\log \sum_{i=1}^{n} \exp(x_i) = c + \log \sum_{i=1}^{n} \exp(x_i - c)$$

其中 $c = \max_i x_i$ 是选择的常数。

**证明**:
$$
\begin{aligned}
\log \sum_{i=1}^{n} \exp(x_i) &= \log \left( \sum_{i=1}^{n} \exp(x_i - c) \cdot \exp(c) \right) \\
&= \log \left( \exp(c) \sum_{i=1}^{n} \exp(x_i - c) \right) \\
&= c + \log \sum_{i=1}^{n} \exp(x_i - c)
\end{aligned}
$$

### 8.3 应用到交叉熵损失

**原始公式**:
$$L = -\log p_y = -\log \frac{\exp(z_y)}{\sum_{v} \exp(z_v)} = \log \sum_{v} \exp(z_v) - z_y$$

**数值稳定版本**:
设 $z_{\max} = \max_v z_v$,则:
$$L = z_{\max} + \log \sum_{v} \exp(z_v - z_{\max}) - z_y$$

**实现示例**:
```python
def cross_entropy_stable(logits, target):
    """
    Args:
        logits: [batch_size, vocab_size]
        target: [batch_size] (目标词汇ID)

    Returns:
        loss: [batch_size]
    """
    # Step 1: 找最大值
    logits_max = torch.max(logits, dim=-1)[0]  # [batch_size]

    # Step 2: 减去最大值(数值稳定)
    logits_shifted = logits - logits_max.unsqueeze(-1)  # [batch_size, vocab_size]

    # Step 3: 计算 log_sum_exp
    exp_logits = torch.exp(logits_shifted)
    sum_exp = torch.sum(exp_logits, dim=-1)  # [batch_size]
    log_sum_exp = logits_max + torch.log(sum_exp)

    # Step 4: 获取目标logit
    batch_idx = torch.arange(logits.size(0))
    target_logits = logits[batch_idx, target]

    # Step 5: 计算损失
    loss = log_sum_exp - target_logits

    return loss
```

### 8.4 Megatron-LM的数值稳定实现

**文件路径**: `megatron/core/tensor_parallel/cross_entropy.py:23-32`

```python
@staticmethod
def calculate_logits_max(vocab_parallel_logits: torch.Tensor):
    """Calculates logits_max."""

    vocab_parallel_logits = vocab_parallel_logits.float()
    # 在分片logits上找最大值
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

    # 在所有GPU上同步最大值(取全局最大)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=get_tensor_model_parallel_group()
    )

    return vocab_parallel_logits, logits_max
```

**关键点**:
1. 在分片的logits上计算本地最大值
2. 通过all-reduce获取全局最大值
3. 所有GPU使用相同的最大值,确保数值稳定性

---

## 9. 分布式损失计算的高效实现

### 9.1 词汇并行(Vocabulary Parallelism)

**概念**: 将词汇表分割到多个GPU,每个GPU只计算分片内的logits,减少内存占用和计算量。

**例子**: 词汇表大小 50000, 4个GPU
- GPU 0: logits for vocab IDs [0, 12500)
- GPU 1: logits for vocab IDs [12500, 25000)
- GPU 2: logits for vocab IDs [25000, 37500)
- GPU 3: logits for vocab IDs [37500, 50000)

### 9.2 VocabParallelCrossEntropy的实现

**文件路径**: `megatron/core/tensor_parallel/cross_entropy.py:16-120`

**核心流程**:

**步骤1: 计算logits最大值** (第23-32行)
```python
vocab_parallel_logits, logits_max = \
    VocabParallelCrossEntropy.calculate_logits_max(vocab_parallel_logits)

# 同步全局最大值
torch.distributed.all_reduce(
    logits_max,
    op=torch.distributed.ReduceOp.MAX,
    group=get_tensor_model_parallel_group()
)
```

**步骤2: 计算目标logit和分母项** (第34-68行)
```python
(target_mask, masked_target_1d, predicted_logits,
 sum_exp_logits, exp_logits) = \
    VocabParallelCrossEntropy.calculate_predicted_logits(
        vocab_parallel_logits, target, logits_max,
        vocab_start_index, vocab_end_index
    )

# 同步目标logit (正确答案的logit)
torch.distributed.all_reduce(
    predicted_logits,
    op=torch.distributed.ReduceOp.SUM,
    group=get_tensor_model_parallel_group()
)

# 同步分母项 (sum of exp)
torch.distributed.all_reduce(
    sum_exp_logits,
    op=torch.distributed.ReduceOp.SUM,
    group=get_tensor_model_parallel_group()
)
```

**步骤3: 计算损失** (第70-82行)
```python
# Loss = log(sum(exp)) - predicted_logit
loss = torch.log(sum_exp_logits) - predicted_logits

# 归一化为概率
exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
```

### 9.3 标签平滑的集成

**文件路径**: `megatron/core/tensor_parallel/cross_entropy.py:165-182`

**定义**: 标签平滑是一种正则化技术,将hard one-hot标签软化为柔和分布。

**原始交叉熵**:
$$L = -\log p_y$$

**标签平滑后的交叉熵**:
$$L_{\text{smoothed}} = (1 - \alpha) L + \alpha H_{\text{uniform}}$$

其中:
- $\alpha$: 平滑系数(通常0.01-0.1)
- $H_{\text{uniform}} = -\frac{1}{V-1} \sum_{v \neq y} \log p_v$: 非目标类的平均交叉熵

**代码实现**:
```python
if label_smoothing > 0:
    smoothing = label_smoothing * vocab_size / (vocab_size - 1)

    # log_probs: 已归一化的概率的对数
    log_probs = torch.log(exp_logits)
    mean_log_probs = log_probs.mean(dim=-1)

    # 混合损失
    loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs
```

**效果**:
- 减少过拟合
- 提高模型的泛化能力
- 使输出概率更均匀(不过度自信)

---

## 10. 损失函数设计原则

### 10.1 数学性质

**原则1: 可微性**
- 损失函数应该在大多数点可微,以便使用梯度下降
- 使用softmax而非argmax,保证可微性

**原则2: 凸性(Convexity)**
- 对于分类问题,交叉熵是凸函数,便于优化
- 凸损失有全局最优解,非凸损失可能陷入局部最优

**原则3: 缩放不变性**
- 理想情况下,损失函数不应该对输入的缩放敏感
- 例如,logits缩放 $\mathbf{z} \to c\mathbf{z}$ 不应该改变top-k排序

### 10.2 机器学习性质

**原则4: 反映任务目标**
- 选择的损失应该直接反映最终的评估指标
- 例如,对于精确度任务,使用精确度损失而非交叉熵

**原则5: 数值稳定性**
- 损失计算应该避免数值溢出/下溢
- 使用LogSumExp技巧等稳定算法

**原则6: 缩放友好**
- 在不同批大小、序列长度、模型大小下都应该稳定
- 使用归一化(如按token数量)而非总和

### 10.3 实践指导

**建议1: 逐token损失vs整体损失**
```python
# 逐token损失: 对每个token单独计算,然后平均
logits = model(input_ids)  # [B, S, V]
loss_per_token = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    reduction='none'
).view(B, S)
loss = loss_per_token.mean()  # 在所有token上平均

# 与整体损失等价,但提供更好的可观测性
```

**建议2: 梯度累积中的损失处理**
```python
# 标准做法: 先计算损失,再反向传播
accumulated_loss = 0
for i in range(num_accumulation_steps):
    logits = model(batch_input[i])
    loss = compute_loss(logits, batch_target[i])
    loss.backward()  # 自动累积梯度
    accumulated_loss += loss.item()

# 分布式训练中应该小心处理batch size
# 不能直接平均损失,应该按照样本数量加权
```

**建议3: 混合精度训练中的损失缩放**
```python
# 在FP16训练中,损失可能太小,导致梯度下溢
# 使用损失缩放:
scaled_loss = loss * loss_scale
scaled_loss.backward()

# 在应用梯度之前,梯度也要缩放回去
grad *= 1.0 / loss_scale
```

---

## 11. 超参数分析与敏感性

### 11.1 标签平滑系数 $\alpha$

**定义**: 标签平滑系数控制真实标签的权重。

**敏感性分析**:

| $\alpha$ | 特点 | 适用场景 |
|---------|------|----------|
| 0.0 | hard one-hot标签,标准交叉熵 | 基准配置 |
| 0.01 | 轻微平滑,主要用于正则化 | 小规模模型 |
| 0.1 | 中等平滑,常见设置 | 中等规模模型 |
| 0.2 | 较强平滑,抑制过拟合 | 大规模模型或有限数据 |
| >0.3 | 过度平滑,可能影响性能 | 不推荐 |

**实验数据** (ImageNet分类):
- 无平滑: Top-1 精度 76.4%
- α=0.1: Top-1 精度 76.9% (+0.5%)
- α=0.2: Top-1 精度 76.6%

### 11.2 词汇表大小的影响

**定义**: 词汇表大小 $V$ 影响logits的维度和softmax的计算复杂度。

**影响分析**:

1. **计算复杂度**
   - Softmax: $O(V)$ 时间
   - 词汇并行: 复杂度降至 $O(V/K)$ (K个GPU)
   - 大词汇表 (50K-250K) 会显著增加计算

2. **数值稳定性**
   - 大词汇表: 指数值范围更大,LogSumExp更重要
   - 小词汇表: 数值问题较少

3. **梯度特性**
   - 大词汇表: 梯度更稀疏(只有目标位置有大梯度)
   - 可能导致某些权重更新缓慢

### 11.3 序列长度的影响

**定义**: 序列长度 $S$ 影响每个样本的token数量,从而影响损失的大小和方差。

**影响分析**:

```python
# 长序列与短序列的损失对比
short_seq_loss = 0.5 * 10 tokens = 总5.0
long_seq_loss = 0.5 * 1000 tokens = 总500.0

# 平均损失
short_seq_avg = 5.0 / 10 = 0.5
long_seq_avg = 500.0 / 1000 = 0.5

# 两者平均损失相同,但总梯度不同
# 长序列产生的梯度10倍强
```

**建议**:
- 使用平均损失(而非总损失)进行比较
- 在梯度累积中注意序列长度的变化
- 对非常长的序列,考虑截断或分段处理

---

## 12. 深入探讨与最佳实践

### 12.1 与其他优化器的交互

**问题**: 损失函数的设计会影响优化器的效果。

**例子1: 学习率的交互**
```python
# 大损失值 → 大梯度 → 需要更小的学习率
loss = 5.0
grad = loss.backward()  # 梯度较大
lr = 1e-4  # 需要较小的学习率

# 小损失值 → 小梯度 → 可以用更大的学习率
loss = 0.5
grad = loss.backward()  # 梯度较小
lr = 1e-3  # 可以用较大的学习率
```

**例子2: 梯度裁剪的交互**
```python
# 梯度裁剪阈值应该根据损失值调整
if use_loss_scaling:
    # FP16训练中,梯度先缩放后再裁剪
    grad *= loss_scale
    grad = torch.clamp(grad, -clip_value, clip_value)
    grad /= loss_scale
else:
    grad = torch.clamp(grad, -clip_value, clip_value)
```

### 12.2 常见问题与解决方案

**问题1: 损失NaN**

可能原因与解决方案:
1. **logits包含Inf**: 检查模型输出
2. **数值下溢**: 使用LogSumExp技巧
3. **梯度爆炸**: 使用梯度裁剪
4. **不兼容的张量类型**: 确保logits和target类型正确

```python
# 诊断代码
def diagnose_nan_loss(logits, target, vocab_size):
    # 检查logits
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # 检查target
    assert target.min() >= 0
    assert target.max() < vocab_size

    # 计算损失,观察中间值
    logits_max = logits.max(dim=-1)[0]
    logits_shifted = logits - logits_max.unsqueeze(-1)

    print(f"logits_max: min={logits_max.min()}, max={logits_max.max()}")
    print(f"logits_shifted range: [{logits_shifted.min()}, {logits_shifted.max()}]")

    loss = F.cross_entropy(logits, target, reduction='none')
    print(f"loss range: [{loss.min()}, {loss.max()}]")
    print(f"loss contains nan: {torch.isnan(loss).any()}")
```

**问题2: 损失不收敛**

可能原因:
1. **学习率设置不当**: 调整lr或使用学习率预热
2. **标签平滑过度**: 减少α
3. **批大小不匹配**: 检查batch size是否与梯度累积步数匹配
4. **数据问题**: 检查标签分布是否异常

**问题3: 验证集损失不稳定**

可能原因:
1. **批大小变化**: 验证集使用不同大小的批
2. **数据顺序影响**: 验证集顺序随机化
3. **批规范化统计**: 使用不同的归一化

### 12.3 最佳实践总结

**推荐配置** (基于Megatron-LM和大规模LLM预训练经验):

1. **基础配置**
   ```python
   loss_fn = VocabParallelCrossEntropy(
       label_smoothing=0.01,  # 轻微平滑
   )
   ```

2. **超参数**
   ```python
   vocab_size = 50000          # 或其他值,取决于数据
   label_smoothing = 0.01      # 或0.02
   loss_scaling = 1024.0       # FP16训练
   ```

3. **实现细节**
   - 使用LogSumExp确保数值稳定
   - 在分布式训练中同步最大值
   - 正确处理梯度累积
   - 监控困惑度作为辅助指标

---

## 13. 总结

### 13.1 核心要点

1. **信息论基础**: 交叉熵损失来自最大似然估计和信息论,最小化交叉熵等价于最小化KL散度

2. **分类损失**:
   - 标准交叉熵是默认选择
   - Focal Loss用于处理类别不平衡,但在LLM预训练中较少使用
   - 标签平滑是简单但有效的正则化技术

3. **语言建模损失**:
   - CLM: 自回归预测,用于GPT系列模型
   - MLM: 双向预测,用于BERT系列模型
   - 困惑度是标准评估指标

4. **数值稳定性**:
   - LogSumExp技巧是必不可少的
   - Megatron-LM通过词汇并行和all-reduce实现高效的分布式计算

5. **实现细节**:
   - 梯度: $\nabla_{z_v} L = p_v - \mathbb{1}(v = y)$
   - 标签平滑: $(1-\alpha)L + \alpha H_{\text{uniform}}$
   - 损失缩放: 用于混合精度训练

### 13.2 优势与局限

**优势**:
- 交叉熵损失凸且可微,理论性质良好
- 梯度形式简洁,易于优化
- 数值稳定的算法已充分研究
- 与最大似然估计理论对应

**局限性**:
- 对离群值(outlier)敏感
- 不能直接优化最终评估指标(如BLEU)
- 高度不平衡数据上表现不佳
- 固定的惩罚机制(所有错误等权)

### 13.3 适用场景

| 任务 | 推荐损失 | 特殊考虑 |
|-----|---------|---------|
| GPT/自回归LLM | CLM交叉熵 | 使用因果掩码 |
| BERT/双向预训练 | MLM交叉熵 | 只计算掩码位置 |
| 多任务学习 | 加权组合 | 可选动态加权 |
| 不平衡数据 | Focal Loss | 较少用于LLM |
| 微调任务 | 与任务相关 | 可能需要调整α |

### 13.4 发展趋势

1. **自适应损失权重**: 动态调整任务权重
2. **对比学习损失**: 用于表示学习
3. **边界样本采样**: 重点关注难样本
4. **多指标优化**: 同时优化多个目标

---

## 14. 参考文献

### 14.1 核心论文

[1] Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *The Annals of Mathematical Statistics*, 22(1), 79-86.

[2] Bridle, J. S. (1989). Probabilistic Interpretation of Feedforward Classification Network Outputs. In *International Conference on Neural Information Processing Systems*.

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

[4] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.

[5] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In *IEEE International Conference on Computer Vision* (pp. 2980-2988).

[6] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*, 1(8), 9.

[7] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narayanan, S., Matena, M., ... & Liu, P. Q. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *arXiv preprint arXiv:1910.10683*.

[8] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp. 7482-7491).

### 14.2 实现参考

[9] NVIDIA Megatron-LM. *GitHub Repository*. https://github.com/NVIDIA/Megatron-LM

[10] PyTorch Documentation. Cross Entropy Loss. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

[11] TensorFlow Documentation. Categorical Crossentropy. https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

### 14.3 相关工作

[12] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. In *IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2818-2826).

[13] Perplexity in Language Models. *Research Papers*. Various sources on language modeling evaluation metrics.

---

## 附录A: 数学推导补充

### A.1 Softmax的完整梯度推导

**定义**: Softmax函数
$$p_i = \text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}$$

**梯度计算**:

对于任意 $z_k$:
$$\frac{\partial p_i}{\partial z_k} = ?$$

**情况1**: $i = k$
$$
\frac{\partial p_k}{\partial z_k} = \frac{\partial}{\partial z_k} \frac{\exp(z_k)}{\sum_j \exp(z_j)}
= \frac{\exp(z_k) \cdot \sum_j \exp(z_j) - \exp(z_k) \cdot \exp(z_k)}{(\sum_j \exp(z_j))^2}
= p_k(1 - p_k)
$$

**情况2**: $i \neq k$
$$
\frac{\partial p_i}{\partial z_k} = \frac{\partial}{\partial z_k} \frac{\exp(z_i)}{\sum_j \exp(z_j)}
= -\frac{\exp(z_i) \cdot \exp(z_k)}{(\sum_j \exp(z_j))^2}
= -p_i p_k
$$

**统一表示**:
$$\frac{\partial p_i}{\partial z_k} = \begin{cases}
p_i(1-p_i) & i = k \\
-p_i p_k & i \neq k
\end{cases} = p_i(\mathbb{1}(i=k) - p_k)$$

**应用到交叉熵**:

$L = -\log p_c$ (其中 $c$ 是正确类)

$$\frac{\partial L}{\partial z_k} = -\frac{1}{p_c} \frac{\partial p_c}{\partial z_k} = -\frac{1}{p_c} \cdot p_c(\mathbb{1}(c=k) - p_k) = p_k - \mathbb{1}(c=k)$$

### A.2 标签平滑的数学推导

**定义**: 标签平滑将one-hot标签 $\mathbf{y}^{\text{hard}}$ 转换为软标签 $\mathbf{y}^{\text{smooth}}$

$$y_i^{\text{smooth}} = \begin{cases}
1 - \alpha & i = c \text{ (真实类)} \\
\frac{\alpha}{V-1} & i \neq c
\end{cases}$$

其中 $\alpha$ 是平滑系数。

**验证概率**: $y_c^{\text{smooth}} + \sum_{i \neq c} y_i^{\text{smooth}} = (1-\alpha) + (V-1) \cdot \frac{\alpha}{V-1} = 1$ ✓

**交叉熵计算**:
$$
\begin{aligned}
L_{\text{smooth}} &= -\sum_i y_i^{\text{smooth}} \log p_i \\
&= -(1-\alpha) \log p_c - \sum_{i \neq c} \frac{\alpha}{V-1} \log p_i \\
&= -(1-\alpha) \log p_c - \frac{\alpha}{V-1} \sum_{i \neq c} \log p_i \\
&= (1-\alpha) L_{\text{hard}} + \frac{\alpha}{V-1} \sum_{i \neq c} (-\log p_i)
\end{aligned}
$$

其中最后一项 $\sum_{i \neq c} (-\log p_i)$ 近似为 $\sum_{i} (-\log p_i) = \sum_i (-\log p_i) - (-\log p_c)$。

在概率归一化的情况下:
$$\sum_{i} (-\log p_i) \approx V \log(1/p_{\text{avg}}) \approx V \log(V)$$

所以:
$$L_{\text{smooth}} \approx (1-\alpha) L_{\text{hard}} + \alpha \log(V)$$

但实际实现通常更简洁:
$$L_{\text{smooth}} = (1-\alpha) L_{\text{hard}} - \alpha \cdot \text{mean}(\log p)$$

---

## 附录B: 完整代码示例

### B.1 从零实现交叉熵损失

```python
import torch
import torch.nn.functional as F

class CrossEntropyLoss:
    """从零实现的交叉熵损失"""

    def __init__(self, label_smoothing=0.0, vocab_size=None):
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        """
        Args:
            logits: [batch_size, vocab_size]
            target: [batch_size]

        Returns:
            loss: 标量
        """
        batch_size, vocab_size = logits.shape

        # Step 1: 数值稳定性 - 减去最大值
        logits_max = logits.max(dim=-1)[0]  # [batch_size]
        logits_shifted = logits - logits_max.unsqueeze(-1)

        # Step 2: 计算 log_sum_exp
        exp_logits = torch.exp(logits_shifted)
        sum_exp = exp_logits.sum(dim=-1)  # [batch_size]
        log_sum_exp = logits_max + torch.log(sum_exp)

        # Step 3: 获取目标位置的logit
        batch_idx = torch.arange(batch_size)
        target_logits = logits[batch_idx, target]

        # Step 4: 计算基础损失
        base_loss = log_sum_exp - target_logits

        # Step 5: 应用标签平滑
        if self.label_smoothing > 0:
            # 软化标签
            log_probs = logits_shifted - log_sum_exp.unsqueeze(-1)
            mean_log_probs = log_probs.mean(dim=-1)

            # 组合损失
            smoothing = self.label_smoothing * vocab_size / (vocab_size - 1)
            loss = (1.0 - smoothing) * base_loss - smoothing * mean_log_probs
        else:
            loss = base_loss

        return loss.mean()

# 测试
if __name__ == "__main__":
    batch_size, vocab_size = 32, 10000
    logits = torch.randn(batch_size, vocab_size)
    target = torch.randint(0, vocab_size, (batch_size,))

    # 自定义实现
    custom_loss = CrossEntropyLoss(label_smoothing=0.01)
    custom_result = custom_loss.forward(logits, target)

    # PyTorch内置(不支持label_smoothing)
    pytorch_loss = F.cross_entropy(logits, target, reduction='mean')

    print(f"Custom loss: {custom_result.item():.4f}")
    print(f"PyTorch loss: {pytorch_loss.item():.4f}")
    print(f"Difference: {abs(custom_result.item() - pytorch_loss.item()):.6f}")
```

### B.2 词汇并行交叉熵的简化实现

```python
import torch
import torch.distributed as dist

class VocabParallelCELoss:
    """词汇并行交叉熵损失(简化版)"""

    def __init__(self, vocab_size, rank, world_size, label_smoothing=0.0):
        self.vocab_size = vocab_size
        self.rank = rank
        self.world_size = world_size
        self.label_smoothing = label_smoothing

        # 计算分片范围
        partition_size = vocab_size // world_size
        self.vocab_start = rank * partition_size
        self.vocab_end = (rank + 1) * partition_size

    def forward(self, vocab_parallel_logits, target):
        """
        Args:
            vocab_parallel_logits: [batch_size, partition_vocab_size]
            target: [batch_size] (全局词汇ID)

        Returns:
            loss: [batch_size]
        """
        batch_size = target.size(0)
        device = vocab_parallel_logits.device

        # Step 1: 计算本地最大值
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        # Step 2: 全局最大值同步
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX)

        # Step 3: 减去最大值(数值稳定)
        logits_shifted = vocab_parallel_logits - logits_max.unsqueeze(-1)

        # Step 4: 计算exp和sum
        exp_logits = torch.exp(logits_shifted)
        sum_exp_logits = exp_logits.sum(dim=-1)

        # Step 5: 同步分母项
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM)

        # Step 6: 获取目标logit
        target_mask = (target < self.vocab_start) | (target >= self.vocab_end)
        masked_target = target.clone() - self.vocab_start
        masked_target[target_mask] = 0  # 占位符

        # 从分片logits中索引
        partition_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.view(-1, partition_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(batch_size, device=device)

        predicted_logits = logits_2d[arange_1d, masked_target_1d]
        predicted_logits[target_mask] = 0

        # Step 7: 同步目标logit
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM)

        # Step 8: 计算损失
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Step 9: 标签平滑
        if self.label_smoothing > 0:
            # 归一化
            softmax = exp_logits / sum_exp_logits.unsqueeze(-1)
            log_probs = torch.log(softmax)
            mean_log_probs = log_probs.mean(dim=-1)

            smoothing = self.label_smoothing * self.vocab_size / (self.vocab_size - 1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        return loss
```

### B.3 评估困惑度

```python
import math

class PerplexityMetric:
    """计算困惑度"""

    def __init__(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, loss_per_token, num_tokens):
        """
        Args:
            loss_per_token: 每个token的损失
            num_tokens: token数量
        """
        self.total_loss += loss_per_token.sum().item()
        self.total_tokens += num_tokens

    def compute(self):
        """计算平均困惑度"""
        if self.total_tokens == 0:
            return float('inf')

        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0

# 使用示例
perplexity = PerplexityMetric()

for batch_logits, batch_target in data_loader:
    loss = model(batch_logits, batch_target)  # [batch_size]
    num_tokens = batch_target.numel()

    perplexity.update(loss, num_tokens)

ppl = perplexity.compute()
print(f"Perplexity: {ppl:.2f}")
```

---

## 附录C: 常见配置速查

### C.1 不同模型的损失函数配置

**GPT模型** (自回归):
```python
loss_config = {
    'loss_type': 'clm',  # Causal Language Modeling
    'label_smoothing': 0.01,
    'vocab_size': 50257,  # GPT-2
}
```

**BERT模型** (双向):
```python
loss_config = {
    'loss_type': 'mlm',  # Masked Language Modeling
    'label_smoothing': 0.0,  # BERT通常不使用平滑
    'vocab_size': 30522,
}
```

**T5模型** (序列到序列):
```python
loss_config = {
    'loss_type': 'seq2seq',
    'label_smoothing': 0.0,
    'vocab_size': 32100,
}
```

**LLaMA模型** (自回归,带改进):
```python
loss_config = {
    'loss_type': 'clm',
    'label_smoothing': 0.0,  # 最新研究建议不使用
    'vocab_size': 32000,
    'use_vocab_parallel': True,
}
```

### C.2 故障排除检查清单

- [ ] logits形状正确: [batch_size, vocab_size]
- [ ] target形状正确: [batch_size]
- [ ] target值在范围: [0, vocab_size)
- [ ] 损失不是NaN: 检查logits是否包含inf
- [ ] 损失在合理范围: 通常在 [0, ln(vocab_size)]
- [ ] 困惑度在合理范围: 通常在 [1, vocab_size]
- [ ] 分布式训练中同步最大值
- [ ] 标签平滑系数合理: 通常 [0, 0.2]
- [ ] 梯度不溢出/下溢: 检查梯度范围

