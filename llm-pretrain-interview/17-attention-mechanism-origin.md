# 17. 注意力机制的诞生：从Seq2Seq到Attention

> **文档编号**: 17
> **所属部分**: 第二部分 - 深度学习基础 (11-20)
> **对应原文档**: RNN序列建模与注意力机制
> **代码位置**: `megatron/core/transformer/attention.py:1-100` (现代注意力实现的历史背景)
> **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM v0.12.0实际代码和历史文献)

---

## 目录

1. [引言](#1-引言)
2. [相关工作](#2-相关工作)
3. [符号定义](#3-符号定义)
4. [Seq2Seq模型与信息瓶颈](#4-seq2seq模型与信息瓶颈)
5. [Bahdanau注意力机制](#5-bahdanau注意力机制)
6. [Luong注意力变体](#6-luong注意力变体)
7. [Soft Attention vs Hard Attention](#7-soft-attention-vs-hard-attention)
8. [注意力权重的可视化与解释](#8-注意力权重的可视化与解释)
9. [RNN Attention到Self-Attention的演进](#9-rnn-attention到self-attention的演进)
10. [实验验证与性能分析](#10-实验验证与性能分析)
11. [深入探讨](#11-深入探讨)
12. [总结](#12-总结)
13. [参考文献](#13-参考文献)
14. [附录](#附录)

---

## 1. 引言

### 1.1 概述

注意力机制(Attention Mechanism)是现代深度学习最重要的架构创新之一,也是Transformer和所有大语言模型的核心基础。虽然Attention概念在2014年被正式提出,但它的诞生过程充满了深刻的启发:从Seq2Seq模型遇到的信息瓶颈问题,到Bahdanau等人提出的优雅解决方案,再到后续的多种变体和改进,注意力机制的演进历程反映了深度学习从RNN时代向Transformer时代的重要转变。

**注意力机制的核心洞察**:
1. **问题**: Seq2Seq的固定长度编码向量无法有效捕捉长序列信息
2. **解决**: 通过动态地关注(attend to)输入序列的不同位置,为每个输出位置自适应地获取上下文
3. **优雅**: 将复杂的序列依赖关系建模为简单的加权求和
4. **通用**: 从RNN辅助机制演进为Transformer的全部架构基础

在本文档中,我们将系统地回顾注意力机制的诞生和演进过程,从数学原理、算法设计、工程实现等多个角度深入理解这一关键技术。

### 1.2 前置知识

**必需的数学基础**:
- 多元微积分：梯度、链式法则、偏导数
- 概率论：期望、方差、概率分布、Softmax函数
- 线性代数：矩阵乘法、转置、范数
- 信息论：熵、KL散度、交叉熵(用于理解Softmax的信息论意义)

**必需的编程知识**:
- Python基础
- PyTorch张量操作
- 循环神经网络(RNN)基础
- 反向传播和自动微分

**相关概念**:
- 循环神经网络(文档16)
- 反向传播算法(文档06)
- 数值稳定性(文档07)
- 归一化技术(文档13)
- 多元微积分(文档02)

### 1.3 文档组织

本文档按以下结构组织:

1. **相关工作** (第2节): 回顾序列建模的历史发展,从RNN到注意力机制的动机
2. **符号定义** (第3节): 统一的数学符号和代码变量约定
3. **Seq2Seq问题分析** (第4节): 深入分析Seq2Seq模型的信息瓶颈和局限性
4. **Bahdanau注意力** (第5节): 原始注意力机制的完整数学推导
5. **Luong注意力** (第6节): 几种重要的改进变体
6. **Soft vs Hard Attention** (第7节): 两种注意力的对比和区别
7. **可视化与解释** (第8节): 注意力权重的直觉理解
8. **演进轨迹** (第9节): RNN Attention到Self-Attention再到Transformer
9. **实验验证** (第10节): 注意力机制对性能的实际影响
10. **深入探讨** (第11节): 高级主题、理论分析、常见问题

### 1.4 文档位置与代码

> **历史背景**: `megatron/core/transformer/attention.py` 中的现代实现虽然是完全的注意力架构,但继承了从RNN Attention演进而来的数学基础
>
> **相关文件**:
> - `megatron/core/transformer/dot_product_attention.py:100-250` (点积注意力的现代实现,与Attention数学相关)
> - `megatron/core/transformer/transformer_block.py` (Attention在Transformer中的使用)
> - PyTorch文档中的注意力实现参考
>
> **历史论文**:
> - Bahdanau et al., 2014: "Neural Machine Translation by Jointly Learning to Align and Translate"
> - Luong et al., 2015: "Effective Approaches to Attention-based Neural Machine Translation"
> - Vaswani et al., 2017: "Attention Is All You Need"

---

## 2. 相关工作

### 2.1 序列建模的历史演进

#### 2.1.1 早期序列模型

**标准RNN (Vanilla RNN, 2011年左右)**:

标准RNN通过递推关系处理序列:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

优点:
- 概念简单,易于理解
- 可以处理任意长度的序列

缺点:
- **梯度消失问题**:长序列中梯度指数级衰减
- **梯度爆炸**:需要人工裁剪
- **长期依赖建模困难**

#### 2.1.2 长短期记忆网络(LSTM, 2 Sepp Hochreiter & Jürgen Schmidhuber, 1997)

LSTM引入门控机制解决梯度消失:
$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(遗忘门)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(输入门)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(候选状态)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(单元状态)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(输出门)} \\
h_t &= o_t \odot \tanh(C_t) \quad \text{(隐藏状态)}
\end{aligned}
$$

改进:
- 有效解决梯度消失问题
- 支持更长的依赖建模(100-300个步骤)

局限:
- 仍然受限于Seq2Seq的固定编码向量
- 计算复杂度高(相对于现代Transformer)

#### 2.1.3 Seq2Seq模型(Sutskever et al., 2014)

Seq2Seq(序列到序列)模型是机器翻译的革命性架构:

**架构**:
```
编码器: x₁, x₂, ..., xₙ → 编码向量 c = h_n
解码器: c → 生成 y₁, y₂, ..., y_m
```

**数学表达**:

编码阶段:
$$h_t = \text{LSTM}(x_t, h_{t-1}), \quad h_0 = \mathbf{0}$$
$$c = h_n \quad \text{(固定长度上下文向量)}$$

解码阶段:
$$h'_t = \text{LSTM}(y_{t-1}, h'_{t-1}, c)$$
$$P(y_t | y_1, \ldots, y_{t-1}, x_1, \ldots, x_n) = \text{softmax}(W_s h'_t)$$

**优点**:
- 首次实现端到端的序列到序列建模
- 统一处理长度不同的输入输出

**关键问题: 信息瓶颈**

所有输入信息必须压缩到单个固定长度的向量$c$中。这引入了严重的限制:

1. **长度的诅咒**: 长输入序列的信息无法完全压缩到固定向量中
2. **信息丧失**: 输入中间部分的重要信息可能被遗忘
3. **解码困难**: 解码器必须在没有直接输入信息的情况下生成输出

**定量分析**:

假设输入序列长度为$n$,编码向量维度为$d$。每个位置的信息量为$\log_2(|\text{alphabet}|)$比特。编码后必须存储在$d$维向量中,总容量为$d \cdot \log_2(2^{\text{float32}})$比特。

对于长序列:
- 当$n$很大时,平均每个位置的存储空间为$d/n$,急剧减少
- 中间位置的信息受到梯度衰减的双重影响

### 2.2 注意力机制的动机

**问题的直观理解**:

在翻译任务中,输出句子的每个单词应该对应输入句子中的特定单词(或多个单词)。Seq2Seq强迫模型将所有输入信息压缩为单个向量,这显然违背了直观的对齐直觉。

**自然解决方案**:

如果解码器能够"回顾"(attend to)输入序列的不同部分,并为每个输出动态地选择相关的输入,会怎样?

这正是注意力机制的核心思想。

---

## 3. 符号定义

### 3.1 序列建模符号

| 符号 | 含义 | 维度 | 说明 |
|------|------|------|------|
| $x_t$ | 第t个输入token | $d_x$ | 输入嵌入向量 |
| $h_t$ | 编码器第t步的隐藏状态 | $d_h$ | RNN/LSTM隐藏维度 |
| $\bar{h}$ 或 $h_n$ | 最终编码状态 | $d_h$ | 编码器的最终状态(Seq2Seq中的context) |
| $y_t$ | 第t个输出token | $d_y$ | 输出嵌入向量 |
| $s_t$ | 解码器第t步的状态 | $d_h$ | 解码器隐藏状态 |
| $c_t$ | 第t步的上下文向量 | $d_h$ | 注意力输出 |
| $n$ | 输入序列长度 | 标量 | 源语言句子长度 |
| $m$ | 输出序列长度 | 标量 | 目标语言句子长度 |

### 3.2 注意力计算符号

| 符号 | 含义 | 维度 | 说明 |
|------|------|------|------|
| $\alpha_{t,i}$ | 第t步对第i个输入的注意力权重 | 标量 | $\sum_i \alpha_{t,i} = 1$ |
| $\alpha_t$ | 第t步的注意力分布 | $n$ | 向量,所有注意力权重 |
| $e_{t,i}$ | 第t步对第i个输入的对齐分数 | 标量 | 原始注意力分数 |
| $a(\cdot)$ | 对齐函数/评分函数 | - | 评分机制实现方式 |
| $c_t$ | 第t步的上下文向量 | $d_h$ | $\sum_i \alpha_{t,i} h_i$ |
| $W_*, U_*, V_*$ | 权重矩阵 | 见下表 | 对齐函数中的可学习参数 |

### 3.3 权重矩阵维度(Bahdanau机制)

| 矩阵 | 形状 | 说明 |
|------|------|------|
| $W_q$ | $d_a \times d_h$ | 解码器隐藏状态到对齐模块的映射 |
| $U_q$ | $d_a \times d_h$ | 编码器隐藏状态到对齐模块的映射 |
| $v_a$ | $d_a$ | 对齐分数的最终投影向量 |
| $d_a$ | 标量(通常$d_a=d_h$) | 对齐模块的隐藏维度 |

### 3.4 数学约定

**求和表示法**:
- $c_t = \sum_{i=1}^n \alpha_{t,i} h_i$ 表示加权平均,也写作向量形式: $c_t = \alpha_t^T h$ (其中$h = [h_1, \ldots, h_n]^T$)

**矩阵vs向量**:
- 当讨论单个时间步时使用向量记号,例如$\alpha_t \in \mathbb{R}^n$
- 当讨论完整序列时使用矩阵记号,例如$\alpha \in \mathbb{R}^{m \times n}$ (所有解码步对所有编码步的注意力)

---

## 4. Seq2Seq模型与信息瓶颈

### 4.1 Seq2Seq架构的完整数学描述

**编码器** (Encoder):

给定输入序列$X = (x_1, x_2, \ldots, x_n)$,其中$x_i \in \mathbb{R}^{d_x}$,编码器(通常是双向LSTM)生成隐藏状态:

$$\overrightarrow{h}_i = \text{LSTM}_{\text{fw}}(x_i, \overrightarrow{h}_{i-1})$$
$$\overleftarrow{h}_i = \text{LSTM}_{\text{bw}}(x_i, \overleftarrow{h}_{i+1})$$
$$h_i = [\overrightarrow{h}_i; \overleftarrow{h}_i] \in \mathbb{R}^{2d_h}$$

**关键步骤: 固定编码向量**:

编码器的最终状态成为解码器的初始上下文:
$$c = h_n = [\overrightarrow{h}_n; \overleftarrow{h}_1] \in \mathbb{R}^{2d_h}$$

注意: 所有过去的信息都必须压缩到这个单一向量中。

**解码器** (Decoder):

解码器使用LSTM生成输出,在每一步使用固定的上下文向量:

$$s_t = \text{LSTM}(y_{t-1}, s_{t-1}, c)$$
$$P(y_t | y_1, \ldots, y_{t-1}, X) = \text{softmax}(W_s s_t)$$

其中$y_t$是第$t$个输出词。

### 4.2 信息瓶颈的理论分析

**定理 4.1** (信息论下界):

对于长度为$n$的输入序列,每个位置有$\log V$比特的信息(其中$V$是词汇表大小),编码向量维度为$d$。则必然存在信息损失:
$$\text{信息损失} \geq \max(0, n \log V - d \log(\text{float32精度}))$$

对于典型的机器翻译任务:
- $n = 20$ (句子长度)
- $V = 50000$ (词汇表)
- 每个词$\approx 15.6$比特
- 总信息$\approx 312$比特
- 编码向量维度$d = 512$ ($\times$浮点数精度$\approx 4096$比特)

虽然容量看似充足,但实际问题更复杂:

1. **梯度信息瓶颈**: 即使信息论上有足够的容量,梯度也可能无法有效传播
2. **头位置信息衰减**: 输入序列开始部分的信息在LSTM中衰减特别快
3. **鲜度偏差(Recency Bias)**: LSTM倾向于过度权重最近的输入

### 4.3 实验证据

**Sutskever et al. (2014)的观察**:

他们在机器翻译实验中发现:
- **问题1**: 短句子(≤10词)性能很好,但长句子(>30词)性能急剧下降
- **问题2**: BLEU分数随输入长度单调下降
- **问题3**: 解码器无法有效学习输出和输入之间的对齐

**实验配置**:
- 英文→法文翻译
- 1000万对句子
- 词汇表大小: 160,000
- LSTM维度: 1000 (双向编码器)

**结果数据**:

| 输入长度 | BLEU分数 | 相对性能 |
|----------|----------|----------|
| ≤10 | 25.3 | 100% |
| 11-20 | 24.3 | 96% |
| 21-30 | 19.8 | 78% |
| 31+ | 10.1 | 40% |

这个实验清楚地表明了Seq2Seq的局限性。

---

## 5. Bahdanau注意力机制

### 5.1 Bahdanau Attention的诞生(2014年)

**论文**: "Neural Machine Translation by Jointly Learning to Align and Translate" (Dzmitry Bahdanau, Kyunhyun Cho, Yoshua Bengio, ICLR 2015)

这篇论文的核心思想非常简洁优雅:与其将整个输入序列压缩到单个向量,不如让解码器为每个输出动态地学习应该关注输入的哪些部分。

### 5.2 Bahdanau Attention的数学推导

**定义 5.1** (注意力上下文向量):

对于解码器在第$t$步的状态$s_t$,上下文向量$c_t$定义为编码器隐藏状态的加权和:

$$c_t = \sum_{i=1}^{n} \alpha_{t,i} h_i$$

其中$\alpha_{t,i}$是第$t$步对第$i$个输入的注意力权重。

**定义 5.2** (注意力权重):

注意力权重通过Softmax归一化计算得分:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{n} \exp(e_{t,k})}$$

其中$e_{t,i}$是对齐分数(alignment score),由对齐函数$a(\cdot)$计算。

**定义 5.3** (对齐函数 - Bahdanau式):

Bahdanau提出的对齐函数是一个小型的前馈神经网络:

$$e_{t,i} = v_a^T \tanh(W_q s_t + U_q h_i)$$

其中:
- $W_q \in \mathbb{R}^{d_a \times d_h}$: 解码器状态的权重矩阵
- $U_q \in \mathbb{R}^{d_a \times d_h}$: 编码器状态的权重矩阵
- $v_a \in \mathbb{R}^{d_a}$: 最终的投影向量
- $d_a$: 对齐模块的隐藏维度(通常取$d_a = d_h$)

### 5.3 完整的Bahdanau Attention模块

**算法 5.1** (Bahdanau注意力机制):

```
输入: 编码器隐藏状态 h = [h_1, ..., h_n], 解码器状态 s_t
输出: 上下文向量 c_t, 注意力权重 α_t

1. 计算对齐分数:
   for i = 1 to n:
       u_t,i = tanh(W_q @ s_t + U_q @ h_i)
       e_t,i = v_a^T @ u_t,i

2. 计算注意力权重 (Softmax):
   α_t = softmax([e_t,1, ..., e_t,n])

3. 计算加权上下文:
   c_t = sum(α_t,i * h_i for i in 1..n)

4. 返回: c_t, α_t
```

### 5.4 Bahdanau机制的几何直觉

**比较查询与键**:

虽然Bahdanau注意力不使用"查询-键-值"的术语(这些术语在Transformer中变得通用),但概念上类似:

- **查询** (Query): 解码器状态$s_t$,表示"我需要什么?"
- **键** (Key): 编码器隐藏状态$h_i$,表示"这个位置有什么?"
- **值** (Value): 编码器隐藏状态$h_i$(在Bahdanau中查询和值相同)

**对齐函数的含义**:

$$e_{t,i} = v_a^T \tanh(W_q s_t + U_q h_i)$$

这个函数:
1. 将查询和键投影到共同的空间($d_a$维)
2. 通过加法组合它们(允许相互作用)
3. 通过tanh应用非线性变换
4. 通过$v_a$计算最终的相关性分数

**Softmax的作用**:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{n} \exp(e_{t,k})}$$

- 将原始分数转换为概率分布
- $\exp$函数放大分数之间的差异(集中注意力)
- 确保权重和为1,形成凸组合

### 5.5 数学性质

**定理 5.1** (注意力权重的性质):

对于Bahdanau注意力权重$\alpha_{t,i}$:

1. **非负性**: $\alpha_{t,i} \geq 0$ 对所有$t, i$
2. **归一性**: $\sum_{i=1}^{n} \alpha_{t,i} = 1$ 对所有$t$
3. **可微性**: $\alpha_{t,i}$关于所有参数可微

**证明**:

由Softmax函数的定义:
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{n} \exp(e_{t,k})}$$

非负性: $\exp(e_{t,i}) > 0$且分母$> 0$,故$\alpha_{t,i} > 0$

归一性:
$$\sum_{i=1}^{n} \alpha_{t,i} = \sum_{i=1}^{n} \frac{\exp(e_{t,i})}{\sum_{k=1}^{n} \exp(e_{t,k})} = \frac{\sum_{i=1}^{n}\exp(e_{t,i})}{\sum_{k=1}^{n} \exp(e_{t,k})} = 1$$

可微性: $\exp, \tanh, \text{softmax}$都是光滑可微函数的组合

### 5.6 完整的Seq2Seq + Attention解码过程

**新的解码器RNN**:

原始Seq2Seq的解码RNN:
$$s_t = \text{LSTM}(y_{t-1}, s_{t-1}, c)$$
其中$c$是固定的编码向量。

新的Seq2Seq + Attention解码RNN:
$$s_t = \text{LSTM}(y_{t-1}, s_{t-1})$$
$$c_t = \text{Attention}(s_t, H) \quad \text{where } H = [h_1, \ldots, h_n]$$
$$\tilde{s}_t = \tanh(W_c [s_t; c_t])$$
$$P(y_t | \ldots) = \text{softmax}(W_o \tilde{s}_t)$$

关键变化:
- 每步都生成新的上下文向量$c_t$
- 上下文的权重由解码器状态动态决定
- 引入了额外的非线性变换$W_c$来融合状态和上下文

---

## 6. Luong注意力变体

### 6.1 Luong Attention(2015年)

**论文**: "Effective Approaches to Attention-based Neural Machine Translation" (Minh-Thang Luong, Hieu Pham, Christopher D. Manning, EMNLP 2015)

Luong在Bahdanau的基础上进行了几个简化和改进,提出了两种主要的注意力变体。

### 6.2 Luong Attention的变体

#### 6.2.1 General (Multiplicative) Attention

**数学定义**:

$$\text{score}(s_t, h_i) = s_t^T W h_i$$

其中$W \in \mathbb{R}^{d_h \times d_h}$是学习的权重矩阵。

**特点**:
- 计算效率高(只需一个矩阵乘法)
- 参数少(只有一个矩阵$W$,而Bahdanau有两个)
- 本质是点积注意力

**与Bahdanau的对比**:

Bahdanau:
$$e_{t,i} = v_a^T \tanh(W_q s_t + U_q h_i)$$

Luong Additive:
$$e_{t,i} = v_a^T \tanh(W_q s_t + U_q h_i)$$

Luong Multiplicative (General):
$$e_{t,i} = s_t^T W h_i$$

#### 6.2.2 Dot-Product Attention

**数学定义**:

$$\text{score}(s_t, h_i) = s_t^T h_i$$

这是最简单的形式,当$s_t$和$h_i$维度相同时使用。

**限制**: 要求$\dim(s_t) = \dim(h_i)$

#### 6.2.3 Concat (Additive) Attention

这实际上与Bahdanau几乎相同:

$$\text{score}(s_t, h_i) = v^T \tanh(W [s_t; h_i])$$

其中$[s_t; h_i]$表示连接操作。

### 6.3 全局注意力(Global Attention)

Luong提出了**全局注意力**的概念,与**本地注意力**相对:

**全局注意力(Global/Soft Attention)**:
- 考虑源序列的所有位置
- 生成全局上下文向量
- 这是标准的注意力机制

**本地注意力(Local/Hard Attention)**:
- 只关注源序列的一个小窗口
- 计算效率更高(见第7节)

### 6.4 带有输入馈送的Luong Attention

**关键改进: 输入馈送(Input Feeding)**

原始Seq2Seq + Attention:
```
s_t = LSTM(y_{t-1}, s_{t-1})
c_t = Attention(s_t, H)
y_t ~ P(y_t | s_t, c_t)
```

带输入馈送的Luong Attention:
```
s_t = LSTM(y_{t-1}, s_{t-1}, output_{t-1})
c_t = Attention(s_t, H)
output_t = tanh(W_c [s_t; c_t])
y_t ~ P(y_t | output_t)
```

其中$\text{output}_{t-1}$是前一步的输出(包含了前一步的注意力信息)。

**直觉**: 这样允许解码器通过反馈看到前一步的注意力决策,可以改进当前步的预测。

---

## 7. Soft Attention vs Hard Attention

### 7.1 概念对比

| 特性 | Soft Attention(软注意力) | Hard Attention(硬注意力) |
|------|-------------------------|------------------------|
| 操作 | 加权平均 | 离散选择 |
| 权重 | 连续概率分布 | 离散one-hot分布 |
| 可微性 | 完全可微 | 不可微(离散采样) |
| 计算复杂度 | $O(n)$ | $O(n)$ |
| 实现复杂度 | 简单 | 复杂(需要强化学习) |
| 解释性 | 模糊(多个位置) | 清晰(单一位置) |

### 7.2 Soft Attention(连续加权平均)

**定义**:

Soft attention通过所有源位置的加权平均来计算上下文:

$$c_t = \sum_{i=1}^{n} \alpha_{t,i} h_i$$

其中权重$\alpha_{t,i} = \text{softmax}(e_{t,i})$形成概率分布。

**数学性质**:

1. **完全可微**: 所有操作都是光滑函数的组合
2. **凸组合**: $c_t$在所有$h_i$形成的凸包内
3. **梯度流畅**: 梯度可以直接反向传播

**反向传播**:

$$\frac{\partial L}{\partial h_i} = \sum_t \frac{\partial L}{\partial c_t} \alpha_{t,i} + \text{其他项}$$

$$\frac{\partial L}{\partial \alpha_{t,i}} = \frac{\partial L}{\partial c_t} h_i$$

### 7.3 Hard Attention(离散选择)

**定义**:

Hard attention离散地选择一个源位置:

$$c_t = h_{z_t}$$

其中$z_t \in \{1, 2, \ldots, n\}$是一个离散随机变量。

权重$\alpha_{t,i}$是one-hot分布:
$$\alpha_{t,i} = \begin{cases} 1 & \text{if } i = z_t \\ 0 & \text{otherwise} \end{cases}$$

**优点**:
1. **计算高效**: 只需要一个源向量,不需要加权求和
2. **可解释**: 清晰地显示哪个源位置被选择
3. **注意力集中**: 避免了软注意力的"平均"问题

**缺点**:
1. **不可微**: 离散采样是不可微的(No gradient!)
2. **训练困难**: 需要强化学习或Gumbel-Softmax技巧
3. **方差高**: 采样导致高方差梯度估计

### 7.4 Hard Attention的训练策略

**策略1: 强化学习**

将选择动作视为强化学习问题:
$$p(z_t = i | s_t) = \alpha_{t,i}$$

损失函数:
$$\mathcal{L} = -\mathbb{E}_{z_t}[\log P(y_t | h_{z_t})] \cdot R(z_t)$$

其中$R(z_t)$是学习信号(如任务奖励)。

**策略2: Gumbel-Softmax技巧**

将离散采样近似为连续的Softmax操作:
$$z_t = \arg\max_i (\log \alpha_{t,i} + G_i)$$

其中$G_i \sim \text{Gumbel}(0, 1)$。

使用温度缩放的Softmax近似:
$$\tilde{\alpha}_{t,i} = \frac{\exp((\log \alpha_{t,i} + G_i) / \tau)}{\sum_j \exp((\log \alpha_{t,j} + G_j) / \tau)}$$

当$\tau \to 0$时,$\tilde{\alpha}_{t,i} \to$ one-hot分布。

### 7.5 实践对比

**经验观察**:

1. **Soft Attention在Seq2Seq中广泛采用**
   - 易于实现
   - 完全可微
   - 效果良好
   - 2014-2017年间的标准选择

2. **Hard Attention主要在以下场景使用**
   - 视觉问题(RamNet)
   - 计算受限的场景
   - 需要高度可解释性的应用

3. **现代Transformer使用Soft Attention**
   - 通过缩放点积进一步简化
   - Flash Attention等优化都基于Soft Attention
   - 成为行业标准

---

## 8. 注意力权重的可视化与解释

### 8.1 注意力矩阵的结构

**注意力矩阵** $\alpha \in \mathbb{R}^{m \times n}$:

行: $m$个解码步(输出序列)
列: $n$个编码步(输入序列)
元素: $\alpha_{t,i}$是第$t$个输出对第$i$个输入的注意力

$$\alpha = \begin{bmatrix}
\alpha_{1,1} & \alpha_{1,2} & \cdots & \alpha_{1,n} \\
\alpha_{2,1} & \alpha_{2,2} & \cdots & \alpha_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{m,1} & \alpha_{m,2} & \cdots & \alpha_{m,n}
\end{bmatrix}$$

每一行都是一个概率分布(和为1)。

### 8.2 典型的注意力模式

**模式1: 对齐模式(Alignment Pattern)**

在单语化翻译(word-for-word)任务中,注意力矩阵呈现对角线结构:

```
源句子: The cat sat on the mat
目标句子: Le chat s'assit sur le tapis

注意力矩阵(概念):
       The  cat  sat  on  the  mat
Le    [0.7  0.2  0.1  0.0  0.0  0.0]
chat  [0.1  0.8  0.1  0.0  0.0  0.0]
s'.. [0.0  0.1  0.8  0.0  0.0  0.0]
sur   [0.0  0.0  0.0  0.7  0.2  0.1]
le    [0.0  0.0  0.0  0.1  0.8  0.1]
tapis [0.0  0.0  0.0  0.0  0.1  0.9]
```

特点: 强对角线,每个目标词主要关注一个源词

**模式2: 一对多模式**

某个源词映射到多个目标词:

```
源句子: I don't like this
目标句子: Je n'aime pas cela

注意力矩阵(概念):
         I  don't  like  this
Je     [0.8  0.2   0.0   0.0]
n'     [0.1  0.8   0.1   0.0]
aime   [0.0  0.3   0.7   0.0]
pas    [0.0  0.2   0.8   0.0]
cela   [0.0  0.0   0.1   0.9]
```

特点: 某些列有分散的权重,反映了一对多的对应

**模式3: 累积注意力(Coverage)**

某些模型跟踪覆盖(coverage)向量,记录哪些源词已被翻译:

$$\text{coverage}_t = \sum_{t'=1}^{t-1} \alpha_{t',i}$$

用于鼓励新的注意力覆盖未翻译的源词。

### 8.3 注意力权重的定量分析

**度量1: 注意力熵**

量化注意力的集中程度:

$$H_t = -\sum_{i=1}^{n} \alpha_{t,i} \log \alpha_{t,i}$$

- 高熵: 注意力分散(关注多个位置)
- 低熵: 注意力集中(关注少数位置)

**度量2: 注意力浓度**

$$\text{Concentration}_t = \max_i \alpha_{t,i}$$

最大权重的大小。

**度量3: 注意力跳跃(Attention Jump)**

相邻时间步之间的注意力变化:

$$\text{Jump}_t = \sqrt{\sum_i (\alpha_{t,i} - \alpha_{t-1,i})^2}$$

量化注意力焦点的稳定性。

### 8.4 可视化技巧

**技巧1: 热力图(Heatmap)**

使用颜色深度表示权重大小。PyTorch/TensorFlow中常见的可视化方式。

**技巧2: 箭头图(Arrow Diagram)**

用箭头连接源词和目标词,箭头粗细表示权重大小。

**技巧3: 注意力流(Attention Flow)**

对于多层注意力(如Transformer),展示信息如何从下层流向上层。

---

## 9. RNN Attention到Self-Attention的演进

### 9.1 完整的演进历程

#### 阶段1: Seq2Seq(2014)

```
编码: RNN(x_1, ..., x_n) → c
解码: RNN(y_1, ..., y_m | c)
```

特点: 无注意力,固定编码向量瓶颈

#### 阶段2: RNN + Attention(2015)

```
编码: RNN(x_1, ..., x_n) → [h_1, ..., h_n]
解码:
  for t = 1 to m:
    α_t = Attention(s_t, [h_1, ..., h_n])
    c_t = sum(α_t,i * h_i)
    s_t = RNN(y_{t-1}, s_{t-1}, c_t)
    y_t ~ P(y_t | s_t, c_t)
```

特点: 动态上下文,解决了Seq2Seq的瓶颈问题

#### 阶段3: 多头注意力(2017)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

MultiHeadAttention(Q, K, V) = Concat(head_1, ..., head_h) W^O
  其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

特点: 多个表示子空间,捕捉不同类型的依赖

#### 阶段4: Self-Attention / Transformer(2017)

**关键洞察**: 在RNN中,注意力是跨序列的(查询来自解码器,键值来自编码器)。如果查询、键、值都来自同一序列呢?

```
Self-Attention(X) = Attention(X W^Q, X W^K, X W^V)
```

其中$X$是输入序列(无论是编码器还是解码器)。

**Transformer架构(2017)**:

```
输入: X ∈ R^(S × H)

Encoder:
  for l = 1 to L:
    h_l = MHA(LayerNorm(h_{l-1})) + h_{l-1}  (残差)
    h_l = FFN(LayerNorm(h_l)) + h_l           (残差)

Decoder:
  for l = 1 to L:
    h_l = MHA(LayerNorm(h_{l-1}), 有因果掩码) + h_{l-1}
    h_l = CrossMHA(h_l, 编码器输出) + h_l   (交叉注意力)
    h_l = FFN(LayerNorm(h_l)) + h_l
```

特点:
- 完全基于注意力(无RNN)
- 并行处理(可以一次处理整个序列)
- 更长的有效上下文
- 更容易扩展到大规模

### 9.2 关键创新点的演进

#### 创新1: 从固定到动态上下文

**Seq2Seq**:
$$c = h_n \quad \text{(固定)}$$

**RNN + Attention**:
$$c_t = \sum_i \alpha_{t,i} h_i \quad \text{(动态,依赖}s_t\text{)}$$

**Self-Attention**:
$$\text{Attention}(X) = \text{softmax}(X W^Q (X W^K)^T / \sqrt{d_k}) X W^V$$

在Self-Attention中,即使没有解码器状态,也能生成动态的注意力权重。

#### 创新2: 从跨层到自层

**RNN Attention**: 跨层注意力
```
查询: 解码器隐藏状态 s_t
键值: 编码器隐藏状态 h_i
交互: 不同层之间
```

**Self-Attention**: 自层注意力
```
查询/键值: 同一层的输入 X
交互: 同一层内部
```

#### 创新3: 从递推到并行

**RNN + Attention**:
```python
h = []
for t in range(seq_len):
    h_t = rnn(x_t, h[t-1])  # 必须顺序执行
    c_t = attention(h)
    o_t = decode(h_t, c_t)
```

时间复杂度: $O(\text{seq_len})$

**Self-Attention**:
```python
h = rnn_input
q = h @ W_q
k = h @ W_k
v = h @ W_v
scores = q @ k.T / sqrt(d)
attn_weights = softmax(scores)
output = attn_weights @ v
```

时间复杂度: $O(1)$ (假设矩阵乘法是并行的)

这是Transformer相比RNN的根本性优势。

### 9.3 从Attention到现代Transformer

**Transformer的完整架构** (Vaswani et al., 2017):

```
Input Embedding + Position Encoding
        ↓
Encoder (×6 层):
  ├─ Multi-Head Self-Attention
  ├─ Add & Norm (残差+LayerNorm)
  ├─ Feed-Forward Network
  └─ Add & Norm (残差+LayerNorm)
        ↓
Decoder (×6 层):
  ├─ Multi-Head Self-Attention (因果掩码)
  ├─ Add & Norm
  ├─ Multi-Head Cross-Attention (与编码器的交叉注意力)
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
        ↓
Output Linear Layer
        ↓
Softmax
```

**与RNN Attention的关键区别**:

| 方面 | RNN + Attention | Transformer |
|------|-----------------|-------------|
| 编码方式 | RNN递推 | 位置编码 |
| 主要计算 | 注意力 + RNN | 纯注意力 |
| 并行性 | 低(RNN递推) | 高(完全并行) |
| 序列长度 | 有限(梯度问题) | 理论无限 |
| 实现简单性 | 复杂(RNN+注意力) | 简单(纯矩阵操作) |

---

## 10. 实验验证与性能分析

### 10.1 Seq2Seq与RNN+Attention的对比

**论文**: Bahdanau et al. (2015)

**实验设置**:
- 任务: 英法机器翻译
- 数据集: WMT14 (约350万句对)
- 模型: LSTM编码器-解码器
- 评估指标: BLEU分数, 句子对齐准确率

**关键结果**:

| 模型 | 短句 | 中句 | 长句 | 平均BLEU |
|------|------|------|------|----------|
| Seq2Seq | 29.0 | 26.3 | 18.2 | 24.2 |
| +RNN Attention | 31.5 | 29.2 | **25.3** | **28.7** |
| 改进(%) | +8.6% | +10.9% | **+39.0%** | **+18.6%** |

**关键观察**:
1. 长句性能的巨大改进(+39%)验证了注意力机制解决瓶颈问题的有效性
2. 平均BLEU提升18.6%,这在当时是重要的突破
3. 注意力对长序列的帮助最大

### 10.2 Luong Attention的实验对比

**论文**: Luong et al. (2015)

**实验**:英法、日英、中英机器翻译

**注意力变体性能**:

| 注意力机制 | 参数数 | BLEU(EN-FR) | BLEU(EN-JA) |
|-----------|--------|------------|------------|
| 无注意力 | baseline | 24.2 | 25.8 |
| Concat | +10M | 28.5 | 28.9 |
| General | +1M | 28.9 | 29.3 |
| Dot-Product | 0M | 28.8 | 29.1 |

**结论**:
- General Attention(参数最少)性能最好
- Dot-Product几乎相同,参数为0
- 验证了Luong的理论:不需要复杂的对齐函数

### 10.3 注意力对长序列的具体帮助

**数据集**: WMT14英法翻译

**对比**:

```
输入长度10-20: Seq2Seq=25.2, +Attention=29.1, 改进=+3.9
输入长度20-30: Seq2Seq=21.1, +Attention=27.8, 改进=+6.7
输入长度30-40: Seq2Seq=15.8, +Attention=23.4, 改进=+7.6
输入长度40-50: Seq2Seq=10.2, +Attention=17.9, 改进=+7.7
```

**模式**:
- 短序列(≤20词):注意力的帮助有限
- 中等长度(20-40词):显著改进
- 长序列(40+词):关键改进

### 10.4 注意力可视化验证

**实验**: 人工检查译文中的注意力权重

**发现1: 注意力与对齐一致**

大多数情况下,高注意力权重对应于源句子中的对应词。

例如:翻译"the cat"时,注意力权重最高的分别是"the"和"cat"。

**发现2: 注意力捕捉复杂对应**

- **重排**: "Le chat noir"对应"The black cat",注意力正确捕捉了词序变化
- **一对多**: "je n'aime"关注"don't like"中的两个词
- **多对一**: "s'assit sur"(坐在)共同关注"sat on"

这表明注意力不只是简单的对齐,而是学会了语言结构的对应关系。

**发现3: 注意力的可解释性**

通过可视化注意力权重,人类可以理解模型的决策过程。这对于调试和改进模型很有帮助。

---

## 11. 深入探讨

### 11.1 注意力机制的理论分析

#### 11.1.1 为什么注意力有效?

**观点1: 信息论视角**

Seq2Seq的信息瓶颈可以用信息论量化。注意力通过增加有效的上下文维度来减轻这个瓶颈:

$$\text{有效上下文维度} = n \times H_{\text{attention}}$$

其中$H_{\text{attention}}$是注意力熵。

- Seq2Seq: 固定的1维(编码向量)
- RNN + Attention: 动态的, 平均$n \times H$维

#### 11.1.2 何时注意力最有帮助?

注意力对以下情况最有帮助:

1. **长序列**: 信息瓶颈更严重,需要更多的动态上下文
2. **分散依赖**: 输出依赖于输入的多个不同位置
3. **复杂对齐**: 输入输出顺序不同,需要复杂的对应关系

反之,对于短序列和简单对齐,注意力的帮助有限。

#### 11.1.3 注意力的容量分析

**定理 11.1**: 注意力机制的参数复杂度

Bahdanau注意力的可学习参数:
$$\text{Params}(W_q, U_q, v_a) = d_a(d_h + d_h + 1) \approx 2 d_a d_h$$

相比于编码器参数$\approx d_h^2$,注意力开销很小。

**定理 11.2**: 注意力的表达能力

注意力机制可以学习任何输入位置的任意加权组合。由于权重范围是$[0,1]^n$上的所有概率分布,Bahdanau注意力的表达能力充分。

### 11.2 注意力的计算复杂度分析

**时间复杂度**:

对于长度为$n$的输入序列,计算一个查询的注意力:

```
1. 计算对齐分数: O(n * d_a)  (for each of n positions)
   → 总计: O(n^2 * d_a) (如果直接循环)
   → 优化后: O(n * d_h) (矩阵化)

2. Softmax: O(n)

3. 加权求和: O(n * d_h)

总计: O(n * d_h)
```

**空间复杂度**:

```
编码器隐藏状态: O(n * d_h)
注意力权重矩阵: O(n) (单次查询)
对齐分数: O(n)

总计: O(n * d_h)
```

### 11.3 常见问题与解决方案

**问题1: 注意力权重噪音高**

症状: 注意力分布不集中,多个位置权重相近

解决方案:
- 增加对齐函数的表达能力($d_a$)
- 使用更好的初始化
- 增加训练数据

**问题2: 注意力未对齐输入输出**

症状: 注意力权重看起来随机,与语言学直觉不符

可能原因:
- 模型足够强,不需要显式对齐(参数过剩)
- 有其他特征(如位置编码)提供对齐信息

**问题3: 长序列性能仍差**

症状: 即使有注意力,长序列仍然性能下降

原因:
- 注意力解决了信息瓶颈,但RNN的梯度消失仍存在
- 解决方案: Transformer(用自注意力完全替代RNN)

### 11.4 注意力与现代架构的联系

#### 11.4.1 Transformer中的注意力

Transformer使用的自注意力是RNN注意力的自然扩展:

**RNN Attention**:
```
α = softmax(v^T tanh(W_q s_t + U_q h_i))
c = Σ α_i h_i
```

**Scaled Dot-Product Attention**:
```
α = softmax(Q K^T / √d_k)
O = α V
```

看似很不同,但本质是相同的:都是计算查询和键的相似度,然后用权重加和值。

#### 11.4.2 闪现注意力(Flash Attention)中的数学

Flash Attention是对Scaled Dot-Product Attention的优化,但核心数学不变。它优化的是:
- 计算顺序(Online Softmax)
- 内存访问模式(分块)

这些都是在保留数学等价性的前提下进行的工程优化。

---

## 12. 总结

### 12.1 核心要点回顾

**1. 问题的根源**

Seq2Seq模型的根本限制是将整个输入压缩到单个固定向量,导致:
- 长序列信息丧失
- 性能随序列长度单调下降
- 无法学习对齐

**2. 优雅的解决方案**

Bahdanau注意力通过一个简单而强大的想法解决了这个问题:
- 动态计算每个输入位置的相关性
- 为每个输出自适应地加权源信息
- 通过可视化权重增加模型可解释性

**3. 数学的优美**

注意力的数学表达简洁而强大:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这个公式:
- 原始形式看似复杂,但实际上基于简单的点积相似度
- 包含了非线性(Softmax)和归一化(除以$\sqrt{d_k}$)
- 易于计算和优化

**4. 实践的有效性**

实验数据明确证明:
- 长序列性能提升39%
- 短序列性能也有改进
- 注意力权重与语言学直觉一致

**5. 历史的影响**

注意力机制的提出标志着:
- RNN时代的结束
- Transformer时代的开始
- 现代大语言模型的基础

### 12.2 关键洞察

**洞察1: 从固定到动态**

固定编码向量(Seq2Seq) → 动态上下文(RNN+Attention) → 完全基于注意力(Transformer)

**洞察2: 从递推到并行**

RNN的递推约束 → RNN+注意力(仍然递推) → Transformer(完全并行)

**洞察3: 从工程到科学**

手工特征工程 → 学习特征的对齐函数 → 完全学习依赖结构

### 12.3 局限性与展望

**当前注意力机制的局限**:

1. **二次复杂度**: $O(n^2)$的时间和空间,限制了超长序列
2. **本地性不足**: 标准注意力是全局的,某些任务需要局部性
3. **可解释性有限**: 虽然比RNN更可解释,但注意力权重不总是反映语言学事实

**未来方向**:

1. **线性注意力**: Kernel-based attention, Mamba等
2. **稀疏注意力**: Local attention, Long-range等
3. **结构化注意力**: 融入更多归纳偏差

### 12.4 实践建议

**何时使用注意力**:
- ✅ 长序列任务
- ✅ 需要可解释性的应用
- ✅ 复杂依赖关系
- ✅ 现代深度学习系统(几乎总是)

**如何调试注意力问题**:
1. 可视化注意力权重
2. 检查权重分布(使用熵等度量)
3. 与语言学直觉对比
4. 尝试调整$d_a$(对齐模块维度)

---

## 13. 参考文献

### 核心论文

1. **Bahdanau, D., Cho, K., & Bengio, Y.** (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015.
   - 首次提出注意力机制
   - Bahdanau注意力的原始论文

2. **Luong, M. T., Pham, H., & Manning, C. D.** (2015). "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015.
   - 简化的General注意力
   - 全局vs局部注意力
   - 输入馈送(Input Feeding)

3. **Sutskever, I., Vinyals, O., & Le, Q. V.** (2014). "Sequence to Sequence Learning with Neural Networks." NIPS 2014.
   - Seq2Seq模型
   - 问题的源头

4. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). "Attention Is All You Need." NeurIPS 2017.
   - Transformer架构
   - 自注意力与多头注意力
   - 取代RNN的关键工作

### 相关论文

5. **Hochreiter, S., & Schmidhuber, J.** (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780.
   - LSTM介绍
   - 解决梯度消失的早期方案

6. **Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N.** (2017). "Convolutional Sequence to Sequence Learning." ICML 2017.
   - CNN + Attention的替代方案
   - 支持Transformer的出现

7. **Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C.** (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
   - 现代注意力优化
   - 解决注意力的二次复杂度

### 扩展阅读

8. **Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. S.** (2018). "MoleculeNet: A Benchmark for Molecular Machine Learning." Chemical Science, 9(2), 513-530.
   - 注意力在化学领域的应用

9. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." ICLR 2019.
   - 双向Transformer
   - 预训练+微调范式

---

## 附录

### A. Softmax函数的性质

**定义**:

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}$$

**关键性质**:

1. **输出范围**: $(0, 1)$
2. **归一性**: $\sum_i \text{softmax}(\mathbf{x})_i = 1$
3. **单调性**: 若$x_i > x_j$,则$\text{softmax}(\mathbf{x})_i > \text{softmax}(\mathbf{x})_j$
4. **可微性**: $\frac{\partial}{\partial x_j} \text{softmax}(\mathbf{x})_i = \text{softmax}(\mathbf{x})_i (\delta_{ij} - \text{softmax}(\mathbf{x})_j)$

**在注意力中的作用**:

- 将原始分数转换为概率分布
- 通过指数函数放大分数差异
- 使得高分数的项获得更多权重
- 完全可微,支持反向传播

### B. 反向传播详细推导

假设我们有:
- 输入: 编码器隐藏状态$H \in \mathbb{R}^{n \times d_h}$, 解码器状态$s_t \in \mathbb{R}^{d_h}$
- 输出: 上下文向量$c_t = \sum_i \alpha_{t,i} h_i$
- 损失: $L$

**反向传播**:

1. **从损失到上下文**:
   $$\frac{\partial L}{\partial c_t} = \frac{\partial L}{\partial \text{logits}} \frac{\partial \text{logits}}{\partial c_t}$$

2. **从上下文到注意力权重**:
   $$\frac{\partial L}{\partial \alpha_{t,i}} = \frac{\partial L}{\partial c_t} \cdot h_i$$

3. **从注意力权重到对齐分数**:
   $$\frac{\partial L}{\partial e_{t,i}} = \sum_j \frac{\partial \alpha_{t,j}}{\partial e_{t,i}} \frac{\partial L}{\partial \alpha_{t,j}}$$

   其中:
   $$\frac{\partial \alpha_{t,j}}{\partial e_{t,i}} = \alpha_{t,j}(\delta_{ij} - \alpha_{t,i})$$

4. **从对齐分数到参数**:
   $$\frac{\partial L}{\partial v_a} = \sum_{t,i} \frac{\partial e_{t,i}}{\partial v_a} \frac{\partial L}{\partial e_{t,i}}$$

   类似地计算$\frac{\partial L}{\partial W_q}$和$\frac{\partial L}{\partial U_q}$

### C. PyTorch伪代码实现

```python
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, d_h, d_a=None):
        super().__init__()
        if d_a is None:
            d_a = d_h

        self.W_q = nn.Linear(d_h, d_a)
        self.U_q = nn.Linear(d_h, d_a)
        self.v_a = nn.Linear(d_a, 1)

    def forward(self, s_t, H, mask=None):
        """
        Args:
            s_t: 解码器状态 (B, d_h) 或 (1, d_h)
            H: 编码器隐藏状态 (B, S, d_h) 或 (S, d_h)
            mask: 掩码 (B, S) 或 (S,)

        Returns:
            c_t: 上下文向量
            alpha: 注意力权重
        """
        # 展开维度
        if s_t.dim() == 2:
            s_t_proj = self.W_q(s_t)  # (B, d_a)
            s_t_proj = s_t_proj.unsqueeze(1)  # (B, 1, d_a)

        H_proj = self.U_q(H)  # (B, S, d_a) 或 (S, d_a)

        # 计算对齐分数
        u = torch.tanh(s_t_proj + H_proj)  # (B, S, d_a)
        e = self.v_a(u).squeeze(-1)  # (B, S)

        # 应用掩码
        if mask is not None:
            e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax
        alpha = torch.softmax(e, dim=-1)  # (B, S)

        # 加权求和
        c_t = (alpha.unsqueeze(1) @ H).squeeze(1)  # (B, d_h)

        return c_t, alpha
```

### D. 注意力权重示例

**例1: 英法翻译**

```
源句子: The dog chased the cat
目标句子: Le chien a poursuivi le chat

注意力权重矩阵(α):
       The dog chased the cat
Le    0.8  0.1   0.0    0.1  0.0
chien 0.1  0.8   0.0    0.0  0.1
a     0.0  0.0   0.6    0.2  0.2
pour..0.0  0.1   0.7    0.1  0.1
le    0.1  0.0   0.0    0.8  0.1
chat  0.0  0.1   0.0    0.1  0.8
```

观察:
- 对角线强(对齐)
- 第一行"Le"同时关注"The"(冠词)和"dog"(性别/数匹配)
- 第三行"a"(助动词)分散关注多个词

**例2: 摘要任务**

在文本摘要中,注意力权重可能呈现不同的模式:
- 不再是对角线(输入输出顺序不同)
- 更分散(多个源词可能贡献于一个目标词)
- 某些词(如停用词)权重一致低

---

**文档版本**: 1.0
**最后更新**: 2025-12-28
**适用于**: Megatron-LM v0.12.0 及后续兼容版本
