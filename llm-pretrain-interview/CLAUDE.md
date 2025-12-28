# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概述

这是一个**基于 Megatron-LM 的 LLM 预训练技术知识库项目**,旨在创建 100 份详细的技术文档,系统梳理大语言模型预训练的完整知识体系。

### 项目特点
- **代码驱动**: 每个知识点都对应 Megatron-LM 实际代码实现
- **数学严谨**: 从数学原理推导到工程实现
- **生产级别**: 基于 NVIDIA 官方生产级代码
- **面试导向**: 涵盖 LLM 预训练技术面试核心知识点

### 目录结构
```
llm-pretrain-interview/
├── CLAUDE.md                              # 本文件
├── OVERVIEW.md                            # 100 文档体系总览
├── TODO.md                                # 进度追踪
├── 21-transformer-architecture.md         # 已完成文档
├── 22-self-attention.md                   # 已完成文档
├── 23-scaled-dot-product-attention.md     # 已完成文档
├── 24-multi-head-attention.md             # 已完成文档
├── 28-rope-positional-embedding.md        # 已完成文档
├── 31-grouped-query-attention.md          # 已完成文档
├── 33-multi-latent-attention.md           # 已完成文档
├── 40-kv-cache-mechanism.md               # 已完成文档
└── [其他待创建文档 01-100]
```

### 与 Megatron-LM 代码仓库的关系
- **代码仓库路径**: `/volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/`
- **文档仓库路径**: `/volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/llm-pretrain-interview/`
- **关系**: 文档仓库是代码仓库的子目录,所有文档都基于父目录中的 Megatron-LM 代码

---

## 核心文档

### OVERVIEW.md (v2.0)
定义完整的 100 文档体系,包含:
- 14 个部分: 数学基础 (1-10)、深度学习基础 (11-20)、Transformer 基础 (21-30)、高级注意力 (31-40)、模型架构 (41-50)、数据并行 (51-55)、张量并行 (56-60)、流水线并行 (61-67)、FSDP 与 ZeRO (68-72)、序列并行 (73-75)、MoE (76-80)、优化器 (81-92)、混合精度 (93-96)、数据工程 (97-100)
- 每个知识点都有对应的代码位置注释 (如: `megatron/core/transformer/attention.py:1014-1349`)
- 所有 100 个知识点均已验证在 Megatron-LM 代码仓库中存在 ✅

### TODO.md (v2.2)
追踪所有 100 份文档的进度:
- 整体进度: 10% (9/100 已完成, 1/100 进行中)
- 优先级标记 (P1/P2)
- 里程碑计划
- 质量检查清单

---

## 文档标准模板

所有文档都遵循统一的模板结构,确保内容完整性和一致性:

### 必需章节
1. **标题与元数据**
   ```markdown
   # [编号]. [标题]

   > **文档编号**: XX
   > **所属部分**: 第X部分 - [部分名称] (XX-XX)
   > **对应原文档**: [如果有]
   > **代码位置**: `megatron/core/path/to/file.py:line_start-line_end`
   > **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM 仓库实际代码)
   ```

2. **引言** - 背景、重要性、学习目标、前置知识
3. **相关工作** - 历史发展、技术对比、Megatron 实现
4. **符号定义** - 数学符号表、代码变量约定
5. **数学原理** - 核心理论、算法推导、复杂度分析
6. **算法伪代码** - 清晰的算法描述
7. **代码实现详解** - 核心类/函数、关键细节、单元测试
8. **实验结果** - 实验设置、性能指标、可视化
9. **消融研究** - 组件消融、设计选择合理性
10. **超参数分析** - 关键超参数、敏感性、调优建议
11. **深入探讨** - 理论深化、与其他技术关系、常见问题、最佳实践
12. **总结** - 核心要点、优势、局限性、适用场景
13. **参考文献** - 核心论文、相关论文、官方文档
14. **附录** - 数学推导补充、代码完整示例、配置文件、术语表

### 关键要求
- **代码位置准确**: 所有代码引用必须基于 Megatron-LM 仓库实际文件路径和函数名
- **代码存在性验证**: 不包含代码仓库中不存在的内容
- **数学严谨**: 公式推导完整无误
- **代码与理论一致**: 代码实现与数学推导对应
- **页数控制**: 约 400-700 行 Markdown, 不超过 32000 token

---

## 创建新文档的工作流程

### 1. 准备阶段
```bash
# 确认要创建的文档编号和主题
# 查阅 OVERVIEW.md 获取知识点描述和代码位置
# 查阅 TODO.md 确认优先级和状态
```

### 2. 代码验证
```bash
# 验证代码位置存在
cd /volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/
ls -la megatron/core/[相关路径]/[文件名].py

# 阅读代码实现
# 确认所有要引用的类、函数、配置项都存在
```

### 3. 文档编写
- 按照标准模板创建文档
- 数学推导严谨完整
- 代码分析基于实际代码
- 不引用不存在的功能
- 使用 MCP 检索相关文献 (如果需要)

### 4. 质量检查
- [ ] 所有代码引用在 Megatron-LM 仓库中存在
- [ ] 数学公式正确且完整
- [ ] 代码示例与理论一致
- [ ] 没有过时或不存在的内容
- [ ] 文档长度适中 (50-80 页)
- [ ] 逻辑连贯,易于理解

### 5. 更新进度
```markdown
# 更新 TODO.md
- 将文档状态从 "📝 待编写" 改为 "✅ 已完成"
- 更新整体进度百分比
- 添加完成记录到"最新完成任务"部分
```

### 6. 文档命名规范
```
格式: {编号:02d}-{英文标题}.md
示例: 23-scaled-dot-product-attention.md
示例: 56-tensor-parallelism-theory.md
```

---

## 重要约束

### ✅ 必须遵守
1. **代码版本锁定**: 所有代码引用必须基于 Megatron-LM
2. **代码存在性**: 不引用代码仓库中不存在的功能
3. **代码位置准确**: 文件路径和行号必须准确
4. **模板完整性**: 所有文档都必须包含完整的章节
5. **数学严谨性**: 数学推导必须严谨无误
6. **交叉引用正确**: 引用其他文档时使用正确的编号

### ❌ 严格禁止
1. **不引用不存在的代码**: 禁止引用代码仓库中不存在的文件或者函数
2. **不编造代码路径**: 所有代码路径必须经过验证
3. **不跳过模板章节**: 每个文档都必须包含所有必需章节
4. **不使用错误的版本**: 代码引用必须基于参考的代码仓库,不能引用其他版本特有的功能

---

## 常用命令

### 查看代码结构
```bash
# 查看 Megatron 核心模块
cd /volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/
ls -la megatron/core/

# 查看特定模块
ls -la megatron/core/transformer/
ls -la megatron/core/optimizer/
ls -la megatron/core/models/
```

### 搜索代码实现
```bash
# 查找类定义
grep -r "class TransformerBlock" megatron/core/

# 查找函数定义
grep -r "def forward" megatron/core/transformer/

# 查找配置参数
grep -r "num_query_groups" megatron/core/
```

### 验证代码位置
```bash
# 查看指定文件的行数
wc -l megatron/core/transformer/attention.py

# 查看文件特定行
sed -n '1014,1349p' megatron/core/transformer/attention.py
```

### 文档进度查询
```bash
# 查看已完成文档列表
ls -la llm-pretrain-interview/*.md | grep "^[0-9]"

# 统计文档数量
ls llm-pretrain-interview/*.md | wc -l
```

---

## 代码架构速查

### 核心目录结构
```
megatron/
├── core/
│   ├── models/                        # 模型实现
│   │   ├── gpt/gpt_model.py          # GPT 模型
│   │   ├── bert/bert_model.py        # BERT 模型
│   │   ├── T5/t5_model.py            # T5 模型
│   │   ├── mamba/mamba_model.py      # Mamba 模型
│   │   └── common/embeddings/        # Embedding 层
│   │       ├── language_model_embedding.py
│   │       ├── rotary_pos_embedding.py      # RoPE
│   │       └── yarn_rotary_pos_embedding.py # YaRN
│   ├── transformer/                   # Transformer 核心组件
│   │   ├── transformer_config.py     # 配置类
│   │   ├── transformer_block.py      # TransformerBlock
│   │   ├── transformer_layer.py      # TransformerLayer
│   │   ├── attention.py              # 注意力机制 (MHA, GQA, MQA)
│   │   ├── dot_product_attention.py  # 点积注意力
│   │   ├── multi_latent_attention.py # MLA
│   │   ├── mlp.py                    # 前馈网络
│   │   ├── torch_layer_norm.py       # LayerNorm
│   │   └── moe/                      # MoE 实现
│   │       ├── router.py
│   │       ├── moe_utils.py
│   │       ├── token_dispatcher.py
│   │       └── shared_experts.py
│   ├── tensor_parallel/               # 张量并行
│   │   ├── layers.py                 # ColumnParallelLinear, RowParallelLinear
│   │   └── cross_entropy.py          # 并行交叉熵
│   ├── pipeline_parallel/             # 流水线并行
│   │   ├── schedules.py              # 调度策略 (1F1B, interleaved)
│   │   └── p2p_communication.py      # P2P 通信
│   ├── distributed/                   # 分布式训练
│   │   ├── data_parallel_base.py
│   │   ├── distributed_data_parallel.py # DDP
│   │   ├── fsdp/                     # FSDP
│   │   └── param_and_grad_buffer.py  # 梯度缓冲
│   ├── optimizer/                     # 优化器
│   │   ├── optimizer_config.py       # 优化器配置 (Adam, AdamW, SGD)
│   │   ├── optimizer.py              # 优化器基类
│   │   ├── distrib_optimizer.py      # 分布式优化器 (ZeRO)
│   │   ├── clip_grads.py             # 梯度裁剪
│   │   ├── grad_scaler.py            # 损失缩放
│   │   └── optimizer_param_scheduler.py # 学习率调度
│   ├── inference/                     # 推理
│   │   └── inference_request.py      # KV Cache 管理
│   ├── datasets/                      # 数据集
│   │   ├── indexed_dataset.py        # 索引化数据集
│   │   └── blended_megatron_dataset_*.py # 数据混合
│   ├── parallel_state.py              # 并行状态管理
│   ├── fp8_utils.py                   # FP8 工具
│   └── fp4_utils.py                   # FP4 工具
├── pretrain_gpt.py                    # GPT 预训练脚本
├── pretrain_bert.py                   # BERT 预训练脚本
├── pretrain_t5.py                     # T5 预训练脚本
├── pretrain_mamba.py                  # Mamba 预训练脚本
└── examples/                          # 示例代码
    ├── gpt3/
    ├── llama/
    └── mixtral/
```

### 关键类与函数

#### Transformer 核心
- `TransformerConfig`: `megatron/core/transformer/transformer_config.py:930-1012`
- `TransformerBlock`: `megatron/core/transformer/transformer_block.py:40-500`
- `TransformerLayer`: `megatron/core/transformer/transformer_layer.py:40-500`

#### 注意力机制
- `Attention` (MHA/GQA/MQA): `megatron/core/transformer/attention.py:1014-1349`
- `DotProductAttention`: `megatron/core/transformer/dot_product_attention.py:100-350`
- `MultiLatentAttention`: `megatron/core/transformer/multi_latent_attention.py`

#### 并行
- `ColumnParallelLinear`: `megatron/core/tensor_parallel/layers.py:200-350`
- `RowParallelLinear`: `megatron/core/tensor_parallel/layers.py:450-600`
- `initialize_model_parallel`: `megatron/core/parallel_state.py:1-150`

#### 优化器
- `AdamOptimizerConfig`: `megatron/core/optimizer/optimizer_config.py:45-118`
- `DistributedOptimizer`: `megatron/core/optimizer/distrib_optimizer.py:126-500`

---

## 文献检索指南

当创建文档时,可能需要检索相关学术文献来支持数学推导和理论分析。使用 MCP (Model Context Protocol) 可以帮助检索相关论文。

### 核心论文按主题分类

#### Transformer 架构
- Attention Is All You Need (Vaswani et al., 2017)
- Pre-LN Transformer (Xiong et al., 2020)

#### 并行训练
- Megatron-LM: 张量并行 (Shoeybi et al., 2019)
- Megatron-LM v2: 流水线并行 (Narayanan et al., 2021)
- ZeRO (Rajbhandari et al., 2020)

#### 注意力优化
- Flash Attention (Dao et al., 2022)
- Flash Attention v2 (Dao, 2023)
- GQA (Ainslie et al., 2023)
- MLA (DeepSeek-V2, 2024)

#### 位置编码
- RoPE (Su et al., 2021)
- YaRN (Peng et al., 2023)

#### MoE
- Switch Transformers (Fedus et al., 2022)
- DeepSeek-V2 (DeepSeek, 2024)

#### 模型架构
- GPT-3 (Brown et al., 2020)
- LLaMA (Touvron et al., 2023)
- Mistral/Mixtral (Mistral AI, 2023)
- Mamba (Gu & Dao, 2023)

---

## 当前进度快照

**最后更新**: 2025-12-27

### 整体统计
- **总文档数**: 100
- **已完成**: 9 (10%)
- **进行中**: 1 (文档 29: LayerNorm 部分完成)
- **待编写**: 90

### 已完成文档列表
1. 21-transformer-architecture.md (1685 行, 60KB)
2. 22-self-attention.md (1805 行, 56KB)
3. 23-scaled-dot-product-attention.md (467 行, 16KB)
4. 24-multi-head-attention.md (473 行, 17KB)
5. 28-rope-positional-embedding.md (467 行, 16KB)
6. 31-grouped-query-attention.md (467 行, 16KB)
7. 33-multi-latent-attention.md (473 行, 17KB)
8. 40-kv-cache-mechanism.md (631 行, 21KB)
9. 29-layernorm.md (部分完成)

### 按部分完成率
- Transformer 基础 (21-30): 70%
- 高级注意力 (31-40): 30%
- 其他部分: 0%

### 近期任务
- [ ] 完成剩余 Transformer 基础文档 (25-27, 30)
- [ ] 完成剩余高级注意力文档 (32, 34-39)
- [ ] 开始模型架构部分 (41-50)

---

## 常见问题

### Q: 如何确定文档编号?
A: 查阅 OVERVIEW.md,每个知识点都有固定编号 (01-100)。

### Q: 如何找到对应的 Megatron 代码?
A: 在 OVERVIEW.md 中每个知识点都标注了代码位置,如 `megatron/core/transformer/attention.py:1014-1349`。

### Q: 文档应该多长?
A: 50-80 页 (约 400-700 行 Markdown),根据主题复杂度调整。

### Q: 如果代码位置找不到怎么办?
A:
1. 确认是否使用了正确的代码库路径 (`/volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/`)
2. 确认是否是 Megatron-LM 代码仓库
3. 如果确实不存在,说明 OVERVIEW.md 中的代码位置可能需要更新,请标记该问题

### Q: 可以参考哪些文档作为模板?
A: 推荐参考已完成的高质量文档:
- 23-scaled-dot-product-attention.md (数学推导完整)
- 24-multi-head-attention.md (代码分析详细)
- 31-grouped-query-attention.md (实验结果丰富)

---

## 贡献指南

### 文档质量标准
每个文档必须通过以下检查:

#### 内容完整性 ✓
- [ ] 引言部分完整 (概述、前置知识、文档组织)
- [ ] 代码位置准确 (Megatron-LM 文件路径和行号)
- [ ] 相关工作部分充分 (历史发展、技术对比、Megatron 创新)
- [ ] 符号定义清晰 (数学符号表、代码变量约定)
- [ ] 数学原理严谨 (核心理论、算法推导、复杂度分析)
- [ ] 算法伪代码完整
- [ ] 代码实现详解 (核心类/函数、关键细节、单元测试)
- [ ] 实验结果充分 (实验设置、性能指标、可视化)
- [ ] 消融研究深入
- [ ] 超参数分析详尽
- [ ] 深入探讨全面
- [ ] 总结到位
- [ ] 参考文献齐全

#### 质量标准 ✓
- [ ] 数学公式推导无误
- [ ] 代码实现与理论一致
- [ ] 所有代码引用都基于 Megatron-LM 实际代码
- [ ] 不包含代码仓库中不存在的内容
- [ ] 图表清晰易懂
- [ ] 语言表达准确专业
- [ ] 适合初学者理解
- [ ] 达到研究著作深度
- [ ] 页数符合预期 (50-80 页)

#### 交叉引用 ✓
- [ ] 正确引用前置文档
- [ ] 标注后续相关文档
- [ ] 与 OVERVIEW.md 中的描述一致

---

## 技术栈

- **Python**: >=3.10
- **PyTorch**: >=2.0
- **CUDA**: >=11.8
- **NCCL**: >=2.18
- **Transformer Engine**: FP8 训练支持
- **Flash Attention**: v2/v3 支持

---

**最后更新**: 2025-12-27
**项目版本**: v2.0
**Megatron-LM 版本**: v0.12.0
