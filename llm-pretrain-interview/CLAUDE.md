# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

这是**基于 Megatron-LM 的 LLM 预训练技术知识库项目**，旨在创建 100 份详细的技术文档，系统梳理大语言模型预训练的完整知识体系。

- **代码仓库**: `/volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/`
- **文档目录**: `/volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/llm-pretrain-interview/`
- **Megatron版本**: v0.12.0

---

## 关键文件说明

| 文件 | 用途 | 受众 |
|------|------|------|
| **CLAUDE.md** | 本文件，知识库开发指南 | Claude模型 |
| **OVERVIEW.md** | 100文档体系总览、学习路径、技术栈 | 读者 |
| **TEMPLATE.md** | 标准文档模板（14章节结构） | Claude模型 |
| **TODO.md** | 进度追踪、里程碑、任务列表 | Claude模型 |
| **REFERENCES.md** | 官方资源、框架对比、核心论文汇总 | Claude模型 |

**重要**: 
- 编写文档时必须参考 `TEMPLATE.md` 的标准结构。
- 文档完成之后同步更新 `TODO.md`
- 同步将重要的参考文献加入 `REFERENCES.md`

---

## 创建新文档的工作流程

### 1. 准备阶段
```bash
# 查阅 OVERVIEW.md 获取知识点描述和代码位置
# 查阅 TODO.md 确认优先级和当前状态
# 查看当前文件夹下知识点文档的进展情况
```

### 2. 代码验证
```bash
# 验证代码位置存在
cd /volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/
ls -la megatron/core/[相关路径]/[文件名].py

# 阅读代码实现，确认所有要引用的类、函数、配置项都存在
```

### 3. 文献检索 (重要!)
**所有参考文献必须通过 MCP 检索验证**:
- 使用 WebSearch 工具检索论文标题、作者、发表年份
- 验证论文的 arXiv ID 或 DOI
- 确保引用格式准确: `作者 (年份). "标题". 会议/期刊`

```
示例检索:
- "Attention Is All You Need Vaswani 2017 arXiv"
- "Megatron-LM Shoeybi 2019 arXiv"
- "ZeRO Rajbhandari 2020 SC"
```

### 4. 文档编写
- 按照 `TEMPLATE.md` 的14章节结构创建文档
- 数学推导严谨完整
- 代码分析基于实际代码
- 不引用不存在的功能

### 5. 质量检查
- [ ] 所有代码引用在 Megatron-LM 仓库中存在
- [ ] 所有参考文献经过 MCP 检索验证
- [ ] 数学公式正确且完整
- [ ] 代码示例与理论一致
- [ ] 文档长度适中 (约2000-3000行)

### 6. 更新进度
```markdown
# 更新 TODO.md
- 将文档状态从 "📝 待编写" 改为 "✅ 已完成"
- 更新整体进度百分比
```

### 7. 更新文献
```markdown
# 更新 REFERENCES.md
- 将新增加知识点中重要的参考文献加入
- 更新相关学习资源

### 8. 文档命名规范
```
格式: {编号:02d}-{英文标题}.md
示例: 23-scaled-dot-product-attention.md
示例: 56-tensor-parallelism-theory.md
```

---

## 重要约束

### ✅ 必须遵守
1. **代码存在性**: 所有代码引用必须在仓库中实际存在
2. **代码路径准确**: 文件路径和行号必须经过验证
3. **模板完整性**: 所有文档必须包含 TEMPLATE.md 中的全部章节
4. **数学严谨性**: 公式推导必须严谨无误
5. **参考文献验证**: 所有论文引用必须通过 MCP 检索验证
6. **交叉引用正确**: 引用其他文档时使用正确的编号

### ❌ 严格禁止
1. **不引用不存在的代码**: 禁止引用不存在的文件或函数
2. **不编造参考文献**: 禁止编造论文标题、作者或发表信息
3. **不跳过模板章节**: 每个文档必须包含所有必需章节
4. **不使用错误的版本**: 代码引用必须基于 Megatron-LM v0.12.0

---

## 代码架构速查

### 核心目录结构
```
megatron/
├── core/
│   ├── models/                        # 模型实现 (GPT, BERT, T5, Mamba)
│   │   └── common/embeddings/         # Embedding层 (RoPE, YaRN)
│   ├── transformer/                   # Transformer核心组件
│   │   ├── attention.py               # 注意力 (MHA, GQA, MQA)
│   │   ├── multi_latent_attention.py  # MLA
│   │   ├── mlp.py                     # FFN
│   │   └── moe/                       # MoE实现
│   ├── tensor_parallel/               # 张量并行
│   ├── pipeline_parallel/             # 流水线并行
│   ├── distributed/                   # 分布式训练 (DDP, FSDP)
│   └── optimizer/                     # 优化器
├── pretrain_*.py                      # 预训练脚本
└── examples/                          # 示例代码
```

### 关键类位置
- `Attention`: `megatron/core/transformer/attention.py:1014-1349`
- `DotProductAttention`: `megatron/core/transformer/dot_product_attention.py:100-350`
- `MLP`: `megatron/core/transformer/mlp.py:24-352`
- `DistributedDataParallel`: `megatron/core/distributed/distributed_data_parallel.py`
- `ColumnParallelLinear`: `megatron/core/tensor_parallel/layers.py:200-350`
- `DistributedOptimizer`: `megatron/core/optimizer/distrib_optimizer.py`

---

## 常用命令

### 查看代码
```bash
cd /volume/pt-train/users/rbliu/github/LLM-Pretrain-Interview/
ls -la megatron/core/transformer/
```

### 搜索实现
```bash
grep -r "class TransformerBlock" megatron/core/
grep -r "num_query_groups" megatron/core/
```

### 查看文档进度
```bash
ls llm-pretrain-interview/*.md | wc -l
```

---

## 参考文献检索指南

> **参考资源汇总**: [REFERENCES.md](REFERENCES.md) 包含本知识库涉及的官方资源、框架对比和核心论文索引。

### 检索工具
- **WebSearch**: 使用 Claude Code 的 WebSearch 工具验证论文信息
- **Google Scholar**: https://scholar.google.com/
- **arXiv**: https://arxiv.org/
- **Semantic Scholar**: https://www.semanticscholar.org/

### 必须通过 MCP 检索验证的信息
1. **论文标题**: 确保准确无误
2. **作者列表**: 至少验证第一作者
3. **发表年份**: 确认正确年份
4. **发表渠道**: 会议名称或期刊名称
5. **arXiv ID**: 如果有的话

### 搜索关键词示例

| 主题 | 搜索关键词 |
|------|------------|
| Transformer | "Vaswani attention transformer 2017" |
| 张量并行 | "Shoeybi Megatron tensor parallel 2019" |
| 流水线并行 | "Huang GPipe pipeline 2019" |
| ZeRO | "Rajbhandari ZeRO memory 2020" |
| Flash Attention | "Dao FlashAttention IO-aware 2022" |
| GQA | "Ainslie GQA grouped query 2023" |
| RoPE | "Su RoPE rotary position 2021" |
| MoE | "Fedus Switch transformer MoE 2022" |

### 检索示例
```
# 使用 WebSearch 工具
query: "Attention Is All You Need Vaswani 2017 arXiv"
预期结果: arXiv:1706.03762, NeurIPS 2017

query: "ZeRO Memory Optimizations Rajbhandari SC 2020"
预期结果: SC'20, Microsoft DeepSpeed

query: "FlashAttention Dao 2022 NeurIPS"
预期结果: arXiv:2205.14135, NeurIPS 2022
```

### 引用格式规范

**标准格式**:
```
作者 (年份). "论文标题". 会议/期刊. arXiv:XXXX.XXXXX
```

**示例**:
```
Vaswani et al. (2017). "Attention Is All You Need". NeurIPS. arXiv:1706.03762
Shoeybi et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". arXiv:1909.08053
Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS. arXiv:2205.14135
```

---

## 常见问题

### Q: 如何确定文档编号?
A: 查阅 `OVERVIEW.md`，每个知识点都有固定编号 (01-100)。

### Q: 如何找到对应的 Megatron 代码?
A: 在 `OVERVIEW.md` 中每个知识点都标注了代码位置。

### Q: 文档应该多长?
A: 约2000-3000行Markdown，根据主题复杂度调整。

### Q: 如果代码位置找不到怎么办?
A: 1) 确认使用正确的代码库路径；2) 代码可能在不同文件；3) 标记该问题并更新 OVERVIEW.md。

### Q: 参考文献如何验证?
A: **必须**使用 WebSearch 工具检索验证论文信息，不可编造。

---

**最后更新**: 2025-12-30
**Megatron-LM 版本**: v0.12.0
