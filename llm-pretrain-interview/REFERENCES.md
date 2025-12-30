# REFERENCES.md

**版本**: 2.1

**最后更新**: 2025-12-30

**用途**: 汇总本知识库涉及的所有官方资源、相关框架和核心论文，供模型编写文档和读者学习时参考

**版本 2.1 新增**:
- 激活检查点（Gradient Checkpointing）相关论文：Chen et al. (2016), Griewank & Walther (2000), Jain et al. (2020)
- 更新55.1扩展文档知识点映射

**版本 2.0 新增**:
- NVIDIA 生态: NCCL、NeMo Framework详细资源
- PyTorch 工具: torchao量化库
- GPU 编程: Triton、cuDNN、FlashInfer
- 边缘推理: ExecuTorch、llama.cpp、MLC LLM
- 量化工具: BitsAndBytes、AutoGPTQ、torchao
- 微调工具: Axolotl
- 推理服务: SGLang、HuggingFace TGI
- MLOps平台: Ray
- 新增论文: LoRA、QLoRA、SmoothQuant、AWQ、Speculative Decoding

---

## 📚 官方资源与文档

### NVIDIA Megatron 生态

| 资源 | 链接 | 说明 |
|------|------|------|
| Megatron-LM GitHub | https://github.com/NVIDIA/Megatron-LM | NVIDIA官方代码仓库 |
| Megatron-Core 文档 | https://docs.nvidia.com/megatron-core/index.html | Megatron Core API文档 |
| Megatron-Core 用户指南 | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html | 开发者指南 |
| NeMo Framework | https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/megatron.html | NeMo Megatron集成 |
| NeMo Framework GitHub | https://github.com/NVIDIA-NeMo/NeMo | 端到端LLM训练框架 |
| NeMo Overview | https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html | NeMo 框架用户指南 |
| Transformer Engine | https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html | FP8训练文档 |
| Transformer Engine GitHub | https://github.com/NVIDIA/TransformerEngine | FP8/FP4训练库 |
| TE FP8 Primer | https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html | FP8入门教程 |
| NVIDIA NCCL | https://developer.nvidia.com/nccl | 集合通信库 |
| NCCL GitHub | https://github.com/NVIDIA/nccl | NCCL代码仓库 |
| NCCL 文档 | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html | NCCL用户指南 |

### PyTorch 分布式训练

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch Documentation | https://pytorch.org/docs/ | PyTorch官方文档 |
| PyTorch FSDP2 教程 | https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html | FSDP2入门 |
| PyTorch FSDP2 API | https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html | FSDP2 API文档 |
| PyTorch FSDP 高级教程 | https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html | FSDP进阶 |
| PyTorch AO (torchao) | https://github.com/pytorch/ao | PyTorch量化和稀疏库 |
| torchao 文档 | https://docs.pytorch.org/ao/stable/ | torchao官方文档 |
| NVIDIA Apex | https://github.com/NVIDIA/apex | 混合精度和分布式训练工具 |
| Apex 文档 | https://nvidia.github.io/apex/amp.html | Apex AMP 文档 |

### DeepSpeed

| 资源 | 链接 | 说明 |
|------|------|------|
| DeepSpeed 官网 | https://www.deepspeed.ai/ | Microsoft DeepSpeed |
| DeepSpeed GitHub | https://github.com/microsoft/DeepSpeed | 代码仓库 |
| ZeRO 教程 | https://www.deepspeed.ai/tutorials/zero/ | ZeRO优化器教程 |
| ZeRO++ 教程 | https://www.deepspeed.ai/tutorials/zeropp/ | ZeRO++通信优化 |
| Megatron-DeepSpeed 集成 | https://www.deepspeed.ai/tutorials/megatron/ | Megatron-LM GPT2教程 |

### 注意力优化

| 资源 | 链接 | 说明 |
|------|------|------|
| Flash Attention GitHub | https://github.com/Dao-AILab/flash-attention | Tri Dao的FA实现 |
| FlashAttention-3 博客 | https://tridao.me/blog/2024/flash3/ | FA3技术博客 |
| vLLM 官网 | https://vllm.ai/ | 高效推理引擎 |
| vLLM 文档 | https://docs.vllm.ai/en/latest/ | vLLM官方文档 |
| vLLM GitHub | https://github.com/vllm-project/vllm | vLLM代码仓库 |
| SGLang GitHub | https://github.com/sgl-project/sglang | 高性能LLM/VLM服务框架 |
| SGLang 学习资料 | https://github.com/sgl-project/sgl-learning-materials | SGLang学习资源 |

### 模型资源

| 资源 | 链接 | 说明 |
|------|------|------|
| HuggingFace Transformers | https://huggingface.co/docs/transformers | 模型参考实现 |
| HuggingFace Accelerate Megatron | https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm | Accelerate Megatron集成 |
| Mamba GitHub | https://github.com/state-spaces/mamba | SSM架构实现 |
| DeepSeek-V2 GitHub | https://github.com/deepseek-ai/DeepSeek-V2 | DeepSeek-V2 MoE模型 |
| DeepSeek-V3 GitHub | https://github.com/deepseek-ai/DeepSeek-V3 | DeepSeek-V3模型 |
| Qwen GitHub | https://github.com/QwenLM/Qwen | 阿里通义千问 |

### GPU 编程与内核优化

| 资源 | 链接 | 说明 |
|------|------|------|
| OpenAI Triton 官网 | https://triton-lang.org/ | Triton编程语言官网 |
| Triton GitHub | https://github.com/triton-lang/triton | GPU编程语言和编译器 |
| OpenAI Triton 博客 | https://openai.com/index/triton/ | Triton介绍博客 |
| NVIDIA Triton 技术博客 | https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/ | Blackwell架构支持 |
| NVIDIA cuDNN | https://developer.nvidia.com/cudnn | GPU加速深度学习基础库 |
| cuDNN 文档 | https://docs.nvidia.com/deeplearning/cudnn/latest/ | cuDNN官方文档 |

### 边缘与移动端推理

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch ExecuTorch | https://pytorch.org/projects/executorch/ | PyTorch边缘设备推理框架 |
| ExecuTorch GitHub | https://github.com/pytorch/executorch | 移动/嵌入式AI推理 |
| llama.cpp GitHub | https://github.com/ggml-org/llama.cpp | C/C++ LLM推理,量化支持 |
| MLC LLM GitHub | https://github.com/mlc-ai/mlc-llm | Apache TVM通用LLM部署引擎 |
| MLC LLM 文档 | https://llm.mlc.ai/docs/get_started/introduction | MLC LLM官方文档 |
| HuggingFace TGI | https://github.com/huggingface/text-generation-inference | 大模型文本生成推理工具包 |
| TGI 文档 | https://huggingface.co/docs/text-generation-inference | TGI官方文档 |

### 量化工具

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch AO (torchao) | https://github.com/pytorch/ao | PyTorch原生量化和稀疏库 |
| torchao 文档 | https://docs.pytorch.org/ao/stable/ | torchao官方文档 |
| BitsAndBytes GitHub | https://github.com/bitsandbytes-foundation/bitsandbytes | k-bit量化(8-bit, 4-bit, QLoRA) |
| BitsAndBytes HF 文档 | https://huggingface.co/docs/transformers/en/quantization/bitsandbytes | BitsAndBytes集成文档 |
| AutoGPTQ GitHub | https://github.com/AutoGPTQ/AutoGPTQ | GPTQ算法量化工具 (已归档) |
| AutoGPTQ 文档 | https://autogptq.github.io/AutoGPTQ/ | AutoGPTQ官方文档 |

### LLM 微调工具

| 资源 | 链接 | 说明 |
|------|------|------|
| Axolotl GitHub | https://github.com/axolotl-ai-cloud/axolotl | LLM微调框架,支持YAML配置 |
| Axolotl 文档 | https://docs.axolotl.ai/ | Axolotl官方文档 |
| Axolotl 官网 | https://axolotl.ai/ | Axolotl开源微调工具 |

### 推理内核库

| 资源 | 链接 | 说明 |
|------|------|------|
| FlashInfer GitHub | https://github.com/flashinfer-ai/flashinfer | LLM推理内核库(MLSys 2025最佳论文) |
| FlashInfer 技术博客 | https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/ | NVIDIA FlashInfer介绍 |

---

## 🔧 相关框架对比

| 框架 | 主要特性 | 开发者 | 链接 |
|------|----------|--------|------|
| **Megatron-LM** | 张量并行、流水线并行、序列并行、MoE | NVIDIA | https://github.com/NVIDIA/Megatron-LM |
| **DeepSpeed** | ZeRO优化器、ZeRO++、3D并行、推理优化 | Microsoft | https://github.com/microsoft/DeepSpeed |
| **Colossal-AI** | 异构训练、自动并行、Gemini内存管理 | HPC-AI Tech | https://github.com/hpcaitech/ColossalAI |
| **Alpa** | 自动并行策略搜索、ILP优化 | UC Berkeley | https://github.com/alpa-projects/alpa |
| **Ray** | 分布式训练、超参数调优、MLOps统一平台 | Anyscale | https://www.ray.io/ |
| **vLLM** | PagedAttention、高效推理、Continuous Batching | UC Berkeley | https://github.com/vllm-project/vllm |
| **SGLang** | 高性能LLM/VLM服务、Day-0模型支持 | SGL Project | https://github.com/sgl-project/sglang |
| **TensorRT-LLM** | 推理优化、量化部署、Tensor Core加速 | NVIDIA | https://github.com/NVIDIA/TensorRT-LLM |
| **LightLLM** | 高吞吐推理、Triton Kernel优化 | ModelTC | https://github.com/ModelTC/lightllm |
| **llama.cpp** | C/C++推理、量化、跨平台支持 | GGML | https://github.com/ggml-org/llama.cpp |
| **ExecuTorch** | 边缘设备AI、50KB起始占用 | PyTorch/Meta | https://github.com/pytorch/executorch |
| **MLC LLM** | Apache TVM编译、跨平台部署 | MLC-AI | https://github.com/mlc-ai/mlc-llm |
| **HuggingFace TGI** | LLM推理服务、多后端支持 | HuggingFace | https://github.com/huggingface/text-generation-inference |
| **Axolotl** | LLM微调框架、YAML配置、QAT支持 | Axolotl AI | https://github.com/axolotl-ai-cloud/axolotl |
| **Triton** | Python风格GPU编程、自定义内核开发 | OpenAI | https://github.com/triton-lang/triton |
| **NVIDIA Apex** | 混合精度训练、分布式优化 (已弃用) | NVIDIA | https://github.com/NVIDIA/apex |

---

## 📖 核心论文索引

### Transformer架构

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | 1706.03762 |
| Pre-LN Transformer | Xiong et al. | 2020 | ICML | 2002.04745 |
| GPT-3: Language Models are Few-Shot Learners | Brown et al. | 2020 | NeurIPS | 2005.14165 |

### 并行训练

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Megatron-LM (张量并行) | Shoeybi et al. | 2019 | arXiv | 1909.08053 |
| Megatron-LM v2 (流水线并行) | Narayanan et al. | 2021 | SC | 2104.04473 |
| GPipe | Huang et al. | 2019 | NeurIPS | 1811.06965 |
| PipeDream | Narayanan et al. | 2019 | SOSP | - |
| ZeRO: Memory Optimizations | Rajbhandari et al. | 2020 | SC | 1910.02054 |
| ZeRO-Offload | Ren et al. | 2021 | ATC | 2101.06840 |
| ZeRO++ | Wang et al. | 2023 | arXiv | 2306.10209 |
| FSDP (Fully Sharded Data Parallel) | Zhao et al. | 2023 | VLDB | 2304.11277 |
| Ring Attention | Liu et al. | 2023 | arXiv | 2310.01889 |
| Context Parallelism for Million-Token Inference | - | 2024 | arXiv | 2411.01783 |

### 注意力优化

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| FlashAttention | Dao et al. | 2022 | NeurIPS | 2205.14135 |
| FlashAttention-2 | Dao | 2023 | ICLR | 2307.08691 |
| **FlashAttention-3** | Shah, Dao et al. | 2024 | arXiv | 2407.08608 |
| GQA (Grouped Query Attention) | Ainslie et al. | 2023 | EMNLP | 2305.13245 |
| MQA (Multi-Query Attention) | Shazeer | 2019 | arXiv | 1911.02150 |
| MLA (Multi-Latent Attention) | DeepSeek | 2024 | arXiv | 2405.04434 |
| PagedAttention (vLLM) | Kwon et al. | 2023 | SOSP | 2309.06180 |

### 位置编码

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| RoPE (Rotary Position Embedding) | Su et al. | 2021 | arXiv | 2104.09864 |
| ALiBi | Press et al. | 2022 | ICLR | 2108.12409 |
| YaRN | Peng et al. | 2023 | arXiv | 2309.00071 |

### MoE (Mixture of Experts)

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Switch Transformers | Fedus et al. | 2022 | JMLR | 2101.03961 |
| GShard | Lepikhin et al. | 2021 | ICLR | 2006.16668 |
| DeepSeek-V2 (MLA + MoE) | DeepSeek | 2024 | arXiv | 2405.04434 |
| **DeepSeek-V3** | DeepSeek | 2024 | arXiv | 2412.19437 |
| Mixtral of Experts | Mistral AI | 2024 | arXiv | 2401.04088 |

### 模型架构

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| BERT | Devlin et al. | 2019 | NAACL | 1810.04805 |
| T5 | Raffel et al. | 2020 | JMLR | 1910.10683 |
| LLaMA | Touvron et al. | 2023 | arXiv | 2302.13971 |
| LLaMA 2 | Touvron et al. | 2023 | arXiv | 2307.09288 |
| **LLaMA 3** | Meta AI | 2024 | arXiv | 2407.21783 |
| Mistral 7B | Mistral AI | 2023 | arXiv | 2310.06825 |
| Mamba | Gu & Dao | 2023 | arXiv | 2312.00752 |
| **Mamba-2 (State Space Duality)** | Dao & Gu | 2024 | arXiv | 2405.21060 |
| **Qwen2** | Alibaba | 2024 | arXiv | 2407.10671 |
| Qwen2.5 | Alibaba | 2024 | arXiv | 2412.15115 |

### 优化器与训练技术

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Adam | Kingma & Ba | 2015 | ICLR | 1412.6980 |
| AdamW (Decoupled Weight Decay) | Loshchilov & Hutter | 2019 | ICLR | 1711.05101 |
| Mixed Precision Training | Micikevicius et al. | 2018 | ICLR | 1710.03740 |
| LoRA (Low-Rank Adaptation) | Hu et al. | 2021 | ICLR | 2106.09685 |
| QLoRA (4-bit Quantization) | Dettmers et al. | 2023 | NeurIPS | 2305.14314 |

### 内存优化：激活检查点（Gradient Checkpointing）

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| **Training Deep Nets with Sublinear Memory Cost** | Chen et al. | 2016 | arXiv | 1604.06174 |
| Algorithm 799: Revolve (Checkpointing基础理论) | Griewank & Walther | 2000 | ACM TOMS | - |
| **Checkmate: Breaking the Memory Wall** | Jain et al. | 2020 | MLSys | - |

### 量化技术

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| SmoothQuant (W8A8 PTQ) | Xiao et al. | 2023 | ICML | 2211.10438 |
| AWQ (Activation-aware Weight Quantization) | Lin et al. | 2024 | MLSys (Best Paper) | 2306.00978 |
| QLoRA (4-bit Quantization) | Dettmers et al. | 2023 | NeurIPS | 2305.14314 |

### 推理优化

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Speculative Decoding | Leviathan et al. | 2023 | ICML | 2211.17192 |
| Speculative Sampling | Chen et al. | 2023 | arXiv | 2302.01318 |
| FlashInfer | - | 2025 | MLSys (Best Paper) | - |

### 归一化技术

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Layer Normalization | Ba et al. | 2016 | arXiv | 1607.06450 |
| RMSNorm | Zhang & Sennrich | 2019 | NeurIPS | 1910.07467 |
| Batch Normalization | Ioffe & Szegedy | 2015 | ICML | 1502.03167 |

### Scaling Laws

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Scaling Laws for Neural LMs | Kaplan et al. | 2020 | arXiv | 2001.08361 |
| Chinchilla (Training Compute-Optimal LLMs) | Hoffmann et al. | 2022 | NeurIPS | 2203.15556 |

---

## 📊 论文按知识点映射

| 文档编号 | 知识点 | 核心论文 |
|----------|--------|----------|
| 21-30 | Transformer基础 | Attention Is All You Need |
| 28 | RoPE | Su et al. (2021) |
| 31 | GQA | Ainslie et al. (2023) |
| 33 | MLA | DeepSeek-V2 (2024) |
| 34-36 | Flash Attention | Dao et al. (2022, 2023, 2024) |
| 39 | 长序列注意力 | Ring Attention (2023) |
| 40 | KV Cache | PagedAttention (2023) |
| 41 | GPT | GPT-3 (Brown et al., 2020) |
| 44 | LLaMA | LLaMA 3 (Meta, 2024) |
| 46 | Mamba | Mamba-2 (Dao & Gu, 2024) |
| 47 | YaRN | Peng et al. (2023) |
| 48 | 量化 | SmoothQuant, AWQ, QLoRA |
| 49 | 推理优化 | vLLM, PagedAttention, Speculative Decoding, FlashInfer |
| 51-55 | 数据并行 | - |
| **55.1** | **梯度累积 + 激活检查点** | **Chen et al. (2016), Griewank & Walther (2000), Jain et al. (2020)** |
| 56-60 | 张量并行 | Megatron-LM (2019) |
| 61-67 | 流水线并行 | GPipe, Megatron-LM v2 |
| 68-72 | ZeRO/FSDP | ZeRO (2020), FSDP (2023) |
| 73-75 | 序列/上下文并行 | Ring Attention (2023) |
| 76-80 | MoE | Switch Transformers, DeepSeek-V3 |
| 84-85 | Adam/AdamW | Adam (2015), AdamW (2019) |
| 86 | LoRA/QLoRA | LoRA (2021), QLoRA (2023) |
| 93-96 | 混合精度 | Mixed Precision Training (2018) |
| 95 | FP8训练 | Transformer Engine文档 |

---

## 🎓 学习资源

### 视频与演讲

| 资源 | 链接 | 说明 |
|------|------|------|
| GTC 2024: Transformer Engine & FP8 | https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/ | FP8训练介绍 |

### 博客与教程

| 资源 | 链接 | 说明 |
|------|------|------|
| Tri Dao 博客 | https://tridao.me/ | FlashAttention作者博客 |
| Mamba-2 技术博客 | https://tridao.me/blog/2024/mamba2-part1-model/ | Mamba-2详解 |
| vLLM 技术博客 | https://blog.vllm.ai/2023/06/20/vllm.html | PagedAttention原理 |

---

**注意**: 所有论文引用在添加到知识库文档前，必须通过 MCP WebSearch 工具验证准确性。
