# REFERENCES.md

**版本**: 3.4

**最后更新**: 2026-01-02

**用途**: 汇总本知识库涉及的所有官方资源、相关框架和核心论文，供模型编写文档和读者学习时参考

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

### 科学计算与数据处理

| 资源 | 链接 | 说明 |
|------|------|------|
| NumPy Documentation | https://numpy.org/doc/stable/ | NumPy官方文档 |
| NumPy memmap | https://numpy.org/doc/stable/reference/generated/numpy.memmap.html | 内存映射数组API |
| NumPy ndarray | https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html | ndarray基础类 |

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
| HuggingFace Accelerate | https://huggingface.co/docs/accelerate/index | Accelerate 集成 |
| Mamba GitHub | https://github.com/state-spaces/mamba | SSM架构实现 |

### GPU 编程与内核优化

| 资源 | 链接 | 说明 |
|------|------|------|
| OpenAI Triton 官网 | https://triton-lang.org/ | Triton编程语言官网 |
| Triton GitHub | https://github.com/triton-lang/triton | GPU编程语言和编译器 |
| OpenAI Triton 博客 | https://openai.com/index/triton/ | Triton介绍博客 |
| NVIDIA cuDNN | https://developer.nvidia.com/cudnn | GPU加速深度学习基础库 |
| cuDNN 文档 | https://docs.nvidia.com/deeplearning/cudnn/latest/ | cuDNN官方文档 |

### 量化工具

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch AO (torchao) | https://github.com/pytorch/ao | PyTorch原生量化和稀疏库 |
| torchao 文档 | https://docs.pytorch.org/ao/stable/ | torchao官方文档 |
| BitsAndBytes GitHub | https://github.com/bitsandbytes-foundation/bitsandbytes | k-bit量化(8-bit, 4-bit, QLoRA) |
| GPTQModel GitHub | https://github.com/ModelCloud/GPTQModel | LLM量化工具 |

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
| **Megatron-LM v2 (序列并行)** | **Narayanan et al.** | **2021** | **SC** | **2104.04473** |
| GPipe | Huang et al. | 2019 | NeurIPS | 1811.06965 |
| PipeDream | Narayanan et al. | 2019 | SOSP | - |
| **Reducing Activation Recomputation** | **Korthikanti et al.** | **2023** | **MLSys** | **2205.05198** |
| **Alpa: Automating Inter- and Intra-Operator Parallelism** | **Zheng et al.** | **2022** | **OSDI** | **2201.12023** |
| **Varuna: Scalable, Low-cost Training** | **Athlur et al.** | **2022** | **EuroSys** | **2111.04007** |
| ZeRO: Memory Optimizations | Rajbhandari et al. | 2020 | SC | 1910.02054 |
| ZeRO-Offload | Ren et al. | 2021 | ATC | 2101.06840 |
| ZeRO++ | Wang et al. | 2023 | arXiv | 2306.10209 |
| FSDP (Fully Sharded Data Parallel) | Zhao et al. | 2023 | VLDB | 2304.11277 |
| **DeepSpeed-MoE: Advancing MoE Inference and Training** | **Rajbhandari et al.** | **2022** | **ICML** | **2201.05596** |
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
| Outrageously Large Neural Networks (Sparsely-Gated MoE) | Shazeer et al. | 2017 | ICLR | 1701.06538 |
| GShard | Lepikhin et al. | 2021 | ICLR | 2006.16668 |
| Switch Transformers | Fedus et al. | 2022 | JMLR | 2101.03961 |
| **ST-MoE: Designing Stable and Transferable MoE** | **Zoph et al.** | **2022** | **arXiv** | **2202.08906** |
| Expert Choice Routing | Zhou et al. | 2022 | NeurIPS | 2202.09368 |
| SMEAR: Soft Merging of Experts with Adaptive Routing | Muqeeth et al. | 2023 | arXiv | 2306.03745 |
| Soft Mixture of Experts | Puigcerver et al. | 2023 | arXiv | 2308.00951 |
| **Global Load Balancing Loss for MoE** | **Qiu et al.** | **2025** | **arXiv** | **2501.11873** |
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
| **A Stochastic Approximation Method** | **Robbins & Monro** | **1951** | **Annals of Mathematical Statistics** | - |
| **Some methods of speeding up the convergence of iteration methods** | **Polyak** | **1964** | **USSR Computational Mathematics** | - |
| A method for solving a convex programming problem with convergence rate O(1/k²) | Nesterov | 1983 | Soviet Mathematics Doklady | - |
| On the importance of initialization and momentum in deep learning | Sutskever et al. | 2013 | ICML | - |
| An overview of gradient descent optimization algorithms | Ruder | 2016 | arXiv | 1609.04747 |
| A differential equation for modeling Nesterov's accelerated gradient method | Su, Boyd, & Candès | 2014 | JMLR | - |
| **AdaGrad: Adaptive Subgradient Methods** | **Duchi, Hazan, & Singer** | **2011** | **JMLR** | - |
| **RMSProp** | **Tieleman & Hinton** | **2012** | **Coursera Lecture 6.5** | - |
| **ADADELTA: An Adaptive Learning Rate Method** | **Zeiler** | **2012** | **arXiv** | **1212.5701** |
| Adam | Kingma & Ba | 2015 | ICLR | 1412.6980 |
| AdamW (Decoupled Weight Decay) | Loshchilov & Hutter | 2019 | ICLR | 1711.05101 |
| **AMSGrad (On the Convergence of Adam and Beyond)** | **Reddi, Kale, & Kumar** | **2018** | **ICLR** | **1904.09237** |
| **Lion: Symbolic Discovery of Optimization Algorithms** | **Chen et al.** | **2023** | **arXiv** | **2302.06675** |
| **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** | **Goyal et al.** | **2017** | **arXiv** | **1706.02677** |
| **On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima** | **Keskar et al.** | **2017** | **ICLR** | **1609.04836** |
| **Cyclical Learning Rates for Training Neural Networks** | **Smith** | **2017** | **IEEE WACV** | **1506.01186** |
| Mixed Precision Training | Micikevicius et al. | 2018 | ICLR | 1710.03740 |
| **FP8 Formats for Deep Learning** | **Micikevicius et al.** | **2022** | **arXiv** | **2209.05433** |
| **A Study of BFLOAT16 for Deep Learning Training** | **Kalamkar et al.** | **2019** | **arXiv** | **1905.12322** |
| **Deep Learning with Limited Numerical Precision** | **Gupta et al.** | **2015** | **ICML** | - |
| **Training Deep Neural Networks with 8-bit Floating Point Numbers** | **Wang et al.** | **2018** | **NeurIPS** | **1812.08011** |
| **Hybrid 8-bit Floating Point (HFP8) Training** | **Sun et al.** | **2019** | **NeurIPS** | **1905.12334** |
| **BinaryConnect: Training DNNs with binary weights** | **Courbariaux et al.** | **2015** | **NeurIPS** | **1511.00363** |
| LoRA (Low-Rank Adaptation) | Hu et al. | 2021 | ICLR | 2106.09685 |
| QLoRA (4-bit Quantization) | Dettmers et al. | 2023 | NeurIPS | 2305.14314 |

### 数值稳定性基础

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| **Rounding Errors in Algebraic Processes** | **Wilkinson, J. H.** | **1963** | **Prentice Hall (专著)** | - |
| **Accuracy and Stability of Numerical Algorithms (2nd ed.)** | **Higham, N. J.** | **2002** | **SIAM (专著)** | - |
| **Accurately computing the log-sum-exp and softmax functions** | **Blanchard, P., Higham, D. J., & Higham, N. J.** | **2019** | **arXiv** | **1909.03469** |
| **Transformers without Tears: Improving the Normalization of Self-Attention** | **Nguyen, T. Q., & Salazar, J.** | **2019** | **arXiv** | **1910.05895** |

### 二阶优化方法

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| **Numerical Optimization (教科书)** | **Nocedal & Wright** | **2006** | **Springer** | - |
| **On the limited memory BFGS method for large scale optimization** | **Liu & Nocedal** | **1989** | **Mathematical Programming** | - |
| **Natural Gradient Works Efficiently in Learning** | **Amari** | **1998** | **Neural Computation** | - |
| Revisiting Natural Gradient for Deep Networks | Pascanu & Bengio | 2013 | arXiv | 1301.3584 |
| **Optimizing Neural Networks with Kronecker-factored Approximate Curvature (K-FAC)** | **Martens & Grosse** | **2015** | **ICML** | **1503.05671** |
| A Kronecker-factored approximate Fisher matrix for convolution layers | Grosse & Martens | 2016 | ICML | - |
| Kronecker-Factored Curvature Approximations for Recurrent Neural Networks | Martens, Ba, & Johnson | 2018 | ICLR | - |
| Deep learning via Hessian-free optimization | Martens | 2010 | ICML | - |
| Identifying and attacking the saddle point problem in high-dimensional non-convex optimization | Dauphin et al. | 2014 | NeurIPS | 1406.2572 |
| Empirical Analysis of the Hessian of Over-Parametrized Neural Networks | Sagun et al. | 2017 | ICLR Workshop | 1706.04454 |
| Shampoo: Preconditioned Stochastic Tensor Optimization | Gupta et al. | 2018 | ICML | 1802.09568 |
| Scalable Second Order Optimization for Deep Learning | Anil et al. | 2020 | arXiv | 2002.09018 |
| ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning | Yao et al. | 2020 | AAAI | 2006.00719 |
| A Progressive Batching L-BFGS Method for Machine Learning | Bollapragada et al. | 2018 | ICML | - |
| A Multi-Batch L-BFGS Method for Machine Learning | Berahas et al. | 2016 | NeurIPS | - |

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

### 数据工程与预处理

| 论文 | 作者 | 年份 | 会议/期刊 | arXiv |
|------|------|------|-----------|-------|
| Where Is My Training Bottleneck? Hidden Trade-Offs in Deep Learning Preprocessing Pipelines | Lim et al. | 2022 | arXiv | 2202.08679 |

### Tokenization技术

| 论文/资源 | 作者 | 年份 | 会议/期刊 | arXiv/链接 |
|------|------|------|-----------|-------|
| **Neural Machine Translation of Rare Words with Subword Units (BPE)** | **Sennrich, Haddow, & Birch** | **2016** | **ACL** | **1508.07909** |
| **Japanese and Korean Voice Search (WordPiece)** | **Schuster & Nakajima** | **2012** | **IEEE ICASSP** | - |
| **SentencePiece: A simple and language independent approach** | **Kudo & Richardson** | **2018** | **EMNLP** | **1808.06226** |
| **TikToken (OpenAI Tokenizer)** | **OpenAI** | **2022** | **GitHub** | https://github.com/openai/tiktoken |
| **HuggingFace Tokenizers** | **HuggingFace** | **2020** | **GitHub** | https://github.com/huggingface/tokenizers |

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
| **73** | **序列并行(Sequence Parallelism)** | **Megatron-LM v2 (Narayanan et al., 2021), Korthikanti et al. (2023)** |
| 74 | 上下文并行(Context Parallelism) | Ring Attention (2023) |
| **75** | **序列并行与张量并行组合** | **Megatron-LM v2 (Narayanan et al., 2021), Korthikanti et al. (2023)** |
| 76-80 | MoE | Switch Transformers, DeepSeek-V3 |
| **77** | **MoE路由算法** | **Shazeer et al. (2017), Fedus et al. (2022), Zhou et al. (2022), Soft MoE (2023)** |
| **78** | **MoE负载均衡技术** | **Switch Transformers (Fedus et al., 2022), ST-MoE (Zoph et al., 2022), Global Load Balancing (Qiu et al., 2025)** |
| **79** | **MoE专家并行实现** | **GShard (Lepikhin et al., 2021), Switch Transformers (Fedus et al., 2022), DeepSpeed-MoE (Rajbhandari et al., 2022)** |
| **80** | **共享专家与稀疏专家** | **DeepSeek-V2 (2024), DeepSeek-V3 (2024), Switch Transformers (Fedus et al., 2022)** |
| **81** | **随机梯度下降(SGD)与动量** | **Robbins & Monro (1951), Polyak (1964), Sutskever et al. (2013), Goyal et al. (2017), Keskar et al. (2017), Smith (2017)** |
| **82** | **Nesterov加速梯度(NAG)** | **Nesterov (1983), Sutskever et al. (2013), Ruder (2016), Su et al. (2014)** |
| **83** | **自适应学习率：AdaGrad/RMSProp** | **Duchi et al. (2011), Tieleman & Hinton (2012), Zeiler (2012), Ruder (2016)** |
| **84** | **Adam优化器详解** | **Kingma & Ba (2015), Reddi et al. (2018)** |
| **87** | **二阶优化方法概览** | **Newton法, L-BFGS (Liu & Nocedal, 1989), 自然梯度 (Amari, 1998), K-FAC (Martens & Grosse, 2015)** |
| **88** | **分布式优化器详解** | **ZeRO (Rajbhandari et al., 2020), FSDP (Zhao et al., 2023), Megatron-LM v2 (2021)** |
| 93-96 | 混合精度 | Mixed Precision Training (2018) |
| **95** | **FP8训练与TransformerEngine** | **FP8 Formats for Deep Learning (Micikevicius et al., 2022)** |
| **96** | **数值稳定性实践** | **Wilkinson (1963), Higham (2002), Blanchard et al. (2019), Nguyen & Salazar (2019)** |
| **97** | **数据预处理与Tokenization** | **Sennrich et al. (2016), Schuster & Nakajima (2012), Kudo & Richardson (2018)** |
| **98** | **数据加载与索引化** | **Lim et al. (2022), NumPy memmap 文档** |
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
| Oxford Protein Informatics - mmap | https://www.blopig.com/blog/2019/08/mmap-vs-zarr-vs-hdf5/ | mmap vs Zarr vs HDF5性能对比 |
| Python Speed - mmap tutorial | https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/ | mmap使用教程 |

---

**注意**: 所有论文引用在添加到知识库文档前，必须通过 MCP WebSearch 工具验证准确性。
