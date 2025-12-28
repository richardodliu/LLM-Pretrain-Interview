import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelConfig:

    # Model Architecture
    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    kv_channels: int = 128
    seq_length: int = 4096
    max_position_embeddings: int = 4096
    
    # Attention & Normalization
    group_query_attention: bool = True
    num_query_groups: int = 8
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    qk_layernorm: bool = True
    use_rotary_position_embeddings: bool = True
    rotary_base: float = 1000000.0
    
    # FFN
    swiglu: bool = True
    disable_bias_linear: bool = True
    
    # Mixture of Experts
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_topk: int = 4
    num_experts: int = 64
    moe_ffn_hidden_size: int = 768
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = True
    moe_router_topk_scaling_factor: float = 2.0
    moe_router_enable_expert_bias: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_router_bias_update_rate: float = 0.001
    
    # Layer frequency
    moe_layer_freq: list = str([0]*1 + [1]*(num_layers - 1))
    
    # Shared Expert Params
    moe_shared_expert_intermediate_size: int = 768
    moe_shared_expert_overlap: bool = True
    moe_shared_expert_gate: bool = True

    # --- Initialization & Optimization ---
    untie_embeddings_and_output_weights: bool = True
    init_method_std: float = 0.02
    split_qkv_init_mode: str = "head"


class SwiGLUBlock(nn.Module):
    def __init__(self, config: ModelConfig):

        super().__init__()
        # 1. 门控投影 (Gate Projection): 负责决定哪些信息通过
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_dim, bias=False)
        
        # 2. 上行投影 (Up Projection): 负责提取特征信息
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_dim, bias=False)
        
        # 3. 下行投影 (Down Projection): 将维度映射回原始大小
        self.down_proj = nn.Linear(config.hidden_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        # 步骤 A: 计算门控分支
        # gate_output 形状: (batch, seq_len, hidden_dim)
        gate_output = self.gate_proj(x)
        
        # 步骤 B: 对门控分支应用 SiLU 激活函数 (也叫 Swish)
        # swish(x) = x * sigmoid(x)
        activated_gate = F.silu(gate_output)
        
        # 步骤 C: 计算特征分支 (Up Projection)
        # up_output 形状: (batch, seq_len, hidden_dim)
        up_output = self.up_proj(x)
        
        # 步骤 D: 逐元素相乘 (Element-wise Multiplication)
        # 这是 SwiGLU 的核心：用激活后的门控信号去“乘”特征信号
        intermediate = activated_gate * up_output
        
        # 步骤 E: 最终线性投影
        # 将 hidden_dim 映射回 args.hidden_size
        output = self.down_proj(intermediate)
        
        return output