from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ModelArgs:

    # Model Architecture
    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    kv_channels: int = 128
    seq_length: int = 4096
    max_position_embeddings: int = 40960
    
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