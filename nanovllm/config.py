import os
import json
from dataclasses import dataclass
import torch
from transformers import AutoConfig

# dataclass自动生成__init__、__repr__、__eq__等方法，并且可以使用slots来减少内存占用（只能拥有声明好的这些字段）
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 一次 batch 最多调度多少 token
    max_num_seqs: int | None = None # Qwen3 默认 512；混合模型默认 8
    max_model_len: int = 4096 # 模型最大长度（prompt tokens + completion tokens）
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1 # 默认不进行多 GPU Tensor Parallel
    enforce_eager: bool = False # eager 指 PyTorch 的 eager execution（即时执行）模式，项目可能允许使用 CUDA Graph 等非纯 eager 的优化路径
    hf_config: AutoConfig | None = None # 自动获取模型配置
    hf_text_config: AutoConfig | None = None
    enable_prefix_caching: bool | None = None
    eos_token_ids: tuple[int, ...] = ()
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model) # 检查模型路径是否存在
        assert self.kvcache_block_size % 256 == 0 # 检查 KV Cache block size 是否是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 检查 Tensor Parallel size 是否在有效范围内
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_text_config = getattr(self.hf_config, "text_config", self.hf_config)
        model_type = self.hf_text_config.model_type
        if model_type not in ("qwen3", "qwen3_5_text"):
            raise ValueError(f"Unsupported model type: {model_type}")
        if self.is_hybrid:
            if self.tensor_parallel_size != 1:
                raise ValueError("Qwen3.5 text inference currently supports tensor_parallel_size=1")
            if self.enable_prefix_caching:
                raise ValueError("Qwen3.5 prefix caching requires GDN state snapshots and is not supported yet")
            self.enforce_eager = True
            self.enable_prefix_caching = False
        elif self.enable_prefix_caching is None:
            self.enable_prefix_caching = True
        if self.max_num_seqs is None:
            self.max_num_seqs = 8 if self.is_hybrid else 512
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1 or self.max_model_len < 1:
            raise ValueError("Sequence, token and context limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.max_model_len = min(self.max_model_len, self.hf_text_config.max_position_embeddings)
        eos = getattr(self.hf_text_config, "eos_token_id", None)
        generation_config = os.path.join(self.model, "generation_config.json")
        if os.path.isfile(generation_config):
            with open(generation_config, encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id", eos)
        self.eos_token_ids = tuple(eos if isinstance(eos, list) else [eos]) if eos is not None else ()

    @property
    def is_hybrid(self):
        return self.hf_text_config.model_type == "qwen3_5_text"

    @property
    def model_dtype(self):
        dtype = getattr(self.hf_text_config, "dtype", None) or torch.bfloat16
        return getattr(torch, dtype) if isinstance(dtype, str) else dtype
