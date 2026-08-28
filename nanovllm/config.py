import os
from dataclasses import dataclass
from transformers import AutoConfig

# dataclass自动生成__init__、__repr__、__eq__等方法，并且可以使用slots来减少内存占用（只能拥有声明好的这些字段）
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 一次 batch 最多调度多少 token
    max_num_seqs: int = 512 # 一次 batch 最多调度多少 sequence
    max_model_len: int = 4096 # 模型最大长度（prompt tokens + completion tokens）
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1 # 默认不进行多 GPU Tensor Parallel
    enforce_eager: bool = False # eager 指 PyTorch 的 eager execution（即时执行）模式，项目可能允许使用 CUDA Graph 等非纯 eager 的优化路径
    hf_config: AutoConfig | None = None # 自动获取模型配置
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model) # 检查模型路径是否存在
        assert self.kvcache_block_size % 256 == 0 # 检查 KV Cache block size 是否是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 检查 Tensor Parallel size 是否在有效范围内
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
