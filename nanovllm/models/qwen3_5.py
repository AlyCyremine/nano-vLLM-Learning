"""Dense Qwen3.5 text inference. Vision and MTP weights are intentionally excluded."""

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.engine.state_manager import StateManager
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.utils.context import get_context


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (output * (1 + self.weight.float())).to(x.dtype)


class TextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        parameters = config.rope_parameters
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("Qwen3.5 currently supports default text RoPE only")
        self.rotary_dim = int(config.head_dim * parameters.get("partial_rotary_factor", 1.0))
        if not 0 < self.rotary_dim <= config.head_dim or self.rotary_dim % 2:
            raise ValueError("Invalid partial rotary dimension")
        inv_freq = 1 / parameters["rope_theta"] ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        frequencies = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = frequencies.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def forward(self, positions, query, key):
        # Text positions are identical on the three MRoPE axes.
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.float()
        cos = frequencies.cos().to(query.dtype).float().unsqueeze(1)
        sin = frequencies.sin().to(query.dtype).float().unsqueeze(1)
        def rotate(x):
            rotary, tail = x[..., :self.rotary_dim].float(), x[..., self.rotary_dim:]
            first, second = rotary.chunk(2, dim=-1)
            return torch.cat((torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype), tail), dim=-1)
        return rotate(query), rotate(key)


class _TorchAttention(nn.Module):
    """Unpaged attention for small reference tests, without FlashAttention kernels."""

    def __init__(self, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.repeats = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.states = {}

    def reset_state(self, slot):
        self.states.pop(slot, None)

    def forward(self, q, k, v):
        context = get_context()
        outputs = []
        for (start, end), slot in zip(context.request_ranges, context.state_indices):
            previous_k, previous_v = self.states.get(slot, (k[:0], v[:0]))
            keys = torch.cat((previous_k, k[start:end]))
            values = torch.cat((previous_v, v[start:end]))
            self.states[slot] = (keys, values)
            query = q[start:end].transpose(0, 1)
            keys = keys.repeat_interleave(self.repeats, dim=1).transpose(0, 1)
            values = values.repeat_interleave(self.repeats, dim=1).transpose(0, 1)
            query_positions = torch.arange(end - start, device=q.device) + previous_k.size(0)
            key_positions = torch.arange(keys.size(1), device=q.device)
            scores = query @ keys.transpose(-1, -2) * self.scale
            scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], -torch.inf)
            probabilities = scores.float().softmax(-1).to(q.dtype)
            outputs.append((probabilities @ values).transpose(0, 1))
        return torch.cat(outputs)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, attention_backend):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.output_gate = getattr(config, "attn_output_gate", True)
        q_size = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config)
        if attention_backend == "flash_attn":
            from nanovllm.layers.attention import Attention
            self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)
        elif attention_backend == "torch":
            self.attn = _TorchAttention(self.num_heads, self.num_kv_heads, self.head_dim)
        else:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")

    def forward(self, positions, hidden_states):
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim * (2 if self.output_gate else 1))
        gate = None
        if self.output_gate:
            q, gate = q.chunk(2, dim=-1)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, self.q_norm(q), self.k_norm(k))
        output = self.attn(q, k, v).reshape(-1, self.num_heads * self.head_dim)
        if gate is not None:
            output = output * gate.reshape_as(output).sigmoid()
        return self.o_proj(output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 text MLP requires silu")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_type, state_capacity, attention_backend):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, state_capacity)
        elif layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, attention_backend)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3_5MLP(config)

    def forward(self, positions, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config, state_capacity, attention_backend):
        super().__init__()
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types must describe every Qwen3.5 text layer")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, kind, state_capacity, attention_backend) for kind in config.layer_types
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config, state_capacity=8, attention_backend="flash_attn"):
        super().__init__()
        self.model = Qwen3_5TextModel(config, state_capacity, attention_backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.state_manager = StateManager(state_capacity)

    @staticmethod
    def map_weight_name(name):
        if name.startswith(("model.visual.", "mtp.")):
            return None
        return name.replace("model.language_model.", "model.", 1)

    def prepare_states(self, sequences):
        slots = []
        for seq_id, cached_tokens in sequences:
            slot = self.state_manager.allocate(seq_id, cached_tokens)
            if cached_tokens == 0:
                for layer in self.model.layers:
                    module = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn.attn
                    if hasattr(module, "reset_state"):
                        module.reset_state(slot)
            slots.append(slot)
        return slots

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        context = get_context()
        if context.is_prefill:
            hidden_states = hidden_states[context.cu_seqlens_q[1:] - 1]
        return self.lm_head(hidden_states)
