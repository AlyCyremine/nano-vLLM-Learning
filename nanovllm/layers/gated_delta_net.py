"""PyTorch Gated DeltaNet kernels: FP32 state, chunked prefill and recurrent decode."""

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def gated_delta_rule(query, key, value, decay, beta, state, chunk_size=64):
    """Evaluate the delta recurrence on one sequence (length, heads, head_dim)."""
    output_dtype = query.dtype
    q, k, v = [x.transpose(0, 1).float() for x in (query, key, value)]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.size(-1) ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g, b = decay.transpose(0, 1).float(), beta.transpose(0, 1).float()
    length = q.size(1)
    if length == 1:
        state = state * g[:, 0].exp()[:, None, None]
        delta = (v[:, 0] - (state * k[:, 0, :, None]).sum(-2)) * b[:, 0, None]
        state = state + k[:, 0, :, None] * delta[:, None, :]
        output = (state * q[:, 0, :, None]).sum(-2).unsqueeze(1)
    else:
        padding = (-length) % chunk_size
        q, k, v = [F.pad(x, (0, 0, 0, padding)) for x in (q, k, v)]
        g, b = [F.pad(x, (0, padding)) for x in (g, b)]
        q, k, v = [x.reshape(x.size(0), -1, chunk_size, x.size(-1)) for x in (q, k, v)]
        g, b = [x.reshape(x.size(0), -1, chunk_size) for x in (g, b)]
        cumulative_decay = g.cumsum(-1)
        upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
        pair_decay = (cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)).masked_fill(upper, -torch.inf).exp()
        k_beta, v_beta = k * b.unsqueeze(-1), v * b.unsqueeze(-1)
        system = (k_beta @ k.transpose(-1, -2)) * pair_decay
        values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
        state_keys = torch.linalg.solve_triangular(
            system, k_beta * cumulative_decay.exp().unsqueeze(-1), upper=False, unitriangular=True
        )
        attention = (q @ k.transpose(-1, -2)) * pair_decay
        chunks = []
        for i in range(q.size(1)):
            updates = values[:, i] - state_keys[:, i] @ state
            chunks.append((q[:, i] @ state) * cumulative_decay[:, i].exp().unsqueeze(-1) + attention[:, i] @ updates)
            end_decay = cumulative_decay[:, i, -1]
            end_keys = k[:, i] * (end_decay[:, None] - cumulative_decay[:, i]).exp().unsqueeze(-1)
            state = state * end_decay.exp()[:, None, None] + end_keys.transpose(-1, -2) @ updates
        output = torch.cat(chunks, dim=1)[:, :length]
    return output.transpose(0, 1).to(output_dtype), state


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        normalized = self.weight * normalized.to(x.dtype)
        return (normalized * F.silu(gate.float())).to(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config, state_capacity):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.kernel_size = config.linear_conv_kernel_dim
        channels = self.key_dim * 2 + self.value_dim
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("GDN value heads must be divisible by key heads")
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 GDN requires silu")
        self.in_proj_qkv = nn.Linear(config.hidden_size, channels, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(channels, channels, self.kernel_size, groups=channels, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = GatedRMSNorm(self.value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.register_buffer("conv_states", torch.zeros(state_capacity, channels, self.kernel_size), persistent=False)
        self.register_buffer("recurrent_states", torch.zeros(
            state_capacity, self.num_v_heads, self.key_head_dim, self.value_head_dim, dtype=torch.float32
        ), persistent=False)

    def _apply(self, fn, recurse=True):
        state = self.recurrent_states
        super()._apply(fn, recurse)
        self.recurrent_states = state.to(device=self.conv_states.device, dtype=torch.float32)
        return self

    def reset_state(self, slot):
        self.conv_states[slot].zero_()
        self.recurrent_states[slot].zero_()

    def forward(self, hidden_states):
        context = get_context()
        projected = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(-1, self.num_v_heads, self.value_head_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias.float())
        outputs = []
        for (start, end), slot in zip(context.request_ranges, context.state_indices):
            raw = projected[start:end].transpose(0, 1)
            history = torch.cat((self.conv_states[slot], raw), dim=-1)
            self.conv_states[slot].copy_(history[:, -self.kernel_size:])
            mixed = F.silu(self.conv1d(history.unsqueeze(0))[0, :, -(end - start):]).transpose(0, 1)
            q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
            q = q.reshape(-1, self.num_k_heads, self.key_head_dim)
            k = k.reshape_as(q)
            v = v.reshape(-1, self.num_v_heads, self.value_head_dim)
            repeats = self.num_v_heads // self.num_k_heads
            if repeats > 1:
                q, k = [x.repeat_interleave(repeats, dim=1) for x in (q, k)]
            output, state = gated_delta_rule(q, k, v, decay[start:end], beta[start:end], self.recurrent_states[slot])
            self.recurrent_states[slot].copy_(state)
            outputs.append(output)
        output = self.norm(torch.cat(outputs), z).reshape(-1, self.value_dim)
        return self.out_proj(output)
