# Qwen3.5-0.8B 纯文本推理：运行与验证

更新日期：2026-10-03。

Qwen3.5-0.8B 的纯文本主干已接入 nano-vLLM，并在 WSL2 的 RTX 5060 Laptop GPU 上使用官方 BF16 权重完成生成及参考对照。MTP 仍为后续计划。

## 1. 当前实现

- 读取多模态 checkpoint 的 `text_config`，选择新的 Qwen3.5 文本实现。
- 严格加载文本参数，映射 `model.language_model.*`，共享 token embedding 与 LM head。
- 根据 `layer_types` 构建 18 层 Gated DeltaNet 和 6 层完整注意力。
- 完整注意力使用现有 FlashAttention/paged KV；加入输出门控、zero-centered RMSNorm 和 partial RoPE。
- GDN 使用 PyTorch 的分块 delta rule 做 prefill，逐步 recurrence 做 decode；卷积历史和 FP32 recurrent state 由每条请求独立持有。
- 支持变长批处理、chunked prefill、请求抢占后的重算、状态 slot 复用、EOS 和长度检查。
- `temperature=0` 选择贪心解码，正温度保留原有随机采样。

默认使用单卡 eager 路径。Qwen3.5 的 CUDA Graph、prefix caching、TP>1 尚未适配；配置会自动使用 eager、关闭 prefix caching，并拒绝显式要求 prefix caching 或多卡 TP 的组合。GDN 状态容量默认对应 `max_num_seqs=8`，避免沿用普通 Transformer 的 512 请求默认值。

本次加载了 320 个文本参数，明确跳过 153 个视觉参数和 15 个 MTP 参数。缺失、形状不符或无法识别的文本权重会报错。

主要代码：

| 文件 | 职责 |
| --- | --- |
| `nanovllm/models/qwen3_5.py` | 语言模型、混合 decoder、注意力、归一化与权重命名映射 |
| `nanovllm/layers/gated_delta_net.py` | 因果卷积、分块 delta rule、循环状态与 gated norm |
| `nanovllm/engine/state_manager.py` | 稳定请求 ID 与状态 slot 的映射 |
| `nanovllm/models/registry.py` | Qwen3 / Qwen3.5 模型选择 |
| `example_qwen3_5.py` | 仿照 `example.py` 的双 prompt 纯文本示例 |
| `scripts/verify_qwen3_5.py` | 真实 checkpoint 的参考对照和生命周期验证 |

## 2. 已验证环境

| 项目 | 值 |
| --- | --- |
| 系统 / 用户 | WSL2 Ubuntu-24.04 / `aly-cyremine` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，约 8 GB |
| Python | WSL Conda `nano-vLLM` 环境中的 Python 3.11 |
| PyTorch / CUDA runtime | `2.11.0+cu128` / CUDA 12.8 |
| Transformers | `5.16.0` |
| FlashAttention / Triton | `2.8.3.post1` / `3.6.0` |
| checkpoint | `Qwen/Qwen3.5-0.8B`，BF16 |
| snapshot revision | `2fc06364715b967f1860aea9cf38778875588b17` |

模型已下载到当前工作区 `.cache/Qwen3.5-0.8B/`，该目录被 Git 忽略。WSL 内的路径为：

```text
/mnt/c/ElyAkasia/Infra/nano-vLLM-Learning/.cache/Qwen3.5-0.8B
```

这里运行的是 Windows 工作区映射到 WSL 的代码；`/home/aly-cyremine/infra/nano-vLLM-Learning` 是另一个 checkout，使用时需确认代码版本。

## 3. 运行文本生成

在 WSL 终端中运行：

```bash
cd /mnt/c/ElyAkasia/Infra/nano-vLLM-Learning
~/miniconda3/envs/nano-vLLM/bin/python example_qwen3_5.py
```

示例与原有 `example.py` 一样，在 `main()` 中设置模型路径、采样参数和两条 prompt，然后调用 `LLM.generate` 并打印 Prompt / Completion。默认模型路径为脚本所在目录下的 `.cache/Qwen3.5-0.8B`，无需命令行参数；模型在其他位置时修改 `path`。

两条输入分别为自我介绍、列出 100 以内的素数。使用非思考模式的 chat template，`temperature=0.6`、`max_tokens=256`；若需要确定性的贪心输出，改为 `temperature=0`。当前示例设定 `max_model_len=1024`、`max_num_seqs=2`、`gpu_memory_utilization=0.6`，并在结束或生成异常时调用 `llm.exit()` 释放运行资源。prompt + 输出预算需不超过上下文限制，实际消耗 KV 的 token 数也需能放入物理 KV cache。

也可直接使用 API：

```python
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

path = ".cache/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(path)
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请只回答一个数字：1+1等于多少？"}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
llm = LLM(path, max_num_seqs=2, max_model_len=512, gpu_memory_utilization=0.6)
try:
    outputs = llm.generate([prompt_ids], SamplingParams(temperature=0, max_tokens=64))
    print(tokenizer.decode(outputs[0]["token_ids"], skip_special_tokens=True))
finally:
    llm.exit()
```

Transformers 5.16 的 `apply_chat_template` 在 `tokenize=True` 时可能返回编码字典；这里显式设置 `return_dict=False`，保证传给 `LLM.generate` 的是 token ID 列表。

## 4. 复现验证

基础测试使用小型随机初始化的 Qwen3.5 配置，与 Transformers 对照 FP32 logits，并测试 delta recurrence、分块、状态重排、权重加载、精度转换和调度容量：

```bash
~/miniconda3/envs/nano-vLLM/bin/python -m unittest discover -s tests -v
```

真实权重对照：

```bash
~/miniconda3/envs/nano-vLLM/bin/python -m scripts.verify_qwen3_5 \
  .cache/Qwen3.5-0.8B
```

实测结果：

- 15 项模型、状态管理及失败边界测试与 1 项真实 Qwen3 回归测试通过，共 16 项。未设置 Qwen3 模型路径时，最后一项会跳过。
- BF16 真实权重的 8 组 greedy 输出与 Transformers 逐 token 一致。
- 输入长度为 20、28、24、422 token，常规输出预算为 32 token；EOS 用例的预算为 128，实际生成 2 token 后结束。
- `max_num_seqs=2`、`max_num_batched_tokens=128`，验证了请求排队、跨 KV block 的分块 prefill、重复调用、slot 复用及抢占重算。
- 只为 6 层完整注意力分配 paged KV；结束后所有请求的 GDN slot 均已释放。
- 该运行条件下，PyTorch 峰值已分配显存约 3.62 GiB。
- 直接运行新版 `example_qwen3_5.py` 成功：第一条生成自我介绍，第二条完整列出 2 至 97 的 25 个素数，进程正常退出。示例采用随机采样，具体措辞可随运行变化。

验证脚本写出 `.cache/qwen3_5_validation.json`。这些数据用于正确性验收；吞吐及延迟基准需要单独固定预热、输入集合和计时方法。

原有 Qwen3 回归可使用本机已有 checkpoint：

```bash
NANOVLLM_QWEN3_MODEL=/home/aly-cyremine/huggingface/Qwen3-0.6B \
  ~/miniconda3/envs/nano-vLLM/bin/python -m unittest discover -s tests -v
```

Qwen3 回归对照的是修改前 `516b9dd` 的贪心 token 记录，并检查 prefix cache 重复运行的一致性。该原有路径在 BF16 下与 Transformers 的后续 token 已存在差异；使用原始代码重跑确认，本次适配保持了原有输出。

## 5. 代码复审修复

本次复审覆盖 `516b9dd..ea2089a` 的所有新增及修改代码，补充的失败路径测试先复现问题，再验证修复：

- 模型创建、权重加载或预热失败时，恢复调用前的 PyTorch 默认 dtype/device，重置推理 context 并销毁已创建的进程组。
- 请求入队前检查实际 KV cache 容量，避免单条请求也放不下时在调度中触发断言；最后一个输出 token 无需写入 KV，容量边界按此计算。
- `generate` 先校验完整批次再入队，避免后续 prompt 非法时遗留前面已入队的请求；采样参数列表长度不匹配会明确报错。
- 权重完整性检查按共享 Parameter 身份识别 embedding / LM head，兼容 checkpoint 只保留任意一方名称的情况。

## 6. 后续工作

先根据需要扩大输入集合及上下文验证，再评估 FLA/Triton GDN kernel、CUDA Graph 和混合 prefix caching。MTP 的预测模块、主模型多 token 验证与回滚尚未实现，继续按[实施计划](qwen3_5_0_8b_mtp_plan.md)管理。
