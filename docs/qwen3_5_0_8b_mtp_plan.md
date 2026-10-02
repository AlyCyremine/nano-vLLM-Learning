# Qwen3.5-0.8B 纯文本推理适配与 MTP 实施计划

创建日期：2026-10-02  
开发分支：`mtp`  
仓库分析基线：`516b9dd`  
状态：阶段 A 的基础实现已完成，Qwen3.5-0.8B 已在 WSL2 GPU 上完成纯文本生成、数值测试与真实权重对照。阶段 B（MTP）尚未开始。详见[运行与验证记录](qwen3_5_text_inference.md)。

## 1. 项目目标与执行顺序

先让 nano-vLLM 正确运行 `Qwen/Qwen3.5-0.8B` 的纯文本自回归推理，建立可复现的正确性和性能基线，再使用模型自带的 MTP 权重实现投机解码。

这里的 MTP 指 Multi-Token Prediction：轻量预测模块提出后续 token，由主模型验证后提交。项目使用已训练的 MTP 权重，工作重点是推理引擎适配。

实施分成两个阶段，每个阶段按里程碑推进：

| 阶段 | 里程碑 | 交付结果 |
| --- | --- | --- |
| A：纯文本推理 | A0：模型与参考环境核验 | 固定模型及依赖版本、权重清单、参考输出 |
| A：纯文本推理 | A1：单请求正确性 | 文本模型、混合状态管理、prefill/decode 数值对齐 |
| A：纯文本推理 | A2：引擎集成 | 连续批处理、分块 prefill、生命周期及回归验证 |
| B：MTP | B0：预测模块 | MTP 权重加载、隐藏状态与 token 的位置对齐 |
| B：MTP | B1：单请求验证闭环 | 单草稿 token、贪心验证、完整状态回滚 |
| B：MTP | B2：扩展与评测 | 多草稿 token、批处理、采样验证与性能报告 |

进入阶段 B 的前提是阶段 A 的正确性验收通过。性能优化以通过验收的实现为基线，逐项加入。

## 2. 范围与首版运行条件

首版面向稠密版 Qwen3.5-0.8B，只接受文本 prompt 或 token ID，提供离线 `LLM.generate` 推理。加载语言模型主干与共享输出头，后续按开关加载 MTP 模块。

首版执行范围：

- NVIDIA CUDA 单卡，`tensor_parallel_size=1`。
- 权重优先使用 BF16，循环状态按照参考实现保留 FP32；其他精度单独验证。
- A1 从单请求开始，A2 再扩展变长批处理。
- 初始上下文上限使用 4096；显存不足时降低，不直接采用模型标称的最大上下文。
- 使用 `enforce_eager=True` 关闭 CUDA Graph。现有层中的 `torch.compile` 另行控制，不能把该参数理解成关闭所有编译。
- Qwen3.5 首版关闭 prefix caching，待混合状态复用方案通过验证后再启用。

图像/视频输入、视觉编码器、MoE、训练、量化、多卡 TP 和 CUDA Graph 优化列为后续扩展。原有 Qwen3 推理路径保留并进行回归验证。

当前代码使用 NCCL、Triton 和 FlashAttention，执行验证优先安排在 Linux CUDA 环境，可采用满足依赖条件的 WSL2 或 GPU 服务器。Windows 工作目录用于开发；具体运行硬件及依赖组合在 A0 中记录。

## 3. 已核实的模型事实

官方 checkpoint 的顶层是多模态配置，文本参数位于 `text_config`，结构类型分别为 `qwen3_5` 和 `qwen3_5_text`。开发时应同时保留完整配置与文本配置，按文本配置构建主干。[官方配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)

| 文本配置项 | 官方值 |
| --- | --- |
| 层数 / hidden size / MLP intermediate size | 24 / 1024 / 3584 |
| 层类型 | 每组 3 层 `linear_attention` + 1 层 `full_attention` |
| 完整注意力 Q heads / KV heads / head dim | 8 / 2 / 256 |
| 注意力输出门控 | `attn_output_gate=true` |
| 线性注意力 key heads / value heads | 16 / 16 |
| 线性注意力 key dim / value dim | 128 / 128 |
| 因果卷积 kernel size | 4 |
| 词表大小 / embedding 与输出头共享 | 248320 / `true` |
| RoPE theta / partial rotary factor | 10000000 / 0.25 |
| MRoPE section / interleaved | `[11, 11, 10]` / `true` |
| MTP 层数 / 专用 embedding | 1 / `false` |
| 主干权重 dtype / 循环状态 dtype | BF16 / FP32 |

由层序列可得：主干有 18 层 Gated DeltaNet 和 6 层完整注意力；完整注意力只旋转每个 head 的 64 个维度。层类型仍以实际 `layer_types` 为准，避免硬编码比例。

官方权重索引已列出语言模型、视觉模型和 MTP 参数，包括 `model.language_model.*`、`model.visual.*`、`mtp.fc.weight`、`mtp.layers.0.*` 及 MTP 的归一化参数。因此阶段 B 具备使用原生 MTP 权重的条件；A0 仍需检查实际下载文件的键名、形状和覆盖率。[官方权重索引](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/model.safetensors.index.json)

Qwen3.5 完整注意力、归一化和 Gated DeltaNet 的语义以 Transformers 官方实现为数值参考；原生 MTP 结构及权重映射参考 vLLM。[Transformers 模型源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[vLLM MTP 源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)

上述资料核查于 2026-10-02。配置文件页面对应提交 `2b48083dfa97c3cbf0220cf1a5e6cffe0a511157`；该值是配置文件的版本记录。A0 需另外固定整个模型 snapshot 的 revision，并记录 Transformers/vLLM 的 tag 或完整 commit，避免后续依赖浮动的 `main`。[配置提交记录](https://huggingface.co/Qwen/Qwen3.5-0.8B/commit/2b48083dfa97c3cbf0220cf1a5e6cffe0a511157)

## 4. 基线仓库与目标之间的差距（516b9dd）

| 当前文件 | 现有行为 | 计划改动 |
| --- | --- | --- |
| `nanovllm/config.py` | 从顶层直接读取最大位置长度 | 提取文本配置，校验模型类型、dtype、层类型和上下文限制 |
| `nanovllm/engine/model_runner.py` | 固定创建 `Qwen3ForCausalLM`；按全部层分配 KV | 增加模型选择，按缓存类型分配，支持多 token 验证 |
| `nanovllm/models/qwen3.py` | 所有层使用完整注意力 | 新增 Qwen3.5 文本模型与混合 decoder |
| `nanovllm/utils/loader.py` | 按 checkpoint 键名查找参数 | 模型专属映射、明确跳过列表、加载覆盖率检查 |
| `nanovllm/layers/rotary_embedding.py` | 要求 rotary dim 等于 head size | 支持 partial RoPE，校验纯文本位置与 MRoPE 等价性 |
| `nanovllm/layers/layernorm.py` | 普通 RMSNorm，直接乘 weight | 新增 Qwen3.5 zero-centered RMSNorm 及 GDN gated norm |
| `nanovllm/layers/embed_head.py` | prefill 只返回每条请求最后位置的 logits | 验证模式可返回全部待验证位置的 logits |
| `nanovllm/utils/context.py` | 主要传递 KV 与注意力元数据 | 加入序列状态索引、执行模式和验证长度等信息 |
| `nanovllm/engine/sequence.py` | 每轮单 token，decode 序列化只携带最后 token | 区分正式输出、草稿、已处理长度；扩展多 token 元数据 |
| `nanovllm/engine/scheduler.py` | 单 token decode；prefix cache 默认参与分配 | 增加模型能力开关、状态生命周期、验证提交逻辑 |
| `nanovllm/engine/block_manager.py` | KV block 分配、复用和 hash | 支持验证所需空间预留、尾部回收与共享块写入保护 |
| `nanovllm/sampling_params.py`<br />`nanovllm/layers/sampler.py` | 禁止贪心采样，使用温度随机采样 | 新增明确的 argmax 路径，再支持投机采样校正 |
| `nanovllm/engine/llm_engine.py` | decode token 统计等于请求数 | 分别记录处理、草稿、接受和正式提交 token 数 |
| `pyproject.toml` | `transformers>=4.51.0` | 验证并固定支持 Qwen3.5 的依赖组合 |

基线中的 `test.py` 是手动生成脚本，当时没有主干数值对齐、缓存回滚或 MTP 正确性测试。当前新增的文本模型测试与真实权重验证见[运行记录](qwen3_5_text_inference.md)；仅查看输出是否通顺不能作为适配成功的依据。

## 5. 阶段 A：纯文本推理

### A0：建立参考与权重清单

1. 固定模型 snapshot、tokenizer/chat template、参考库版本以及 CUDA/PyTorch 环境。
2. 检查 `text_config`、safetensors 键名与形状，生成文本主干和 MTP 的参数清单。
3. 用 Transformers 跑纯文本参考 forward 与生成；固定输入 token ID、position、dtype、attention backend 和生成参数。
4. 保存 prefill 最后位置的 logits、指定层 hidden states、固定续写 token 下的逐步 decode logits，以及贪心输出。
5. 加入明确的 greedy 模式。设计为 `temperature=0` 时直接 argmax，禁止先做除以零的温度变换。
6. 检查实际 tokenizer 与可用的生成配置，统一 EOS/结束标记集合及 chat template；测试时两端使用完全相同的模板及思考模式。

当前依赖下限和 checkpoint 中的 `transformers_version` 字段都不能证明本机支持 Qwen3.5。A0 的判断条件是配置类、模型类可用且官方参考实际运行成功；确认后记录精确版本。

交付：权重清单、参考数据生成脚本、环境记录和小型固定输入集合，文档放在 `docs/`。

### A1：新增文本模型并对齐单请求

#### 配置与加载

- 新增模型工厂或小型注册表，分别选择原有 Qwen3 与新增 `Qwen3_5ForCausalLM`。
- 同时保留顶层 HF 配置和有效文本配置；runner 中最大位置、dtype、head 数等统一读取有效文本配置。
- 将 `model.language_model.*` 映射到本地文本主干。阶段 A 明确跳过 `model.visual.*` 和 `mtp.*`，并统计跳过项。
- 检查加载后全部必需参数；除明确的共享权重别名外，缺失或未知文本参数必须报错。
- 确认 embedding 与 LM head 的共享关系，避免依赖 checkpoint 一定包含独立 `lm_head.weight`。
- 初版保留易核对的独立 projection；完成数值对齐后再考虑 packed projection 与融合算子。

#### 完整注意力层

实现输出门控、Q/K 归一化、partial RoPE 和完整注意力计算。Q projection 同时包含 query 与 gate，拆分需遵守参考实现的逐 head 排布，不能简单假设整个张量前半是 Q、后半是 gate。

Qwen3.5 的主干 RMSNorm 使用 `1 + weight`；GDN 内的 gated norm 有自己的归一化与门控语义，分别实现，避免统一修改现有 Qwen3 RMSNorm。纯文本路径先用相同文本位置构造各 MRoPE 分量，再与参考结果对齐后采用简化位置计算。[归一化与注意力参考](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

#### Gated DeltaNet 层

实现 projection、因果 depthwise conv、Q/K 归一化、衰减与更新门、delta recurrence、gated norm 及输出 projection。先写可逐 token 检查的 PyTorch 参考路径，再评估引入经过验证的 FLA/因果卷积 kernel。

每条序列、每个线性注意力层独立维护卷积历史和 recurrent state。prefill、分块 prefill 与单 token decode 必须延续同一状态；拼接 batch 中的序列边界必须重置或切换状态，避免不同请求相互污染。

#### 混合缓存与显存预算

设计分开管理两种缓存：

| 缓存 | 管理方式 | 必须保证的行为 |
| --- | --- | --- |
| 6 层完整注意力的 KV | 复用 paged KV block | 层索引准确、长度一致、分配及释放可回溯 |
| 18 层 GDN 的卷积/循环状态 | 按 sequence 分配 state slot | 请求隔离、可快照/恢复、结束及抢占时清理 |

模型层编号与完整注意力缓存编号显式映射。显存预算扣除权重、workspace、线性状态及预留空间后，再决定 KV block 数量；MTP 阶段还需计入草稿 KV 和回滚临时状态。

按当前配置计算，仅 FP32 recurrent state 就需要：

```text
18 layers × 16 heads × 128 × 128 × 4 bytes = 18 MiB / active sequence
```

这还不包括卷积状态、KV、激活和快照。沿用默认 `max_num_seqs=512` 会仅为循环状态预留约 9 GiB，因此 A1 使用 1，A2 从小批量增加，并按实际显存限制活跃请求数。该估算是本方案根据官方配置推导的容量预算，运行时以实际 tensor 分配为准。

warmup 使用临时状态；warmup 结束后清理，保证真实请求从正确的初始状态开始。

交付：单请求 prefill/decode、文本权重严格加载、混合缓存和逐层/逐步对齐报告。

### A2：接入调度并稳定运行

1. 把 state slot 生命周期接入请求分配、运行、抢占、重新 prefill 与结束流程；slot 由稳定序列标识关联，不能依赖 batch 中的排列位置。
2. 抢占时释放 KV 和线性状态；恢复请求时从正式 token 历史重新构建两种缓存。
3. 接入变长批处理与 chunked prefill，确保分块边界的卷积历史和循环状态正确衔接。
4. 为 Qwen3.5 增加显式关闭 prefix caching 的路径。只命中 KV prefix 而缺少对应边界的 GDN 状态会产生错误结果。
5. 校验 prompt + completion 的总长度，正常处理 EOS、最大输出长度和连续多次 `generate`。
6. 记录普通解码的 TTFT、decode 时间、正式输出吞吐和峰值显存，为 MTP 评测准备对照组。

后续启用混合 prefix caching 时，需要在同一 prefix 边界缓存全部相关 GDN 状态，并处理状态复制及共享 KV 的写入保护；单纯复用现有 KV hash 不能满足条件。

阶段 A 验收：

- [x] 所有必需文本权重加载完成；跳过项与共享别名可解释。
- [ ] 同一输入下，逐层 hidden states、prefill/decode logits 与参考实现完成数值比较。
- [x] 固定输入集合的 greedy token 序列一致；不一致时定位首个分歧及对应 logits。
- [x] 一次完整 prefill 与不同分块大小的 prefill 结果一致。
- [x] 请求单独运行与变长 batch 运行一致，batch 重排不会串用状态。
- [x] 覆盖请求结束、状态 slot 复用、抢占重算、EOS、最大长度和连续调用。
- [x] 原有 Qwen3 示例及固定输入回归通过。
- [ ] 保存关闭 MTP 的性能基线和运行条件。

数值报告分别记录 FP32/BF16 等设置下的最大绝对误差、平均误差及关键位置的 argmax。容差在参考环境固定后按算子制定并记录；随机采样不以同 seed 输出完全一致作为数值对齐标准。

## 6. 阶段 B：原生 MTP 投机解码

### B0：实现预测模块并锁定位置契约

新增 MTP 模块，加载官方 `mtp.*`，并复用文本 embedding 与 LM head。结构包含两路输入归一化、拼接投影、完整注意力 decoder 和输出归一化；MTP 层需要单独的注意力 KV。[vLLM 原生实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)

官方配置的 MTP 层数为 1，草稿长度 K 是推理参数。先实现 `K=1`，之后验证递归复用该层生成 `K=2/3/4` 的结果；不能把 MTP 层数当作草稿长度上限。[MTP 算法说明](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)

约定 `x_t` 为已经提交、但主模型尚未消费的最后一个 token：

- 主模型缓存处理到 `x_(t-1)`，保存对应位置的 target hidden state。
- MTP 输入对齐为前一位置的 hidden state 与当前 token embedding：`(h_(t-1), E(x_t))`，在位置 t 预测第一个草稿 `d_1`。
- 更长草稿按已验证的递归规则继续提出 `d_2 ... d_K`。
- target hidden state 的取值位置及 final norm 语义必须与选定参考实现核对，通过固定输入比较后冻结接口。
- 首次 MTP prefill 用正确移位的 token/hidden 对建立历史；接受或拒绝后，草稿缓存需恢复并补齐到实际正式前缀。

接口区分 `decode` 与 `verify`。主模型验证时返回每个验证位置的 logits，以及下一轮 MTP 所需的 hidden states。现有 LM head 的“prefill 只取最后位置”逻辑必须在验证模式中绕过。

交付：可独立调用的 MTP forward、参数覆盖报告、token/position/hidden 的对齐用例，以及草稿历史初始化测试。

### B1：单请求、单草稿 token、greedy 验证闭环

先实现最容易定位错误的配置：单卡、单请求、`K=1`、greedy、CUDA Graph 与 prefix caching 关闭。

一轮验证采用以下契约，之后可扩展到 K 个草稿：

```text
正式历史：  ... x_(t-1), x_t       x_t 已提交，尚未被 target 消费
草稿：                 d_1, ... d_K
target 输入：   [x_t, d_1, ... d_K]
target logits： [p_1, p_2, ... p_(K+1)]
验证对应：        d_1↔p_1，d_2↔p_2，...，d_K↔p_K
```

对每个位置比较草稿与主模型 argmax，仅接受连续匹配的前缀。若接受 r 个后发生拒绝，则提交 `d_1 ... d_r` 及该位置的主模型修正 token；若全部接受，则提交全部草稿及 `p_(K+1)` 的 bonus token。遇到 EOS 或长度上限时立即截断提交。

主模型完整注意力可以一次验证多位置；GDN 在这些位置上仍按因果 recurrence 更新。先用参考多 token 路径验证正确性，再优化执行速度。

#### 状态提交与回滚

把一轮解码作为“试运行 → 验证 → 提交/恢复”的事务。草稿 token 在通过验证前不写入正式 `Sequence.token_ids`。

| 对象 | 一轮结束时的要求 |
| --- | --- |
| 正式 token 历史 | 只包含接受的前缀和有效修正/bonus token |
| 主模型 KV | 有效长度对应实际消费的正式 token；拒绝尾部不可读 |
| 主模型 GDN 状态 | 卷积历史与 recurrent state 同时恢复到接受边界 |
| MTP KV 与 hidden state | 恢复并补齐真实前缀；下一轮使用 target 对齐的 hidden |
| block/state slot 管理 | 释放无效预留空间，保留引用计数与请求隔离 |
| prefix hash | 仅发布正式且已消费的完整块 |

GDN 是覆盖更新的循环状态，不能通过减少 `num_cached_tokens` 回滚。首版在验证前保存卷积状态与 recurrent state；验证后恢复并仅重放实际接受的输入前缀。该策略便于验证，但会增加时间和显存，B2 再评估边界快照或支持回滚的 kernel。

对于仍活跃的请求，保持以下不变量：

```text
num_cached_tokens = target 实际消费的正式 token 数
num_tokens = 正式 token 历史长度
num_tokens = num_cached_tokens + 1
```

最后一个 token 是修正或 bonus token，留给下一轮消费。接受 r 个草稿时，target 有效缓存只前进 `1 + r` 个输入位置；本轮新增正式输出通常是 `r + 1` 个 token。输出计数与缓存计数分别更新，EOS 截断及结束清理另行处理。

MTP 缓存的已消费位置与 target 不一定相同，尤其在全部接受、首次调用及拒绝时。显式记录其长度并补齐缺失历史；首版可用重放验证同步逻辑，之后优化。

交付：开启/关闭 MTP 的 greedy 一致性、单草稿验证、真实模型状态恢复及资源释放测试。

### B2：多草稿、批处理与随机采样

#### 多 token 验证与调度

- 为每条请求预留 anchor + K 个验证输入所需的 KV 容量，同时考虑 MTP KV、临时状态及剩余输出预算。
- block manager 按多 token 输入跨度计算新增块数。空间不足时减小 K 或退回普通解码。
- 使用 packed/varlen 验证元数据描述各请求长度、position、block table、state slot 和 logits 行索引。
- runner 返回每条请求的提交 token 列表、接受长度、缓存更新量和统计信息，替换原有单 token 返回契约。
- scheduler 按请求分别提交，处理不同接受长度、EOS、结束释放及 batch 重排。
- 多卡恢复时还需扩展 `Sequence.__getstate__` / `__setstate__` 的传输字段；首版对未实现的 TP 组合明确报错。

#### 保持分布正确的随机采样

greedy 闭环通过后再实现随机采样。目标分布记为 p，草稿分布记为 q；二者均是在相应前缀下应用约定采样处理后的归一化分布。对于从 q 抽取的候选 d：

```text
接受概率：min(1, p(d) / q(d))
拒绝后的修正分布：normalize(max(p - q, 0))
全部接受后的 bonus：从最后位置的 target 分布采样
```

首次拒绝后停止使用该轮后续草稿。实现需要保存相应 q 分布或支持等价的严格校正算法，处理数值稳定性和 q 为零等边界；不能用简单 token 相等判断代替随机采样验证。[投机采样原始论文](https://arxiv.org/abs/2211.17192)

现有温度采样先作为基础；未来加入 top-k/top-p 等处理时，p/q 与修正步骤的定义需要一同更新。相同 seed 可能因随机数消费次序不同而产生不同文本，随机模式验收关注分布而非逐 token 相等。

阶段 B 验收：

- [ ] MTP 权重与共享 embedding/head 加载正确，K=1 的预测数值对齐。
- [ ] 开启/关闭 MTP 的固定测试 greedy 输出完全一致。
- [ ] 强制覆盖全部接受、首个拒绝、中间拒绝和最后位置拒绝。
- [ ] 回滚后继续 decode，与从相同正式历史重建的状态/logits 一致。
- [ ] 覆盖 EOS 出现在接受前缀、修正及 bonus，且不会超过 `max_tokens`。
- [ ] 覆盖 KV block 跨界、空间不足降级、请求抢占及 state slot 复用。
- [ ] K=1/2/3/4 与 batch=1/2/4/8 按显存条件逐步验证。
- [ ] 随机接受/修正逻辑通过可枚举小词表分布测试，再验证真实模型。
- [ ] 记录性能、接受率、额外显存及不适合开启 MTP 的工作负载。

## 7. 验证与性能评测

正确性验证固定真实文本 token、dtype、position 与模型版本，先逐层定位误差，再比较整个生成。缓存测试同时比较“完整 prefill”“分块 prefill”“逐 token decode”“已缓存前缀后的多 token forward”，重点排查多 token 路径是否使用完整卷积历史并保持因果性。

性能比较使用相同输入集合、输出长度、batch、运行参数与 warmup 方式。CUDA 计时使用 event 或显式同步，预热与正式测量分开，多次测量报告中位数及波动范围。

| 指标 | 定义 / 用途 |
| --- | --- |
| TTFT | 请求开始到首个正式输出 token 的延迟 |
| Decode 时间 / TPOT | 首 token 后的生成时间及平均每 token 延迟 |
| 正式输出吞吐 | 实际提交给用户的 token 数 / 测量时间 |
| 接受率 | 接受草稿数 / 提出草稿数 |
| 每轮提交量 | 正式提交 token 数 / 验证轮数，单独报告截断情况 |
| 分项时间 | drafting、target verification、回滚/重放、采样、调度 |
| 峰值显存 | 权重、KV、循环状态、草稿与临时缓存占用 |

输入长度先覆盖 32/256/1024，输出长度覆盖 64/256/512；增加中文、英文、代码与多轮文本对话，分别报告。接受率依赖输入任务，不从随机 token benchmark 推断真实文本表现。

0.8B 模型上 MTP 的 drafting、大词表输出头与回滚开销可能抵消收益。成功标准先是结果正确和过程可解释，再以实测报告说明适用区间；不预设固定加速倍数。`LLMEngine` 的现有 decode 统计必须同步修改，不能继续把请求数当作正式输出 token 数。

## 8. 计划文件与接口

以下为最初规划的文件拆分。文本模型、模型选择、GDN 和状态管理已落地；相关基础测试集中在 `tests/test_qwen3_5.py`，真实权重对照位于 `scripts/verify_qwen3_5.py`。MTP 与性能评测文件仍待实现。

| 文件 | 职责 |
| --- | --- |
| `nanovllm/models/registry.py` | 选择模型实现与声明能力 |
| `nanovllm/models/qwen3_5.py` | 文本主干、混合 decoder、权重映射 |
| `nanovllm/models/qwen3_5_mtp.py` | MTP forward、共享参数及草稿输出 |
| `nanovllm/layers/gated_delta_net.py` | 参考 recurrence 与后续 kernel 接口 |
| `nanovllm/engine/state_manager.py` | 线性状态 slot、快照、恢复和生命周期 |
| `nanovllm/engine/speculative.py` | 草稿、验证、采样校正和提交结果 |
| `tests/test_qwen3_5_reference.py` | 层、logits 与生成对齐 |
| `tests/test_hybrid_state.py` | 分块、隔离、抢占及状态恢复 |
| `tests/test_mtp_verify.py` | 接受/拒绝、位置契约、EOS 和缓存边界 |
| `tests/test_speculative_sampling.py` | 小词表分布与随机校正 |
| `bench_qwen3_5.py` | 关闭/开启 MTP 的同条件评测 |

已新增 `enable_prefix_caching`，Qwen3.5 当前固定关闭该功能，线性状态容量由 `max_num_seqs` 控制。`enable_mtp` 和 `num_speculative_tokens` 尚未实现。引擎会拒绝当前未支持的 Qwen3.5 prefix caching 与多卡 TP 组合。

## 9. 建议提交顺序与近期任务

每次提交保持原有 Qwen3 路径可运行，完成一个可验证的行为，再更新本文对应状态：

1. `docs: add Qwen3.5 text inference and MTP plan`。
2. 配置归一化、模型选择、权重核验与 greedy 基础支持，完成 A0。
3. Qwen3.5 完整注意力、归一化、partial RoPE 与 GDN 参考层。
4. 单请求混合状态及 prefill/decode 数值对齐，完成 A1。
5. 生命周期、分块与 batch 支持，保存普通解码基线，完成 A2。
6. 原生 MTP 模块及位置契约对齐，完成 B0。
7. K=1 greedy 验证、提交和回滚，完成 B1。
8. 多草稿、变长 batch 与准确统计。
9. 投机随机采样及性能报告，完成 B2。
10. 根据分项耗时选择 kernel、CUDA Graph、prefix caching 或 TP 扩展。

当前继续完善阶段 A 的参考数据和性能基线；暂不进入 MTP。已实现功能、运行命令和验证范围以[运行与验证记录](qwen3_5_text_inference.md)为准。
