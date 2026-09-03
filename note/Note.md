### ,Day 1

装环境，编译flash atten，把nano-vLLM跑起来

### Day 2

#### generate() 运行逻辑：

```python
generate(prompts)
        │
        ▼
for prompt
        │
        ▼
add_request()
        │
        ├─ str?
        │    ↓
        │ tokenizer.encode()
        │
        ▼
Sequence(prompt, sampling_params) # prompt 须是 encode 过后的 token_ids 的 List
        │           			 # 每一个 prompt 及其对应的 sp 构成一个 Sequence
        ▼
scheduler.add(seq)

然后进入主循环：

while not scheduler.is_finished()
        │
        ▼
      step()
        │
        ▼
scheduler.schedule()
        │
        ├── seqs
        └── is_prefill
              │
              ▼
       计算 num_tokens
              │
              ▼
model_runner.call("run", seqs, is_prefill)
              │
              ▼
          token_ids
              │
              ▼
scheduler.postprocess(...)
              │
              ▼
检查 seq.is_finished
              │
              ▼
(seq_id, completion_token_ids)
              │
              ▼
返回 generate()
```

#### Key word arguments:

```python
# kwargs: 在“定义函数/类”时，** 是打包（收进来）；在“调用函数/类”时，** 是解包（放出去）。
config_fields = {field.name for field in fields(Config)}  # 从config里面筛
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)

# 正负编码区分 prefill 和 decode 的小巧思
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
if num_tokens > 0:
    prefill_throughput = num_tokens / (perf_counter() - t)
    else:
        decode_throughput = -num_tokens / (perf_counter() - t)
```

#### 父子类

```python
class BaseModel:
    def __init__(self, model_name, device="cpu", **kwargs):  # 基类收下自己认识的
        self.model_name = model_name
        self.device = device
        # 注意：基类不认识的参数，可以无视，也可以存起来
        print(f"基类拿到了: model_name={model_name}, device={device}")

class AdvancedModel(BaseModel):
    def __init__(self, extra_feature, **kwargs):  # 子类只暴露自己特有的参数
        # 关键魔法：把自己不认识的剩下所有参数，原封不动扔给父类
        super().__init__(**kwargs)  
        self.extra_feature = extra_feature
        print(f"子类特有: {extra_feature}")
```

1. 打包阶段：Python 看到 **，把所有 key=value 打包成一个字典：
   kwargs = {'extra_feature': 'flash_attn', 'model_name': 'Llama-3', 'device': 'mps', 'temperature': 0.7}

2. 子类取值：子类的 __init__ 显式声明了 extra_feature，所以 Python 会从字典里弹出 'extra_feature' 赋给这个参数。此时 kwargs 里剩下：{'model_name': 'Llama-3', 'device': 'mps', 'temperature': 0.7}

3. 向上透传：执行 super().__init__(**kwargs)，** 把字典解包成关键字参数传进去，等价于调用：
   BaseModel.__init__(model_name='Llama-3', device='mps', temperature=0.7)

4. 父类取值：父类的 __init__ 显式声明了 model_name 和 device，吃掉这两个。剩下的 temperature=0.7 被父类的 **kwargs 捕获（虽然父类没用到，但不会报错）。

   ​




### Day 3

#### Sequence 类

```python
# @property 的作用是：方法内部可以执行计算，但调用时看起来像普通属性。
# 正常：seq.is_finished() / 有property：seq.is_finished
```

```python
# __getstate__ / __setstate__ 通常不是业务代码直接调用的，而是在对象被 pickle 序列化/反序列化时，由 Python 自动调用。最典型的形式是：
import pickle

seq = Sequence([10, 20, 30])

data = pickle.dumps(seq)
new_seq = pickle.loads(data)
```

```python
背后大致发生的是：

pickle.dumps(seq)
      │
      └──→ seq.__getstate__()
                │
                ↓
          得到 state tuple
                │
                ↓
          序列化成 bytes


pickle.loads(data)
      │
      └──→ 创建 Sequence 对象
                │
                ↓
          obj.__setstate__(state)
```

#### Config 类

```python
max_num_batched_tokens: int = 16384 # 一次 batch 最多调度多少 token
max_num_seqs: int = 512 # 一次 batch 最多调度多少 sequence
max_model_len: int = 4096 # 模型最大长度（prompt tokens + completion tokens）
    
# Sequence = 一条prompt + 它对应的SP + 后续生成出来的completion + 这条请求的运行状态/KV Cache信息
# Batch = 某一次模型前向计算（forward）中，被放在一起处理的一组 Sequence / token。
# Scheduler 每一轮决定哪些 Sequence 组成这一轮要送给模型计算的 batch。
# 每条 Sequence 的长度、开始时间和结束时间都不一样，但 GPU 又希望尽可能把很多工作拼成 batch 一起算。

### 噢，所以有可能一句话被拆到了多个batch里面，才需要seq_id来复原？
```

### Day 4

#### Scheduler

```python
# Scheduler 的任务是维护 waiting 和 running 两组 Sequence，每次 schedule() 从中挑出一批 Sequence 组成当前 batch，并决定这一轮做 prefill 还是 decode。
self.waiting: deque[Sequence] = deque() # 等待 / 需要 prefill 的 Sequence
self.running: deque[Sequence] = deque() # 已经完成 prefill，正在 decode 的 Sequence
    
#目前对 BlockManager 的理解是：负责管理 KV Cache block 的分配和释放
Scheduler
   │
   │ "seq A 需要 KV Cache"
   ↓
BlockManager
   │
   │ 分配物理 block
   ↓
seq.block_table

# Scheduler 优先处理 prefill
# 只要这一轮成功调度了 waiting 中的 Sequence，这一轮就是纯 prefill，不会再混入 decode Sequence。
```

### Day 5

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O \text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

#### 对于Attention的一些理解： 

- Q、K、V 不是原始 token，分别有$W_i^Q$、$W_i^K$、$W_i^V$三个可学习参数对其进行投影：
  - Q：token经过$W_i^Q$投影得到的**“查询向量”**。表达**此token要从其他token中寻找什么信息。**
  - K：token经过$W_i^K$投影得到的**“索引/匹配向量”**。表达**此token自己具有什么特征，以便其他token的Q判断“我是否值得被关注”。**
  - V：token经过$W_i^V$投影得到的**“内容向量”**。表达**如果此token被某个Q关注了，那么它实际要提供给对方什么信息。**
  - 三个可学习的参数分别让 token 在当前层的 hidden state（只有第一层可能是 embedding）被**特化**到三个功能方向。
  - 同时**实现多头注意力按头的数量进行降维的操作**（$W_i^Q \in \mathbb R^{d_{\text{model}}\times d_k}, d_k=\frac {d_{model}}{n_{head}} $)，但是工程上通常把所有的 $W_i^Q$ concat 成一个矩阵，方便运算，不会单独对每个头做运算。

- Vector的点积运算是在做 **相关性/相似性** 运算，所以整个Attention的过程就是用$QK^T$来计算某一个token的Q对另一个（也可能是同一个）token的K的相似度，然后**对这个结果矩阵的每一个Q行做softmax运算**，这个softmax值表示一个Query对所有Key的注意力分布（决定对每个V注意多少）。

- 上述$QK^T$完成后应该进行 scale 也就是 $\frac {QK^T}{\sqrt{d_{k}}}$ （缩放点积 相比 cos相似度 并没有完全消除向量长度的信息）

  - 因为当维度越来越高的时候，点积的值也会越来越大（有期望和方差的证明，大概是假设期望为0方差为1且各个维度独立分布的时候，高纬度会让方差变成$d_k$，标准差变为$\sqrt{d_k}$，也就是注意力Score更容易分布在$\sqrt{d_k}$附近）
  - 当值之间数量级差距较大的时候，softmax会更尖锐，比如：
    - $[0.2,\ 0.5,\ 0.8,\ 1.0,\ 1.2]$ 的 softmax 为 $[0.10,\ 0.14,\ 0.19,\ 0.26,\ 0.31]$
    - $[2,\ 5,\ 8,\ 10,\ 12]$ 的 softmax 为 $[0.00004,\ 0.0008,\ 0.016,\ 0.117,\ 0.866]$
    - 当数值过于大，就会出现类似 $softmax=[0.00001,\ 0.00002,\ 0.99997]$ 的这种情况，非常近似于 one-hot 分布 $[0, 0, 1]$。这会让 softmax 的导数 $p_i(1-p_i)$ 的值非常小，梯度的值非常小，就会让训练的 back propagation 变得非常困难（每次只动一点点）。

- 上述为Encode的双向Self-Attention过程，对于Decode的Causal Self-Attention，**不可以看到未来的信息**，需要加一个Causal Mask:

  ​                                                  $M=\begin{bmatrix}0&-\infty&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty&-\infty\\0&0&0&-\infty&-\infty\\0&0&0&0&-\infty\\0&0&0&0&0\end{bmatrix}$

- 在mask之后，将 softmax 之后的值对 V 重新进行加权求和，获得**真正需要读取的信息**。

- Attention中每个位置都在聚合**自己能够看到的前缀**，于是整个句子的信息都被聚合到了最后一个token这里，然后过多次transformer layer，FFN之类的操作之后拿去做对下一个token的预测（最后给一个Linear层升维做分类器）。

- 完整数据流为：
  $$
  X\xrightarrow{W_Q,W_K,W_V}Q,K,V
  \\
  S=\frac{QK^T}{\sqrt{d_k}}+M   
  \\
  A=\operatorname{softmax}(S,\text{dim}=-1)
  \\
  O=AV
  $$

  然后 Multi-Head：

  ​                                                        $O_1,\ldots,O_h\xrightarrow{\text{Concat}}O\xrightarrow{W_O}\text{Attention Output}$

  再经过：

  ​                                                    $text{Residual / Norm / FFN / 多层 Transformer}$

  最后取：$h_{\text{last}}$

  ​                                               $\boxed{h_{\text{last}}\xrightarrow{\text{LM Head}}\text{Vocabulary Logits}\xrightarrow{\text{Sampling}}\text{Next Token}}$

  - M为Mask矩阵
  - dim=-1是指只对每一个Q行进行softmax（对Tensor的最后一个维度做Softmax）

#### KV Cache

- KV Cache 成立的基础是 **causal attention**，最后的 token 可以同时看到前面所有的信息，但是**前面的 token 不可以看到后面的 token 的信息**，前面的 Hidden State 并不会更新，所以**前面的 KV 可以复用**。（prefill 也是 causal 的）
- Decode 时，新 token 只需要计算自己的 Q/K/V。新 Q 查询所有历史 K，并根据 attention weight 读取所有历史 V；新 K/V 则追加进 KV Cache。
- 不 cache Q 是因为过去的 Q 再也用不到了，只有新生成的 token 需要去前面找注意力信息。
- KV Cache 会因为一次请求的结束而被释放，也会因为每一次的 Prefill 不同而不同。
```
        Transformer Layer
                │
          hidden states
     ┌──────────┼──────────┐
    Wq         Wk         Wv
     │          │          │
     Q          K          V
     │          └────┬─────┘
     │      KV Cache [past + new]
     │               │
 current Q        K Cache
     └───────┬───────┘
            QKᵀ
             │ (Softmax)
             ▼
      attention weights
             │ (V Cache)
             ▼
      attention output   
```
### Day 6

大部分理解都在注释里，拉高了 Block Manager 的优先级，感觉不懂怎么分配 KV Cache 块的话没法弄懂 Scheduler。

#### assert

```python
# Python 的 assert（断言）核心机制就是：条件为真（True）时，程序静默通过，继续执行下一行代码；条件为假（False）时，立即抛出 AssertionError 异常并终止程序（除非被捕获）。

# 条件为真 -> 什么都不发生，程序继续
assert 1 + 1 == 2
print("这里会被执行")

# 条件为假 -> 抛出 AssertionError
# assert 1 + 1 == 3  # 报错：AssertionError

# 带自定义报错信息
# assert 1 + 1 == 3, "数学计算错误"  # 报错：AssertionError: 数学计算错误

# 1. assert 可能被全局禁用：如果你在启动 Python 时带上了 -O（优化）或 -OO 参数，所有 assert 语句会被完全忽略，相当于被删除了。此时无论条件成不成立，它都不会报错（也不会执行）。
# 即使断言失败，加了 -O 后也不会报错，会直接跳过
python -O my_script.py

# 2. 不要在断言里写会改变状态的代码：因为断言可能被跳过，如果你在里面修改变量或执行函数，程序的行为会变得不可预测。
# 危险写法：千万不要这样写！
# assert (x := x + 1) > 0  # 如果开启了 -O，这行赋值就不会执行
```
### Day 7

```python
BlockManager line 114: h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
# 找到 start 上一个块的 hash 值，因为 hash 值是链式的，需要上一个的值来更新这一个块
# h0 = hash(block0)
# h1 = hash(h0 + block1)
# h2 = hash(h1 + block2)
```

### Day 8

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
	break
    
# 这里的意思是：如果本轮剩下的可以 prefill 的额度 < 本轮需要 prefill 的 token 数量，并且已经有 seq 需要进行 prefill 了，就不强行把手上这轮的 token 截成两段送进去 prefill 了。也就是注释里说的，只允许第一个 seq 被截断后送进 prefill。
# 本质是为了简化实现而做的性能妥协（如：限1000，第一轮100，第二轮901，两轮总共效率 1001/2000 = 50%).
# 一旦允许 batch 里任意 Sequence 被截断，系统就必须支持更一般的 每条 Sequence 都处于不同 prefill 进度 的 batch，实现就会更复杂。
```

### Day 9

```python
@classmethod # 类方法可以通过类名直接调用，而不需要实例化对象。
def compute_hash(cls, token_ids: list[int], prefix: int = -1): # 计算 token_ids+prefix 的哈希值
	h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little")) 
	h.update(np.array(token_ids).tobytes())
	return h.intdigest()

# 写死 小端序"little" 是为了保证确定性和跨平台一致性
# 直接把上一轮算好的 64 位整数哈希值（代表了前面所有的历史）作为本轮的输入前缀，本轮只需要处理“8 个字节的前缀 + 当前这一小块的 token”，而不需要把前几轮的所有 token 拼成一个超长列表，计算所有数据块的总哈希。
# to_bytes(8)，不管上一轮多少 token 都把它压缩成 8 字节，让内存不会无休止增长。
```

#### python 的 while … else … 特殊语法

```python
while ...:
    ...
else:
    ...
# while 如果是因为 “条件自然变成 False” 而结束，则执行 else；如果因为 break 结束，则不执行。
```

#### 关于 decode

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
scheduled_seqs.append(seq)
# 因为 schedule 在本轮实际运行之前，所以我们这里 may_append 是在看上一轮新生成的 token。因为上一轮新生成的 token 是从最后的 classifier 出来的，并没有计算 QKV，所以也还没分配 KV Cache块 所需要的空间。又因为 decode 是一个一个出新 token，所以如果刚好超过了 Cache 块的大小，需要新块的时候，may_append 那边会因为 mod 为 1 而分配新的 KV Cache块 空间。
```

#### 关于 postprocess

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens: # 还没 prefill 完，继续等待调度
    continue 
seq.append_token(token_id) # 把新生成的 token 加到 seq 里
# 一次完整 prefill 后就已经能得到 completion 的第一个 token 了

if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
# 判断生成是否结束（EOS / 已经达到 seq 允许的 token 上限了）就 释放 KV Cache + 从 running 移除

# 而 deallocate 释放的是 “Seq 对 KV block 的占用权” ，而不是立即擦除 “block 中的 KV Cache 内容”
```

