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

