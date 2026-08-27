### Day 2

#### generate() 运行逻辑：

```
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
Sequence(prompt, sampling_params)
        │
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




### Day 3