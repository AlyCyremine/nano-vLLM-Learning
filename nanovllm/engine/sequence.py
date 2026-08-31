from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto() # 自动标号


class Sequence:
    block_size = 256
    counter = count() # 所有 Sequence 实例都使用同一个计数器对象

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING # 新请求首先处于 Waiting 等待调度
        self.token_ids = copy(token_ids) #修改 self.token_ids 时不会影响原来的 token_ids
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids) # 还未创建 completion
        self.num_cached_tokens = 0 # 已经存在 KV Cache 的 token 数
        self.num_scheduled_tokens = 0 # scheduler 本轮安排模型处理的 token 数
        self.is_prefill = True # 当前是不是 prompt 的 prefill 阶段
        self.block_table = [] # 这条 sequence 使用哪些物理 KV Cache block
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self): # python 预留的接口，调用 Sequence.len 实际是调用这个
        return self.num_tokens

    def __getitem__(self, key): # 让 Sequence 像 list 一样索引
        return self.token_ids[key]

    # @property 的作用是：方法内部可以执行计算，但调用时看起来像普通属性。
    # 正常：seq.is_finished() / 有property：seq.is_finished
    @property 
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens] # 左闭右开

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self): # 算要几个 KV Cache block 才能存下所有 token
        return (self.num_tokens + self.block_size - 1) // self.block_size # 向上取整除法

    @property
    def last_block_num_tokens(self): # 计算最后一个 block 中的 token 数量
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks # 左闭右开检查下标是否合法
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # 取某一个逻辑 block

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self): # pickle 序列化时调用，返回一个可序列化的对象 "pickle.dumps()"
        last_state = self.last_token if not self.is_prefill else self.token_ids # pre-fill阶段返回 token_ids，decode阶段返回 last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state) # pack 成 tuple

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state # unpack
        if isinstance(last_state, list): # prefill 阶段
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else: # decode 阶段
            self.token_ids = []
            self.last_token = last_state
