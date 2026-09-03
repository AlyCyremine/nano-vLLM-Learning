from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = [] # 空 batch，存放本轮调度的所有 sequence
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0: # 因为 min(num_tokens, remaining)
                break
            if not seq.block_table: # 第一次进入 prefill 的新 Sequence
                num_cached_blocks = self.block_manager.can_allocate(seq) # 有无可复用的 prefix cache 块数量，是否可以分配 KV Cache块
                if num_cached_blocks == -1: # 无法分配
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size # 计算真正需要 prefill 的 token
            else: # 之前已经 prefill 过的 Sequence
                num_cached_blocks = len(seq.block_table) # 已经 prefill 的 KV Cache块 的数量
                num_tokens = seq.num_tokens - seq.num_cached_tokens # 这一轮需要 prefill 的 token 数量
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 这一轮计划完成的 prefill
            num_batched_tokens += seq.num_scheduled_tokens # 统计调度的总 token 数
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens: # 如果 prefill 完成了
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq) # 不管这轮有没有 prefill 完成，都标记下

        if scheduled_seqs: # prefill 和 decode 不会出现在同一个 batch
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs: # 每条 Seq 一轮只生成一个新 token
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):# 如果这个 Seq 再生成一个 token，KV Cache 是否有足够空间？
                if self.running: # 如果空间不够，然后 running 里还有其他 seq
                    self.preempt(self.running.pop()) # 抢占队尾的空间
                else: # 已经没有可以抢的了
                    self.preempt(seq) #那就只能把自己也踢回去了
                    break
            else:
                seq.num_scheduled_tokens = 1 # decode 一次只调度 1 token
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq) # GPU 可以并行跑多条 seq 的 decode，但是每条 seq 一轮只生成一个 token
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) # 因为之前被 pop 出去了，所以要放回去，保持顺序不变
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq) # 把这个 Seq 当前占用的 KV Cache 释放掉
        self.waiting.appendleft(seq) # 让这个 Seq 优先被调度，避免它被饿死

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool): # 执行完 GPU 后收尾（更新 Seq/BM 状态）
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq) # 把新形成的完整 block计算 hash，登记进 prefix cache
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens: # 还没 prefill 完，继续等待调度
                continue 
            seq.append_token(token_id) # 把新生成的 token 加到 seq 里
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # 判断生成是否结束
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
