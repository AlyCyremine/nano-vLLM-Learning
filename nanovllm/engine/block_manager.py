from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block: # 描述一个 物理KV Cache块 的管理信息

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0 # 这个块被多少 seq 使用（为0时可以c拿去重新分配）
        self.hash = -1 # ？
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]): #当某个块的 KV 已经计算完成，可以进入 prefix cache 时
        self.hash = hash
        self.token_ids = token_ids

    def reset(self): # 物理 block 被重新分配
        self.ref_count = 1 # 重新分配时，ref_count 置为 1，表示这个块被新分配的 seq 使用
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 对象: 类型 = 初始值
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod # 类方法可以通过类名直接调用，而不需要实例化对象。
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): # 计算 token_ids+prefix 的哈希值
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 将 prefix 转换为 8 字节的小端字节序，并更新哈希对象
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int: # 重置一个 block
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0 # 既然是 free，就不应该还有 Seq 使用它。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id: # 如果这个 block 之前已经计算过 prefix cache，并且还在 hash_to_block_id 中（有可能出现相同哈希值导致被覆盖）
            del self.hash_to_block_id[block.hash] # 删除这个 block 的哈希值映射，避免后续错误地认为这个 block 还在 prefix cache 中。
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int): # 把一个已经没人使用的 block 放回 free pool。
        assert self.blocks[block_id].ref_count == 0 # 确保没人用了
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int: # 1.找 prefix cache； 2.检查剩余 block 是否够用
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks # 假设这条 Seq 的所有 block 都需要从 free_block_ids 里拿
        for i in range(seq.num_blocks - 1): # 检查除了最后一个之外的所有 block（最后一个很可能不全是 prefill）
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1) # 寻找以前是否计算过这个 prefix，没有赋值 -1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # 没算过 / 算过但是哈希不匹配
                break # 因为 prefix cache 必须连续，所以一旦发现不匹配，就不再继续找了。
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1 # 因为这个 block 已经被使用了，不需要从 free_block_ids 里拿了
        if len(self.free_block_ids) < num_new_blocks:
            return -1 # 无法分配
        return num_cached_blocks # 可以分配，并且返回有多少个 block 可以从 prefix cache 里拿到。

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table # 确保 seq 还没有分配过 block
        h = -1 
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block()) # 分配新的 block
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence): 
        for block_id in reversed(seq.block_table): # 从最后一个开始往前
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) # 如果decode的时候下一个token需要新分配一个块，就需要至少1个free块。

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block()) # 分配一个新块

    def hash_blocks(self, seq: Sequence): # prefill计算完一些完整 block后，把它们登记进 prefix cache。
        start = seq.num_cached_tokens // self.block_size # 前面已经 cached 到哪个 block（向下取整）
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return # 没有完整的 block 可以登记进 prefix cache
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end): # 逐个处理新的完整 block
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
