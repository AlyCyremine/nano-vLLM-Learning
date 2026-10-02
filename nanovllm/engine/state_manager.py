from collections import deque


class StateManager:
    """Map stable request IDs to fixed GDN state slots, independent of batch order."""

    def __init__(self, capacity: int):
        self.slots: dict[int, int] = {}
        self.free_slots = deque(range(capacity))

    def allocate(self, seq_id: int, cached_tokens: int) -> int:
        if seq_id not in self.slots:
            if cached_tokens:
                raise RuntimeError(f"Missing GDN state for cached sequence {seq_id}")
            if not self.free_slots:
                raise RuntimeError("GDN state capacity exhausted")
            self.slots[seq_id] = self.free_slots.popleft()
        return self.slots[seq_id]

    def release(self, seq_ids):
        for seq_id in seq_ids:
            slot = self.slots.pop(seq_id, None)
            if slot is not None:
                self.free_slots.append(slot)

    def retain(self, active_seq_ids):
        active = set(active_seq_ids)
        self.release([seq_id for seq_id in self.slots if seq_id not in active])

    def clear(self):
        self.release(list(self.slots))
