"""A small LRU cache that maps a token prefix -> its precomputed KV states."""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from typing import Optional, Sequence


def hash_tokens(token_ids: Sequence[int]) -> str:
    """Stable hash of a token-id sequence."""
    h = hashlib.sha256()
    h.update((",".join(map(str, token_ids))).encode())
    return h.hexdigest()


class PrefixKVCache:
    """Fixed-size LRU dict: hash(prefix tokens) -> (KV cache, prefix length).

    The stored KV object is never handed out directly. `get` returns a deep
    copy, because HF caches are extended in place while decoding and would
    otherwise corrupt the stored prefix.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, token_ids: Sequence[int]) -> Optional[tuple]:
        key = hash_tokens(token_ids)
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)  # mark most recently used
        self.hits += 1
        kv, n = self._store[key]
        return copy.deepcopy(kv), n

    def put(self, token_ids: Sequence[int], kv) -> None:
        key = hash_tokens(token_ids)
        self._store[key] = (copy.deepcopy(kv), len(token_ids))
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # evict least recently used

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
