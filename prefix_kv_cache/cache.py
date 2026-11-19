
"""Thread-safe bounded LRU cache for reusable Transformer KV prefixes."""

from __future__ import annotations

import copy
import hashlib
import math
import threading
from collections import OrderedDict
from typing import Optional, Sequence


def hash_tokens(token_ids: Sequence[int]) -> str:
    """Return a stable SHA-256 digest for token IDs."""
    h = hashlib.sha256()
    h.update(
        ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    )
    return h.hexdigest()


def estimate_kv_bytes(value, seen=None) -> int:
    """
    Estimate memory occupied by KV tensors.

    Supports:
    - PyTorch tensors
    - Nested tuples, lists and dictionaries
    - Older Hugging Face cache objects
    - Newer Hugging Face DynamicCache objects
    - Newer Hugging Face DynamicLayer objects

    Counts tensor payload bytes, not Python object overhead.
    """
    if seen is None:
        seen = set()

    object_id = id(value)

    if object_id in seen:
        return 0

    seen.add(object_id)

    # PyTorch tensors
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)

    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())

    # Dictionaries
    if isinstance(value, dict):
        return sum(
            estimate_kv_bytes(k, seen)
            + estimate_kv_bytes(v, seen)
            for k, v in value.items()
        )

    # Tuples and lists
    if isinstance(value, (tuple, list)):
        return sum(
            estimate_kv_bytes(item, seen)
            for item in value
        )

    # Newer Hugging Face DynamicCache:
    # DynamicCache contains a list of DynamicLayer objects.
    layers = getattr(value, "layers", None)

    if layers is not None:
        return estimate_kv_bytes(layers, seen)

    # Older Hugging Face cache format.
    key_cache = getattr(value, "key_cache", None)
    value_cache = getattr(value, "value_cache", None)

    if key_cache is not None or value_cache is not None:
        return (
            estimate_kv_bytes(key_cache, seen)
            + estimate_kv_bytes(value_cache, seen)
        )

    # Newer Hugging Face DynamicLayer:
    # Each layer exposes keys and values tensors.
    keys = getattr(value, "keys", None)
    values = getattr(value, "values", None)

    if keys is not None or values is not None:
        return (
            estimate_kv_bytes(keys, seen)
            + estimate_kv_bytes(values, seen)
        )

    return 0


class PrefixKVCache:
    """Thread-safe LRU cache with pinned entries and memory limits."""

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_mb: Optional[float] = None,
    ):
        if (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")

        if max_memory_mb is not None:
            if (
                isinstance(max_memory_mb, bool)
                or not isinstance(max_memory_mb, (int, float))
                or not math.isfinite(max_memory_mb)
                or max_memory_mb <= 0
            ):
                raise ValueError(
                    "max_memory_mb must be a positive finite number "
                    "or None"
                )

        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb

        self.max_memory_bytes = (
            int(max_memory_mb * 1024 * 1024)
            if max_memory_mb is not None
            else None
        )

        # key -> (KV state, prefix length, pinned, memory bytes)
        self._store = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self._lock = threading.RLock()

    @staticmethod
    def _key(token_ids: Sequence[int]):
        tokens = tuple(int(token) for token in token_ids)
        return hash_tokens(tokens), tokens

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def contains(self, token_ids: Sequence[int]) -> bool:
        key = self._key(token_ids)

        with self._lock:
            return key in self._store

    def is_pinned(self, token_ids: Sequence[int]) -> bool:
        key = self._key(token_ids)

        with self._lock:
            item = self._store.get(key)
            return item is not None and item[2]

    def get(self, token_ids: Sequence[int]) -> Optional[tuple]:
        key = self._key(token_ids)

        with self._lock:
            item = self._store.get(key)

            if item is None:
                self.misses += 1
                return None

            self._store.move_to_end(key)
            self.hits += 1

            kv, prefix_length, _, _ = item

            return copy.deepcopy(kv), prefix_length

    def _memory_used_bytes(self) -> int:
        """Return total estimated KV tensor memory. Lock must be held."""
        return sum(item[3] for item in self._store.values())

    def _pinned_memory_bytes(self) -> int:
        """Return memory occupied by pinned entries. Lock must be held."""
        return sum(
            item[3]
            for item in self._store.values()
            if item[2]
        )

    def _evict_one(self, exclude_key=None) -> bool:
        """Evict oldest unpinned entry. Lock must be held."""

        victim = None

        for key, item in self._store.items():
            if key == exclude_key:
                continue

            if not item[2]:
                victim = key
                break

        if victim is None:
            return False

        del self._store[victim]
        self.evictions += 1

        return True

    def _make_room(
        self,
        key,
        required_bytes: int,
        pinned: bool,
    ) -> None:
        """
        Make room for an entry while respecting count and memory limits.

        Existing entry replacement is accounted for before eviction.
        The entry being replaced is never selected as an eviction victim.

        Lock must be held.
        """

        existing = self._store.get(key)

        old_bytes = existing[3] if existing is not None else 0
        old_pinned = existing[2] if existing is not None else False

        current_pinned_count = sum(
            1 for item in self._store.values() if item[2]
        )

        current_pinned_bytes = self._pinned_memory_bytes()

        projected_pinned_count = (
            current_pinned_count
            - (1 if old_pinned else 0)
            + (1 if pinned else 0)
        )

        projected_pinned_bytes = (
            current_pinned_bytes
            - (old_bytes if old_pinned else 0)
            + (required_bytes if pinned else 0)
        )

        # Pinned entries cannot be evicted.
        # Reject impossible requests before evicting ordinary entries.
        if projected_pinned_count > self.max_entries:
            raise RuntimeError(
                "Cannot insert prefix: pinned entries exceed cache capacity"
            )

        if (
            self.max_memory_bytes is not None
            and projected_pinned_bytes > self.max_memory_bytes
        ):
            raise RuntimeError(
                "Cannot insert prefix: pinned KV states exceed "
                "the configured memory budget"
            )

        while True:
            projected_count = (
                len(self._store)
                - (1 if existing is not None else 0)
                + 1
            )

            projected_bytes = (
                self._memory_used_bytes()
                - old_bytes
                + required_bytes
            )

            count_exceeded = projected_count > self.max_entries

            memory_exceeded = (
                self.max_memory_bytes is not None
                and projected_bytes > self.max_memory_bytes
            )

            if not count_exceeded and not memory_exceeded:
                return

            if not self._evict_one(exclude_key=key):
                raise RuntimeError(
                    "Cannot insert prefix: insufficient cache capacity "
                    "or memory budget; remaining entries are pinned"
                )

    def put(self, token_ids: Sequence[int], kv) -> None:
        key = self._key(token_ids)

        with self._lock:
            existing = self._store.get(key)

            # Preserve existing pinned status.
            pinned = existing[2] if existing is not None else False

            kv_copy = copy.deepcopy(kv)
            kv_bytes = estimate_kv_bytes(kv_copy)

            self._make_room(key, kv_bytes, pinned)

            self._store[key] = (
                kv_copy,
                len(key[1]),
                pinned,
                kv_bytes,
            )

            self._store.move_to_end(key)

    def pin(self, token_ids: Sequence[int], kv=None) -> None:
        key = self._key(token_ids)

        with self._lock:
            existing = self._store.get(key)

            if existing is not None:
                old_kv, prefix_length, _, old_bytes = existing

                if kv is not None:
                    new_kv = copy.deepcopy(kv)
                    new_bytes = estimate_kv_bytes(new_kv)
                else:
                    new_kv = old_kv
                    new_bytes = old_bytes

                self._make_room(key, new_bytes, pinned=True)

                self._store[key] = (
                    new_kv,
                    prefix_length,
                    True,
                    new_bytes,
                )

                self._store.move_to_end(key)
                return

            if kv is None:
                raise ValueError(
                    "KV state is required when pinning a new prefix"
                )

            kv_copy = copy.deepcopy(kv)
            kv_bytes = estimate_kv_bytes(kv_copy)

            self._make_room(key, kv_bytes, pinned=True)

            self._store[key] = (
                kv_copy,
                len(key[1]),
                True,
                kv_bytes,
            )

            self._store.move_to_end(key)

    def unpin(self, token_ids: Sequence[int]) -> bool:
        key = self._key(token_ids)

        with self._lock:
            item = self._store.get(key)

            if item is None or not item[2]:
                return False

            kv, prefix_length, _, kv_bytes = item

            self._store[key] = (
                kv,
                prefix_length,
                False,
                kv_bytes,
            )

            # Unpinning moves the entry to most recently used.
            self._store.move_to_end(key)

            return True

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total else 0.0

    @property
    def stats(self) -> dict:
        with self._lock:
            requests = self.hits + self.misses

            pinned = sum(
                1 for item in self._store.values() if item[2]
            )

            memory_bytes = self._memory_used_bytes()
            pinned_bytes = self._pinned_memory_bytes()

            memory_mb = memory_bytes / (1024 ** 2)
            pinned_memory_mb = pinned_bytes / (1024 ** 2)

            if self.max_memory_bytes is None:
                memory_utilization = None
            else:
                memory_utilization = (
                    memory_bytes / self.max_memory_bytes
                )

            return {
                "entries": len(self._store),
                "capacity": self.max_entries,
                "pinned_entries": pinned,
                "ordinary_entries": len(self._store) - pinned,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "requests": requests,
                "hit_rate": (
                    self.hits / requests if requests else 0.0
                ),
                "memory_bytes": memory_bytes,
                "memory_mb": round(memory_mb, 4),
                "pinned_memory_mb": round(pinned_memory_mb, 4),
                "memory_capacity_mb": self.max_memory_mb,
                "memory_utilization": (
                    round(memory_utilization, 4)
                    if memory_utilization is not None
                    else None
                ),
            }

    def clear(self, reset_stats: bool = True) -> None:
        with self._lock:
            self._store.clear()

            if reset_stats:
                self.hits = 0
                self.misses = 0
                self.evictions = 0
