import pytest

from prefix_kv_cache.cache import PrefixKVCache


def test_lru_eviction_and_stats():
    cache = PrefixKVCache(max_entries=2)
    cache.put([1], "one")
    cache.put([2], "two")
    assert cache.get([1]) == ("one", 1)  # [1] becomes most recently used
    cache.put([3], "three")             # evicts [2]
    assert cache.get([2]) is None
    assert cache.get([1]) == ("one", 1)
    assert cache.stats == {
        "entries": 2, "capacity": 2, "hits": 2, "misses": 1,
        "evictions": 1, "requests": 3, "hit_rate": 2 / 3,
    }


def test_exact_prefixes_are_distinct():
    cache = PrefixKVCache()
    cache.put([1, 23], "a")
    cache.put([12, 3], "b")
    assert cache.get([1, 23]) == ("a", 2)
    assert cache.get([12, 3]) == ("b", 2)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        PrefixKVCache(0)


def test_clear_resets_entries_and_stats():
    cache = PrefixKVCache()
    cache.put([1], "value")
    cache.get([1])
    cache.clear()
    assert len(cache) == 0
    assert cache.stats["hits"] == 0
    assert cache.stats["requests"] == 0


def test_clear_can_preserve_stats():
    cache = PrefixKVCache()
    cache.get([1])
    cache.clear(reset_stats=False)
    assert cache.stats["misses"] == 1
