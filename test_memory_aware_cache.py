
import torch

from prefix_kv_cache.cache import PrefixKVCache


def make_kv(size_mb):
    """Create a KV pair occupying approximately size_mb MiB in total."""
    # Divide the requested memory equally between the two tensors.
    num_elements = size_mb * 1024 * 1024 // 8

    key = torch.zeros(num_elements, dtype=torch.float32)
    value = torch.zeros(num_elements, dtype=torch.float32)

    return ((key, value),)


def test_memory_eviction():
    print("\nTEST 1: MEMORY-BASED LRU EVICTION")

    cache = PrefixKVCache(max_entries=10, max_memory_mb=10)

    cache.put([1], make_kv(4))
    cache.put([2], make_kv(4))

    print("Before third insertion:", cache.stats)

    cache.put([3], make_kv(4))

    assert not cache.contains([1])
    assert cache.contains([2])
    assert cache.contains([3])

    print("After third insertion:", cache.stats)
    print("TEST 1 PASSED")


def test_pinned_protection():
    print("\nTEST 2: PINNED ENTRY PROTECTION")

    cache = PrefixKVCache(max_entries=10, max_memory_mb=10)

    cache.pin([10], make_kv(4))
    cache.put([20], make_kv(4))
    cache.put([30], make_kv(4))

    assert cache.contains([10])
    assert not cache.contains([20])
    assert cache.contains([30])
    assert cache.is_pinned([10])

    print("Cache stats:", cache.stats)
    print("TEST 2 PASSED")


def test_pinned_memory_limit():
    print("\nTEST 3: PINNED MEMORY BUDGET")

    cache = PrefixKVCache(max_entries=10, max_memory_mb=5)

    cache.pin([100], make_kv(4))

    try:
        cache.pin([200], make_kv(4))
    except RuntimeError as error:
        print("Expected error:", error)
    else:
        raise AssertionError(
            "Expected insertion to fail because pinned memory exceeds budget"
        )

    assert cache.contains([100])
    assert not cache.contains([200])

    print("Cache stats:", cache.stats)
    print("TEST 3 PASSED")



def test_unpin_allows_eviction():
    print("\nTEST 4: UNPIN ALLOWS EVICTION")

    cache = PrefixKVCache(max_entries=10, max_memory_mb=8)

    cache.pin([1], make_kv(4))
    cache.put([2], make_kv(4))

    assert cache.unpin([1])

    # Unpinning moves [1] to the most-recently-used position.
    # Therefore [2] is the oldest ordinary entry.
    cache.put([3], make_kv(4))

    assert cache.contains([1])
    assert not cache.contains([2])
    assert cache.contains([3])
    assert not cache.is_pinned([1])

    print("Cache stats:", cache.stats)
    print("TEST 4 PASSED")



if __name__ == "__main__":
    test_memory_eviction()
    test_pinned_protection()
    test_pinned_memory_limit()
    test_unpin_allows_eviction()

    print("\nALL MEMORY-AWARE CACHE TESTS PASSED.")
