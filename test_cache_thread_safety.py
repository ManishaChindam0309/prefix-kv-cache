
from concurrent.futures import ThreadPoolExecutor

from prefix_kv_cache import PrefixKVCache


def main():
    print("\nTEST 1: CONCURRENT CACHE OPERATIONS")

    # Enough capacity to prevent eviction during this test.
    cache = PrefixKVCache(max_entries=5000)

    # Insert and pin one shared entry.
    cache.pin([1, 2, 3], {"kv": [10, 20, 30]})

    def worker(worker_id):
        for i in range(500):
            # Repeated access to the shared pinned prefix.
            result = cache.get([1, 2, 3])

            assert result is not None
            assert result[0]["kv"] == [10, 20, 30]

            # Insert and retrieve a unique ordinary entry.
            tokens = [worker_id, i + 1000]

            cache.put(tokens, {"value": i})

            retrieved = cache.get(tokens)

            assert retrieved is not None
            assert retrieved[0]["value"] == i

            # Read statistics while other threads are active.
            _ = cache.stats

    print("Starting 8 concurrent worker threads...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(8)
        ]

        for future in futures:
            future.result()

    assert cache.is_pinned([1, 2, 3])
    assert cache.contains([1, 2, 3])

    stats = cache.stats

    assert stats["pinned_entries"] == 1
    assert stats["entries"] == 4001
    assert stats["capacity"] == 5000
    assert stats["hits"] == 8000
    assert stats["misses"] == 0

    print("\nPASS: All worker threads completed.")
    print("PASS: Shared pinned entry survived.")
    print("PASS: All 4000 ordinary entries were stored.")
    print("PASS: All concurrent retrievals succeeded.")
    print("PASS: Cache statistics remained consistent.")

    print("\nFinal statistics:")
    print(stats)

    print("\nTEST 1 PASSED")

    # --------------------------------------------------
    # TEST 2: CONCURRENT WORKLOAD WITH BOUNDED CAPACITY
    # --------------------------------------------------

    print("\nTEST 2: BOUNDED LRU CAPACITY")

    bounded_cache = PrefixKVCache(max_entries=20)

    bounded_cache.pin([999], {"shared": True})

    for i in range(100):
        bounded_cache.put([i + 2000], {"value": i})

    bounded_stats = bounded_cache.stats

    assert bounded_stats["entries"] == 20
    assert bounded_stats["entries"] <= bounded_stats["capacity"]
    assert bounded_cache.contains([999])
    assert bounded_cache.is_pinned([999])

    print("PASS: Cache capacity remained bounded.")
    print("PASS: Pinned entry survived eviction.")
    print("PASS: Ordinary entries were evicted as needed.")

    print("\nBounded cache statistics:")
    print(bounded_stats)

    print("\n================================")
    print("THREAD-SAFETY STRESS TEST PASSED")
    print("================================")


if __name__ == "__main__":
    main()
