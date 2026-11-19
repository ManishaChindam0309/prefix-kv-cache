
from prefix_kv_cache import PrefixKVCache

cache = PrefixKVCache(max_entries=3)

# Insert and pin the system prompt.
cache.pin([1, 2, 3], "SYSTEM_KV")

# Add two ordinary prefixes.
cache.put([4, 5], "PREFIX_A")
cache.put([6, 7], "PREFIX_B")

assert len(cache) == 3
assert cache.is_pinned([1, 2, 3])

# Insert another prefix. The oldest ordinary entry should be evicted.
cache.put([8, 9], "PREFIX_C")

assert cache.contains([1, 2, 3])
assert cache.is_pinned([1, 2, 3])
assert not cache.contains([4, 5])
assert cache.contains([8, 9])

# Verify that the pinned KV can still be retrieved.
result = cache.get([1, 2, 3])

assert result is not None
assert result[0] == "SYSTEM_KV"

print("PASS: Pinned prefix survived eviction.")
print("PASS: Ordinary prefix was evicted.")
print("PASS: Pinned KV remains retrievable.")
print("Cache statistics:", cache.stats)

# Unpin and verify it can now be evicted.
assert cache.unpin([1, 2, 3])
assert not cache.is_pinned([1, 2, 3])

print("PASS: Prefix can be unpinned.")
