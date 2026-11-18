import random

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prefix_kv_cache import CachedGenerator, PrefixKVCache


def make_gen(max_entries=100):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=300))
    return CachedGenerator(model, eos_token_id=None, cache=PrefixKVCache(max_entries))


def test_cached_output_matches_uncached():
    g = make_gen()
    prefix = [random.randrange(300) for _ in range(50)]
    for _ in range(3):  # 1st = miss, rest = hits; all must equal baseline
        suffix = [random.randrange(300) for _ in range(5)]
        a = g.generate_ids(prefix, suffix, 10, use_cache=True)
        b = g.generate_ids(prefix, suffix, 10, use_cache=False)
        assert a == b


def test_hit_miss_counts_and_prefix_not_corrupted():
    g = make_gen()
    prefix = list(range(30))
    for s in ([1, 2, 3], [4, 5, 6], [7, 8, 9]):
        g.generate_ids(prefix, s, 5)
    assert g.cache.misses == 1 and g.cache.hits == 2
    kv, n = g.cache.get(prefix)
    assert n == 30
    assert kv.get_seq_length() == 30  # decoding must not extend the stored copy


def test_lru_eviction():
    c = PrefixKVCache(max_entries=2)
    c.put([1], "a"); c.put([2], "b")
    c.get([1])            # [1] becomes most recent
    c.put([3], "c")       # evicts [2]
    assert c.get([2]) is None
    assert c.get([1]) is not None and c.get([3]) is not None
    assert len(c) == 2
