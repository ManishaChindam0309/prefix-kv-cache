
import torch

from transformers import GPT2Config, GPT2LMHeadModel
from prefix_kv_cache import CachedGenerator, PrefixKVCache


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    config = GPT2Config(
        vocab_size=5000,
        n_positions=256,
        n_ctx=256,
        n_embd=128,
        n_layer=2,
        n_head=4,
        bos_token_id=0,
        eos_token_id=1,
    )

    model = GPT2LMHeadModel(config).to(device)
    model.eval()

    cache = PrefixKVCache(max_entries=3)

    generator = CachedGenerator(
        model,
        eos_token_id=None,
        cache=cache,
    )

    # Two prefixes that must remain protected.
    prefix_a = [10, 11, 12, 13]
    prefix_b = [20, 21, 22, 23]

    # Two ordinary prefixes that compete for cache space.
    prefix_c = [30, 31, 32, 33]
    prefix_d = [40, 41, 42, 43]

    suffix = [100, 101]

    print("\nPinning two real model prefixes...")

    generator.pin_prefix(prefix_a)
    generator.pin_prefix(prefix_b)

    assert cache.is_pinned(prefix_a)
    assert cache.is_pinned(prefix_b)

    print("PASS: Prefix A pinned.")
    print("PASS: Prefix B pinned.")

    print("\nInserting ordinary prefix C...")

    generator.generate_ids(
        prefix_c,
        suffix,
        max_new_tokens=2,
        use_cache=True,
    )

    print("Cache:", cache.stats)

    print("\nInserting ordinary prefix D...")

    generator.generate_ids(
        prefix_d,
        suffix,
        max_new_tokens=2,
        use_cache=True,
    )

    print("Cache:", cache.stats)

    # Both pinned entries must survive.
    assert cache.contains(prefix_a)
    assert cache.contains(prefix_b)

    assert cache.is_pinned(prefix_a)
    assert cache.is_pinned(prefix_b)

    print("\nPASS: Both pinned prefixes survived.")

    # Oldest ordinary entry C should have been evicted.
    assert not cache.contains(prefix_c)

    print("PASS: Old ordinary prefix C was evicted.")

    assert cache.contains(prefix_d)

    print("PASS: New ordinary prefix D was stored.")

    # Verify that the pinned KV states can still be retrieved.
    assert cache.get(prefix_a) is not None
    assert cache.get(prefix_b) is not None

    print("PASS: Both pinned KV states are retrievable.")

    print("\nFinal cache statistics:")
    print(cache.stats)

    print("\n================================")
    print("PASS: MULTIPLE PINNED PREFIX TEST")
    print("================================")


if __name__ == "__main__":
    main()
