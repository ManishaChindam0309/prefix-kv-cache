
import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from prefix_kv_cache.cache import (
    PrefixKVCache,
    estimate_kv_bytes,
)


def count_tensors(value, seen=None):
    """
    Independently count tensor payload bytes.

    Supports DynamicCache, DynamicLayer, tuples,
    lists, dictionaries and PyTorch tensors.
    """
    if seen is None:
        seen = set()

    object_id = id(value)

    if object_id in seen:
        return 0, 0

    seen.add(object_id)

    # PyTorch tensor
    if isinstance(value, torch.Tensor):
        return (
            1,
            value.numel() * value.element_size(),
        )

    # Dictionary
    if isinstance(value, dict):
        results = [
            count_tensors(v, seen)
            for v in value.values()
        ]

        return (
            sum(x[0] for x in results),
            sum(x[1] for x in results),
        )

    # Tuple or list
    if isinstance(value, (tuple, list)):
        results = [
            count_tensors(v, seen)
            for v in value
        ]

        return (
            sum(x[0] for x in results),
            sum(x[1] for x in results),
        )

    # Newer Hugging Face DynamicCache
    layers = getattr(value, "layers", None)

    if layers is not None:
        return count_tensors(layers, seen)

    # Older Hugging Face cache
    key_cache = getattr(value, "key_cache", None)
    value_cache = getattr(value, "value_cache", None)

    if key_cache is not None or value_cache is not None:
        a = count_tensors(key_cache, seen)
        b = count_tensors(value_cache, seen)

        return (
            a[0] + b[0],
            a[1] + b[1],
        )

    # Newer Hugging Face DynamicLayer
    keys = getattr(value, "keys", None)
    values = getattr(value, "values", None)

    if keys is not None or values is not None:
        a = count_tensors(keys, seen)
        b = count_tensors(values, seen)

        return (
            a[0] + b[0],
            a[1] + b[1],
        )

    return 0, 0


def main():
    print("\nREAL GPT-2 MEMORY ACCOUNTING TEST")

    print("Transformers version:", transformers.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading GPT-2...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    model.to(device)
    model.eval()

    prefix = (
        "You are a helpful assistant. "
        "Answer questions clearly and accurately. "
        "Explain technical topics step by step."
    )

    suffix = "What is machine learning?"

    prefix_ids = tokenizer(
        prefix,
        return_tensors="pt",
    ).input_ids.to(device)

    suffix_ids = tokenizer(
        suffix,
        return_tensors="pt",
    ).input_ids.to(device)

    print("Prefix tokens:", prefix_ids.shape[1])
    print("Suffix tokens:", suffix_ids.shape[1])

    print("\nComputing real KV cache...")

    with torch.no_grad():
        output = model(
            input_ids=prefix_ids,
            use_cache=True,
            return_dict=True,
        )

    kv = output.past_key_values

    print("KV cache type:", type(kv))

    # Independent memory measurement
    tensor_count, actual_bytes = count_tensors(kv)

    # Project memory estimator
    estimated_bytes = estimate_kv_bytes(kv)

    print("\nMEMORY ACCOUNTING")

    print("Tensor count:", tensor_count)

    print(
        "Independent memory (MiB):",
        round(actual_bytes / (1024 ** 2), 4),
    )

    print(
        "Project estimate (MiB):",
        round(estimated_bytes / (1024 ** 2), 4),
    )

    if tensor_count == 0:
        raise AssertionError(
            "No KV tensors were found."
        )

    if actual_bytes <= 0:
        raise AssertionError(
            "Actual KV memory is zero."
        )

    if estimated_bytes <= 0:
        raise AssertionError(
            "Project estimator returned zero."
        )

    if estimated_bytes != actual_bytes:
        raise AssertionError(
            "Project estimator does not match "
            "independent tensor accounting."
        )

    print("\nInserting KV cache into PrefixKVCache...")

    cache = PrefixKVCache(
        max_entries=10,
        max_memory_mb=100,
    )

    prefix_list = prefix_ids[0].tolist()

    cache.pin(prefix_list, kv)

    stats = cache.stats

    print("Cache statistics:", stats)

    assert stats["entries"] == 1
    assert stats["pinned_entries"] == 1
    assert stats["memory_bytes"] == actual_bytes

    print("\nRetrieving cached KV state...")

    retrieved = cache.get(prefix_list)

    assert retrieved is not None

    retrieved_kv, prefix_length = retrieved

    assert prefix_length == len(prefix_list)

    print("Retrieved prefix length:", prefix_length)
    print("Retrieved KV type:", type(retrieved_kv))

    print("\nTesting generation using retrieved KV...")

    with torch.no_grad():
        result = model(
            input_ids=suffix_ids,
            past_key_values=retrieved_kv,
            use_cache=True,
            return_dict=True,
        )

    assert result.logits.shape[1] == suffix_ids.shape[1]

    print("Generation test: PASSED")
    print("Logits shape:", tuple(result.logits.shape))

    print("\nFINAL CACHE STATISTICS")
    print(cache.stats)

    print("\nALL REAL GPT-2 MEMORY TESTS PASSED!")


if __name__ == "__main__":
    main()
