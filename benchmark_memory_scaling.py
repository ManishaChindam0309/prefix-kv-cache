
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prefix_kv_cache.cache import PrefixKVCache
from prefix_kv_cache.engine import CachedGenerator


MODEL_NAME = "gpt2"
DEVICE = "cuda"

PREFIX_LENGTHS = [32, 64, 128, 256, 512]
CACHE_CAPACITY = 10
MAX_NEW_TOKENS = 4


def memory_mb(value):
    return value / (1024 ** 2)


def get_memory():
    torch.cuda.synchronize()

    return {
        "allocated": memory_mb(torch.cuda.memory_allocated()),
        "reserved": memory_mb(torch.cuda.memory_reserved()),
        "peak": memory_mb(torch.cuda.max_memory_allocated()),
    }


def make_prefix(tokenizer, length, prefix_number):
    text = (
        f"This is reusable system instruction number {prefix_number}. "
        "You are a helpful assistant. Give accurate, clear and concise "
        "answers. Follow the user's instructions carefully. "
        "Explain important details and avoid unnecessary information. "
    )

    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Repeat natural-language tokens and truncate to the exact length.
    repetitions = (length // len(tokens)) + 1
    return (tokens * repetitions)[:length]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("\nPREFIX KV CACHE: MEMORY SCALING TEST")
    print("=" * 65)
    print("Device:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)
    print("Prefix lengths:", PREFIX_LENGTHS)
    print("Cache capacity:", CACHE_CAPACITY)

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    model.eval()

    cache = PrefixKVCache(max_entries=CACHE_CAPACITY)

    generator = CachedGenerator(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
    )

    # Warm up the model before measuring memory.
    print("\nWarming up model...")

    warmup_ids = tokenizer.encode(
        "Hello, how are you?",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        model(input_ids=warmup_ids, use_cache=True)

    torch.cuda.synchronize()

    del warmup_ids
    gc.collect()
    torch.cuda.empty_cache()

    baseline = get_memory()

    print("\nBASELINE GPU MEMORY")
    print("-" * 65)
    print(f"Allocated: {baseline['allocated']:.2f} MiB")
    print(f"Reserved:  {baseline['reserved']:.2f} MiB")

    results = []

    print("\nPINNING PREFIXES")
    print("=" * 65)

    for index, length in enumerate(PREFIX_LENGTHS, start=1):

        print(f"\nTesting prefix {index}: {length} tokens")

        prefix_ids = make_prefix(
            tokenizer,
            length,
            index
        )

        gc.collect()
        torch.cuda.synchronize()

        before = get_memory()

        torch.cuda.reset_peak_memory_stats()

        # Compute and pin the reusable prefix KV state.
        generator.pin_prefix(prefix_ids)

        gc.collect()
        torch.cuda.synchronize()

        after = get_memory()

        incremental = after["allocated"] - before["allocated"]
        peak_incremental = after["peak"] - before["allocated"]

        print(f"Prefix length:       {length} tokens")
        print(f"Memory before:       {before['allocated']:.2f} MiB")
        print(f"Memory after:        {after['allocated']:.2f} MiB")
        print(f"Additional memory:   {incremental:.2f} MiB")
        print(f"Peak increase:       {peak_incremental:.2f} MiB")
        print(f"Total allocated:     {after['allocated']:.2f} MiB")

        # Verify that the pinned prefix can be reused for generation.
        suffix_ids = tokenizer.encode(
            "Please answer briefly:",
            add_special_tokens=False
        )

        try:
            generated = generator.generate_ids(
                prefix_ids,
                suffix_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )

            generation_ok = True
            print("Generation test:    PASSED")
            print("Generated tokens:", len(generated))

        except Exception as error:
            generation_ok = False
            print("Generation test:    FAILED")
            print("Error:", str(error))

        stats = cache.stats

        results.append({
            "length": length,
            "incremental": incremental,
            "total": after["allocated"],
            "peak": peak_incremental,
            "generation_ok": generation_ok,
        })

        print("Pinned entries:", stats["pinned_entries"])
        print("Cache entries:", stats["entries"])

    print("\n\nFINAL MEMORY SCALING RESULTS")
    print("=" * 80)

    print(
        f"{'Prefix':>10} "
        f"{'Additional MiB':>18} "
        f"{'Total MiB':>15} "
        f"{'Peak increase':>16} "
        f"{'Generation':>14}"
    )

    print("-" * 80)

    for result in results:
        status = "PASS" if result["generation_ok"] else "FAIL"

        print(
            f"{result['length']:>10} "
            f"{result['incremental']:>18.2f} "
            f"{result['total']:>15.2f} "
            f"{result['peak']:>16.2f} "
            f"{status:>14}"
        )

    final_memory = get_memory()

    print("\nFINAL CACHE STATISTICS")
    print("-" * 65)
    print(cache.stats)

    print("\nFINAL GPU MEMORY")
    print("-" * 65)
    print(f"Allocated: {final_memory['allocated']:.2f} MiB")
    print(f"Reserved:  {final_memory['reserved']:.2f} MiB")

    print("\nMEMORY SCALING TEST COMPLETED.")


if __name__ == "__main__":
    main()
