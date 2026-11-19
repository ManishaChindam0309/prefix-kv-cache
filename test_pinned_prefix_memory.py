
import gc
import torch
from transformers import AutoModelForCausalLM

from prefix_kv_cache import PrefixKVCache, CachedGenerator


def gpu_memory(label):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    free, total = torch.cuda.mem_get_info()

    print(f"\n{label}")
    print(f"GPU allocated: {allocated:.2f} MiB")
    print(f"GPU reserved:  {reserved:.2f} MiB")
    print(f"GPU free:      {free / (1024 ** 2):.2f} MiB")
    print(f"GPU total:     {total / (1024 ** 2):.2f} MiB")


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    device = torch.device("cuda")

    print("Loading GPT-2 pretrained model...")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()

    cache = PrefixKVCache(max_entries=10)

    generator = CachedGenerator(
        model,
        eos_token_id=None,
        cache=cache
    )

    gc.collect()
    torch.cuda.empty_cache()

    gpu_memory("BASELINE: Model loaded, no prefixes pinned")

    prefix_length = 512
    number_of_prefixes = 5

    suffix = [100, 200, 300]

    prefixes = []

    # Create five different 512-token prefixes.
    for i in range(number_of_prefixes):
        prefix = [
            1000 + i * 1000 + j
            for j in range(prefix_length)
        ]
        prefixes.append(prefix)

    print("\nStarting multiple pinned-prefix memory test")

    successful_pins = 0

    for i, prefix in enumerate(prefixes):

        print(f"\nPinning prefix {i + 1}...")

        try:
            generator.pin_prefix(prefix)

            successful_pins += 1

            assert cache.is_pinned(prefix)

            print(f"PASS: Prefix {i + 1} pinned successfully.")

            gpu_memory(
                f"GPU memory after {successful_pins} pinned prefixes"
            )

            print("Cache statistics:", cache.stats)

        except torch.cuda.OutOfMemoryError:
            print("\nGPU OUT OF MEMORY!")
            print(f"Could not pin prefix {i + 1}.")

            gc.collect()
            torch.cuda.empty_cache()
            break

    print("\n===================================")
    print("PINNED PREFIX GENERATION TEST")
    print("===================================")

    for i, prefix in enumerate(prefixes[:successful_pins]):

        try:
            output = generator.generate_ids(
                prefix,
                suffix,
                max_new_tokens=2,
                use_cache=True
            )

            assert len(output) > 0
            assert cache.is_pinned(prefix)

            print(
                f"PASS: Prefix {i + 1} generated successfully. "
                f"Output tokens: {output}"
            )

        except Exception as error:
            print(f"FAIL: Prefix {i + 1}: {error}")

    gpu_memory("FINAL GPU MEMORY")

    print("\nFinal cache statistics:")
    print(cache.stats)

    print("\n===================================")
    print(f"Successfully pinned: {successful_pins}")
    print(f"Cache capacity: {cache.max_entries}")
    print("===================================")


if __name__ == "__main__":
    main()
