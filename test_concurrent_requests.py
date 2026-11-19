
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM

from prefix_kv_cache import PrefixKVCache, CachedGenerator


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    device = torch.device("cuda")

    print("Loading GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()

    cache = PrefixKVCache(max_entries=10)

    generator = CachedGenerator(
        model,
        eos_token_id=None,
        cache=cache
    )

    # One shared system prefix.
    prefix = [100 + i for i in range(128)]

    # Four different request suffixes.
    suffixes = [
        [200, 201, 202],
        [300, 301, 302],
        [400, 401, 402],
        [500, 501, 502],
    ]

    print("\nPinning shared prefix...")

    generator.pin_prefix(prefix)

    assert cache.is_pinned(prefix)

    print("PASS: Shared prefix pinned.")

    # Generate reference outputs sequentially.
    print("\nGenerating reference outputs sequentially...")

    expected_outputs = []

    for i, suffix in enumerate(suffixes):
        output = generator.generate_ids(
            prefix,
            suffix,
            max_new_tokens=4,
            use_cache=True
        )

        expected_outputs.append(output)

        print(f"Reference request {i + 1}: {output}")

    assert cache.is_pinned(prefix)

    # Run the same requests concurrently.
    print("\nRunning 4 requests with 2 worker threads...")

    concurrent_outputs = [None] * len(suffixes)

    def run_request(index):
        output = generator.generate_ids(
            prefix,
            suffixes[index],
            max_new_tokens=4,
            use_cache=True
        )
        return index, output

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_request, i)
            for i in range(len(suffixes))
        ]

        for future in as_completed(futures):
            index, output = future.result()
            concurrent_outputs[index] = output

            print(
                f"Concurrent request {index + 1}: {output}"
            )

    # Compare sequential and concurrent outputs.
    print("\nChecking output correctness...")

    all_match = True

    for i in range(len(suffixes)):
        matches = (
            expected_outputs[i] == concurrent_outputs[i]
        )

        print(f"Request {i + 1} matches: {matches}")

        if not matches:
            all_match = False

    assert all_match, "Concurrent outputs differ from reference outputs!"

    print("PASS: All concurrent outputs match.")

    # Check cache integrity.
    assert cache.is_pinned(prefix)
    assert cache.contains(prefix)

    print("PASS: Shared pinned prefix survived.")
    print("PASS: Cache entry remains available.")

    print("\nFinal cache statistics:")
    print(cache.stats)

    print("\n===================================")
    print("CONCURRENT REQUEST TEST PASSED")
    print("===================================")

    gc.collect()
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"GPU allocated memory: {allocated:.2f} MiB")


if __name__ == "__main__":
    main()
