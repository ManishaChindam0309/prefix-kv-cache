
"""
GPU-aware benchmark for Prefix KV Cache.

Examples:
    python benchmark.py
    python benchmark.py --device cuda
    python benchmark.py --device cuda --runs 20 --new-tokens 8
    python benchmark.py --device cpu
    python benchmark.py --offline-random --device cuda

Compares:
    1. Standard inference (without prefix KV caching)
    2. Inference with reusable prefix KV caching

Measures:
    - Mean latency
    - Median latency
    - Speedup
    - Latency reduction
    - Cache hit rate
    - GPU memory usage
"""

import argparse
import random
import statistics
import time

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

from prefix_kv_cache import CachedGenerator, PrefixKVCache


SYSTEM = (
    "You are a helpful support assistant for Acme Corp. "
    "Follow the policy. "
) * 40

QUESTIONS = [
    "How do I reset my password?",
    "Where is my order?",
    "Can I get a refund?",
    "What are your opening hours?",
]


# --------------------------------------------------
# CUDA synchronization
# --------------------------------------------------

def synchronize(device):
    """
    CUDA operations are asynchronous.
    Synchronization ensures accurate timing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# Timing helper
# --------------------------------------------------

def timed(fn, device):
    synchronize(device)

    start = time.perf_counter()

    result = fn()

    synchronize(device)

    elapsed = time.perf_counter() - start

    return result, elapsed


# --------------------------------------------------
# Device selection
# --------------------------------------------------

def get_device(requested_device):

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. "
                "Check your CUDA-enabled PyTorch installation."
            )

        return torch.device("cuda")

    return torch.device("cpu")


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Benchmark Prefix KV Cache on CPU or GPU."
    )

    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model name."
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of measured requests per mode."
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=8,
        help="Number of tokens to generate."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device."
    )

    parser.add_argument(
        "--offline-random",
        action="store_true",
        help="Use a randomly initialized GPT-2 model without downloading."
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.new_tokens < 1:
        parser.error("--new-tokens must be at least 1")

    # --------------------------------------------------
    # Select device
    # --------------------------------------------------

    device = get_device(args.device)

    print("\n===================================")
    print(" PREFIX KV CACHE GPU BENCHMARK")
    print("===================================\n")

    print("PyTorch version:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("Selected device:", device)

    if device.type == "cuda":

        print("GPU:", torch.cuda.get_device_name(device))

        total_memory = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )

        print(f"GPU total memory: {total_memory:.2f} GB")

    else:
        print("GPU: Not being used")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading model...")

    if args.offline_random:

        torch.manual_seed(0)

        model = GPT2LMHeadModel(
            GPT2Config(
                n_layer=6,
                n_head=8,
                n_embd=512,
                vocab_size=5000,
            )
        )

        prefix_ids = [
            random.Random(42).randrange(5000)
            for _ in range(800)
        ]

        suffixes = [
            [random.Random(100 + i).randrange(5000) for _ in range(12)]
            for i in range(len(QUESTIONS))
        ]

        tokenizer = None

    else:

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        model = AutoModelForCausalLM.from_pretrained(
            args.model
        )

        prefix_ids = None
        suffixes = None

    # IMPORTANT: Move model to selected device.
    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")
    print("Model device:", next(model.parameters()).device)

    # --------------------------------------------------
    # Initialize cache and generator
    # --------------------------------------------------

    cache = PrefixKVCache(max_entries=50)

    if args.offline_random:

        generator = CachedGenerator(
            model,
            eos_token_id=None,
            cache=cache,
        )

        def run(use_cache, request_index):

            suffix = suffixes[
                request_index % len(suffixes)
            ]

            return generator.generate_ids(
                prefix_ids,
                suffix,
                args.new_tokens,
                use_cache,
            )

    else:

        generator = CachedGenerator(
            model,
            tokenizer,
            cache,
        )

        def run(use_cache, request_index):

            question = QUESTIONS[
                request_index % len(QUESTIONS)
            ]

            return generator.generate(
                SYSTEM,
                question,
                args.new_tokens,
                use_cache,
            )

    # --------------------------------------------------
    # Warm up the prefix cache
    # --------------------------------------------------

    print("\nWarming up prefix cache...")

    _, cold_time = timed(
        lambda: run(True, 0),
        device,
    )

    print(
        f"Initial prefix computation: "
        f"{cold_time * 1000:.2f} ms"
    )

    # Warm up a cache hit.
    timed(
        lambda: run(True, 0),
        device,
    )

    # Warm up uncached inference.
    timed(
        lambda: run(False, 0),
        device,
    )

    print("Warm-up completed.")

    # --------------------------------------------------
    # Reset peak GPU memory measurement
    # --------------------------------------------------

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    baseline_times = []
    cached_times = []

    matching_outputs = True

    print("\nRunning benchmark...")
    print(f"Requests per mode: {args.runs}")

    for i in range(args.runs):

        if i % 2 == 0:

            # Baseline first.
            baseline_output, baseline_time = timed(
                lambda: run(False, i),
                device,
            )

            cached_output, cached_time = timed(
                lambda: run(True, i),
                device,
            )

        else:

            # Cached first.
            cached_output, cached_time = timed(
                lambda: run(True, i),
                device,
            )

            baseline_output, baseline_time = timed(
                lambda: run(False, i),
                device,
            )

        baseline_times.append(baseline_time)
        cached_times.append(cached_time)

        if baseline_output != cached_output:
            matching_outputs = False

        print(
            f"Request {i + 1:02d}/{args.runs}: "
            f"Standard = {baseline_time * 1000:.2f} ms | "
            f"Cached = {cached_time * 1000:.2f} ms"
        )

    # --------------------------------------------------
    # Calculate results
    # --------------------------------------------------

    baseline_mean = statistics.mean(baseline_times)
    cached_mean = statistics.mean(cached_times)

    baseline_median = statistics.median(baseline_times)
    cached_median = statistics.median(cached_times)

    speedup = (
        baseline_mean / cached_mean
        if cached_mean > 0
        else 0
    )

    latency_reduction = (
        (1 - cached_mean / baseline_mean) * 100
        if baseline_mean > 0
        else 0
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n===================================")
    print(" BENCHMARK RESULTS")
    print("===================================\n")

    print("Execution device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    print("\nMean latency:")

    print(
        f"Without cache: "
        f"{baseline_mean * 1000:.2f} ms/request"
    )

    print(
        f"With KV cache: "
        f"{cached_mean * 1000:.2f} ms/request"
    )

    print("\nMedian latency:")

    print(
        f"Without cache: "
        f"{baseline_median * 1000:.2f} ms/request"
    )

    print(
        f"With KV cache: "
        f"{cached_median * 1000:.2f} ms/request"
    )

    print("\nPerformance:")

    print(f"Speedup: {speedup:.2f}x")

    print(
        f"Latency reduction: "
        f"{latency_reduction:.2f}%"
    )

    print("\nCache information:")

    print(
        f"Cache hit rate: "
        f"{generator.cache.hit_rate * 100:.2f}%"
    )

    print("Cache statistics:", generator.cache.stats)

    print("\nOutput correctness:")

    print(
        "Cached and uncached outputs match:",
        matching_outputs
    )

    # --------------------------------------------------
    # GPU memory
    # --------------------------------------------------

    if device.type == "cuda":

        peak_allocated = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(device)
            / (1024 ** 2)
        )

        current_allocated = (
            torch.cuda.memory_allocated(device)
            / (1024 ** 2)
        )

        print("\nGPU memory usage:")

        print(
            f"Peak allocated: "
            f"{peak_allocated:.2f} MiB"
        )

        print(
            f"Peak reserved: "
            f"{peak_reserved:.2f} MiB"
        )

        print(
            f"Current allocated: "
            f"{current_allocated:.2f} MiB"
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n===================================")
    print(" BENCHMARK SUMMARY")
    print("===================================\n")

    if matching_outputs:
        print("PASS: Cached and uncached outputs match.")
    else:
        print(
            "WARNING: Some cached and uncached outputs differ."
        )

    print(f"Device: {device}")
    print(f"Runs per mode: {args.runs}")
    print(f"Generated tokens: {args.new_tokens}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Latency reduction: {latency_reduction:.2f}%")

    print(
        "\nNote: These are measurements from one benchmark run. "
        "Repeat the experiment in fresh processes and across "
        "different workloads before making general performance claims."
    )


if __name__ == "__main__":
    main()
