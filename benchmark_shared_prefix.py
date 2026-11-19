
import argparse
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


QUESTIONS = [
    "What is artificial intelligence?",
    "Explain machine learning in simple words.",
    "What is deep learning?",
    "How do neural networks work?",
    "What are the applications of NLP?",
    "Explain the difference between AI and ML.",
    "What is a transformer model?",
    "Why is model inference expensive?",
    "What is attention in a transformer?",
    "How does KV caching improve performance?",
]


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer questions clearly and accurately.
Explain technical concepts using simple language.
Provide useful examples whenever appropriate.
"""


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    synchronize(device)
    start = time.perf_counter()

    output = fn()

    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, output


def outputs_equal(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)

    return a == b


def get_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    return device


def load_model(args, device):
    if args.offline_random:
        config = GPT2Config(
            vocab_size=5000,
            n_positions=2048,
            n_ctx=2048,
            n_embd=512,
            n_layer=6,
            n_head=8,
            bos_token_id=0,
            eos_token_id=1,
        )

        model = GPT2LMHeadModel(config)
        tokenizer = None

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)

    model.to(device)
    model.eval()

    return model, tokenizer


def create_prefix(tokenizer, length, vocab_size):
    if tokenizer is not None:
        base = tokenizer.encode(
            SYSTEM_PROMPT,
            add_special_tokens=False,
        )
    else:
        base = list(range(2, min(vocab_size, 500)))

    if not base:
        raise RuntimeError("Prefix tokenization returned no tokens.")

    repetitions = (length // len(base)) + 1

    return (base * repetitions)[:length]


def create_questions(tokenizer):
    suffixes = []

    for index, question in enumerate(QUESTIONS):
        if tokenizer is not None:
            ids = tokenizer.encode(
                question,
                add_special_tokens=False,
            )
        else:
            # Different deterministic token sequences for each question.
            ids = [
                10 + index * 10,
                20 + index * 10,
                30 + index * 10,
                40 + index * 10,
                50 + index * 10,
            ]

        if not ids:
            raise RuntimeError(
                f"Question {index + 1} produced no tokens."
            )

        suffixes.append(ids)

    return suffixes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple questions sharing one cached prefix"
    )

    parser.add_argument("--model", default="gpt2")

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=512)

    parser.add_argument(
        "--offline-random",
        action="store_true",
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("Runs must be positive.")

    if args.new_tokens < 1 or args.prefix_length < 1:
        raise ValueError("Token counts must be positive.")

    device = get_device(args.device)

    print("\n==========================================")
    print(" PREFIX KV CACHE SHARED-PREFIX BENCHMARK")
    print("==========================================")

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

        memory = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )

        print("GPU memory:", round(memory, 2), "GB")

    print("\nLoading model...")

    model, tokenizer = load_model(args, device)

    print("Model loaded.")
    print("Model device:", next(model.parameters()).device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prefix_ids = create_prefix(
        tokenizer,
        args.prefix_length,
        model.config.vocab_size,
    )

    suffixes = create_questions(tokenizer)

    max_positions = getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "n_positions", None),
    )

    if max_positions is not None:
        longest_suffix = max(len(x) for x in suffixes)

        required = (
            len(prefix_ids)
            + longest_suffix
            + args.new_tokens
        )

        if required > max_positions:
            raise ValueError(
                f"Prompt requires {required} positions, "
                f"but model supports only {max_positions}. "
                "Reduce prefix length or generated tokens."
            )

    print("\nBenchmark configuration:")
    print("Shared prefix tokens:", len(prefix_ids))
    print("Different questions:", len(suffixes))
    print("Requests per question:", args.runs)
    print("Total measured requests:", len(suffixes) * args.runs)
    print("Generated tokens:", args.new_tokens)

    cache = PrefixKVCache(max_entries=50)

    if tokenizer is not None:
        generator = CachedGenerator(
            model,
            tokenizer,
            cache,
        )
    else:
        generator = CachedGenerator(
            model,
            eos_token_id=None,
            cache=cache,
        )

    # Warm up uncached generation for all questions.
    print("\nWarming up standard generation...")

    for suffix in suffixes:
        generator.generate_ids(
            prefix_ids,
            suffix,
            max_new_tokens=args.new_tokens,
            use_cache=False,
        )

    # First cached request computes and stores the shared prefix.
    print("\nComputing shared prefix cache...")

    cold_ms, first_output = timed_call(
        lambda: generator.generate_ids(
            prefix_ids,
            suffixes[0],
            max_new_tokens=args.new_tokens,
            use_cache=True,
        ),
        device,
    )

    print(f"Initial prefix-cache request: {cold_ms:.2f} ms")

    # Warm up cached generation.
    generator.generate_ids(
        prefix_ids,
        suffixes[0],
        max_new_tokens=args.new_tokens,
        use_cache=True,
    )

    standard_times = []
    cached_times = []
    correctness = True

    print("\nRunning shared-prefix benchmark...")

    for run in range(args.runs):
        print(f"\nRound {run + 1}/{args.runs}")

        for index, suffix in enumerate(suffixes):

            if (run + index) % 2 == 0:

                standard_ms, standard_output = timed_call(
                    lambda: generator.generate_ids(
                        prefix_ids,
                        suffix,
                        max_new_tokens=args.new_tokens,
                        use_cache=False,
                    ),
                    device,
                )

                cached_ms, cached_output = timed_call(
                    lambda: generator.generate_ids(
                        prefix_ids,
                        suffix,
                        max_new_tokens=args.new_tokens,
                        use_cache=True,
                    ),
                    device,
                )

            else:

                cached_ms, cached_output = timed_call(
                    lambda: generator.generate_ids(
                        prefix_ids,
                        suffix,
                        max_new_tokens=args.new_tokens,
                        use_cache=True,
                    ),
                    device,
                )

                standard_ms, standard_output = timed_call(
                    lambda: generator.generate_ids(
                        prefix_ids,
                        suffix,
                        max_new_tokens=args.new_tokens,
                        use_cache=False,
                    ),
                    device,
                )

            standard_times.append(standard_ms)
            cached_times.append(cached_ms)

            matches = outputs_equal(
                standard_output,
                cached_output,
            )

            correctness = correctness and matches

            print(
                f"  Question {index + 1:02d}: "
                f"Standard = {standard_ms:.2f} ms | "
                f"Cached = {cached_ms:.2f} ms | "
                f"Match = {matches}"
            )

    standard_mean = statistics.mean(standard_times)
    cached_mean = statistics.mean(cached_times)

    standard_median = statistics.median(standard_times)
    cached_median = statistics.median(cached_times)

    speedup = standard_mean / cached_mean

    reduction = (
        (standard_mean - cached_mean)
        / standard_mean
    ) * 100

    print("\n==========================================")
    print(" SHARED-PREFIX BENCHMARK RESULTS")
    print("==========================================")

    print("\nLatency:")
    print(f"Without cache: {standard_mean:.2f} ms/request")
    print(f"With cache:    {cached_mean:.2f} ms/request")

    print("\nMedian latency:")
    print(f"Without cache: {standard_median:.2f} ms")
    print(f"With cache:    {cached_median:.2f} ms")

    print("\nPerformance:")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Latency reduction: {reduction:.2f}%")

    print("\nCache statistics:")
    print("Hit rate:", f"{cache.hit_rate * 100:.2f}%")
    print("Statistics:", cache.stats)

    print("\nCorrectness:")
    print("All cached outputs match:", correctness)

    if device.type == "cuda":
        print("\nGPU memory:")
        print(
            "Peak allocated:",
            round(
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2),
                2,
            ),
            "MiB",
        )

        print(
            "Peak reserved:",
            round(
                torch.cuda.max_memory_reserved(device)
                / (1024 ** 2),
                2,
            ),
            "MiB",
        )

    print("\n==========================================")

    if correctness:
        print("PASS: Shared-prefix correctness verified.")
    else:
        print("WARNING: Output mismatch detected.")

    print("Questions:", len(suffixes))
    print("Rounds:", args.runs)
    print("Total measured requests:", len(standard_times))
    print("Speedup:", f"{speedup:.2f}x")
    print("==========================================")


if __name__ == "__main__":
    main()
