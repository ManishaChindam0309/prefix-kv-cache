
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


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    synchronize(device)
    start = time.perf_counter()

    result = fn()

    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, result


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
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

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


def make_prefix(tokenizer, length, vocab_size):
    if tokenizer is not None:
        base = tokenizer.encode(
            "This is a reusable system instruction for a language model. "
            "It contains context that can be shared across many requests.",
            add_special_tokens=False,
        )
    else:
        base = list(range(2, min(vocab_size, 500)))

    if not base:
        raise RuntimeError(
            "Could not create prefix tokens."
        )

    repetitions = (length // len(base)) + 1

    return (base * repetitions)[:length]


def make_suffix(tokenizer):
    if tokenizer is not None:
        ids = tokenizer.encode(
            "What is the main idea?",
            add_special_tokens=False,
        )
    else:
        ids = [10, 20, 30, 40, 50, 60]

    return ids[:8]


def run_length_test(
    generator,
    prefix_ids,
    suffix_ids,
    new_tokens,
    runs,
    device,
):
    # First call computes and stores the prefix KV cache.
    warmup_ms, _ = timed_call(
        lambda: generator.generate_ids(
            prefix_ids,
            suffix_ids,
            max_new_tokens=new_tokens,
            use_cache=True,
        ),
        device,
    )

    # Warm up the uncached path too.
    generator.generate_ids(
        prefix_ids,
        suffix_ids,
        max_new_tokens=new_tokens,
        use_cache=False,
    )

    standard_times = []
    cached_times = []
    correctness = True

    for i in range(runs):

        # Alternate order to reduce ordering bias.
        if i % 2 == 0:

            standard_ms, standard_output = timed_call(
                lambda: generator.generate_ids(
                    prefix_ids,
                    suffix_ids,
                    max_new_tokens=new_tokens,
                    use_cache=False,
                ),
                device,
            )

            cached_ms, cached_output = timed_call(
                lambda: generator.generate_ids(
                    prefix_ids,
                    suffix_ids,
                    max_new_tokens=new_tokens,
                    use_cache=True,
                ),
                device,
            )

        else:

            cached_ms, cached_output = timed_call(
                lambda: generator.generate_ids(
                    prefix_ids,
                    suffix_ids,
                    max_new_tokens=new_tokens,
                    use_cache=True,
                ),
                device,
            )

            standard_ms, standard_output = timed_call(
                lambda: generator.generate_ids(
                    prefix_ids,
                    suffix_ids,
                    max_new_tokens=new_tokens,
                    use_cache=False,
                ),
                device,
            )

        standard_times.append(standard_ms)
        cached_times.append(cached_ms)

        correctness = (
            correctness
            and outputs_equal(
                standard_output,
                cached_output,
            )
        )

        print(
            f"  Run {i + 1:02d}/{runs}: "
            f"Standard = {standard_ms:.2f} ms | "
            f"Cached = {cached_ms:.2f} ms"
        )

    standard_mean = statistics.mean(standard_times)
    cached_mean = statistics.mean(cached_times)

    speedup = standard_mean / cached_mean

    reduction = (
        (standard_mean - cached_mean)
        / standard_mean
    ) * 100

    return {
        "warmup_ms": warmup_ms,
        "standard_mean": standard_mean,
        "cached_mean": cached_mean,
        "standard_median": statistics.median(standard_times),
        "cached_median": statistics.median(cached_times),
        "speedup": speedup,
        "reduction": reduction,
        "correctness": correctness,
        "stats": generator.cache.stats,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Prefix KV Cache benchmark across prefix lengths"
    )

    parser.add_argument(
        "--model",
        default="gpt2",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--prefix-lengths",
        default="128,256,512,768",
    )

    parser.add_argument(
        "--offline-random",
        action="store_true",
    )

    args = parser.parse_args()

    if args.runs < 1 or args.new_tokens < 1:
        raise ValueError(
            "Runs and new tokens must be positive."
        )

    lengths = [
        int(x.strip())
        for x in args.prefix_lengths.split(",")
        if x.strip()
    ]

    if not lengths or any(x < 1 for x in lengths):
        raise ValueError(
            "Prefix lengths must be positive integers."
        )

    if len(set(lengths)) != len(lengths):
        raise ValueError(
            "Prefix lengths must not contain duplicates."
        )

    device = get_device(args.device)

    print("\n======================================")
    print(" PREFIX KV CACHE PREFIX-LENGTH TEST")
    print("======================================")

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

        total_memory = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )

        print(
            "GPU memory:",
            round(total_memory, 2),
            "GB",
        )

    print("\nLoading model...")

    model, tokenizer = load_model(args, device)

    print("Model loaded.")
    print(
        "Model device:",
        next(model.parameters()).device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    vocab_size = model.config.vocab_size

    max_positions = getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "n_positions", None),
    )

    suffix_ids = make_suffix(tokenizer)

    available_prefix = None

    if max_positions is not None:
        available_prefix = (
            max_positions
            - len(suffix_ids)
            - args.new_tokens
        )

    print("\nModel context limit:", max_positions)
    print("Suffix tokens:", len(suffix_ids))
    print("Generated tokens:", args.new_tokens)

    results = []

    for length in lengths:

        print("\n--------------------------------------")
        print(f"Testing prefix length: {length} tokens")
        print("--------------------------------------")

        if (
            available_prefix is not None
            and length > available_prefix
        ):
            print(
                "SKIPPED: Prefix exceeds the available "
                f"context budget of {available_prefix} tokens."
            )
            continue

        prefix_ids = make_prefix(
            tokenizer,
            length,
            vocab_size,
        )

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

        result = run_length_test(
            generator,
            prefix_ids,
            suffix_ids,
            args.new_tokens,
            args.runs,
            device,
        )

        result["length"] = length
        results.append(result)

        print("\nResults for", length, "tokens:")

        print(
            f"  Initial cache computation: "
            f"{result['warmup_ms']:.2f} ms"
        )

        print(
            f"  Standard mean: "
            f"{result['standard_mean']:.2f} ms"
        )

        print(
            f"  Cached mean: "
            f"{result['cached_mean']:.2f} ms"
        )

        print(
            f"  Speedup: "
            f"{result['speedup']:.2f}x"
        )

        print(
            f"  Latency reduction: "
            f"{result['reduction']:.2f}%"
        )

        print(
            f"  Correctness: "
            f"{result['correctness']}"
        )

        print(
            f"  Cache statistics: "
            f"{result['stats']}"
        )

    print("\n======================================")
    print(" FINAL PREFIX-LENGTH COMPARISON")
    print("======================================")

    print(
        f"{'Prefix':>10} | "
        f"{'Standard':>12} | "
        f"{'Cached':>12} | "
        f"{'Speedup':>9} | "
        f"{'Reduction':>10}"
    )

    print("-" * 68)

    for result in results:

        print(
            f"{result['length']:>10} | "
            f"{result['standard_mean']:>9.2f} ms | "
            f"{result['cached_mean']:>9.2f} ms | "
            f"{result['speedup']:>8.2f}x | "
            f"{result['reduction']:>9.2f}%"
        )

    if results:

        all_correct = all(
            result["correctness"]
            for result in results
        )

        print("\nOverall correctness:", all_correct)

        if all_correct:
            print("PASS: All tested outputs match.")
        else:
            print(
                "WARNING: At least one output did not match."
            )

    else:
        print("\nNo prefix lengths were tested.")

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


if __name__ == "__main__":
    main()
