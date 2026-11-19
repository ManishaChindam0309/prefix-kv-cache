
"""Natural-language prefix-length scaling benchmark."""

import math
import statistics
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prefix_kv_cache.cache import PrefixKVCache
from prefix_kv_cache.engine import CachedGenerator


MODEL_NAME = "gpt2"
PREFIX_LENGTHS = [32, 64, 128, 256, 512]
CACHE_CAPACITY = 4
ROUNDS = 8
TRIALS = 3
MAX_NEW_TOKENS = 8

HOT_QUESTIONS = [
    "What is a transformer model?",
    "Why do transformer models use attention?",
    "What is the purpose of a key-value cache?",
    "How does prefix reuse improve inference speed?",
    "What is the difference between training and inference?",
    "Why do long prompts increase inference latency?",
    "How does greedy decoding work?",
    "What is an autoregressive language model?",
]

COMPETING_QUESTIONS = [
    "Write a short professional email requesting a meeting.",
    "Explain what a Python dictionary is with an example.",
    "Explain how to calculate the average of five numbers.",
    "What is the difference between a planet and a star?",
]

PROMPT_CONTENT = {
    "hot": [
        "Explain technical concepts clearly and accurately.",
        "Use simple language when describing machine learning.",
        "Transformer models process sequences of tokens.",
        "Attention mechanisms help models use contextual information.",
        "A key-value cache stores intermediate attention states.",
        "Reusable prefix states can reduce repeated computation.",
        "Inference latency depends on prompt length and model size.",
        "Autoregressive models generate one token at a time.",
        "Greedy decoding selects the token with the highest score.",
        "Memory usage is an important inference constraint.",
        "GPU execution can accelerate neural network computation.",
        "Shared prompts can occur across many user requests.",
        "A cache can avoid recomputing an identical prefix.",
        "Cache capacity determines how many entries can remain.",
        "An eviction policy selects entries when space is limited.",
        "Pinned entries are protected from ordinary eviction.",
        "Correctness must be checked when reusing cached states.",
        "Performance should be measured across repeated requests.",
        "Latency measurements should include median and tail values.",
        "A realistic workload contains different user questions.",
    ],
    "writing": [
        "Help users prepare clear professional written communication.",
        "Use a polite, concise, and respectful tone.",
        "Preserve the original meaning of every request.",
        "Organize longer messages into readable paragraphs.",
        "Use appropriate greetings and closing sentences.",
        "Avoid unnecessary repetition and vague expressions.",
        "Adapt the writing style to the intended audience.",
    ],
    "python": [
        "Teach Python programming using understandable examples.",
        "Explain variables, functions, and data structures.",
        "Use readable code and meaningful variable names.",
        "Describe errors and explain how to fix them.",
        "Prefer simple examples before advanced techniques.",
        "Explain the purpose of each important code statement.",
    ],
    "math": [
        "Teach mathematics through careful step by step reasoning.",
        "Show important calculations and intermediate results.",
        "Explain formulas before applying them.",
        "Check calculations for mistakes.",
        "Use simple examples to explain abstract concepts.",
        "Keep the final answer clear and easy to understand.",
    ],
    "science": [
        "Explain scientific concepts using accurate information.",
        "Use accessible language and practical examples.",
        "Distinguish established facts from uncertain claims.",
        "Explain important terms when they first appear.",
        "Connect scientific ideas to everyday observations.",
        "Keep explanations structured and understandable.",
    ],
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def make_natural_prefix(tokenizer, role, target_length):
    """Build a natural-language prefix of exactly target_length tokens."""
    sentences = PROMPT_CONTENT[role]

    if role == "hot":
        heading = (
            "System instructions: You are a helpful technical assistant. "
            "Answer user questions accurately and clearly.\n"
        )
    else:
        headings = {
            "writing": "System instructions: You are a professional writing assistant.\n",
            "python": "System instructions: You are a Python programming tutor.\n",
            "math": "System instructions: You are a mathematics tutor.\n",
            "science": "System instructions: You are a science learning assistant.\n",
        }
        heading = headings[role]

    text = heading
    token_ids = encode(tokenizer, text)
    sentence_index = 0

    # Extend the prompt using readable natural-language guidance.
    # Stop before exceeding the target by a large amount.
    while len(token_ids) < target_length:
        sentence = sentences[sentence_index % len(sentences)]
        note_number = sentence_index + 1

        text += f"Guidance note {note_number}: {sentence}\n"
        token_ids = encode(tokenizer, text)
        sentence_index += 1

    return token_ids[:target_length]


def build_workload(tokenizer, prefix_length):
    hot_prefix = make_natural_prefix(
        tokenizer, "hot", prefix_length
    )

    competitor_roles = [
        "writing", "python", "math", "science"
    ]

    competitor_prefixes = [
        make_natural_prefix(tokenizer, role, prefix_length)
        for role in competitor_roles
    ]

    workload = []

    for round_index in range(ROUNDS):
        first_question = (
            "\nUser question: "
            + HOT_QUESTIONS[round_index]
            + "\nAssistant:"
        )

        workload.append({
            "prefix": hot_prefix,
            "suffix": encode(tokenizer, first_question),
        })

        # Four competing prefixes put pressure on the cache.
        for index in range(4):
            question = (
                "\nUser question: "
                + COMPETING_QUESTIONS[index]
                + "\nAssistant:"
            )

            workload.append({
                "prefix": competitor_prefixes[index],
                "suffix": encode(tokenizer, question),
            })

        second_question = (
            "\nUser question: "
            + HOT_QUESTIONS[(round_index + 1) % len(HOT_QUESTIONS)]
            + "\nAssistant:"
        )

        workload.append({
            "prefix": hot_prefix,
            "suffix": encode(tokenizer, second_question),
        })

    return hot_prefix, workload


def percentile(values, percentage):
    ordered = sorted(values)
    index = max(
        0,
        math.ceil(percentage / 100 * len(ordered)) - 1
    )
    return ordered[index]


def run_experiment(model, tokenizer, device, workload, mode):
    cache = PrefixKVCache(max_entries=CACHE_CAPACITY)
    generator = CachedGenerator(model, tokenizer, cache)

    pin_ms = 0.0

    if mode == "pinned":
        synchronize(device)
        start = time.perf_counter()

        generator.pin_prefix(workload[0]["prefix"])

        synchronize(device)
        pin_ms = (time.perf_counter() - start) * 1000

    latencies = []
    outputs = []

    for item in workload:
        synchronize(device)
        start = time.perf_counter()

        output = generator.generate_ids(
            item["prefix"],
            item["suffix"],
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=(mode != "no_cache"),
        )

        synchronize(device)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        outputs.append(output)

    return {
        "latencies": latencies,
        "outputs": outputs,
        "pin_ms": pin_ms,
        "stats": cache.stats,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPREFIX KV CACHE: PREFIX-LENGTH SCALING")
    print("=" * 60)
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Prefix lengths:", PREFIX_LENGTHS)
    print("Cache capacity:", CACHE_CAPACITY)
    print("Rounds:", ROUNDS)
    print("Trials:", TRIALS)
    print("Requests per trial:", ROUNDS * 6)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Warm up before measurement.
    print("\nWarming up...")
    warmup_cache = PrefixKVCache(max_entries=CACHE_CAPACITY)
    warmup_generator = CachedGenerator(
        model, tokenizer, warmup_cache
    )

    warmup_ids = encode(
        tokenizer,
        "You are a helpful assistant."
    )
    warmup_suffix = encode(
        tokenizer,
        "\nUser question: What is a transformer?\nAssistant:"
    )

    warmup_generator.generate_ids(
        warmup_ids,
        warmup_suffix,
        max_new_tokens=2,
        use_cache=False,
    )
    synchronize(device)

    all_results = {}

    for prefix_length in PREFIX_LENGTHS:
        print("\n" + "=" * 60)
        print(f"TESTING PREFIX LENGTH: {prefix_length} TOKENS")
        print("=" * 60)

        hot_prefix, workload = build_workload(
            tokenizer, prefix_length
        )

        actual_lengths = {
            len(item["prefix"]) for item in workload
        }

        if actual_lengths != {prefix_length}:
            raise AssertionError(
                f"Unexpected prefix lengths: {actual_lengths}"
            )

        max_input = max(
            len(item["prefix"]) + len(item["suffix"])
            for item in workload
        )

        if max_input + MAX_NEW_TOKENS > model.config.n_positions:
            raise ValueError(
                "Input plus generated tokens exceeds GPT-2 context."
            )

        print("Shared prefix length:", len(hot_prefix))
        print("Requests per trial:", len(workload))

        modes = ["no_cache", "lru", "pinned"]
        results_for_length = {
            mode: [] for mode in modes
        }

        for trial in range(TRIALS):
            # Rotate experiment order to reduce order bias.
            offset = trial % len(modes)
            order = modes[offset:] + modes[:offset]

            print(
                f"\nTrial {trial + 1}/{TRIALS}; order: {order}"
            )

            for mode in order:
                print("Running:", mode)

                result = run_experiment(
                    model,
                    tokenizer,
                    device,
                    workload,
                    mode,
                )

                results_for_length[mode].append(result)

        # Check exact token output equality within each trial.
        for trial in range(TRIALS):
            reference = results_for_length["no_cache"][trial]["outputs"]

            for mode in ["lru", "pinned"]:
                actual = results_for_length[mode][trial]["outputs"]

                if reference != actual:
                    raise AssertionError(
                        f"Output mismatch: length={prefix_length}, "
                        f"trial={trial + 1}, mode={mode}"
                    )

                print(
                    f"Correctness trial {trial + 1} {mode}: True"
                )

        all_results[prefix_length] = results_for_length

    # Summarize all prefix lengths.
    print("\n\nFINAL SCALING RESULTS")
    print("=" * 90)

    summary = {}

    for prefix_length in PREFIX_LENGTHS:
        results = all_results[prefix_length]
        summary[prefix_length] = {}

        print(f"\nPREFIX LENGTH: {prefix_length} TOKENS")
        print("-" * 75)

        for mode in ["no_cache", "lru", "pinned"]:
            trials = results[mode]

            latencies = [
                value
                for result in trials
                for value in result["latencies"]
            ]

            pin_costs = [
                result["pin_ms"] for result in trials
            ]

            mean_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = percentile(latencies, 95)
            average_pin = statistics.mean(pin_costs)

            inclusive = (
                sum(latencies) + sum(pin_costs)
            ) / len(latencies)

            summary[prefix_length][mode] = {
                "mean": mean_latency,
                "median": median_latency,
                "p95": p95_latency,
                "pin": average_pin,
                "inclusive": inclusive,
                "stats": trials[-1]["stats"],
            }

            print(f"\n{mode.upper()}")
            print(f"Mean:   {mean_latency:.2f} ms")
            print(f"Median: {median_latency:.2f} ms")
            print(f"P95:    {p95_latency:.2f} ms")

            if mode == "pinned":
                print(f"Pin preparation: {average_pin:.2f} ms")
                print(f"Including setup: {inclusive:.2f} ms")

            print("Cache stats:", trials[-1]["stats"])

        baseline = summary[prefix_length]["no_cache"]["mean"]
        lru_mean = summary[prefix_length]["lru"]["mean"]
        pinned_mean = summary[prefix_length]["pinned"]["mean"]
        pinned_inclusive = summary[prefix_length]["pinned"]["inclusive"]

        print("\nCOMPARISON")
        print(
            "LRU improvement vs no cache: "
            f"{(baseline - lru_mean) / baseline * 100:.2f}%"
        )
        print(
            "Pinned improvement vs no cache: "
            f"{(baseline - pinned_mean) / baseline * 100:.2f}%"
        )
        print(
            "Pinned improvement vs LRU: "
            f"{(lru_mean - pinned_mean) / lru_mean * 100:.2f}%"
        )
        print(
            "Pinned including setup vs no cache: "
            f"{(baseline - pinned_inclusive) / baseline * 100:.2f}%"
        )

    # Compact comparison table.
    print("\n\nSCALING SUMMARY")
    print("=" * 100)
    print(
        f"{'Prefix':>8} {'No cache':>12} {'LRU':>12} "
        f"{'Pinned':>12} {'Pinned+setup':>15} {'Pinned vs LRU':>15}"
    )

    for prefix_length in PREFIX_LENGTHS:
        row = summary[prefix_length]

        lru_mean = row["lru"]["mean"]
        pinned_mean = row["pinned"]["mean"]
        improvement = (
            (lru_mean - pinned_mean) / lru_mean * 100
        )

        print(
            f"{prefix_length:>8} "
            f"{row['no_cache']['mean']:>11.2f} "
            f"{lru_mean:>11.2f} "
            f"{pinned_mean:>11.2f} "
            f"{row['pinned']['inclusive']:>14.2f} "
            f"{improvement:>14.2f}%"
        )

    print("\nALL SCALING EXPERIMENTS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
