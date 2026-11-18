"""Measure latency with and without the prefix KV cache.

  python benchmark.py                         # downloads gpt2 from Hugging Face
  python benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct
  python benchmark.py --offline-random        # no download, random tiny model
"""
import argparse
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from prefix_kv_cache import CachedGenerator, PrefixKVCache

SYSTEM = "You are a helpful support assistant for Acme Corp. Follow the policy. " * 40
QUESTIONS = ["How do I reset my password?", "Where is my order?",
             "Can I get a refund?", "What are your opening hours?"]


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--offline-random", action="store_true")
    a = ap.parse_args()

    if a.offline_random:
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(n_layer=6, n_head=8, n_embd=512, vocab_size=5000))
        gen = CachedGenerator(model, eos_token_id=None, cache=PrefixKVCache(50))
        prefix = [random.randrange(5000) for _ in range(800)]

        def run(use_cache):
            suffix = [random.randrange(5000) for _ in range(12)]
            return gen.generate_ids(prefix, suffix, a.new_tokens, use_cache)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model)
        gen = CachedGenerator(model, tok, PrefixKVCache(50))

        def run(use_cache):
            return gen.generate(SYSTEM, random.choice(QUESTIONS), a.new_tokens, use_cache)

    run(True)  # warm the cache (first request is always a miss)
    base = [timed(lambda: run(False))[1] for _ in range(a.runs)]
    fast = [timed(lambda: run(True))[1] for _ in range(a.runs)]
    b, f = sum(base) / a.runs, sum(fast) / a.runs
    print(f"no cache : {b*1000:8.1f} ms/request")
    print(f"KV cache : {f*1000:8.1f} ms/request")
    print(f"speedup  : {b/f:5.2f}x  ({(1-f/b)*100:.0f}% lower latency)")
    print(f"hit rate : {gen.cache.hit_rate:.0%}")


if __name__ == "__main__":
    main()
