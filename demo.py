"""Chat demo with a real instruct model: same system prompt, many questions.

  python demo.py
  python demo.py --model Qwen/Qwen2.5-0.5B-Instruct --new-tokens 40

Prints each answer plus the time with and without the prefix KV cache.
"""
import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from prefix_kv_cache import CachedGenerator, PrefixKVCache

SYSTEM = (
    "You are a concise support assistant for Acme Corp. "
    "Answer in one or two sentences. Policies: refunds within 30 days with receipt; "
    "support hours 9am-5pm Monday to Friday; passwords reset via the login page. "
) * 6  # repeated so the prefix is long enough for the cache to matter

QUESTIONS = [
    "How do I reset my password?",
    "Can I get a refund after 3 weeks?",
    "When is support open?",
    "Do I need a receipt for a refund?",
]


def split_chat_prompt(tok, system, user):
    """Return (prefix_text, suffix_text) so prefix_text is a true string prefix."""
    prefix = tok.apply_chat_template(
        [{"role": "system", "content": system}], tokenize=False)
    full = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    assert full.startswith(prefix), "chat template prefix is not stable for this model"
    return prefix, full[len(prefix):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--new-tokens", type=int, default=40)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model)
    gen = CachedGenerator(model, tok, PrefixKVCache(50))

    for q in QUESTIONS:
        prefix, suffix = split_chat_prompt(tok, SYSTEM, q)
        t0 = time.perf_counter()
        slow = gen.generate(prefix, suffix, a.new_tokens, use_cache=False)
        t_slow = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast = gen.generate(prefix, suffix, a.new_tokens, use_cache=True)
        t_fast = time.perf_counter() - t0
        print(f"\nQ: {q}\nA: {fast.strip()}")
        print(f"   no cache {t_slow*1000:6.0f} ms | cache {t_fast*1000:6.0f} ms"
              f"{'  (first request: prefix computed + stored)' if q == QUESTIONS[0] else ''}")
        if slow != fast:
            print("   WARNING: cached and uncached outputs differ")

    print(f"\ncache hit rate: {gen.cache.hit_rate:.0%}")


if __name__ == "__main__":
    main()
