"""Greedy generation with and without a pinned prefix KV cache."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .cache import PrefixKVCache


class CachedGenerator:
    def __init__(self, model, tokenizer=None, cache: Optional[PrefixKVCache] = None,
                 eos_token_id: Optional[int] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.cache = cache if cache is not None else PrefixKVCache()
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device

    # ---- core, token-id level API (used by tests and benchmark) ----------
    @torch.no_grad()
    def _decode(self, past, new_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        ids = torch.tensor([list(new_ids)], device=self.device)
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        generated: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(out.logits[0, -1].argmax())
            generated.append(nxt)
            if self.eos_token_id is not None and nxt == self.eos_token_id:
                break
            step = torch.tensor([[nxt]], device=self.device)
            out = self.model(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
        return generated

    @torch.no_grad()
    def generate_ids(self, prefix_ids: Sequence[int], suffix_ids: Sequence[int],
                     max_new_tokens: int = 32, use_cache: bool = True) -> List[int]:
        prefix_ids = list(prefix_ids)
        if not use_cache:  # baseline: recompute everything every time
            return self._decode(None, prefix_ids + list(suffix_ids), max_new_tokens)

        hit = self.cache.get(prefix_ids)
        if hit is None:  # miss: compute the prefix once and store it
            ids = torch.tensor([prefix_ids], device=self.device)
            past = self.model(input_ids=ids, use_cache=True).past_key_values
            self.cache.put(prefix_ids, past)
            hit = (past, len(prefix_ids))  # `put` stored its own copy
        past, _ = hit
        return self._decode(past, suffix_ids, max_new_tokens)

    # ---- text-level API --------------------------------------------------
    def generate(self, system_prompt: str, user_message: str,
                 max_new_tokens: int = 64, use_cache: bool = True) -> str:
        assert self.tokenizer is not None, "tokenizer required for text API"
        # Tokenize separately so the prefix boundary is exactly token-aligned.
        prefix_ids = self.tokenizer(system_prompt, add_special_tokens=False).input_ids
        suffix_ids = self.tokenizer(user_message, add_special_tokens=False).input_ids
        new = self.generate_ids(prefix_ids, suffix_ids, max_new_tokens, use_cache)
        return self.tokenizer.decode(new, skip_special_tokens=True)
