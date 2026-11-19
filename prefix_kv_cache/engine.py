
"""Greedy generation with reusable and pinned prefix KV caching."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .cache import PrefixKVCache


class CachedGenerator:
    def __init__(
        self,
        model,
        tokenizer=None,
        cache: Optional[PrefixKVCache] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.cache = cache if cache is not None else PrefixKVCache()

        if eos_token_id is None and tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id

        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _decode(
        self,
        past,
        new_ids: Sequence[int],
        max_new_tokens: int
    ) -> List[int]:

        ids = torch.tensor([list(new_ids)], device=self.device)

        out = self.model(
            input_ids=ids,
            past_key_values=past,
            use_cache=True
        )

        past = out.past_key_values
        generated: List[int] = []

        for _ in range(max_new_tokens):
            nxt = int(out.logits[0, -1].argmax())
            generated.append(nxt)

            if self.eos_token_id is not None and nxt == self.eos_token_id:
                break

            step = torch.tensor([[nxt]], device=self.device)

            out = self.model(
                input_ids=step,
                past_key_values=past,
                use_cache=True
            )

            past = out.past_key_values

        return generated

    @torch.no_grad()
    def pin_prefix(self, prefix_ids: Sequence[int]) -> None:
        """Compute and permanently protect a prefix while it remains pinned."""
        prefix_ids = list(prefix_ids)

        if self.cache.contains(prefix_ids):
            self.cache.pin(prefix_ids)
            return

        ids = torch.tensor([prefix_ids], device=self.device)

        past = self.model(
            input_ids=ids,
            use_cache=True
        ).past_key_values

        self.cache.pin(prefix_ids, past)

    @torch.no_grad()
    def pin_system_prompt(self, system_prompt: str) -> None:
        """Tokenize and pin a text system prompt."""
        assert self.tokenizer is not None, "tokenizer required"

        prefix_ids = self.tokenizer(
            system_prompt,
            add_special_tokens=False
        ).input_ids

        self.pin_prefix(prefix_ids)

    def unpin_prefix(self, prefix_ids: Sequence[int]) -> bool:
        """Allow a previously pinned prefix to be evicted."""
        return self.cache.unpin(prefix_ids)

    @torch.no_grad()
    def generate_ids(
        self,
        prefix_ids: Sequence[int],
        suffix_ids: Sequence[int],
        max_new_tokens: int = 32,
        use_cache: bool = True
    ) -> List[int]:

        prefix_ids = list(prefix_ids)
        suffix_ids = list(suffix_ids)

        if not use_cache:
            return self._decode(
                None,
                prefix_ids + suffix_ids,
                max_new_tokens
            )

        hit = self.cache.get(prefix_ids)

        if hit is None:
            ids = torch.tensor([prefix_ids], device=self.device)

            past = self.model(
                input_ids=ids,
                use_cache=True
            ).past_key_values

            self.cache.put(prefix_ids, past)
            hit = (past, len(prefix_ids))

        past, _ = hit

        return self._decode(
            past,
            suffix_ids,
            max_new_tokens
        )

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 64,
        use_cache: bool = True
    ) -> str:

        assert self.tokenizer is not None, "tokenizer required"

        prefix_ids = self.tokenizer(
            system_prompt,
            add_special_tokens=False
        ).input_ids

        suffix_ids = self.tokenizer(
            user_message,
            add_special_tokens=False
        ).input_ids

        new = self.generate_ids(
            prefix_ids,
            suffix_ids,
            max_new_tokens,
            use_cache
        )

        return self.tokenizer.decode(
            new,
            skip_special_tokens=True
        )
