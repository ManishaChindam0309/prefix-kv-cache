# Prefix KV Cache

Pin the KV states of a fixed prompt prefix (e.g. the system prompt) in a small
LRU cache, so each request only runs the model on its new tokens.

## How it works
1. Tokenize the system prompt and the user message separately (token-aligned boundary).
2. Hash the prefix tokens and look them up in `PrefixKVCache`.
3. **Hit:** load a copy of the stored KV states and run the model only on the user tokens.
4. **Miss:** compute the prefix once, store its KV states, then continue.
5. The cache is size-limited (default 100 entries, LRU eviction).

## Setup (VS Code)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then pick the `.venv` interpreter in VS Code (Ctrl/Cmd+Shift+P -> "Python: Select Interpreter").

## Run
```bash
pytest                              # correctness: cached output == uncached output
python benchmark.py --offline-random   # no download needed
python benchmark.py                 # real gpt2 from Hugging Face
python benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct
python demo.py                      # chat demo with Qwen2.5-0.5B-Instruct
```
Use `torch>=2.6` with `transformers<5` (pinned in requirements.txt); transformers 5.x is untested.
Or use the Run and Debug panel: launch configs are in `.vscode/launch.json`.

## Use it
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from prefix_kv_cache import CachedGenerator

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
gen = CachedGenerator(model, tok)
print(gen.generate("You are a helpful assistant.", "Hi!"))
```

## Notes
- `get()` returns a deep copy because HF caches grow in place during decoding.
- Only exact prefix matches hit. Keep the system prompt byte-identical between requests.
- Memory per entry = 2 x layers x heads x head_dim x prefix_len x dtype bytes; size `max_entries` accordingly.
- For production serving, vLLM (`enable_prefix_caching=True`) and SGLang provide this built in.
