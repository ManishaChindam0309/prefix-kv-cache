
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Transformers version:", transformers.__version__)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

inputs = tokenizer(
    "This is a test prefix for inspecting KV cache.",
    return_tensors="pt"
)

with torch.no_grad():
    output = model(**inputs, use_cache=True, return_dict=True)

cache = output.past_key_values

print("\nCache type:", type(cache))
print("Cache attributes:", [
    name for name in dir(cache)
    if not name.startswith("__")
])

print("\nCache object dictionary:")
print(getattr(cache, "__dict__", "No __dict__"))

if hasattr(cache, "layers"):
    print("\nNumber of layers:", len(cache.layers))

    for i, layer in enumerate(cache.layers[:2]):
        print(f"\nLayer {i} type:", type(layer))
        print("Layer attributes:", [
            name for name in dir(layer)
            if not name.startswith("__")
        ])

        for name in ("keys", "values", "key_cache", "value_cache"):
            value = getattr(layer, name, None)

            if isinstance(value, torch.Tensor):
                print(
                    name,
                    "shape:", tuple(value.shape),
                    "bytes:", value.numel() * value.element_size()
                )
            else:
                print(name, "type:", type(value))

if hasattr(cache, "to_legacy_cache"):
    legacy = cache.to_legacy_cache()
    print("\nLegacy cache type:", type(legacy))
    print("Legacy layers:", len(legacy))

    if legacy:
        print("First layer tensor shapes:", [
            tuple(t.shape) for t in legacy[0]
            if isinstance(t, torch.Tensor)
        ])
