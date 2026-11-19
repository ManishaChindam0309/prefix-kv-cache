# Prefix KV Cache
### An AI Systems Project for LLM Inference Optimization

**Prefix KV Cache** is a Python-based AI systems project that improves Transformer inference efficiency by reusing previously computed Key-Value (KV) states for repeated input prefixes.

Instead of recomputing the same prefix for every request, the system stores and reuses its KV states to reduce redundant computation and inference latency.

The project focuses on reusable prefix caching, pinned KV states, LRU eviction, memory-aware cache management, and concurrent inference.

---

## Project Overview

Large Language Models repeatedly process common context, such as system prompts and instructions. Recomputing this context for every request introduces unnecessary computation.

This project addresses that problem through a reusable prefix KV cache.

**Objective:** Reduce repeated Transformer computation and improve inference efficiency through intelligent KV cache management.

## Architecture

![Prefix KV Cache Architecture](prefix_kv_cache_architecture.png)

The system has two main components:

### 1. PrefixKVCache

Manages storage and lifecycle of prefix KV states.

Responsibilities:
- Store and retrieve KV states.
- Pin and unpin prefixes.
- Track cache hits and misses.
- Perform LRU eviction.
- Enforce entry-count and optional memory limits.
- Track memory usage and cache statistics.

### 2. CachedGenerator

Connects the prefix cache to Transformer inference.

Responsibilities:
- Compute prefix KV states.
- Retrieve reusable cached prefixes.
- Generate tokens using cached KV states.
- Support token-ID-based and tokenizer-based generation.

## Key Features

- **Prefix KV Caching:** Reuse previously computed attention Key-Value states.
- **Pinned Prefixes:** Protect frequently used prefixes from LRU eviction.
- **Multiple Pinned Prefixes:** Support multiple reusable prefixes simultaneously.
- **LRU Eviction:** Remove least-recently-used unpinned entries when capacity is reached.
- **Memory-Aware Eviction:** Enforce configurable KV memory budgets.
- **KV Memory Accounting:** Estimate tensor memory usage, including Hugging Face `DynamicCache` objects.
- **Concurrent Cache Access:** Synchronize cache operations for concurrent callers.
- **Cached Transformer Generation:** Reuse cached prefix states during text generation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Tensor operations and model inference |
| CUDA | GPU acceleration |
| Hugging Face Transformers | Transformer model integration |
| GPT-2 | Benchmark and validation model |
| ThreadPoolExecutor | Concurrent request testing |
| Pytest | Test framework |

## Repository Structure

```text
prefix-kv-cache/
│
├── prefix_kv_cache/
│   └── Core cache implementation
│
├── tests/
│   └── Additional test directory
│
├── benchmark.py
├── benchmark_lru_vs_pinned.py
├── benchmark_memory_scaling.py
├── benchmark_pinned_prefix.py
├── benchmark_prefix_lengths.py
├── benchmark_shared_prefix.py
│
├── test_cache_thread_safety.py
├── test_concurrent_requests.py
├── test_memory_aware_cache.py
├── test_multiple_pinned_prefixes.py
├── test_pinned_cache.py
├── test_pinned_prefix_memory.py
├── test_real_model_memory.py
│
├── demo.py
├── inspect_dynamic_cache.py
├── prefix_kv_cache_architecture.png
├── prefix_kv_cache_results.xlsx
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ManishaChindam0309/prefix-kv-cache.git

cd prefix-kv-cache
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Dependencies:

```text
torch>=2.6
transformers>=4.45,<5
pytest>=8.0
```

For GPU acceleration, install a PyTorch build compatible with your NVIDIA GPU and CUDA environment.

## Usage

### Initialize the cache

```python
from prefix_kv_cache import PrefixKVCache

cache = PrefixKVCache(
    max_entries=100,
    max_memory_mb=100
)
```

The `max_memory_mb` parameter is optional. It limits the estimated memory occupied by cached KV tensor payloads.

### Initialize CachedGenerator

After loading a compatible Transformer model and tokenizer:

```python
generator = CachedGenerator(
    model,
    tokenizer,
    cache
)
```

### Pin a reusable system prompt

```python
generator.pin_system_prompt(
    "You are a helpful assistant."
)
```

Pinned prefixes are protected from normal LRU eviction.

### Generate a response

```python
response = generator.generate(
    "You are a helpful assistant.",
    "Explain how Transformer attention works.",
    max_new_tokens=50,
    use_cache=True
)

print(response)
```

### Work with token IDs

```python
generator.pin_prefix(prefix_ids)

output_ids = generator.generate_ids(
    prefix_ids,
    suffix_ids,
    max_new_tokens=20,
    use_cache=True
)
```

### Unpin a prefix

```python
generator.unpin_prefix(prefix_ids)
```

Unpinned entries become eligible for normal LRU eviction.

### Monitor cache statistics

```python
print(cache.stats)
```

Statistics include:
- Total cache entries
- Pinned entries
- Ordinary entries
- Cache hits and misses
- Eviction count
- Cache hit rate
- Estimated memory usage
- Pinned memory usage
- Memory capacity and utilization

## Performance Evaluation

Experiments were conducted using GPT-2 on an NVIDIA GeForce RTX 2050 GPU with 4 GB VRAM.

### 1. Pinned Prefix Benchmark

| Prefix Length | Standard Latency | Pinned Latency | Speedup | Latency Reduction |
|---|---:|---:|---:|---:|
| 512 tokens | 155.05 ms | 93.40 ms | 1.66× | 39.76% |
| 768 tokens | 180.20 ms | 98.57 ms | 1.83× | 45.30% |

Prefix preparation time was measured separately:

- 512-token prefix: 58.82 ms
- 768-token prefix: 90.67 ms

Steady-state latency measurements exclude initial prefix preparation.

### 2. Shared-Prefix Benchmark

A workload of 10 questions sharing a common prefix was repeated over 10 rounds, with results averaged across three runs.

| Prefix Length | Standard Latency | Cached Latency | Speedup |
|---|---:|---:|---:|
| 512 tokens | 155.82 ms | 98.67 ms | 1.58× |
| 768 tokens | 177.09 ms | 96.81 ms | 1.83× |

Measured cache hit rate: **99.02%**

### 3. Natural-Language Workload

A separate benchmark used a 46-token system prompt, four competing prompts, and 48 requests per trial across three trials.

| Strategy | Mean Latency |
|---|---:|
| No Cache | 238.90 ms |
| Ordinary LRU | 168.41 ms |
| Pinned LRU | 157.71 ms |
| Pinned LRU including average setup time | 158.63 ms |

Pinned caching achieved a measured 33.60% latency reduction compared with no caching, including average pin preparation time.

### 4. KV Memory Accounting

A real GPT-2 KV cache was tested using Hugging Face Transformers `DynamicCache`.

| Measurement | Result |
|---|---:|
| Prefix length | 19 tokens |
| Tensor count | 24 |
| Independent memory measurement | 1.3359 MiB |
| Project memory estimate | 1.3359 MiB |

The measurements matched at the reported precision.

## Testing and Validation

The following tests passed in the latest recorded regression run:

| Test | Validation |
|---|---|
| `test_pinned_cache.py` | Pinning, unpinning, and LRU protection |
| `test_multiple_pinned_prefixes.py` | Multiple pinned prefix protection |
| `test_memory_aware_cache.py` | Memory limits and memory-based eviction |
| `test_real_model_memory.py` | Real GPT-2 KV memory accounting |
| `test_concurrent_requests.py` | Concurrent inference output correctness |
| `test_cache_thread_safety.py` | Concurrent cache operations and bounded capacity |

Run the regression tests:

```powershell
python test_pinned_cache.py
python test_multiple_pinned_prefixes.py
python test_memory_aware_cache.py
python test_real_model_memory.py
python test_concurrent_requests.py
python test_cache_thread_safety.py
```

**All six tests passed in the latest recorded regression run.**

## Hardware and Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| GPU Memory | 4 GB |
| PyTorch | 2.14.1+cu126 |
| CUDA Build | 12.6 |
| Transformers | 4.57.6 |
| Benchmark Model | GPT-2 |
| Operating System | Windows |

Performance may vary depending on hardware, model architecture, prefix length, workload, and GPU utilization.

## Limitations

- Cache storage is currently in-memory and does not persist across application restarts.
- Memory accounting estimates KV tensor payloads, not total GPU memory consumption.
- Performance experiments primarily use GPT-2 and controlled workloads.
- Prefix preparation introduces an initial computational cost.
- Concurrent inference validation has been performed on limited workloads.
- Broader testing across different Transformer architectures and production workloads remains future work.

## Future Improvements

- Persistent KV cache storage.
- Support for additional Transformer architectures.
- More extensive concurrent inference benchmarking.
- Cache monitoring and visualization.
- Workload-aware cache replacement strategies.
- Integration with larger LLM inference systems.

## Project Classification

**Domain:** AI Systems and AI Infrastructure

**Specialization:** LLM Inference Optimization

**Implementation:** Python

This project demonstrates practical concepts in Transformer inference, GPU memory management, cache design, performance optimization, and concurrent systems.

## Author

**Manisha Chindam**

GitHub: [ManishaChindam0309](https://github.com/ManishaChindam0309)

**Repository:** [Prefix KV Cache](https://github.com/ManishaChindam0309/prefix-kv-cache)

---

*Prefix KV Cache — Exploring efficient and reusable KV state management for Transformer inference.*
