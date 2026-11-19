# Prefix KV Cache

**Prefix KV Cache** is a Python-based AI systems project that improves Transformer language-model inference efficiency by reusing previously computed Key-Value (KV) states for repeated prompt prefixes.

When multiple requests begin with the same system prompt or shared context, the system reuses the cached KV states instead of recomputing the prefix. This reduces redundant prefill computation and can lower inference latency for repeated-prefix workloads.

The project includes prefix KV caching, pinned system prompts, LRU eviction, memory-aware cache management, cached-versus-uncached inference comparison, benchmarking, and an interactive Streamlit dashboard.

---

## Overview

Large language models repeatedly process common context, such as system prompts, instructions, and shared document context. Recomputing this context for every request creates unnecessary computation and increases inference latency.

This project addresses that problem through a reusable prefix KV cache.

**Objective:** Reduce repeated Transformer computation and improve inference efficiency through intelligent KV-state caching and cache management.

### How It Works

1. A prompt prefix is processed by the Transformer model.
2. The resulting KV states are stored in the prefix cache.
3. When a later request shares the same prefix, the cached KV states are reused.
4. The model processes only the new suffix and generates the response.
5. Cache statistics, memory usage, and inference latency are tracked for analysis.

---

## Features

### Prefix KV Caching
Stores reusable Key-Value states for prompt prefixes, avoiding repeated computation for requests that share the same beginning.

### Pinned Prefixes
Allows frequently used system prompts to remain in the cache and reduces the likelihood of eviction.

### LRU Eviction
Removes the least recently used unpinned cache entries when the cache reaches its configured capacity.

### Memory-Aware Cache Management
Supports both entry-count and estimated KV-memory limits, helping control retained cache memory.

### Inference Comparison
Compares cached and uncached generation and reports:

- Inference latency
- Speedup
- Latency reduction
- Generated response comparison

### Streamlit Dashboard
Provides an interactive interface for running experiments and inspecting cache behavior.

The dashboard includes five sections:

| Section | Purpose |
|---|---|
| **Inference Lab** | Run cached, uncached, or comparison inference |
| **Prefix Vault** | Pin, view, and unpin reusable system prompts |
| **Benchmark Arena** | Review measured inference latency |
| **GPU Monitor** | View GPU memory information when CUDA is available |
| **Experiment Results** | View documented benchmark results and charts |

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Tensor operations and model inference |
| CUDA | GPU acceleration, when available |
| Hugging Face Transformers | Transformer model integration |
| GPT-2 | Benchmark and validation model |
| Streamlit | Interactive dashboard |
| Pandas | Benchmark data handling |
| Pytest | Testing |

---

## Project Structure

```text
prefix-kv-cache/
│
├── app.py                     # Streamlit dashboard
├── prefix_kv_cache/
│   ├── cache.py               # Prefix cache implementation
│   └── engine.py              # Cached inference engine
│
├── benchmarks/                # Benchmark scripts and results
├── tests/                     # Test suite
├── screenshots/               # Dashboard screenshots
├── requirements.txt
└── README.md
```

> Note: Exact file and folder names may vary depending on the project version.

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/ManishaChindam0309/prefix-kv-cache.git](https://github.com/ManishaChindam0309/prefix-kv-cache.git)
cd prefix-kv-cache
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser. If it does not open automatically, open the local URL printed in the terminal.

---

## Usage

### Run an inference comparison

1. Open **Inference Lab**.
2. Enter a system prompt.
3. Enter a user message.
4. Select **Compare both**.
5. Click **Run Inference**.
6. Review the generated responses and latency comparison.

### Pin a reusable system prompt

1. Open **Prefix Vault**.
2. Enter a reusable system prompt.
3. Click **Pin Prefix**.
4. Review the pinned prefix.
5. Unpin it when it is no longer needed.

Pinned prefixes are protected from normal LRU eviction.

### View benchmark results

Open **Benchmark Arena** to review measured inference latency and available benchmark statistics from the current session.

Open **Experiment Results** to view documented benchmark charts and tables.

### Monitor GPU memory

Open **GPU Monitor** to view available GPU memory information. GPU metrics depend on your hardware and PyTorch CUDA availability.

---

## Experimental Results

The following results are historical experimental measurements from the project’s benchmark environment. They are not guaranteed results for every computer, model configuration, or workload.

### LRU vs. Pinned Prefix

| Method | Average Latency |
|---|---:|
| No Cache | 238.90 ms |
| LRU Cache | 168.41 ms |
| Pinned Prefix | 157.71 ms |

Pinned prefix caching achieved approximately **33.98% lower latency** than the no-cache baseline in this experiment.

### Pinned Prefix Performance

| Prefix Length | Standard Latency | Pinned Latency | Speedup |
|---:|---:|---:|---:|
| 512 tokens | 155.05 ms | 93.40 ms | 1.66x |
| 768 tokens | 180.20 ms | 98.57 ms | 1.83x |

### Shared Prefix Performance

| Prefix Length | Standard Latency | Cached Latency | Speedup |
|---:|---:|---:|---:|
| 512 tokens | 155.82 ms | 98.67 ms | 1.58x |
| 768 tokens | 177.09 ms | 96.81 ms | 1.83x |

### GPU Memory Scaling

| Prefix Length | Incremental Retained KV Memory |
|---:|---:|
| 32 tokens | 2.25 MiB |
| 64 tokens | 4.50 MiB |
| 128 tokens | 9.00 MiB |
| 256 tokens | 18.00 MiB |
| 512 tokens | 37.00 MiB |

Five pinned prefixes across the tested lengths retained approximately **70.75 MiB** of additional allocated GPU memory.

---

## Application Screenshots

Add screenshots of the running application to a folder named `screenshots/`.

Recommended screenshots:

- `01-inference-lab.png` — Inference Lab with generated output
- `02-inference-comparison.png` — Cached versus uncached latency comparison
- `03-benchmark-arena.png` — Benchmark Arena with measured results
- `04-prefix-vault.png` — Prefix Vault showing a pinned prompt
- `05-gpu-monitor.png` — GPU Monitor
- `06-experiment-results.png` — Experiment Results charts and tables

### Inference Lab

![Inference Lab](screenshots/01-inference-lab.png)

### Inference Comparison

![Inference Comparison](screenshots/02-inference-comparison.png)

### Benchmark Arena

![Benchmark Arena](screenshots/03-benchmark-arena.png)

### Prefix Vault

![Prefix Vault](screenshots/04-prefix-vault.png)

### GPU Monitor

![GPU Monitor](screenshots/05-gpu-monitor.png)

### Experiment Results

![Experiment Results](screenshots/06-experiment-results.png)

---

## Limitations

- Experiments are primarily based on GPT-2.
- Performance depends on hardware, model configuration, and workload characteristics.
- Results from controlled workloads may differ from real-world serving environments.
- Memory measurements focus on retained KV-cache memory and do not represent total model or allocator memory.
- The project is experimental and is not a complete distributed production-serving system.

---

## Future Improvements

- Support additional language models and Transformer architectures.
- Improve cache-memory optimization and eviction strategies.
- Explore larger-scale and more realistic inference workloads.
- Add persistent KV-cache storage.
- Expand automated performance and correctness testing.
- Add deeper cache monitoring and visualization.

---

## Repository

GitHub: [Prefix KV Cache](https://github.com/ManishaChindam0309/prefix-kv-cache)

## License

Add the license that applies to this repository.

---

**Prefix KV Cache — Improving language-model inference through reusable prefix Key-Value states.**
