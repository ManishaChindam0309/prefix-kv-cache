# Prefix KV Cache

**Reducing Transformer LLM inference latency through reusable Key-Value (KV) states.**

Prefix KV Cache is a Python-based implementation that improves Large Language Model (LLM) inference efficiency by caching and reusing previously computed Key-Value (KV) states for identical system prompt prefixes.

Instead of processing the same system prompt repeatedly, the model reuses cached KV states and processes only the new user input, reducing redundant computation and inference latency.

## 🚀 Project Overview

In conversational AI applications, multiple requests often share the same system prompt. A Transformer model normally processes this repeated prefix for every request, resulting in unnecessary computation.

This project addresses this problem using **Prefix KV Caching**.

### Key Features

- Reuses previously computed KV states for identical prompt prefixes.
- Uses hash-based prefix identification for cache lookup.
- Implements an LRU (Least Recently Used) eviction strategy.
- Uses deep copies of cached KV states to prevent unintended modifications.
- Compares inference latency with and without caching.
- Tracks cache hits and calculates cache hit rate.

## 🏗️ Architecture

![Prefix KV Cache Architecture](prefix_kv_cache_architecture.png)

The flowchart illustrates the complete caching workflow:

1. Receive a new user request.
2. Tokenize the system prompt and identify its prefix.
3. Check whether the prefix exists in the cache.
4. **Cache Hit:** Retrieve and reuse previously computed KV states.
5. **Cache Miss:** Process the prefix, compute KV states, and store them in the cache.
6. Process new user tokens and generate the response.

## ⚙️ How It Works

### 1. Tokenization
The system prompt and user message are converted into tokens separately.

### 2. Prefix Identification
The system identifies the system prompt prefix using its token IDs and a hash-based lookup.

### 3. Cache Lookup
The cache checks whether the prefix has already been processed.

### 4. Cache Hit
If the prefix exists:
- Retrieve the stored KV states.
- Create a deep copy to avoid modifying cached states.
- Process only the new user tokens.
- Generate the response.

### 5. Cache Miss
If the prefix does not exist:
- Process the system prompt.
- Compute its KV states.
- Store the states in the cache.
- Continue generating the response.

### 6. Cache Management
An LRU eviction strategy manages cache entries. When the cache reaches its maximum capacity, the least recently used entry is removed.

## 📊 Performance Results

The project was tested using:

| Configuration | Details |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Hardware | CPU |
| New tokens | 10 |
| Cache capacity | 100 entries (default) |

### Benchmark Comparison

Results obtained by running:

`python demo.py --new-tokens 10`

| Request | Without Cache | With Cache | Latency Reduction |
|---|---:|---:|---:|
| First request (cache miss) | 2854 ms | 2695 ms | 6% |
| Refund after 3 weeks | 2749 ms | 1427 ms | 48% |
| When is support open? | 2596 ms | 1406 ms | 46% |
| Do I need a receipt? | 2683 ms | 1426 ms | 47% |

### Key Observations

- Cache hits achieved approximately **1.9x faster inference** in these measurements.
- Repeated requests showed approximately **46–48% latency reduction**.
- Around **1.2 seconds were saved per cache hit**.
- The first request was a cache miss because the prefix had to be computed and stored.
- The demo reported a **75% cache hit rate**.

**Note:** These are observed CPU benchmark results from the project demo. Actual performance may vary depending on hardware, model, prompt length, and generation settings.

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- pytest
- LRU Cache

## 📁 Project Structure

```text
prefix-kv-cache/
│
├── prefix_kv_cache/       # KV cache implementation
├── tests/                 # Unit tests
├── demo.py                # Interactive demo
├── benchmark.py           # Performance benchmarking
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── prefix_kv_cache_architecture.png
└── prefix_kv_cache_results.xlsx
```

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ManishaChindam0309/prefix-kv-cache.git
cd prefix-kv-cache
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Run the Demo

```bash
python demo.py
```

To generate 10 new tokens:

```bash
python demo.py --new-tokens 10
```

### Run Tests

```bash
pytest
```

### Run the Benchmark

```bash
python benchmark.py
```

For offline random-input benchmarking:

```bash
python benchmark.py --offline-random
```

To use Qwen2.5-0.5B-Instruct:

```bash
python benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## 🧠 Important Implementation Details

- **Exact Prefix Matching:** Cache reuse requires an identical tokenized prefix.
- **Hash-Based Lookup:** Prefix token IDs are used to identify cache entries.
- **Deep Copy of KV States:** Prevents unintended changes to stored cache states.
- **LRU Eviction:** Removes the least recently used entry when capacity is reached.
- **Configurable Cache Capacity:** Uses a default maximum of 100 entries.

## ⚠️ Limitations

- Only identical tokenized prefixes are reused.
- Different system prompts require separate cache entries.
- KV states consume memory, especially for longer prompts and larger models.
- The current implementation focuses on prefix caching rather than complete production inference optimization.
- Performance depends on model architecture, hardware, and prompt length.

For production-grade inference optimization, explore:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## 🔮 Future Improvements

- Add cache hit, miss, and eviction statistics.
- Experiment with different system prompt lengths.
- Compare CPU and GPU performance.
- Evaluate different model sizes.
- Compare against production inference frameworks.
- Explore more advanced prefix caching strategies.

## 📚 Key Learnings

Through this project, I explored:

- Transformer attention mechanisms.
- Key-Value caching in autoregressive language models.
- Reducing redundant computation during LLM inference.
- Hash-based cache lookup.
- LRU cache management.
- Model inference benchmarking and latency evaluation.
- Maintaining correctness when reusing cached KV states.

## 👩‍💻 Author

**Manisha Chindam**

GitHub: [ManishaChindam0309](https://github.com/ManishaChindam0309)


