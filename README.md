# Homework 1: KV Cache Compression

## Overview

Week 2 established that autoregressive decode is memory-bandwidth-bound and
that the KV cache grows linearly with sequence length, driving both memory
footprint and latency. This assignment asks you to get hands-on: implement
and evaluate **three KV cache compression algorithms** on top of the
TinyLlama inference server from Week 2, then quantify the memory / latency /
quality tradeoffs each one makes.

## Setup

You need **Python 3.10+**. Pick **one** of the two setups below.

### On the course server (`mv.cs.columbia.edu`)

**Do not create a virtual environment or `pip install` anything on the course
server.** A ready-made environment with torch, transformers, and the TinyLlama
weights is shared for the whole class, and your disk quota there is small
(500 MB soft / 1 GB hard), so installing torch yourself will fail and can leave
your account over quota. Load the shared environment instead:

```bash
source /home/coms6998-shared/env.sh
```

The model does not fit on the login node, so run it on the GPU through Slurm
(`srun` or `sbatch`, at most 15 minutes per job). See the "Course Server Access"
post on Ed for the exact commands.

### On your own computer

From the repository root, create a virtual environment and install
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt
```

Run the server with the environment loaded (e.g.
`python code/less_minimal_full_server.py --verbose`). A CUDA GPU is optional
but recommended for timing runs; `--device cpu` works for debugging on your own
computer.

## Starting Code

Start from `code/less_minimal_full_server.py`, the unwrapped, layer-by-layer
TinyLlama inference server from Week 2. Before writing any code, get
familiar with:

- `kv_seq_len()` / `kv_bytes()` — how KV cache size is measured today.
- The `--verbose` path, which manually loops through Q/K/V projections,
  RoPE, GQA expansion, and attention per layer. This is where you will
  intercept and modify the KV cache.
- The `DynamicCache` object, populated via `kv_cache.update(k, v, layer_idx)`
  during both prefill and decode.

## Overview

There are **three** KV cache compression algorithms being implemented: 

1. **Eviction / windowing** — e.g., a sliding window over the most recent
   tokens, optionally keeping a few fixed "attention sink" tokens
   (StreamingLLM-style).
2. **Importance-based selection** — e.g., score cached tokens by cumulative
   attention weight and evict the lowest-scoring ones once the cache
   exceeds a fixed budget (heavy-hitter style).
3. **Numerical compression** — e.g., quantize cached K/V tensors (int8 or
   lower) instead of storing them in fp16/fp32.

