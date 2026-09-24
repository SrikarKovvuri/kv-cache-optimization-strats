# Homework 1: KV Cache Compression

## Overview

Week 2 established that autoregressive decode is memory-bandwidth-bound and
that the KV cache grows linearly with sequence length, driving both memory
footprint and latency. This assignment asks you to get hands-on: implement
and evaluate **three KV cache compression algorithms** on top of the
TinyLlama inference server from Week 2, then quantify the memory / latency /
quality tradeoffs each one makes.

## Setup

You need **Python 3.10+**. From the repository root, create a virtual
environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt
```

Run the server with the venv activated (e.g.
`python code/less_minimal_full_server.py --verbose`). A CUDA GPU is optional
but recommended for timing runs; `--device cpu` works for debugging.

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

## Task

Implement **three** KV cache compression algorithms. You may choose which
three, drawing on the Week 2 readings (Pope et al., FlexGen) and lecture
discussion of KV cache reduction techniques. The categories below are
**examples only** — other well-motivated approaches are fine as long as your
three methods are clearly distinct and you can compare them to the baseline.
Three well-separated categories to consider:

1. **Eviction / windowing** — e.g., a sliding window over the most recent
   tokens, optionally keeping a few fixed "attention sink" tokens
   (StreamingLLM-style).
2. **Importance-based selection** — e.g., score cached tokens by cumulative
   attention weight and evict the lowest-scoring ones once the cache
   exceeds a fixed budget (heavy-hitter style).
3. **Numerical compression** — e.g., quantize cached K/V tensors (int8 or
   lower) instead of storing them in fp16/fp32.

Implement each algorithm as a modification to (or wrapper around) the KV
cache used in the prefill/decode loop, selectable via a flag (e.g.
`--compression {none,window,importance,quant}`). Your three algorithm names
can differ from these examples; include a **`none`** baseline so you can
compare runs directly.

## Evaluation

For each of your three algorithms, plus the uncompressed baseline, sweep at
least **three** sequence lengths or cache budgets and measure:

- **KV cache memory footprint** (bytes). `kv_bytes()` in the starter code is
  the **dense** GQA formula — use it for the uncompressed baseline. For
  compressed caches, measure the **total bytes of all K and V data you keep in
  memory** after compression (count every tensor you store per layer: number
  of elements × bytes per element, then add them up; a small helper function
  is fine). Do not use the dense formula if your cache holds fewer tokens or
  smaller dtypes than the baseline. **Include these byte totals in your
  `UNI-hw1-report.md`** (in a plot and/or a small results table).
- **Decode latency / throughput** (ms per token, extending the existing timing
  instrumentation).
- **Output quality degradation** relative to the uncompressed baseline
  (e.g., same prompt and `max-tokens`, compare generated text, token
  divergence, or edit distance; perplexity on a held-out prompt is optional
  if you want a stricter metric). `--max-tokens` is the total number of
  **new** tokens, including the one chosen during prefill. The decode loop
  then runs at most `max-tokens - 1` steps, and stops early on EOS or a
  new chat-role tag, so a run can be shorter than the flag.

## Deliverable

Produce a single markdown report, `UNI-hw1-report.md`, in the root folder,
containing:

1. A short description of each of your three algorithms and how you
   implemented it.
2. **Evaluation results** from the section above: KV footprint (bytes),
   decode latency/throughput, and quality vs. baseline. At least **three**
   settings per algorithm plus `none`. Show the numbers in **at least two
   plots** (e.g., cache bytes vs. sequence length; latency vs. compression)
   and/or a **compact markdown table**; plots alone are fine if every series
   is labeled with the measured values you used.
3. A short discussion of trade-offs: which algorithm compressed the most,
   which degraded quality the least, and why.

Embed plots as images committed alongside the report (e.g.
`![](plots/latency.png)`). Do not submit only a wall of unlabeled numbers —
tie each figure or table to a sentence of interpretation.

Come to class ready to discuss and present your code and analysis.

### Submission Steps:

1. Pull the latest instructions: `git pull origin main`
2. Implement your three compression algorithms in `code/`, extending
   `less_minimal_full_server.py`.
3. Create `UNI-hw1-report.md` in the root folder (replace `UNI` with your
   Columbia UNI) with the writeup described above, plus a `plots/` folder
   for your figures.
4. Title header: `# Homework 1: KV Cache Compression`
5. Include: UNI, Full Name, GitHub handle, and Date.
6. Commit and push your code, plots, and report to your private repository
   by **Thursday, October 1, 2026, 11:55 PM Eastern**. Your repo will lock at the deadline, and no extensions will be given. 
