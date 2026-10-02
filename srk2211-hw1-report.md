# Homework 1: KV Cache Compression

- **UNI:** srk2211
- **Name:** Srikar Kovvuri
- **GitHub:** SrikarKovvuri
- **Date:** October 1, 2026

## 1. Overview

Autoregressive language-model inference stores a key/value (KV) cache so that previously processed tokens do not need to be recomputed during each decode step. However, the cache grows linearly with sequence length and can become a significant source of memory use.

I implemented three KV-cache compression strategies on top of the TinyLlama inference server:

1. **Sliding-window eviction** — retain only the most recent \(W\) cached token positions.
2. **Importance-based eviction** — retain the \(B\) token positions with the highest cumulative attention received.
3. **Per-KV-head INT8 quantization** — retain all token positions but store K/V tensors using INT8 values with per-head scaling factors.

I compared each method with the uncompressed baseline in terms of KV-cache memory, decode latency, and generated-output quality.

All experiments reported here were run locally on **CPU** using `torch.float32`. Absolute latency results reflect CPU execution rather than GPU serving performance.

## 2. Compression Algorithms

### 2.1 Sliding Window

The sliding-window implementation limits each layer's KV cache to at most \(W\) positions.

Prefill is performed normally using the entire prompt. After prefill completes, each layer's persistent K/V cache is truncated to the most recent \(W\) positions. During decode, each new token's K/V vectors are appended and the oldest cached positions are removed whenever the cache exceeds the budget.

For cache tensors with shape \([B, H_{kv}, S, D_h]\), the implementation retains:

```text
k = k[:, :, -W:, :]
v = v[:, :, -W:, :]
```

Absolute token positions continue increasing normally for RoPE even when old K/V entries are evicted.

This method is inexpensive and gives a predictable memory bound, but it assumes that recency is a good proxy for usefulness.

### 2.2 Importance-Based Eviction

Each transformer layer maintains a cumulative importance score for every cached position. During prefill, attention weights have shape \([B, H, S_q, S_k]\). I aggregate the attention received by each key position across heads and query positions to initialize one importance value per cached token.

During decode, each new query produces attention weights over the current cache. I average these weights across attention heads and add them to the cumulative importance scores. When the cache exceeds budget \(B\), I retain the top-\(B\) positions by cumulative attention score. Surviving indices are sorted back into chronological order before slicing the K/V tensors.

This is a simple heavy-hitter-style policy: frequently attended positions survive; positions receiving little attention are evicted.

### 2.3 Per-KV-Head INT8 Quantization

This method does not remove tokens. It reduces bytes per K/V value.

For each KV head, symmetric INT8 scale \(s = \max|x| / 127\) and quantized values \(q = \mathrm{round}(x/s)\), clamped to \([-127, 127]\) and stored as `torch.int8`. Each layer stores quantized K, K scales (one per KV head), quantized V, and V scales.

Reconstruction before attention: \(\hat{x} = q \cdot s\).

TinyLlama uses GQA (32 query heads, 4 KV heads), so four K scales and four V scales per layer suffice.

On decode, the simple implementation dequantizes the existing layer cache, appends new K/V, recomputes scales, requantizes the full layer, then dequantizes for attention. Persistent storage stays compressed; computation increases.

## 3. Experimental Setup

Model: **TinyLlama/TinyLlama-1.1B-Chat-v1.0**

- 22 transformer layers
- hidden size 2048
- 32 query heads, 4 KV heads, head dimension 64 (GQA)

Experiments on **CPU**, **FP32** (4 bytes per baseline K/V element; INT8 uses 1 byte per value plus scales).

**Window / importance:** same 84-token TCP prompt, 50 total new tokens (`--max-tokens`), budgets **16, 32, 64**.

**Quantization:** three prompt lengths (**22**, **84**, **102** tokens) with INT8 per-KV-head; decode length varied by run (19, 49, and 79 decode tokens respectively in the logged runs below).

Decode latency: total decode time and **ms per decode token** (excluding the token chosen at prefill).

## 4. Results

### 4.1 Sliding-Window Eviction

| Configuration | KV positions after prefill | KV storage | Decode time | ms/token | Output quality |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 84 | 3696 KB | 4409.6 ms | 90.0 | Coherent |
| Window 16 | 16 | 704 KB | 4491.4 ms | 91.7 | Severe degradation |
| Window 32 | 32 | 1408 KB | 4356.7 ms | 88.9 | Severe degradation |
| Window 64 | 64 | 2816 KB | 4249.0 ms | 86.7 | Severe degradation |

KV footprint decreases almost linearly with retained positions (e.g. 32 → 16 positions halves storage from 1408 KB to 704 KB). Aggressive truncation caused severe quality degradation on this prompt—even \(W=64\) eventually produced repetitive output.

CPU decode latency changed little; attention over fewer keys did not dominate total time versus the rest of the transformer and Python overhead.

### 4.2 Importance-Based Eviction

| Configuration | KV positions after prefill | KV storage | Decode time | ms/token | Output quality |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 84 | 3696 KB | 4289.3 ms | 87.5 | Coherent |
| Importance 16 | 16 | 704 KB | 4854.7 ms | 99.1 | Severe degradation |
| Importance 32 | 32 | 1408 KB | 4931.7 ms | 100.6 | Severe degradation |
| Importance 64 | 64 | 2816 KB | 4809.5 ms | 98.2 | Severe degradation |

Same memory reductions as sliding window at equal budgets. Decode was **slower** than baseline on CPU due to score accumulation, top-\(k\), sort, gather, and cache rewrite.

Quality also degraded severely. Raw cumulative attention during causal prefill favors early keys (more queries can attend to them), which can evict useful recent positions. Normalized scores or less frequent eviction could help.

### 4.3 INT8 Quantization

INT8 per-KV-head quantization at three sequence lengths (same algorithm; varying prompt size). FP32 baseline KV storage uses the measured/stored dense cache size after prefill; INT8 uses actual quantized tensors plus per-head scales.

| Configuration | Prompt tokens | FP32 KV storage | INT8 KV storage | Compression | Decode time | ms/token | Output quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 per-KV-head | 22 | 968.0 KB | 242.7 KB | ~4.0× | 2271.3 ms | 119.5 | Coherent |
| INT8 per-KV-head | 84 | 3696.0 KB | 924.7 KB | ~4.0× | 6888.6 ms | 140.6 | Coherent / near baseline |
| INT8 per-KV-head | 102 | 4488.0 KB | 1122.7 KB | ~4.0× | 11132.6 ms | 140.9 | Same generated output as baseline |

KV footprint scales roughly linearly with prompt length and stays at a **consistent ~4× reduction** versus FP32 at each length (968.0 → 242.7 KB, 3696.0 → 924.7 KB, 4488.0 → 1122.7 KB), as expected from 4 bytes per FP32 value versus 1 byte per INT8 value plus small scale overhead.

Unlike eviction, all cached token positions are retained. The 22-token and 84-token INT8 runs remained coherent; the 84-token run matched baseline content very closely, and the 102-token run produced the same generated sequence as the FP32 baseline.

Decode latency is higher than an uncompressed run at comparable length because each step dequantizes the layer cache, appends new K/V, recomputes scales, requantizes, and dequantizes again before attention.

## 5. Trade-Off Discussion

| Method | Mechanism | Memory | Quality (these runs) | Latency (CPU) |
| --- | --- | --- | --- | --- |
| Sliding window | Drop oldest | Bounded by \(W\) | Poor at small \(W\) | ~baseline |
| Importance | Drop low-attention | Bounded by \(B\) | Poor with simple scores | Slower |
| INT8 quant | Fewer bits per value | ~4× vs FP32 KV | Best here | Slower (naive requant) |

**Sliding window:** simplest strict cap; discards long-range context.

**Importance:** more selective than recency, but scoring overhead and causal bias hurt both speed and quality in this implementation.

**INT8:** strongest memory–quality tradeoff in these experiments (~4× KV reduction at 22/84/102 prompt tokens, with coherent or matching output); cost is extra compute from full-layer requantization each step.

Reducing KV memory does **not** automatically reduce decode latency on CPU: compression bookkeeping and conversion can dominate. On GPU, KV bandwidth matters more; these CPU numbers are mainly for **relative** comparison across methods.

## 6. Limitations

- Timings are **CPU / FP32**, not CUDA serving.
- Implementations are intentionally simple: no attention sinks, raw cumulative importance, full-layer requant every decode step.
- Quality labels are qualitative; a stricter metric (token match rate, edit distance) could strengthen the evaluation.
