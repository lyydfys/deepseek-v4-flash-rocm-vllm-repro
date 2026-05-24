# vLLM ROCm Reproduction Report Draft

This is a draft comment/report that can be posted to the vLLM DeepSeek-V4 ROCm
tracking issue or discussion after manual review.

## Suggested title

Reproduction report: DeepSeek-V4-Flash on AMD ROCm ModelScope DSW with 32K
needle correctness and 8K top-k sweep

## Draft comment

Hi vLLM maintainers,

I am sharing a user-side reproduction report for `deepseek-ai/DeepSeek-V4-Flash`
on an AMD ROCm ModelScope DSW instance. This is not an upstream implementation
claim; it is a reproducibility and correctness report from an application-side
deployment study.

### Environment

- Platform: ModelScope DSW AMD GPU instance
- Runtime: ROCm-enabled torch build
- vLLM: 0.20.1-era ROCm environment during the experiment
- AITER: installed and used where available
- Served model name: `deepseek-v4-flash-amd-32k-batch8-16384`
- Main service mode: OpenAI-compatible vLLM API server
- Main flags:
  - `--max-model-len 32768`
  - `--max-num-seqs 8`
  - `--max-num-batched-tokens 16384`
  - `--kv-cache-dtype fp8`
  - `--block-size 256`
  - `--enforce-eager`
  - `--moe-backend triton`

The experiment used a fallback-heavy path to prioritize correctness and
reproducibility before performance tuning. Several unstable ROCm paths were
guarded or routed through AITER/Triton/PyTorch fallback variants.

### Correctness gates

The service passed basic OpenAI API health checks and semantic needle retrieval
tests:

| Gate | Result |
|---|---:|
| `/v1/models` | HTTP 200 |
| Short completion | HTTP 200, `Paris`, 13.540s |
| 2K needle retrieval | PASS, 53.376s |
| 8K needle retrieval | PASS, 151.976s |
| 32K needle retrieval | PASS at `index_topk=4096`, 497.470s after restart restore |

For the 32K begin-position needle gate, the observed top-k boundary was:

| index_topk | 32K begin needle result | Latency |
|---:|---|---:|
| 2048 | FAIL, missing key fragment | 303.88s |
| 3072 | FAIL, incorrect token | 519.05s |
| 3584 | FAIL, incomplete token | 549.69s |
| 4096 | PASS | 572.91s |

### 8K top-k sweep

The 8K single-request benchmark used an 8192-token prompt and one output token.
This is not comparable to high-concurrency public serving benchmarks, but it is
useful as a local prefill probe.

| index_topk | 8K needle | Gate latency | TTFT | Effective prefill |
|---:|---|---:|---:|---:|
| 4096 | PASS | 151.976s | 96.589s | 84.740 tok/s |
| 1024 | PASS | 138.563s | 95.217s | 85.962 tok/s |
| 1536 | PASS | 154.255s | 94.250s | 86.844 tok/s |
| 2048 | PASS | 156.403s | 80.313s | 101.914 tok/s |

For this run, `index_topk=2048` was the best 8K TTFT/prefill point, while
`index_topk=4096` was needed for the verified 32K begin-position needle test.

### Negative finding

Increasing `--max-num-batched-tokens` to 32768 failed during KV cache sizing:

```text
To serve max seq len 32768, 13.17 GiB KV cache is needed.
Available KV cache memory: 1.39 GiB.
Estimated maximum model length: 3448.
```

This suggests that scheduler tuning cannot be considered independently from KV
cache planning, fallback path memory behavior, and the model-specific sparse MLA
path.

### Current bottleneck interpretation

The bottleneck does not appear to be the OpenAI API layer or model loading. The
main performance limit appears to come from DeepSeek-V4-specific ROCm execution
paths:

- sparse MLA / sparse attention indexer fallback;
- top-k candidate count required for long-context correctness;
- FlashMLA prefill/decode fallback;
- qnorm, RoPE, KV, and mHC paths that are not fully fused on this setup;
- MXFP4/FP8 MoE path maturity on ROCm;
- `--enforce-eager` chosen for stability, giving up graph capture speedups.

### Reproducibility package

I prepared a public repository with:

- the Gallery-ready notebook;
- lightweight CSV result data;
- launch and probe scripts;
- 8K top-k sweep data;
- 32K correctness/performance report;
- figures and evidence snippets;
- no model weights.

Repository URL: https://github.com/lyydfys/deepseek-v4-flash-rocm-vllm-repro

I would appreciate any guidance on whether the current `index_topk` behavior and
fallback-heavy bottleneck interpretation match the expected ROCm DeepSeek-V4
roadmap, especially around sparse MLA/top-k and AITER MHC paths.
