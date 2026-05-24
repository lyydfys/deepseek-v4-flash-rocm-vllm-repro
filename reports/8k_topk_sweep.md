# DeepSeek-V4-Flash ROCm 8K topk performance report - 2026-05-23

## Executive conclusion

This round tested whether the current AMD ROCm DSW deployment can reach Nvidia-class serving performance at an 8K context length.

Result: the 8K context path is functionally correct and can be tuned modestly, but it does **not** reach the public high-end Nvidia benchmark level.

The best measured 8K point in this run is `index_topk=2048`:

- 8K needle retrieval: PASS, `156.403s`
- 8K one-token streaming bench: `TTFT=80.313s`
- effective prefill rate: `101.914 prompt tokens/s`

Compared with the restored `index_topk=4096` baseline (`TTFT=96.589s`, `84.740 prompt tokens/s`), this is about a 16.9% TTFT improvement and about a 20.3% prefill-rate improvement. It is still far from public B200/H100 serving references.

## Environment

- Remote instance: `dsw-1920667-57f4686c47-xrr7c`
- Runtime: AMD ROCm DSW image, vLLM OpenAI API server
- Served model: `deepseek-v4-flash-amd-32k-batch8-16384`
- Model tree: `/home/deepseek_hybrid/DeepSeek-V4-Flash`
- Model cache: a non-persistent local model cache outside the reproducibility bundle
- Repro bundle: persistent scripts, reports, manifests, and small evidence files
- Launch script: `scripts/launch_32k_batch8_16384_system.sh`
- Sweep script: `scripts/run_8k_topk_sweep.sh`
- Sweep log root: `/home/amd_research_logs/8k_topk_sweep_20260523_201422`

Model weights remain outside the persistent disk. Only scripts, patches, reports, manifests, and small evidence files are backed up.

## Method

Each topk value was tested by:

1. patching `/home/deepseek_hybrid/DeepSeek-V4-Flash/config.json`;
2. restarting vLLM with the same stable 32K batch8/16384 launch script;
3. waiting for `/v1/models`;
4. running `needle8192_tokens` correctness gate;
5. running an 8K one-token streaming bench with `target_tokens=8192`, `max_tokens=1`, `runs=1`.

Command:

```bash
BASE=/path/to/deepseek-v4-flash-rocm-vllm-repro
TOPKS="1024 1536 2048" \
  nohup bash "$BASE/scripts/run_8k_topk_sweep.sh" \
  > /home/amd_research_logs/8k_topk_sweep_wrapper_latest.log 2>&1 &
```

## Results

| index_topk | 8K needle | Gate latency | TTFT / elapsed | Effective prefill |
|---:|---:|---:|---:|---:|
| 4096 | PASS | 151.976s | 96.589s | 84.740 tok/s |
| 1024 | PASS | 138.563s | 95.217s | 85.962 tok/s |
| 1536 | PASS | 154.255s | 94.250s | 86.844 tok/s |
| 2048 | PASS | 156.403s | 80.313s | 101.914 tok/s |

Notes:

- `index_topk=4096` is the prior restored 8K/32K correctness baseline, not re-run inside this exact sweep.
- `index_topk=2048` is best for 8K performance in this run.
- Previous 32K tests showed that `index_topk=2048` does not preserve 32K begin-position retrieval correctness, so it should be treated as an 8K-specific tuning point.

## Nvidia reference boundary

Public Lambda vLLM results for DeepSeek-V4-Flash report a B200 benchmark with `8192 input / 1024 output`, `512 prompts`, and `32` concurrent requests, where vLLM on NVIDIA HGX B200 reports mean TTFT around `1.452s`.

That is not the same workload as this single-request AMD DSW probe (`8192 input / 1 output`, `1` request), so it is not a strict apples-to-apples benchmark. Still, the gap is large enough to conclude that the current ROCm fallback-safe baseline is not Nvidia-class.

## Bottleneck interpretation

The topk sweep improved 8K latency only modestly, which suggests the main bottleneck is not just the scalar value of `index_topk`.

Likely bottlenecks remain:

- sparse MLA / sparse attention indexer still relying on fallback-heavy execution;
- qnorm, RoPE, KV, and mHC paths not fully fused for ROCm;
- MXFP4 / FP8 MoE paths not using a Blackwell-equivalent native kernel stack;
- `enforce-eager` preserving stability while giving up graph-capture speedups;
- no production-grade ROCm speculative decoding path is enabled in this baseline.

## Current best operating points

- For 8K performance exploration: `index_topk=2048`
- For 32K correctness preservation: `index_topk=4096`

Do not claim a single topk value is universally optimal. The correct setting depends on the target context length and correctness gate.

## Next steps

1. Run the same 8K stream bench on an Nvidia baseline before making any "not worse than Nvidia" claim.
2. Add a true 8K batch/concurrency benchmark using `vllm bench serve` style parameters.
3. Fix ROCm profiler output so the run produces useful kernel CSV or trace files.
4. Replace sparse MLA/top-k fallback with ROCm-native Triton/AITER kernels.
5. Re-test `index_topk=2048` at 16K/32K to quantify exactly where long-context correctness starts to fail.

## References

- vLLM DeepSeek-V4-Flash recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM ROCm DeepSeek-V4 tracking issue: https://github.com/vllm-project/vllm/issues/41820
- Lambda DeepSeek-V4-Flash benchmark page: https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash
