# DeepSeek-V4-Flash ROCm 32K performance report - 2026-05-23

## Executive conclusion

DeepSeek-V4-Flash is now verified to run a 32K semantic retrieval case on the AMD ROCm DSW instance, but the measured performance does **not** match an Nvidia-class serving target.

The best correctness-first configuration found in this round is `index_topk=4096`. It passes the 32K begin-position needle test, but the single request takes `572.91s`. Streaming prefill measurement for a 32K prompt reports `TTFT=440.04s` and about `74.34 prompt tokens/s`.

This is a useful research baseline, not a production-performance result.

## Environment

- Instance: `dsw-1919578`
- Runtime: torch ROCm 7.2, vLLM 0.20.1, AITER installed
- Model: `/home/deepseek_hybrid/DeepSeek-V4-Flash`
- Served model: `deepseek-v4-flash-amd-32k-batch8-16384`
- Main launch: `scripts/launch_32k_batch8_16384_system.sh`
- Current restored correctness baseline: model `config.json` has `index_topk=4096`
- Important flags: `max_model_len=32768`, `max_num_seqs=8`, `max_num_batched_tokens=16384`, `kv_cache_dtype=fp8`, `enforce-eager`

## Correctness gates

| Config | Test | Result | Time | Notes |
|---|---:|---:|---:|---|
| topk512 | 8K begin needle | FAIL | 70.92s | output `SECRET_CODE: KITE-1984` |
| topk512 | 8K middle needle | PASS | 175.50s | exact code |
| topk512 | 8K end needle | PASS | 175.50s | exact code |
| topk2048 | 8K begin needle | PASS | 139.51s | exact code |
| topk2048 | 16K begin needle | PASS | 176.77s | contains code |
| topk2048 | 32K begin needle | FAIL | 303.88s | output missing `_ORBIT` |
| topk3072 | 32K begin needle | FAIL | 519.05s | output `KITE_7912` |
| topk3584 | 32K begin needle | FAIL | 549.69s | output `KITE_ORBIT_7913` |
| topk4096 | 32K begin needle | PASS | 572.91s | contains `KITE_7913_ORBIT` |

## Performance measurements

| Workload | Metric | Result |
|---|---:|---:|
| restart restore short gate | capital completion | HTTP 200, 13.540s, `Paris` |
| restart restore 2K needle | end-to-end latency | HTTP 200, 53.376s |
| restart restore 8K needle | end-to-end latency | HTTP 200, 151.976s |
| restart restore 32K needle, 32768 prompt tokens | end-to-end latency | HTTP 200, 497.470s |
| restart restore post-32K health | count completion | HTTP 200, 1.652s, `5` |
| 8K stream bench after restore, 8185 prompt tokens | TTFT / elapsed | 96.589s |
| 8K stream bench after restore, 8185 prompt tokens | prefill rate | 84.740 prompt tokens/s |
| full-prefill trial, `max_num_batched_tokens=32768` | result | failed at startup; available KV cache only 1.39 GiB, estimated max model length 3448 |
| short streaming prompt, 121 prompt tokens | TTFT | 2.515s |
| short streaming prompt, 121 prompt tokens | prefill rate | 48.10 prompt tokens/s |
| 32K streaming prompt, 32713 prompt tokens | TTFT | 440.04s |
| 32K streaming prompt, 32713 prompt tokens | prefill rate | 74.34 prompt tokens/s |
| 32K semantic needle, 32717 prompt tokens | end-to-end latency | 572.91s |
| historical batch8 short throughput | aggregate output tok/s | 5.568 tok/s |
| short prompt long decode, max_tokens=256 | result | hung/timed out; client killed |
| short prompt non-stream decode, max_tokens=64 | result | ReadTimeout at 60s |

## Negative finding

After a long decode probe, `/v1/models` stayed healthy but generation requests timed out. This is a service-level stability issue in the current fallback-safe stack. It reinforces that the current state is not production-grade and not Nvidia-class.

Additional top-k sweep after the first 32K pass:

- `index_topk=3072`: 32K begin returned HTTP 200 in `519.05s`, but semantic output was `SECRET_CODE: KITE_7912`; failed.
- `index_topk=3584`: 32K begin returned HTTP 200 in `549.69s`, but semantic output was `SECRET_CODE: KITE_ORBIT_7913`; failed.
- `index_topk=4096` remains the smallest verified 32K begin correctness baseline in this run.

Profiling attempt:

- `rocprofv3` is available at `/opt/rocm/bin/rocprofv3`, and attach mode is available.
- An attach run was started against the vLLM server while running an 8K begin needle request.
- The 8K request passed in `136.04s`, but this first profiler invocation only preserved stdout/request logs and did not produce a useful kernel-summary CSV. The next profiling run should launch or attach with explicit output-format/domain options and verify output files before the long request completes.
- A later scripted attach run against the restored service also completed the 8K stream request successfully (`TTFT=96.589s`, `prefill_tok_s=84.740`), but this ROCm attach mode still only preserved `rocprof_stdout.log`, `request.log`, and `request.jsonl`. Kernel CSV output remains unresolved.

Full-prefill tuning attempt:

- `MAX_NUM_BATCHED_TOKENS=32768` was tested to see if a 32K prompt could be prefetched in one scheduling chunk.
- The service failed during KV cache sizing: serving max sequence length `32768` required `13.17 GiB` KV cache, but only `1.39 GiB` was available, with estimated maximum model length `3448`.
- The original `max_num_batched_tokens=16384` batch8 baseline was restored and passed a short post-restore health gate.

## Bottleneck interpretation

The performance ceiling is dominated by fallback-heavy DeepSeek-V4 paths:

- sparse MLA / sparse attention indexer remains the main 32K prefill bottleneck;
- increasing `index_topk` fixes long-distance semantic retrieval but increases prefill cost;
- FlashMLA prefill/decode fallback is still enabled;
- qnorm+RoPE+KV and other model-specific pieces still use conservative fallback paths;
- `enforce-eager` avoids graph-related failures but gives up graph capture speedups.

Scheduler and batch tuning improved short concurrent throughput, but did not change the 32K prefill order of magnitude.

## Restart restore note

After a clean DSW restart, the model cache was re-downloaded to a non-persistent local model cache, the hybrid model tree was rebuilt, and `index_topk=4096` was restored.

The restart exposed several patch-order gaps that are now recorded as reproducible scripts:

- `MODELSCOPE_CACHE` is kept outside the small reproducibility bundle when model weights are re-downloadable;
- `patch_home_model_config_index_topk.sh` now accepts a positional value;
- MHC tilelang import fallback is required before importing `vllm.model_executor.layers.mhc`;
- MXFP4 public `triton` route is mapped to `TRITON_UNFUSED` for this ROCm/SILU baseline;
- W8A8 FP8 scale upcast, sparse skip-cache, kernel IR priority, and AITER MQA tile patches must be restored before launch.

With those fixes applied, the service passed 2K, 8K, and 32K correctness gates. The best restart 32K gate took `497.470s`, which reinforces the same conclusion: correctness is restored, but performance is still far from an Nvidia-class serving target.

## Nvidia comparison status

There is no same-hardware, same-model, same-script Nvidia result in this workspace, so a strict apples-to-apples comparison cannot be claimed. Public vLLM materials target high-end Nvidia configurations such as B200/B300 multi-GPU recipes, which are not equivalent to this single AMD DSW setup.

Based on the AMD measurements above, the current ROCm fallback baseline is **not** competitive with an Nvidia-class deployment target.

## Next research path

1. Profile topk4096 32K prefill with `rocprof` to quantify time in top-k, sparse MLA, qnorm/RoPE/KV, MoE, and fallback copies.
2. Replace PyTorch sparse top-k fallback with ROCm-native Triton/AITER kernels that support 4096 candidates correctly.
3. Bring up the ROCm DeepSeek-V4 sparse MLA / FlashMLA DSV4 backend from newer vLLM/AITER, but only accept it after the 32K semantic gates pass.
4. Re-run the same streaming benchmark script on an Nvidia baseline before making any "not worse than N card" claim.

Sources for external comparison context:

- vLLM DeepSeek-V4-Flash recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM ROCm DeepSeek-V4 tracking issue: https://github.com/vllm-project/vllm/issues/41820
