# Remote DSW Validation Result - 2026-05-24

## Instance

- DSW URL family: `dsw-1923511`
- Remote validation directory: `/mnt/workspace/publish_modelscope_gallery_notebook_20260524_remote_validated_v2`
- Validation mode: default Gallery Notebook cells only
- Heavy model download: not run
- vLLM service launch: not run

## Environment Captured By Notebook

```json
{
  "python": "3.12.13",
  "platform": "Linux-5.10.134-008.14.kangaroo.al8.x86_64-x86_64-with-glibc2.35",
  "cwd": "/mnt/workspace/publish_modelscope_gallery_notebook_20260524_remote_validated_v2",
  "torch": "2.10.0+git8514f05",
  "torch_hip": "7.2.53211",
  "cuda_available_api": true,
  "device_count": 1,
  "device_name_0": ""
}
```

## Result

- Code cells executed: 11
- Failures: `[]`
- Default run status: `REMOTE_VALIDATION_OK`
- Generated output files:
  - `outputs/8k_topk_sweep.png`
  - `outputs/context_correctness_matrix.png`
  - `outputs/env_report.json`
  - `outputs/gallery_run_summary.json`
  - `outputs/optional_heavy_reproduce_commands.sh`
  - `outputs/optional_live_vllm_check.json`
  - `outputs/remote_validation_log.txt`
  - `outputs/result_summary.json`
  - `outputs/topk_microbenchmark.json`

## Key Results Reproduced

```json
{
  "best_8k_topk_by_ttft": 2048,
  "best_8k_ttft_s": 80.313,
  "best_8k_prefill_tok_s": 101.914,
  "first_verified_32k_begin_topk": 4096,
  "service_checks_passed": 5
}
```

## Note

The first upload used a Windows-created zip and extracted paths with literal backslashes on Linux. The validated v2 upload used a POSIX-path zip. Future upload package should use:

```text
modelscope_gallery_deepseek_v4_flash_amd_rocm_notebook_package_posix.zip
```
