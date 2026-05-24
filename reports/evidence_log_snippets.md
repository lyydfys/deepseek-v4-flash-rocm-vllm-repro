# 实验证据摘录 - 2026-05-23

这些片段来自已保存的实验报告和终端证据，刻意保持短小，方便放入 ModelScope 研习社文章中作为证据链。

## 服务健康

```text
/v1/models -> HTTP 200
服务模型名 -> deepseek-v4-flash-amd-32k-batch8-16384
扫参后 index_topk -> 2048
短问答校验 -> HTTP 200，生成 'Paris'，校验通过
```

## 32K correctness 基线

```text
32K 隐针检索，prompt_tokens=32768
HTTP 200
输出命中 cobalt-7391
elapsed=497.470s
32K 后健康检查 -> HTTP 200，输出 5，elapsed=1.652s
```

## 8K topk 扫参

```text
topk1024 -> 8K 隐针通过，gate=138.563s，首 token=95.217s，prefill=85.962 tok/s
topk1536 -> 8K 隐针通过，gate=154.255s，首 token=94.250s，prefill=86.844 tok/s
topk2048 -> 8K 隐针通过，gate=156.403s，首 token=80.313s，prefill=101.914 tok/s
topk4096 -> 8K 隐针通过，gate=151.976s，首 token=96.589s，prefill=84.740 tok/s
```

## 32K topk 负例

```text
topk2048 -> 32K 开头 needle 失败，elapsed=303.88s，输出缺失关键片段
topk3072 -> 32K 开头 needle 失败，elapsed=519.05s，输出 token 不完整
topk3584 -> 32K 开头 needle 失败，elapsed=549.69s，输出 token 不完整
topk4096 -> 32K 开头 needle 通过，elapsed=572.91s，输出命中完整 token
```

## full-prefill 负例

```text
MAX_NUM_BATCHED_TOKENS=32768
To serve max seq len 32768, 13.17 GiB KV cache is needed.
Available KV cache memory: 1.39 GiB.
Estimated maximum model length: 3448.
```

## 证据文件

- screenshots/8k_topk_sweep_terminal_20260523.png
- figures/01_rocm_fallback_pipeline.png
- figures/02_8k_topk_perf.png
- figures/03_context_correctness_matrix.png
- figures/04_service_and_correctness_evidence.png
- figures/05_topk_and_negative_evidence.png
- REPORT_20260523_8K_TOPK_PERFORMANCE.md
- REPORT_20260523_32K_PERFORMANCE_COMPARISON.md
