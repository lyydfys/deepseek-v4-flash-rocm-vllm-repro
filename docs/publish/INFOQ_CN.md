# 在 AMD ROCm 上复现 DeepSeek-V4-Flash：vLLM 兼容部署、32K 正确性与 8K TopK 扫参

这是一篇面向 InfoQ 的深度技术稿，重点不在“模型是否能起服务”，而在于：
如何把一个在 AMD ROCm 上不容易直接跑稳的 DeepSeek-V4-Flash 过程，
整理成一条可复查、可复现、可继续优化的研究 baseline。

## 文章主旨

DeepSeek-V4-Flash 的高性能路径涉及 FP4/FP8、FlashMLA、sparse MLA、
MoE kernel 和 graph capture 等能力。在 AMD ROCm 环境中，直接按常规
vLLM 方式启动时，容易遇到 kernel、dtype、mHC、MXFP4 和 sparse top-k
等兼容问题。

这篇稿子的核心不是“我证明了某个上游 ROCm 能力已完整可用”，
而是说明：

- vLLM OpenAI API 服务可以启动；
- 短问答不会乱码；
- 2K、8K、32K needle retrieval 可以形成可复现的 correctness gate；
- 8K 场景下的 `index_topk` 存在明显性能拐点；
- 某些看似直观的 scheduler 调整会在 KV cache 规划阶段直接失败。

## 可直接保留的结论

| Gate | Result |
|---|---:|
| `/v1/models` | HTTP 200 |
| Short completion | `Paris`, 13.540s |
| 2K needle retrieval | PASS, 53.376s |
| 8K needle retrieval | PASS, 151.976s |
| 32K needle retrieval | PASS at `index_topk=4096`, 497.470s |
| Best 8K TTFT point | `index_topk=2048`, 80.313s |

## 建议的稿件结构

1. 选题背景：为什么 DeepSeek-V4-Flash 适合作为 AMD ROCm 研究对象。
2. 复现口径：ModelScope DSW、ROCm、vLLM、AITER、PyTorch fallback。
3. 从“能启动”到“能验证”：服务健康、短输出、长上下文 correctness。
4. `index_topk` 的边界：32K correctness 与 8K 性能并不共用同一个最优点。
5. 负例：`--max-num-batched-tokens 32768` 在 KV cache 规划阶段失败。
6. 结论：当前 baseline 的价值在于可复现、可继续做 kernel 优化。

## 建议保留的参考图

- ROCm fallback pipeline
- service and correctness evidence
- context correctness matrix
- 8K top-k sweep
- negative evidence

## 适合 InfoQ 的口吻

- 少一点“我做了什么”，多一点“问题是什么、怎么验证、得到什么边界”。
- 少一点宣传性总结，多一点方法、数据和结论。
- 可以明确边界：这是研究型 baseline，不是生产级高吞吐 serving。

## 推荐作者简介

来自 AMD ROCm / ModelScope DSW 的 DeepSeek-V4-Flash 复现实践，关注
vLLM 兼容部署、长上下文 correctness 和 ROCm kernel 优化。
