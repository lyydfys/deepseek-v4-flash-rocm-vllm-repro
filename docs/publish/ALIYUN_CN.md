# 我在 ModelScope DSW AMD GPU 上跑通 DeepSeek-V4-Flash：vLLM 兼容部署、长上下文验证与 8K 性能扫参

这篇稿子适合阿里云开发者社区。它更强调实践、可复现和开发者价值。
可以直接围绕 ModelScope DSW、ROCm、vLLM、Notebook、Space 和复现资料来写。

## 推荐摘要

本文记录我在 ModelScope DSW AMD GPU 实例上完成的一轮 DeepSeek-V4-Flash
部署研究。整个过程不是简单启动一个服务，而是经历了模型准备、ROCm
兼容排查、vLLM 服务启动、实例重启后的恢复、补丁验证、长上下文
correctness 测试，以及 8K 场景下的 top-k 性能扫参。

我把这次工作整理成一套可以复现的 baseline，重点沉淀脚本、补丁、
日志摘要、实验结果和截图，而不是保留大模型权重本身。

## 适合保留的正文主线

### 1. 为什么选 DeepSeek-V4-Flash

- 模型新；
- 执行路径复杂；
- 在 AMD ROCm 上会暴露真实工程问题；
- 适合作为 ROCm 大模型部署研究对象。

### 2. 复现环境怎么写

- ModelScope DSW AMD GPU；
- ROCm、PyTorch、ModelScope 基础组件；
- `MODEL_DIR`、`RUN_MODEL_DIR`、`REPRO_DIR` 变量化路径；
- 复现资料优先持久化，模型权重可按需重下。

### 3. 怎么验证真的跑通

- `/v1/models` 返回 HTTP 200；
- 短问答不乱码；
- 2K / 8K / 32K needle retrieval 通过；
- 8K `index_topk` sweep 找到最佳点；
- 负例也保留。

### 4. 能直接放进文中的结果

| 项目 | 结果 |
|---|---:|
| vLLM 服务 | 可启动 |
| 32K correctness | `index_topk=4096` 通过 |
| 8K 最优点 | `index_topk=2048` |
| 8K TTFT | 80.313s |
| 8K 有效 prefill | 101.914 tok/s |

### 5. 结尾怎么收

可以落到这几点：

- 这不是生产级高性能 serving；
- 这是一个可运行、可验证、可继续优化的 AMD ROCm baseline；
- 后续重点是 sparse MLA、top-k、mHC、MoE、FlashMLA；
- Space、GitHub Pages 和仓库已经把复现入口统一好了。

## 推荐配图

- 封面图；
- 服务健康与 correctness 证据图；
- 32K 正确性矩阵；
- 8K top-k 扫参图；
- 负例图。

## 推荐标签

ROCm，AMD GPU，vLLM，DeepSeek-V4-Flash，大模型推理优化，ModelScope DSW

## 适合阿里云社区的写法

- 多写步骤；
- 多写环境和结果；
- 多写复现和调试；
- 少写过强的结论。

## 可附上的链接

- GitHub 仓库
- GitHub Pages
- Hugging Face Space
- vLLM 复现评论
