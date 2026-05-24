# 在 AMD ROCm 上复现 DeepSeek-V4-Flash vLLM 部署

这个仓库整理的是一次基于 ModelScope DSW AMD GPU 实例的
DeepSeek-V4-Flash 工程复现记录。项目重点不是宣称实现了新的上游 ROCm
后端，而是把一次用户侧部署研究沉淀成可查看、可复跑、可继续优化的
baseline。

本项目覆盖：

- vLLM OpenAI API 服务启动与健康检查；
- ROCm / AITER / PyTorch fallback 兼容路径；
- 2K、8K、32K 长上下文 needle retrieval 正确性验证；
- 8K `index_topk` 性能扫参；
- 负例记录、瓶颈分析和后续 ROCm kernel 优化方向。

![封面](assets/cover_zh.png)

## 公开链接

- GitHub 仓库：  
  https://github.com/lyydfys/deepseek-v4-flash-rocm-vllm-repro

- GitHub Pages 英文页：  
  https://lyydfys.github.io/deepseek-v4-flash-rocm-vllm-repro/

- GitHub Pages 中文页：  
  https://lyydfys.github.io/deepseek-v4-flash-rocm-vllm-repro/zh.html

- Hugging Face Dataset 镜像：  
  https://huggingface.co/datasets/lyydfys/deepseek-v4-flash-rocm-vllm-repro

- vLLM ROCm tracking issue 复现报告评论：  
  https://github.com/vllm-project/vllm/issues/41820#issuecomment-4529203579

## 关键结果

| 验证项 | 结果 |
|---|---:|
| vLLM OpenAI API 服务 | `/v1/models` 返回 HTTP 200 |
| 短问答 gate | HTTP 200，输出 `Paris`，13.540s |
| 2K needle retrieval | 通过，53.376s |
| 8K needle retrieval | 通过，151.976s |
| 32K needle retrieval | `index_topk=4096` 下通过，重启恢复后 497.470s |
| 8K 最优扫参点 | `index_topk=2048`，TTFT 80.313s，101.914 prompt tok/s |

当前结果是 correctness-first 的研究型部署，不是生产级高吞吐 serving
结果。它的价值在于把 AMD ROCm 上的 DeepSeek-V4-Flash 部署问题拆成了可复现的
服务健康检查、语义正确性验证、top-k 扫参和瓶颈定位。

## 仓库结构

```text
.
├── assets/                 # 封面图
├── data/                   # 轻量 CSV 结果数据
├── docs/                   # GitHub Pages、HF 文章、vLLM 报告草稿
├── figures/                # 证据图和结果图
├── notebooks/              # ModelScope Gallery Notebook
├── reports/                # 32K、8K、远端验证报告
├── scripts/                # 启动、健康检查、needle、topk sweep 脚本
└── requirements.txt
```

模型权重没有放入仓库。DeepSeek-V4-Flash 权重体积很大，复现实验时应根据目标环境单独下载或挂载。

## 轻量 Notebook 运行

Notebook 默认 Cell 不下载大模型、不启动重型 vLLM 服务，只读取仓库里的 CSV 数据并生成图表，适合 ModelScope Gallery 自动审核和公开复现。

```bash
python3 -m pip install -r requirements.txt
jupyter lab notebooks/DeepSeek_V4_Flash_AMD_ROCm_vLLM_Gallery.ipynb
```

默认 Cell 会：

- 检查 Python / torch / ROCm 环境；
- 读取 `data/` 中整理好的实验结果；
- 生成 8K topk sweep 和 correctness matrix 图；
- 输出轻量运行摘要。

远端 Gallery 验证记录：

```text
Python: 3.12.13
torch: 2.10.0+git8514f05
torch HIP: 7.2.53211
Code cells executed: 11
Failures: []
Default run status: REMOTE_VALIDATION_OK
```

## 可选 live vLLM 检查

如果本机或 DSW 实例里已经启动了 DeepSeek-V4-Flash vLLM 服务，可以运行：

```bash
python3 scripts/run_openai_gate.py \
  --model deepseek-v4-flash-amd-32k-batch8-16384 \
  --name capital_chat \
  --timeout 120
```

## 可选 32K 服务启动

这一步是重型实验，需要模型权重、兼容的 ROCm vLLM 环境，以及相关兼容补丁已经应用完成。

```bash
export MODEL_DIR=/path/to/DeepSeek-V4-Flash
export SERVED_MODEL_NAME=deepseek-v4-flash-amd-32k-batch8-16384
bash scripts/launch_32k_batch8_16384_system.sh
```

然后运行长上下文 needle retrieval：

```bash
DSV4_PROBE_MODEL=deepseek-v4-flash-amd-32k-batch8-16384 \
python3 scripts/run_needle_retrieval_probe.py \
  --targets 2048 8192 32768 \
  --needle-position begin \
  --timeout 1800
```

## 8K top-k 扫参

```bash
export MODEL_DIR=/path/to/DeepSeek-V4-Flash
export TOPKS="1024 1536 2048 4096"
bash scripts/run_8k_topk_sweep.sh
```

本轮结果中，`index_topk=2048` 是 8K 单请求 TTFT / prefill 最优点；但
32K 开头 needle correctness 需要 `index_topk=4096` 才通过。因此不能简单把一个
top-k 参数当成所有上下文长度的最优配置。

## 结果解读

这次复现最重要的结论不是速度，而是验证链路：

1. 服务健康检查；
2. 短输出 sanity check；
3. 2K / 8K / 32K semantic needle retrieval；
4. 8K `index_topk` 扫参；
5. 32K 负例和 top-k 边界；
6. 后续 ROCm-native kernel 优化方向。

目前主要瓶颈集中在 DeepSeek-V4-Flash 的底层执行路径：

- sparse MLA / sparse attention indexer fallback；
- long-context correctness 对 top-k 候选规模敏感；
- FlashMLA prefill / decode fallback；
- qnorm、RoPE、KV、mHC 融合不足；
- MXFP4 / FP8 MoE ROCm 路径仍需优化；
- `--enforce-eager` 提高稳定性，但牺牲 graph capture 性能空间。

这个 baseline 更适合作为后续 ROCm kernel、AITER、Triton 和投机解码实验的起点。

## 参考资料

- vLLM DeepSeek-V4-Flash recipe:  
  https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM ROCm DeepSeek-V4 tracking issue:  
  https://github.com/vllm-project/vllm/issues/41820
- DeepSeek-V4-Flash model card:  
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- Lambda DeepSeek-V4-Flash benchmark reference:  
  https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash
