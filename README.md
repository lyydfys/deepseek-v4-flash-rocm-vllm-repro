# Reproducing DeepSeek-V4-Flash on AMD ROCm with vLLM

This repository is a reproducibility package for a user-side engineering study of
DeepSeek-V4-Flash on an AMD ROCm ModelScope DSW instance.

Public repository:
https://github.com/lyydfys/deepseek-v4-flash-rocm-vllm-repro

GitHub Pages:
https://lyydfys.github.io/deepseek-v4-flash-rocm-vllm-repro/

vLLM ROCm tracking issue reproduction comment:
https://github.com/vllm-project/vllm/issues/41820#issuecomment-4529203579

The goal is not to claim a new upstream ROCm backend implementation. The goal is
to preserve a runnable, reviewable baseline that shows how the model was brought
up through vLLM, ROCm/AITER/PyTorch fallback paths, long-context correctness
checks, and an 8K `index_topk` sweep.

![Cover](assets/cover.png)

## What is included

- A ModelScope Gallery-ready notebook:
  `notebooks/DeepSeek_V4_Flash_AMD_ROCm_vLLM_Gallery.ipynb`
- Lightweight CSV result data under `data/`
- Reproducibility scripts under `scripts/`
- Evidence figures under `figures/`
- Experiment reports under `reports/`
- Draft public writeups under `docs/`

Model weights are intentionally not included. DeepSeek-V4-Flash weights are large
and should be downloaded or mounted separately according to the target runtime.

## Key results from this run

| Gate | Result |
|---|---:|
| vLLM OpenAI API server | `/v1/models` returned HTTP 200 |
| Short completion gate | HTTP 200, `Paris`, 13.540s |
| 2K needle retrieval | PASS, 53.376s |
| 8K needle retrieval | PASS, 151.976s |
| 32K needle retrieval | PASS at `index_topk=4096`, 497.470s restart run |
| Best 8K sweep point | `index_topk=2048`, TTFT 80.313s, 101.914 prompt tok/s |

The current baseline is correctness-first and fallback-heavy. It is useful for
ROCm research and reproducibility, but it is not a production-grade high
throughput serving result.

## Hardware and software context

The validation run used a ModelScope DSW AMD GPU instance with ROCm enabled.
The Gallery notebook default cells were also validated remotely without
downloading model weights or starting the full vLLM service.

Remote Gallery validation summary:

```text
Python: 3.12.13
torch: 2.10.0+git8514f05
torch HIP: 7.2.53211
Code cells executed: 11
Failures: []
Default run status: REMOTE_VALIDATION_OK
```

See `reports/modelscope_gallery_remote_validation.md` for details.

## Repository layout

```text
.
├── assets/
│   └── cover.png
├── data/
│   ├── 8k_topk_sweep.csv
│   ├── context_correctness.csv
│   ├── negative_cases.csv
│   ├── service_checks.csv
│   └── compat_patch_summary.csv
├── docs/
│   ├── HF_ARTICLE.md
│   ├── GITHUB_PAGES_ARTICLE.md
│   ├── PUBLISHING_GUIDE.md
│   └── VLLM_ROCM_REPRO_REPORT.md
├── figures/
├── notebooks/
├── reports/
├── scripts/
└── requirements.txt
```

## Run the lightweight notebook

The notebook is designed to pass in a normal ModelScope DSW AMD GPU environment
without downloading the model or launching a heavyweight service.

```bash
python3 -m pip install -r requirements.txt
jupyter lab notebooks/DeepSeek_V4_Flash_AMD_ROCm_vLLM_Gallery.ipynb
```

The default cells:

- inspect the Python / ROCm / torch environment when available;
- load the recorded result CSV files;
- regenerate the correctness and top-k analysis charts;
- write lightweight output summaries.

## Optional live vLLM service check

If a DeepSeek-V4-Flash vLLM server is already running locally, enable the live
check cell or run:

```bash
python3 scripts/run_openai_gate.py \
  --model deepseek-v4-flash-amd-32k-batch8-16384 \
  --name capital_chat \
  --timeout 120
```

## Optional 32K service launch

This command is heavyweight and requires model weights, a compatible ROCm vLLM
environment, and the relevant compatibility patches already applied.

```bash
export MODEL_DIR=/path/to/DeepSeek-V4-Flash
export SERVED_MODEL_NAME=deepseek-v4-flash-amd-32k-batch8-16384
bash scripts/launch_32k_batch8_16384_system.sh
```

Then run a semantic needle test:

```bash
DSV4_PROBE_MODEL=deepseek-v4-flash-amd-32k-batch8-16384 \
python3 scripts/run_needle_retrieval_probe.py \
  --targets 2048 8192 32768 \
  --needle-position begin \
  --timeout 1800
```

## Optional 8K top-k sweep

```bash
export MODEL_DIR=/path/to/DeepSeek-V4-Flash
export TOPKS="1024 1536 2048 4096"
bash scripts/run_8k_topk_sweep.sh
```

This script patches `index_topk`, restarts the local server, runs an 8K needle
gate, and records an 8K one-token streaming benchmark.

## Interpretation

The strongest result in this package is not raw serving speed. The value is the
reproducible chain:

1. service health gate;
2. short output sanity check;
3. 2K/8K/32K semantic needle retrieval;
4. 8K `index_topk` sweep;
5. negative cases that show why a lower top-k can fail at 32K;
6. clear bottleneck mapping for future ROCm-native kernel work.

In this run, `index_topk=2048` was best for the 8K single-request TTFT probe, but
`index_topk=4096` was needed for the verified 32K begin-position needle case.

## References

- vLLM DeepSeek-V4-Flash recipe:
  https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM ROCm DeepSeek-V4 tracking issue:
  https://github.com/vllm-project/vllm/issues/41820
- DeepSeek-V4-Flash model card:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- Lambda DeepSeek-V4-Flash benchmark reference:
  https://lambda.ai/inference-models/deepseek-ai/deepseek-v4-flash
