# Publishing Guide

This guide describes the safest publication path for the reproducibility package.

## 1. GitHub repository

Recommended repository name:

```text
deepseek-v4-flash-rocm-vllm-repro
```

Suggested description:

```text
Reproducing DeepSeek-V4-Flash on AMD ROCm with vLLM: 32K correctness, 8K top-k sweep, and ModelScope DSW notebook baseline.
```

Before publishing, run:

```bash
git status
git lfs ls-files
find . -type f -size +50M -print
```

This repository should not contain model weights, caches, or local service logs.

If GitHub CLI is available, the publish commands are:

```bash
gh auth login
gh repo create deepseek-v4-flash-rocm-vllm-repro --public --source . --remote origin --push
```

If the repository is created manually on GitHub first, use:

```bash
git remote add origin https://github.com/<your-user>/deepseek-v4-flash-rocm-vllm-repro.git
git push -u origin main
```

## 2. vLLM issue or discussion comment

Use `docs/VLLM_ROCM_REPRO_REPORT.md` as the source. After the GitHub repository
is public, replace:

```text
Repository URL: <add GitHub URL after publishing>
```

with the real URL, then post it under the relevant vLLM DeepSeek-V4 ROCm
tracking issue or discussion.

The tone should stay as a reproduction report, not an upstream support claim.

Published comment:

```text
https://github.com/vllm-project/vllm/issues/41820#issuecomment-4529203579
```

## 3. Hugging Face article

Use `docs/HF_ARTICLE.md` as the article body.

Suggested title:

```text
Reproducing DeepSeek-V4-Flash on AMD ROCm with vLLM: 32K Correctness and TopK Sweep
```

Suggested tags:

```text
ROCm, AMD GPU, vLLM, DeepSeek-V4-Flash, LLM inference
```

Published Dataset mirror:

```text
https://huggingface.co/datasets/lyydfys/deepseek-v4-flash-rocm-vllm-repro
```

## 4. GitHub Pages

Use `docs/GITHUB_PAGES_ARTICLE.md` as a simple page source. It can be copied into
`docs/index.md` if GitHub Pages is enabled from the `docs/` directory.
