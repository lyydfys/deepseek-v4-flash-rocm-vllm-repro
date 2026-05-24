# DeepSeek-V4-Flash 为什么值得在 AMD ROCm 上复现？

我做完这轮实验后，最想分享的不是“跑起来了”，而是下面这个判断：

> 对复杂大模型来说，服务能启动，不等于长上下文真的可用。

## 适合知乎的表达方式

知乎更适合写成“问题 + 结论 + 细节”的结构，所以这篇可以先讲三个问题：

1. 为什么选 DeepSeek-V4-Flash。
2. 为什么在 AMD ROCm 上不是直接起个服务就算完。
3. 我最后是怎么判断“真的跑通了”。

## 可以直接讲的结论

- 我不是只看 `/v1/models`。
- 我还做了短问答、2K、8K、32K 的 needle retrieval。
- 8K 的最优 `index_topk` 和 32K 的正确性 `index_topk` 不是同一个值。
- 有些参数看起来更大，实际会在 KV cache 规划阶段失败。

## 读者容易看懂的叙述

你可以把它讲成一个普通但真实的工程故事：

- 先让服务起来；
- 再让输出别乱码；
- 再让长上下文真能找回隐藏 token；
- 最后才谈 8K 场景下的性能扫参。

这样读者很容易理解，为什么“跑通”不是一个单一动作，而是一串门槛。

## 可保留的关键数据

| 项目 | 结果 |
|---|---:|
| 短问答 | `Paris`, 13.540s |
| 2K needle retrieval | PASS |
| 8K needle retrieval | PASS |
| 32K needle retrieval | PASS at `index_topk=4096` |
| 8K 最优点 | `index_topk=2048` |

## 适合知乎的收束方式

最后可以落到三句话：

- 这次复现让我确认了 DeepSeek-V4-Flash 在 AMD ROCm 上可以形成一条研究 baseline。
- 这条 baseline 更适合做 correctness 和 kernel 优化研究，而不是直接拿来和高端 Nvidia serving 做口径不一致的比较。
- 真正有价值的不是“模型能不能起”，而是“复现链路有没有被整理成别人也能继续做的样子”。

## 适合加的链接

- GitHub 仓库
- GitHub Pages
- Hugging Face Space
