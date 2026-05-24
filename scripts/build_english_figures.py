#!/usr/bin/env python3
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures_en"


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    boxes = [
        (0.4, 3.6, 2.0, 1.2, "ModelScope DSW\nAMD GPU + ROCm", "#e7f0ff"),
        (3.0, 3.6, 2.0, 1.2, "Model cache\nDeepSeek-V4-Flash", "#e8f7ee"),
        (5.6, 3.6, 2.0, 1.2, "Runtime model tree\nsymlinked weights", "#fff4dd"),
        (8.2, 3.6, 2.0, 1.2, "vLLM patch chain\nROCm/AITER fallback", "#f5e8ff"),
        (10.8, 3.6, 2.0, 1.2, "vLLM OpenAI API\n/v1/models OK", "#e6f7ff"),
        (4.3, 1.0, 2.1, 1.1, "8K semantic gate\nPASS", "#e8f7ee"),
        (6.9, 1.0, 2.1, 1.1, "32K semantic gate\nPASS @ topk4096", "#e8f7ee"),
        (9.5, 1.0, 2.1, 1.1, "Best 8K sweep point\ntopk2048", "#fff4dd"),
    ]

    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.08",
                linewidth=1.3,
                edgecolor="#273142",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#273142"))

    arrow(2.45, 4.2, 2.95, 4.2)
    arrow(5.05, 4.2, 5.55, 4.2)
    arrow(7.65, 4.2, 8.15, 4.2)
    arrow(10.25, 4.2, 10.75, 4.2)
    arrow(11.8, 3.55, 10.6, 2.15)
    arrow(11.0, 3.55, 8.0, 2.15)
    arrow(10.2, 3.55, 5.4, 2.15)

    ax.text(
        0.4,
        5.35,
        "Figure 1. Reproducible deployment path: keep weights separate, preserve scripts, patches, reports, and evidence",
        fontsize=12.5,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.4,
        0.25,
        "Core idea: vLLM remains the serving layer; unstable DeepSeek-V4-Flash ROCm paths are guarded with AITER/PyTorch fallback, then verified by semantic gates.",
        fontsize=10.2,
        color="#374151",
    )
    save(fig, "01_rocm_fallback_pipeline.png")


def draw_8k_topk_perf() -> None:
    df = pd.read_csv(DATA_DIR / "8k_topk_sweep.csv")
    order = [1024, 1536, 2048, 4096]
    df = df.set_index("index_topk").loc[order].reset_index()
    labels = [str(x) if x != 4096 else "4096\nbaseline" for x in df["index_topk"]]
    x = np.arange(len(df))
    colors = ["#74a9ff", "#74a9ff", "#28b487", "#b8c2cc"]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    bars = ax1.bar(x, df["ttft_s"], width=0.55, color=colors)
    ax1.set_ylabel("Time to first token (s)")
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("index_topk")
    ax1.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, df["ttft_s"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}s", ha="center", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, df["effective_prefill_tok_s"], color="#e26d5a", marker="o", linewidth=2.5)
    ax2.set_ylabel("Effective prefill rate (tok/s)")
    ax2.set_ylim(70, 110)
    for xi, value in zip(x, df["effective_prefill_tok_s"]):
        ax2.text(xi, value + 1.2, f"{value:.1f}", ha="center", fontsize=10, color="#9b2c1f")

    ax1.set_title("Figure 2. 8K top-k sweep: topk2048 was the best single-request point in this run")
    ax1.text(
        -0.45,
        -18,
        "All 8K needle gates passed. topk2048 is not a 32K-safe setting; topk4096 remains the 32K correctness baseline.",
        transform=ax1.transData,
        fontsize=9.8,
        color="#374151",
    )
    fig.tight_layout()
    save(fig, "02_8k_topk_perf.png")


def draw_correctness_matrix() -> None:
    rows = ["topk512", "topk2048", "topk3072", "topk3584", "topk4096"]
    cols = ["8K begin", "8K middle", "8K end", "16K begin", "32K begin"]
    data = np.array(
        [
            [0, 1, 1, -1, -1],
            [1, -1, -1, 1, 0],
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, 0],
            [-1, -1, -1, -1, 1],
        ]
    )
    colors = {1: "#33a36f", 0: "#d95f5f", -1: "#e5e7eb"}
    labels = {1: "PASS", 0: "FAIL", -1: "N/T"}

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)

    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(data[i, j])
            ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=colors[val], edgecolor="white", linewidth=2))
            ax.text(
                j + 0.5,
                i + 0.5,
                labels[val],
                ha="center",
                va="center",
                color="#111827" if val == -1 else "white",
                fontsize=11,
                weight="bold",
            )

    ax.set_title("Figure 3. Semantic correctness matrix: 8K and 32K require different top-k settings")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, "03_context_correctness_matrix.png")


def draw_evidence_card(title: str, sections: list[tuple[str, list[str], str]], name: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    ax.axis("off")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.8)
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.25, 0.25),
            12.0,
            7.3,
            boxstyle="round,pad=0.12,rounding_size=0.08",
            linewidth=1.3,
            edgecolor="#273142",
            facecolor="#fbfdff",
        )
    )
    ax.text(0.55, 7.15, title, fontsize=17, weight="bold", color="#111827")
    y = 6.4
    for heading, body, color in sections:
        ax.add_patch(
            patches.FancyBboxPatch((0.55, y - 0.22), 11.4, 0.45, boxstyle="round,pad=0.05,rounding_size=0.04", linewidth=0, facecolor=color)
        )
        ax.text(0.75, y, heading, fontsize=12.5, weight="bold", va="center", color="#111827")
        y -= 0.55
        for line in body:
            ax.text(0.85, y, line, fontsize=10.8, va="center", color="#1f2937")
            y -= 0.42
        y -= 0.25
    save(fig, name)


def draw_evidence_cards() -> None:
    draw_evidence_card(
        "Figure 4. Service health and 32K correctness evidence",
        [
            (
                "Service health",
                ["/v1/models -> HTTP 200", "served model -> deepseek-v4-flash-amd-32k-batch8-16384", "short QA gate -> generated 'Paris', check passed"],
                "#e6f7ff",
            ),
            (
                "32K semantic path",
                ["32K needle retrieval: prompt_tokens=32768", "HTTP 200, output contained cobalt-7391", "elapsed=497.470s", "post-32K health gate -> HTTP 200, generated 5, elapsed=1.652s"],
                "#e8f7ee",
            ),
            (
                "Conclusion",
                ["Service startup is not enough; long context needs semantic retrieval evidence.", "The current 32K correctness baseline uses index_topk=4096."],
                "#fff4dd",
            ),
        ],
        "04_service_and_correctness_evidence.png",
    )

    draw_evidence_card(
        "Figure 5. Top-k sweep and negative evidence",
        [
            (
                "8K sweep",
                [
                    "topk1024 -> 8K PASS, TTFT=95.217s, prefill=85.962 tok/s",
                    "topk1536 -> 8K PASS, TTFT=94.250s, prefill=86.844 tok/s",
                    "topk2048 -> 8K PASS, TTFT=80.313s, prefill=101.914 tok/s",
                    "topk4096 -> 8K PASS, TTFT=96.589s, prefill=84.740 tok/s",
                ],
                "#e7f0ff",
            ),
            (
                "32K negative cases",
                [
                    "topk2048 -> 32K begin FAIL, missing key fragment",
                    "topk3072 -> 32K begin FAIL, incorrect token",
                    "topk3584 -> 32K begin FAIL, incomplete token",
                    "topk4096 -> 32K begin PASS, full key matched",
                ],
                "#fde8e8",
            ),
            (
                "Tuning boundary",
                ["The best 8K point is not the stable 32K setting.", "Use topk2048 for 8K exploration and topk4096 for the 32K correctness baseline."],
                "#fff4dd",
            ),
        ],
        "05_topk_and_negative_evidence.png",
    )


def main() -> None:
    setup()
    draw_pipeline()
    draw_8k_topk_perf()
    draw_correctness_matrix()
    draw_evidence_cards()


if __name__ == "__main__":
    main()
