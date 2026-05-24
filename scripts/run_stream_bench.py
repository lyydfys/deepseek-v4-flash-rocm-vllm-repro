#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from typing import Iterable

import requests


def post_json(base: str, path: str, payload: dict, timeout: int) -> dict:
    response = requests.post(base + path, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def token_count(base: str, model: str, prompt: str, timeout: int) -> int:
    return int(post_json(base, "/tokenize", {"model": model, "prompt": prompt}, timeout)["count"])


def make_prompt(base: str, model: str, target_tokens: int, timeout: int) -> tuple[str, int]:
    segment = "AMD ROCm DeepSeek V4 Flash 32K benchmark filler text. "
    suffix = "\nFinal instruction: answer with exactly PASS."
    lo, hi = 1, max(2, target_tokens)
    best_prompt = segment + suffix
    best_count = token_count(base, model, best_prompt, timeout)
    while token_count(base, model, segment * hi + suffix, timeout) <= target_tokens:
        best_prompt = segment * hi + suffix
        best_count = token_count(base, model, best_prompt, timeout)
        hi *= 2
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = segment * mid + suffix
        count = token_count(base, model, prompt, timeout)
        if count <= target_tokens:
            best_prompt, best_count = prompt, count
            lo = mid + 1
        else:
            hi = mid - 1
    return best_prompt, best_count


def iter_sse_lines(response: requests.Response) -> Iterable[str]:
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith("data: "):
            yield raw[6:]


def stream_completion(base: str, payload: dict, timeout: int) -> dict:
    start = time.time()
    first = None
    last = None
    chunks = 0
    text_parts: list[str] = []
    usage = None
    with requests.post(base + "/v1/completions", json=payload, timeout=timeout, stream=True) as response:
        status = response.status_code
        response.raise_for_status()
        for data in iter_sse_lines(response):
            now = time.time()
            if data.strip() == "[DONE]":
                break
            chunks += 1
            if first is None:
                first = now
            last = now
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                text_parts.append(choice.get("text") or "")
    elapsed = time.time() - start
    generated = "".join(text_parts)
    completion_tokens = None
    if usage:
        completion_tokens = usage.get("completion_tokens")
    return {
        "status": status,
        "elapsed_s": elapsed,
        "ttft_s": None if first is None else first - start,
        "last_token_s": None if last is None else last - start,
        "chunks": chunks,
        "generated": generated,
        "usage": usage,
        "completion_tokens": completion_tokens,
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    p50 = statistics.median(values)
    p95 = values[int((len(values) - 1) * 0.95)]
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "p50": p50,
        "p95": p95,
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tokens", type=int, default=32720)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-jsonl")
    args = parser.parse_args()

    prompt, prompt_tokens = make_prompt(args.base, args.model, args.target_tokens, args.timeout)
    print(f"STREAM_BENCH_CONFIG model={args.model} target={args.target_tokens} prompt_tokens={prompt_tokens} max_tokens={args.max_tokens} runs={args.runs}", flush=True)

    rows = []
    for run_id in range(args.runs):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.time()
        try:
            row = stream_completion(args.base, payload, args.timeout)
            row["ok"] = True
        except Exception as exc:
            row = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "elapsed_s": time.time() - started,
                "ttft_s": None,
            }
        row["run_id"] = run_id
        row["prompt_tokens_expected"] = prompt_tokens
        row["prefill_tok_s"] = None if not row.get("ttft_s") else prompt_tokens / row["ttft_s"]
        if row.get("last_token_s") and row.get("ttft_s") and row.get("completion_tokens"):
            decode_tokens = max(int(row["completion_tokens"]) - 1, 0)
            decode_span = max(row["last_token_s"] - row["ttft_s"], 1e-9)
            row["decode_tok_s"] = decode_tokens / decode_span
        else:
            row["decode_tok_s"] = None
        rows.append(row)
        compact = dict(row)
        compact["generated"] = (compact.get("generated") or "")[:120].replace("\n", "|")
        print("STREAM_BENCH_ROW " + json.dumps(compact, ensure_ascii=False, sort_keys=True), flush=True)

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("STREAM_BENCH_SUMMARY " + json.dumps({
        "runs": len(rows),
        "ok": sum(1 for row in rows if row.get("ok")),
        "prompt_tokens": prompt_tokens,
        "ttft_s": summarize([row["ttft_s"] for row in rows if row.get("ttft_s") is not None]),
        "prefill_tok_s": summarize([row["prefill_tok_s"] for row in rows if row.get("prefill_tok_s") is not None]),
        "decode_tok_s": summarize([row["decode_tok_s"] for row in rows if row.get("decode_tok_s") is not None]),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
