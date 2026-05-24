#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request


MODEL = os.environ.get("DSV4_PROBE_MODEL", "deepseek-v4-flash-amd-32k-batch8-16384")
TOKENIZE_URL = os.environ.get("DSV4_TOKENIZE_URL", "http://127.0.0.1:8000/tokenize")
COMPLETIONS_URL = os.environ.get("DSV4_COMPLETIONS_URL", "http://127.0.0.1:8000/v1/completions")
NEEDLE = os.environ.get("DSV4_NEEDLE", "KITE_7913_ORBIT")
FILLER = os.environ.get(
    "DSV4_NEEDLE_FILLER",
    "This is neutral AMD ROCm deployment filler text for a long-context retrieval benchmark. ",
)


def post_json(url: str, payload: dict, timeout: int) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, repr(exc)


def token_count(prompt: str) -> int:
    status, text = post_json(TOKENIZE_URL, {"model": MODEL, "prompt": prompt}, 120)
    if status != 200:
        raise RuntimeError(f"tokenize failed status={status}: {text[-1000:]}")
    return int(json.loads(text)["count"])


def build_prompt(target_tokens: int, needle_position: str) -> tuple[str, int, int]:
    intro = (
        "You are running a strict retrieval test. Somewhere in this context there is one SECRET_CODE.\n"
        "Return only the exact SECRET_CODE and no other words.\n\n"
    )
    needle = f"BEGIN_NEEDLE\nSECRET_CODE: {NEEDLE}\nEND_NEEDLE\n\n"
    question = "\nQuestion: What is the exact SECRET_CODE?\nAnswer:"

    def assemble(repeats: int) -> str:
        if needle_position == "end":
            return intro + (FILLER * repeats) + "\n" + needle + question
        if needle_position == "middle":
            left = repeats // 2
            right = repeats - left
            return intro + (FILLER * left) + "\n" + needle + (FILLER * right) + question
        return intro + needle + (FILLER * repeats) + question

    lo, hi = 0, max(2, target_tokens)
    best_repeat, best_count = 0, token_count(assemble(0))
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = assemble(mid)
        n = token_count(prompt)
        if n <= target_tokens:
            best_repeat, best_count = mid, n
            lo = mid + 1
        else:
            hi = mid - 1
    return assemble(best_repeat), best_count, best_repeat


def run_case(target_tokens: int, max_tokens: int, timeout: int, needle_position: str) -> None:
    build_started = time.time()
    prompt, actual, repeats = build_prompt(target_tokens, needle_position)
    print(
        f"NEEDLE_BUILT target={target_tokens} actual={actual} repeats={repeats} position={needle_position} "
        f"build_seconds={time.time() - build_started:.2f}",
        flush=True,
    )
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    started = time.time()
    print(f"NEEDLE_START target={target_tokens} actual={actual} max_tokens={max_tokens}", flush=True)
    status, text = post_json(COMPLETIONS_URL, payload, timeout)
    elapsed = time.time() - started
    parsed_text = ""
    try:
        parsed = json.loads(text)
        parsed_text = parsed.get("choices", [{}])[0].get("text", "")
    except Exception:
        pass
    print(f"NEEDLE_DONE target={target_tokens} actual={actual} seconds={elapsed:.2f} status={status}", flush=True)
    print(f"NEEDLE_EXPECTED {NEEDLE}", flush=True)
    print(f"NEEDLE_OUTPUT {parsed_text!r}", flush=True)
    print(f"NEEDLE_CONTAINS {NEEDLE in parsed_text}", flush=True)
    print(text[-4000:], flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Needle retrieval semantic probe for DeepSeek V4 Flash long context")
    parser.add_argument("--target", type=int, action="append", dest="target")
    parser.add_argument("--targets", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--needle-position", choices=["begin", "middle", "end"], default="begin")
    args = parser.parse_args()
    merged = []
    if args.targets:
        merged.extend(args.targets)
    if args.target:
        merged.extend(args.target)
    if not merged:
        merged = [8192]
    args.targets = merged
    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        f"NEEDLE_CONFIG model={MODEL} targets={args.targets} max_tokens={args.max_tokens} "
        f"timeout={args.timeout} needle={NEEDLE} position={args.needle_position}",
        flush=True,
    )
    for target in args.targets:
        run_case(target, args.max_tokens, args.timeout, args.needle_position)
