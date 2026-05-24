#!/usr/bin/env python3
import argparse
import json
import sys
import time

import requests


PROMPTS = {
    "capital_completion": {
        "endpoint": "/v1/completions",
        "payload": {
            "prompt": "Question: What is the capital of France? Answer:",
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
        "expect": "Paris",
    },
    "count_completion": {
        "endpoint": "/v1/completions",
        "payload": {
            "prompt": "Question: Count 2 plus 3. Answer:",
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
        "expect": "5",
    },
    "needle0_completion": {
        "endpoint": "/v1/completions",
        "payload": {
            "prompt": "Needle: cobalt-7391.\nQuestion: What is the needle token? Return only the token.\nAnswer:",
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        },
        "expect": "cobalt-7391",
    },
    "capital_chat": {
        "endpoint": "/v1/chat/completions",
        "payload": {
            "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        },
        "expect": "Paris",
    },
}


def make_needle_tokens(tokens: int) -> str:
    filler = " ".join(["padding"] * max(tokens - 40, 1))
    return (
        "Needle: cobalt-7391.\n"
        f"{filler}\n"
        "Question: What is the needle token? Return only the token.\n"
        "Answer:"
    )


def extract_text(endpoint: str, body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if endpoint == "/v1/chat/completions":
        return (choice.get("message") or {}).get("content") or ""
    return choice.get("text") or ""


def run_one(base: str, model: str, name: str, timeout: int) -> int:
    if name.startswith("needle") and name.endswith("_tokens"):
        token_count = int(name.removeprefix("needle").removesuffix("_tokens"))
        spec = {
            "endpoint": "/v1/completions",
            "payload": {
                "prompt": make_needle_tokens(token_count),
                "max_tokens": 16,
                "temperature": 0,
                "stream": False,
            },
            "expect": "cobalt-7391",
        }
    else:
        spec = PROMPTS[name]

    payload = dict(spec["payload"])
    payload["model"] = model
    endpoint = spec["endpoint"]
    start = time.time()
    print(f"GATE_START name={name} endpoint={endpoint} timeout={timeout}", flush=True)
    try:
        response = requests.post(base + endpoint, json=payload, timeout=timeout)
    except Exception as exc:
        print(f"GATE_EXCEPTION name={name} elapsed={time.time()-start:.3f} type={type(exc).__name__} msg={str(exc)[:240]}", flush=True)
        return 2

    elapsed = time.time() - start
    text = response.text[:500].replace("\n", "|")
    print(f"GATE_HTTP name={name} status={response.status_code} elapsed={elapsed:.3f} body={text}", flush=True)
    if response.status_code != 200:
        return 3
    try:
        body = response.json()
    except json.JSONDecodeError:
        return 4
    generated = extract_text(endpoint, body)
    compact = generated.replace("\n", "|")[:240]
    ok = spec["expect"].lower() in generated.lower()
    print(f"GATE_TEXT name={name} ok={ok} generated={compact!r}", flush=True)
    return 0 if ok else 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    return run_one(args.base, args.model, args.name, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
