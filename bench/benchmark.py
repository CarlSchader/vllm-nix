#!/usr/bin/env python3
"""Stdlib OpenAI SSE benchmark; token rates use server usage, never SSE chunks."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


TASKS = (
    "Implement a Python TTL cache with bounded capacity, monotonic expiry, and unit tests.",
    "Review and implement an async job queue with cancellation, retries, and graceful shutdown.",
    "Write a streaming CSV aggregator that validates rows and reports errors with line numbers.",
    "Design a SQLite migration runner with transactions, version tracking, and rollback tests.",
    "Implement a file synchronization planner using hashes; cover renames and conflict detection.",
    "Write an HTTP retry policy with exponential backoff, jitter, and Retry-After handling.",
)


def make_payload(args, index, warmup=False):
    phase = "warmup" if warmup else "measured"
    identity = f"{args.seed}:{phase}:{index}"
    nonce = hashlib.sha256(identity.encode()).hexdigest()[:24]
    rng = random.Random(identity)
    task = rng.choice(TASKS)
    details = []
    for section in range(args.prompt_repeat):
        details.append(
            f"Specification {section + 1}: {task} "
            f"Use workload size {rng.randint(100, 10000)} and budget "
            f"{rng.randint(10, 500)} ms. Explain edge cases, provide runnable code, "
            "and include deterministic tests. Prefer standard library solutions."
        )
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": f"{nonce}\n" + "\n".join(details)}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
        "ignore_eos": not args.allow_eos,
    }


def encode_payload(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sse_events(response, raw):
    """Parse SSE framing (including multi-line data); retain exact UTF-8 wire text."""
    data = []
    for line_bytes in response:
        line = line_bytes.decode("utf-8")
        raw.append(line)
        text = line.rstrip("\r\n")
        if not text:
            if data:
                yield "\n".join(data)
                data = []
        elif text.startswith("data:"):
            value = text[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        raise ValueError("truncated SSE event (missing blank line)")


def run_request(base_url, payload, timeout=600, opener=None, clock=None):
    opener = opener or urllib.request.urlopen
    clock = clock or time.perf_counter
    wire = encode_payload(payload)
    result = {
        "payload": payload, "request_body": wire.decode(),
        "payload_sha256": hashlib.sha256(wire).hexdigest(),
        "success": False, "errors": [], "content": "", "reasoning": "",
        "usage": None, "prompt_tokens": None, "completion_tokens": None,
        "reasoning_tokens": None, "ttft_s": None, "latency_s": None,
        "decode_tokens_per_s_approx": None,
    }
    raw = []
    start = clock()
    first = None
    last = None
    done = False
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=wire,
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        with opener(req, timeout=timeout) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"HTTP status {response.status}")
            for event in sse_events(response, raw):
                if event.strip() == "[DONE]":
                    done = True
                    break
                chunk = json.loads(event)
                if chunk.get("error"):
                    raise ValueError(f"server error: {chunk['error']}")
                if chunk.get("usage") is not None:
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content or reasoning:
                        now = clock()
                        if first is None:
                            first = now
                        last = now
                        result["content"] += content
                        result["reasoning"] += reasoning
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
        if not done:
            result["errors"].append("truncated SSE: missing [DONE]")
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["http_error_body"] = exc.read().decode("utf-8", errors="replace")
        result["errors"].append(f"HTTP {exc.code}: {exc.reason}")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["latency_s"] = clock() - start
    result["raw_sse"] = "".join(raw)
    if first is not None:
        result["ttft_s"] = first - start
    else:
        result["errors"].append("empty output")
    usage = result["usage"]
    if not isinstance(usage, dict):
        result["errors"].append("missing usage")
    else:
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if type(value) is not int or value < 0:
                result["errors"].append(f"missing/invalid usage {key}")
            else:
                result[key] = value
        details = usage.get("completion_tokens_details") or {}
        result["reasoning_tokens"] = details.get("reasoning_tokens", usage.get("reasoning_tokens")) if isinstance(details, dict) else None
    if result.get("finish_reason") not in ("stop", "length"):
        result["errors"].append(f"missing/invalid finish_reason: {result.get('finish_reason')!r}")
    tokens = result["completion_tokens"]
    if tokens is not None:
        if tokens == 0:
            result["errors"].append("zero completion tokens")
        if tokens > payload["max_tokens"]:
            result["errors"].append(f"output exceeds max_tokens: {tokens} > {payload['max_tokens']}")
        if payload["ignore_eos"] and tokens < payload["max_tokens"]:
            result["errors"].append(f"short output with ignore_eos: {tokens} < {payload['max_tokens']}")
        if first is not None and last > first:
            result["decode_tokens_per_s_approx"] = tokens / (last - first)
    result["success"] = not result["errors"]
    return result


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {key: None for key in ("p50", "p90", "p95", "p99")}
    def at(p):
        pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {f"p{p}": at(p / 100) for p in (50, 90, 95, 99)}


def summarize(results, wall):
    good = [r for r in results if r["success"]]
    return {
        "requests": len(results), "succeeded": len(good), "failed": len(results) - len(good),
        "wall_time_s": wall,
        "successful_output_tokens": sum(r["completion_tokens"] for r in good),
        "aggregate_output_tokens_per_s": sum(r["completion_tokens"] for r in good) / wall if wall > 0 else None,
        "percentiles_successful_only": {key: percentiles([r[key] for r in good]) for key in
                                       ("ttft_s", "latency_s", "decode_tokens_per_s_approx")},
    }


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI API URL ending in /v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--requests", type=positive_int, default=3)
    parser.add_argument("--max-tokens", type=positive_int, default=512)
    parser.add_argument("--prompt-repeat", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--allow-eos", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_url.rstrip("/").endswith("/v1"):
        parser.error("--base-url must end in /v1")
    warmup = run_request(args.base_url, make_payload(args, 0, warmup=True))
    payloads = [make_payload(args, i) for i in range(args.requests)]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda payload: run_request(args.base_url, payload), payloads))
    wall = time.perf_counter() - start
    report = {
        "schema_version": 1, "label": args.label, "options": vars(args),
        "effective_generation_options": {key: payloads[0][key] for key in
                                         ("temperature", "seed", "max_tokens", "ignore_eos", "chat_template_kwargs", "stream_options")},
        "metric_notes": {
            "decode_tokens_per_s_approx": "Server completion_tokens / time from first to last nonempty content or reasoning delta. Includes first-chunk tokens; chunk boundaries are not token boundaries (especially with speculative decoding). Undefined for one chunk.",
            "aggregate_output_tokens_per_s": "Successful measured completion_tokens / measured wall time, including failed request time; warmup excluded.",
            "reasoning_tokens": "Server-reported only; null means unavailable. Not added to completion_tokens.",
            "percentiles": "Linear interpolation over successful requests only; failed runs are not valid comparisons.",
        },
        "warmup": warmup, "results": results, "summary": summarize(results, wall),
        "success": warmup["success"] and all(r["success"] for r in results),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "output": args.output, **report["summary"]}))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
