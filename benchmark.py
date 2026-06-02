#!/usr/bin/env python3
"""Benchmark script for Ollama models — measures TTFT, tokens/sec, and throughput."""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

DEFAULT_HOST = "http://172.26.112.1:11434"
DEFAULT_MODEL = "qwen3:30b-a3b"

PROMPTS = {
    "short": "What is 2 + 2? Answer in one word.",
    "medium": (
        "Explain the difference between a process and a thread in operating systems. "
        "Be concise but thorough."
    ),
    "long": (
        "You are an expert software engineer. Write a Python function that implements "
        "merge sort. Include type hints, a brief docstring, and a short example showing "
        "it works on a list of integers. Then explain the time and space complexity."
    ),
    "reasoning": (
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? "
        "Think step by step."
    ),
    "code": (
        "Write a Python function that checks if a string is a valid IPv4 address "
        "without using any external libraries. Include edge cases in the implementation."
    ),
}

GENERATION_TOKENS = 512  # max tokens to generate per run


@dataclass
class RunResult:
    prompt_name: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float           # time to first token
    total_time_s: float
    gen_tokens_per_sec: float
    prompt_tokens_per_sec: float
    full_response: str = field(repr=False)


def stream_generate(host: str, model: str, prompt: str, num_predict: int) -> Iterator[dict]:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,
        },
    }
    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def run_single(host: str, model: str, prompt_name: str, prompt: str, num_predict: int) -> RunResult:
    first_token_time: float | None = None
    start = time.perf_counter()
    chunks = []
    final_stats: dict = {}

    for chunk in stream_generate(host, model, prompt, num_predict):
        now = time.perf_counter()
        if first_token_time is None and chunk.get("response"):
            first_token_time = now
        if chunk.get("response"):
            chunks.append(chunk["response"])
        if chunk.get("done"):
            final_stats = chunk
            break

    end = time.perf_counter()

    if first_token_time is None:
        first_token_time = end

    total_time = end - start
    ttft_ms = (first_token_time - start) * 1000

    prompt_tokens = final_stats.get("prompt_eval_count", 0)
    generated_tokens = final_stats.get("eval_count", 0)

    # Ollama also reports these in nanoseconds
    eval_duration_ns = final_stats.get("eval_duration", 0)
    prompt_eval_duration_ns = final_stats.get("prompt_eval_duration", 0)

    if eval_duration_ns > 0:
        gen_tps = generated_tokens / (eval_duration_ns / 1e9)
    elif total_time > 0:
        gen_tps = generated_tokens / total_time
    else:
        gen_tps = 0.0

    if prompt_eval_duration_ns > 0:
        prompt_tps = prompt_tokens / (prompt_eval_duration_ns / 1e9)
    else:
        prompt_tps = 0.0

    return RunResult(
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        total_time_s=total_time,
        gen_tokens_per_sec=gen_tps,
        prompt_tokens_per_sec=prompt_tps,
        full_response="".join(chunks),
    )


def check_model_available(host: str, model: str) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json()
        names = [m["name"] for m in tags.get("models", [])]
        return any(model == n or model == n.split(":")[0] for n in names)
    except Exception:
        return False


def pull_model(host: str, model: str) -> None:
    print(f"Pulling {model} (this may take a while)...")
    url = f"{host}/api/pull"
    with requests.post(url, json={"name": model}, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                if total:
                    pct = completed / total * 100
                    print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
                else:
                    print(f"\r  {status}", end="", flush=True)
    print()


def print_result(r: RunResult, show_response: bool = False) -> None:
    print(f"\n  [{r.prompt_name}]")
    print(f"    Prompt tokens      : {r.prompt_tokens}")
    print(f"    Generated tokens   : {r.generated_tokens}")
    print(f"    Time to first token: {r.ttft_ms:.0f} ms")
    print(f"    Total time         : {r.total_time_s:.2f} s")
    print(f"    Generation speed   : {r.gen_tokens_per_sec:.1f} tok/s")
    if r.prompt_tokens_per_sec > 0:
        print(f"    Prompt eval speed  : {r.prompt_tokens_per_sec:.1f} tok/s")
    if show_response:
        print(f"\n    Response:\n    {r.full_response[:400]}{'...' if len(r.full_response) > 400 else ''}")


def print_summary(results: list[RunResult]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    tps_values = [r.gen_tokens_per_sec for r in results if r.gen_tokens_per_sec > 0]
    ttft_values = [r.ttft_ms for r in results]
    if tps_values:
        print(f"  Generation speed  — avg: {statistics.mean(tps_values):.1f} tok/s"
              f"  min: {min(tps_values):.1f}  max: {max(tps_values):.1f}")
    if ttft_values:
        print(f"  Time to 1st token — avg: {statistics.mean(ttft_values):.0f} ms"
              f"  min: {min(ttft_values):.0f}  max: {max(ttft_values):.0f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an Ollama model")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS),
                        help="Which prompts to run (default: all)")
    parser.add_argument("--num-predict", type=int, default=GENERATION_TOKENS,
                        help=f"Max tokens to generate (default: {GENERATION_TOKENS})")
    parser.add_argument("--show-responses", action="store_true", help="Print model responses")
    parser.add_argument("--pull", action="store_true", help="Pull the model before benchmarking")
    parser.add_argument("--localhost", action="store_true",
                        help="Use localhost instead of Windows host IP (if Ollama runs in WSL)")
    args = parser.parse_args()

    host = "http://localhost:11434" if args.localhost else args.host

    print(f"Ollama host : {host}")
    print(f"Model       : {args.model}")
    print(f"Max tokens  : {args.num_predict}")

    # Connectivity check
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"\nError: cannot reach Ollama at {host}")
        print(f"  {e}")
        print("\nMake sure Ollama is running. If on Windows with WSL, start it with:")
        print("  $env:OLLAMA_HOST='0.0.0.0'; ollama serve")
        sys.exit(1)

    if args.pull:
        pull_model(host, args.model)

    if not check_model_available(host, args.model):
        print(f"\nModel '{args.model}' not found. Available models:")
        tags = requests.get(f"{host}/api/tags", timeout=5).json()
        for m in tags.get("models", []):
            print(f"  {m['name']}")
        print(f"\nRun with --pull to download it, or pass --model <name> to pick one above.")
        sys.exit(1)

    print(f"\nRunning {len(args.prompts)} prompt(s)...\n" + "-" * 60)

    results: list[RunResult] = []
    for name in args.prompts:
        prompt = PROMPTS[name]
        print(f"  Running: {name}", end="", flush=True)
        try:
            result = run_single(host, args.model, name, prompt, args.num_predict)
            results.append(result)
            print(f" — {result.gen_tokens_per_sec:.1f} tok/s  TTFT {result.ttft_ms:.0f} ms")
            if args.show_responses:
                print_result(result, show_response=True)
        except Exception as e:
            print(f"\n  Error on '{name}': {e}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
