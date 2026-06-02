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

def _default_host() -> str:
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                return "http://172.26.112.1:11434"
    except OSError:
        pass
    return "http://localhost:11434"

DEFAULT_HOST = _default_host()

DEFAULT_MODELS = [
    "qwen3:30b-a3b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b",
    "mistral:latest",
    "phi:latest",
]

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

GENERATION_TOKENS = 512


@dataclass
class RunResult:
    model: str
    prompt_name: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_time_s: float
    gen_tokens_per_sec: float
    prompt_tokens_per_sec: float
    full_response: str = field(repr=False)


@dataclass
class ModelSummary:
    model: str
    results: list[RunResult]
    errors: list[str]

    @property
    def avg_gen_tps(self) -> float:
        vals = [r.gen_tokens_per_sec for r in self.results if r.gen_tokens_per_sec > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def min_gen_tps(self) -> float:
        vals = [r.gen_tokens_per_sec for r in self.results if r.gen_tokens_per_sec > 0]
        return min(vals) if vals else 0.0

    @property
    def max_gen_tps(self) -> float:
        vals = [r.gen_tokens_per_sec for r in self.results if r.gen_tokens_per_sec > 0]
        return max(vals) if vals else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self.results]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_prompt_tps(self) -> float:
        vals = [r.prompt_tokens_per_sec for r in self.results if r.prompt_tokens_per_sec > 0]
        return statistics.mean(vals) if vals else 0.0


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
    with requests.post(url, json=payload, stream=True, timeout=600) as resp:
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
    eval_duration_ns = final_stats.get("eval_duration", 0)
    prompt_eval_duration_ns = final_stats.get("prompt_eval_duration", 0)

    gen_tps = generated_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else (
        generated_tokens / total_time if total_time > 0 else 0.0
    )
    prompt_tps = prompt_tokens / (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns > 0 else 0.0

    return RunResult(
        model=model,
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        total_time_s=total_time,
        gen_tokens_per_sec=gen_tps,
        prompt_tokens_per_sec=prompt_tps,
        full_response="".join(chunks),
    )


def get_available_models(host: str) -> dict[str, dict]:
    resp = requests.get(f"{host}/api/tags", timeout=10)
    resp.raise_for_status()
    return {m["name"]: m for m in resp.json().get("models", [])}


def check_model_available(available: dict[str, dict], model: str) -> bool:
    return any(model == n or model == n.split(":")[0] for n in available)


def pull_model(host: str, model: str) -> None:
    print(f"  Pulling {model}...")
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
                    print(f"\r    {status}: {completed / total * 100:.1f}%", end="", flush=True)
                else:
                    print(f"\r    {status}", end="", flush=True)
    print()


def fmt_ms(ms: float) -> str:
    if ms >= 10_000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.0f} ms"


def print_comparison_table(summaries: list[ModelSummary]) -> None:
    col_model = max(len(s.model) for s in summaries)
    col_model = max(col_model, len("Model"))

    header = (
        f"{'Model':<{col_model}}  "
        f"{'Avg tok/s':>10}  "
        f"{'Min tok/s':>10}  "
        f"{'Max tok/s':>10}  "
        f"{'Avg TTFT':>10}  "
        f"{'Prefill tok/s':>14}  "
        f"{'Runs':>4}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("COMPARISON SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    # Sort by avg gen tps descending
    for s in sorted(summaries, key=lambda x: x.avg_gen_tps, reverse=True):
        runs = len(s.results)
        errors = len(s.errors)
        run_str = f"{runs}" if not errors else f"{runs}+{errors}err"
        prefill = f"{s.avg_prompt_tps:.1f}" if s.avg_prompt_tps > 0 else "n/a"
        print(
            f"{s.model:<{col_model}}  "
            f"{s.avg_gen_tps:>10.1f}  "
            f"{s.min_gen_tps:>10.1f}  "
            f"{s.max_gen_tps:>10.1f}  "
            f"{fmt_ms(s.avg_ttft_ms):>10}  "
            f"{prefill:>14}  "
            f"{run_str:>4}"
        )

    print(sep)
    print()


def benchmark_model(
    host: str,
    model: str,
    prompt_names: list[str],
    num_predict: int,
    show_responses: bool,
) -> ModelSummary:
    summary = ModelSummary(model=model, results=[], errors=[])

    for name in prompt_names:
        prompt = PROMPTS[name]
        print(f"    {name:<12}", end="", flush=True)
        try:
            result = run_single(host, model, name, prompt, num_predict)
            summary.results.append(result)
            print(f" {result.gen_tokens_per_sec:>6.1f} tok/s   TTFT {fmt_ms(result.ttft_ms)}")
            if show_responses:
                preview = result.full_response[:300]
                suffix = "..." if len(result.full_response) > 300 else ""
                print(f"              → {preview}{suffix}\n")
        except Exception as e:
            summary.errors.append(f"{name}: {e}")
            print(f" ERROR: {e}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one or more Ollama models")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, metavar="MODEL",
        help="Model tags to benchmark (default: preset list)",
    )
    # kept for backwards compatibility
    parser.add_argument("--model", metavar="MODEL", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS),
        help="Which prompts to run (default: all)",
    )
    parser.add_argument("--num-predict", type=int, default=GENERATION_TOKENS,
                        help=f"Max tokens to generate per prompt (default: {GENERATION_TOKENS})")
    parser.add_argument("--show-responses", action="store_true", help="Print model responses")
    parser.add_argument("--pull", action="store_true", help="Pull missing models before benchmarking")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    host = args.host
    models = [args.model] if args.model else args.models

    # Connectivity check
    try:
        available = get_available_models(host)
    except Exception as e:
        print(f"Error: cannot reach Ollama at {host}\n  {e}")
        print("\nMake sure Ollama is running. On Windows/WSL: $env:OLLAMA_HOST='0.0.0.0'; ollama serve")
        sys.exit(1)

    if args.list:
        print(f"Available models on {host}:")
        for name, info in sorted(available.items()):
            size_gb = info.get("size", 0) / 1e9
            params = info.get("details", {}).get("parameter_size", "")
            quant = info.get("details", {}).get("quantization_level", "")
            print(f"  {name:<40} {size_gb:>6.1f} GB  {params:<8}  {quant}")
        sys.exit(0)

    # Resolve / pull models
    runnable: list[str] = []
    for model in models:
        if check_model_available(available, model):
            runnable.append(model)
        elif args.pull:
            pull_model(host, model)
            runnable.append(model)
        else:
            print(f"Skipping '{model}' — not found (use --pull to download, --list to see what's available)")

    if not runnable:
        print("No models to run. Use --list to see available models.")
        sys.exit(1)

    total_runs = len(runnable) * len(args.prompts)
    print(f"Ollama host : {host}")
    print(f"Models      : {', '.join(runnable)}")
    print(f"Prompts     : {', '.join(args.prompts)}")
    print(f"Max tokens  : {args.num_predict}")
    print(f"Total runs  : {total_runs}\n")

    all_summaries: list[ModelSummary] = []

    for i, model in enumerate(runnable, 1):
        print(f"[{i}/{len(runnable)}] {model}")
        print(f"  {'Prompt':<12}  {'Speed':>10}   {'TTFT':>10}")
        print(f"  {'-'*40}")
        summary = benchmark_model(host, model, args.prompts, args.num_predict, args.show_responses)
        all_summaries.append(summary)
        if summary.results:
            tps_vals = [r.gen_tokens_per_sec for r in summary.results]
            print(f"  → avg {statistics.mean(tps_vals):.1f} tok/s  "
                  f"avg TTFT {fmt_ms(summary.avg_ttft_ms)}")
        print()

    if len(all_summaries) > 1:
        print_comparison_table(all_summaries)
    elif all_summaries:
        s = all_summaries[0]
        print(f"\nSUMMARY — {s.model}")
        print(f"  Avg generation speed : {s.avg_gen_tps:.1f} tok/s")
        print(f"  Avg time to 1st token: {fmt_ms(s.avg_ttft_ms)}")
        if s.avg_prompt_tps > 0:
            print(f"  Avg prefill speed    : {s.avg_prompt_tps:.1f} tok/s")
        print()


if __name__ == "__main__":
    main()
