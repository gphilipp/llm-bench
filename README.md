# qwen-benchmark

Benchmark local LLMs running in [Ollama](https://ollama.com). Measures **generation speed** (tok/s), **time to first token** (TTFT), and **prefill speed** across five prompt categories: `short`, `medium`, `long`, `reasoning`, and `code`.

## Requirements

- Python 3.10+
- `pip install requests`
- Ollama running locally

> **WSL2 on Windows:** Ollama must be started with `$env:OLLAMA_HOST='0.0.0.0'; ollama serve` so WSL can reach it. The script auto-detects WSL and uses the Windows gateway IP.

## Usage

```bash
# Benchmark the default model set
python3 benchmark.py

# List available models
python3 benchmark.py --list

# Pick specific models
python3 benchmark.py --models mistral:latest phi:latest qwen2.5-coder:7b

# Pull a model if not already downloaded, then run
python3 benchmark.py --models qwen3:30b-a3b --pull

# Run only specific prompts
python3 benchmark.py --prompts short reasoning code

# Show model responses alongside results
python3 benchmark.py --show-responses

# Point at a remote Ollama instance
python3 benchmark.py --host http://192.168.1.10:11434
```

## Sample output

### Single model

```
Ollama host : http://172.26.112.1:11434
Model       : qwen3:30b-a3b
Max tokens  : 512

Running 5 prompt(s)...
------------------------------------------------------------
  short          — 4.7 tok/s   TTFT 71128 ms
  medium         — 6.1 tok/s   TTFT 87915 ms
  long           — 5.9 tok/s   TTFT 25087 ms
  reasoning      — 5.6 tok/s   TTFT 93217 ms
  code           — 5.3 tok/s   TTFT 100353 ms

SUMMARY — qwen3:30b-a3b
  Avg generation speed : 5.5 tok/s
  Avg time to 1st token: 75.5 s
  Avg prefill speed    : 8.2 tok/s
```

### Multi-model comparison

```
Ollama host : http://172.26.112.1:11434
Models      : qwen3:30b-a3b, qwen2.5-coder:14b, qwen2.5-coder:7b, mistral:latest, phi:latest
Prompts     : short, medium, long, reasoning, code
Max tokens  : 512
Total runs  : 25

[1/5] phi:latest
  Prompt         Speed         TTFT
  ----------------------------------------
  short          44.1 tok/s    823 ms
  medium         43.8 tok/s    901 ms
  long           42.5 tok/s    912 ms
  reasoning      41.9 tok/s    934 ms
  code           43.2 tok/s    889 ms
  → avg 43.1 tok/s  avg TTFT 892 ms

[2/5] mistral:latest
  ...

===========================================================================
COMPARISON SUMMARY
===========================================================================
Model                    Avg tok/s   Min tok/s   Max tok/s     Avg TTFT  Prefill tok/s  Runs
---------------------------------------------------------------------------
phi:latest                    43.1        41.9        44.1      892 ms            89.4     5
mistral:latest                31.4        28.2        34.0    1.2 s               72.1     5
qwen2.5-coder:7b              28.7        25.1        31.8    1.4 s               68.3     5
qwen2.5-coder:14b             14.2        12.8        15.9    3.1 s               31.7     5
qwen3:30b-a3b                  5.5         4.7         6.1   75.5 s                8.2     5
---------------------------------------------------------------------------
```

> Results above from an RTX 2070 (8 GB VRAM). `qwen3:30b-a3b` is 18.5 GB so it spills onto CPU RAM, which explains the low speed and high TTFT.

## Prompts

| Name | Description |
|---|---|
| `short` | One-word answer — baseline latency |
| `medium` | Technical explanation — typical chat |
| `long` | Code + explanation — extended generation |
| `reasoning` | Step-by-step problem — chain-of-thought |
| `code` | Function implementation — coding assistant |
