# LLM Benchmark

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

```
OS          : Linux 6.6.87.2-microsoft-standard-WSL2
CPU         : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12 threads)
RAM         : 31.3 GB
GPU         : NVIDIA GeForce RTX 2070 (8.0 GB)

Ollama host : http://172.26.112.1:11434
Models      : qwen3:30b-a3b, qwen2.5-coder:14b, qwen2.5-coder:7b, mistral:latest, phi:latest
Prompts     : short, medium, long, reasoning, code
Max tokens  : 512
Total runs  : 25

[1/5] qwen3:30b-a3b
  Prompt             Speed         TTFT
  ----------------------------------------
    short           7.9 tok/s   TTFT 15.1 s
    medium          6.5 tok/s   TTFT 81.7 s
    long            7.2 tok/s   TTFT 20.5 s
    reasoning       5.7 tok/s   TTFT 92.6 s
    code            5.4 tok/s   TTFT 98.4 s
  → avg 6.5 tok/s  avg TTFT 61.7 s

[2/5] qwen2.5-coder:14b
  Prompt             Speed         TTFT
  ----------------------------------------
    short           6.6 tok/s   TTFT 43.9 s
    medium          2.5 tok/s   TTFT 5058 ms
    long            2.3 tok/s   TTFT 1156 ms
    reasoning       2.4 tok/s   TTFT 833 ms
    code            2.3 tok/s   TTFT 927 ms
  → avg 3.2 tok/s  avg TTFT 10.4 s

[3/5] qwen2.5-coder:7b
  Prompt             Speed         TTFT
  ----------------------------------------
    short          25.1 tok/s   TTFT 10.1 s
    medium          9.1 tok/s   TTFT 1137 ms
    long            8.6 tok/s   TTFT 701 ms
    reasoning       9.3 tok/s   TTFT 439 ms
    code            9.1 tok/s   TTFT 477 ms
  → avg 12.2 tok/s  avg TTFT 2579 ms

[4/5] mistral:latest
  Prompt             Speed         TTFT
  ----------------------------------------
    short          20.2 tok/s   TTFT 44.6 s
    medium          8.4 tok/s   TTFT 1455 ms
    long            8.0 tok/s   TTFT 286 ms
    reasoning       7.8 tok/s   TTFT 1821 ms
    code            7.7 tok/s   TTFT 288 ms
  → avg 10.4 tok/s  avg TTFT 9686 ms

[5/5] phi:latest
  Prompt             Speed         TTFT
  ----------------------------------------
    short          26.1 tok/s   TTFT 32.3 s
    medium         84.8 tok/s   TTFT 95 ms
    long           86.0 tok/s   TTFT 132 ms
    reasoning      78.9 tok/s   TTFT 115 ms
    code           78.0 tok/s   TTFT 110 ms
  → avg 70.8 tok/s  avg TTFT 6559 ms

=======================================================================================
COMPARISON SUMMARY
=======================================================================================
Model               Avg tok/s   Min tok/s   Max tok/s    Avg TTFT   Prefill tok/s  Runs
---------------------------------------------------------------------------------------
phi:latest               70.8        26.1        86.0     6559 ms          2473.2     5
qwen2.5-coder:7b         12.2         8.6        25.1     2579 ms           246.1     5
mistral:latest           10.4         7.7        20.2     9686 ms            96.4     5
qwen3:30b-a3b             6.5         5.4         7.9      61.7 s            24.3     5
qwen2.5-coder:14b         3.2         2.3         6.6      10.4 s            72.8     5
---------------------------------------------------------------------------------------
```

### Notes on these results

Machine: **Intel i7-9750H / 32 GB RAM / RTX 2070 8 GB VRAM** running Ollama on Windows, benchmarked from WSL2.

- **phi:latest** (3B) runs fully on GPU → very fast generation (70–86 tok/s), but the `short` prompt has a high TTFT (32 s) because it's the first run and the model is loading cold.
- **qwen2.5-coder:7b** fits in VRAM → consistent ~9 tok/s generation and sub-second TTFT after the cold start.
- **qwen3:30b-a3b** and **qwen2.5-coder:14b** exceed 8 GB VRAM and spill to CPU RAM → low generation speed and high TTFT. The first prompt per model always includes model-load time.
- High `short` TTFTs across the board (10–45 s) are the model cold-start; subsequent prompts on the same model are much faster.

## Prompts

| Name | Description |
|---|---|
| `short` | One-word answer — baseline latency |
| `medium` | Technical explanation — typical chat |
| `long` | Code + explanation — extended generation |
| `reasoning` | Step-by-step problem — chain-of-thought |
| `code` | Function implementation — coding assistant |
