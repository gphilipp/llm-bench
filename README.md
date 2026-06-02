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

# Per-prompt detail (speed + TTFT for each prompt)
python3 benchmark.py --verbose

# Show model responses alongside results
python3 benchmark.py --show-responses

# Point at a remote Ollama instance
python3 benchmark.py --host http://192.168.1.10:11434
```

## Sample output

My old but trusty Zotac EN72070, beefed up with 64GB:
```
OS          : Linux 6.6.87.2-microsoft-standard-WSL2
CPU         : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (12 threads)
RAM         : 63.9 GB
GPU         : NVIDIA GeForce RTX 2070 (8.0 GB)

[1/5] qwen3:30b-a3b       .....  4m 47s  — avg 8.1 tok/s  avg TTFT 42.6 s
[2/5] qwen2.5-coder:14b   .....  10m 03s  — avg 3.2 tok/s  avg TTFT 4840 ms
[3/5] qwen2.5-coder:7b    .....  2m 34s  — avg 12.6 tok/s  avg TTFT 1616 ms
[4/5] mistral:latest      .....  2m 48s  — avg 11.7 tok/s  avg TTFT 1835 ms
[5/5] phi:latest          .....  27.6s  — avg 104.1 tok/s  avg TTFT 814 ms

=======================================================================================
COMPARISON SUMMARY
=======================================================================================
Model               Avg tok/s   Min tok/s   Max tok/s    Avg TTFT   Prefill tok/s  Runs
---------------------------------------------------------------------------------------
phi:latest              104.1        88.2       153.4      814 ms          2834.6     5
qwen2.5-coder:7b         12.6         8.9        26.1     1616 ms           225.7     5
mistral:latest           11.7         8.8        21.9     1835 ms           100.1     5
qwen3:30b-a3b             8.1         7.3         9.4      42.6 s            27.9     5
qwen2.5-coder:14b         3.2         2.5         6.0     4840 ms            66.4     5
---------------------------------------------------------------------------------------
```

### Notes on these results

- **phi:latest** (3B) fits fully in VRAM → 104 tok/s avg, peaking at 153 tok/s.
- **qwen2.5-coder:7b** and **mistral:latest** (7B) both fit in VRAM → consistent ~12 tok/s with sub-2s TTFT.
- **qwen3:30b-a3b** (18.6 GB) and **qwen2.5-coder:14b** (9 GB) exceed VRAM and spill to CPU RAM → low generation speed and high TTFT.
- The first prompt per model includes cold model-load time; subsequent prompts on the same model are faster.

## Prompts

| Name | Description |
|---|---|
| `short` | One-word answer — baseline latency |
| `medium` | Technical explanation — typical chat |
| `long` | Code + explanation — extended generation |
| `reasoning` | Step-by-step problem — chain-of-thought |
| `code` | Function implementation — coding assistant |
