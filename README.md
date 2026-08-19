# DeepSeek V4 Flash on A800 / A100 (SM80) Deployment

**English** · [中文](README.zh-CN.md)

Complete deployment recipe for running DeepSeek-V4-Flash-0731 on 8× NVIDIA A800-SXM4-80GB (SM80).

> **SM90+ users (H100 / H800 / H20) do not need this repository** — use the official upstream weights and stock SGLang directly.

## Hardware

- 8× A800-SXM4-80GB (SM80, Compute Capability 8.0)
- NVLink 8×25 GB/s bidirectional
- CUDA 13.0+ / Driver 595.71+

## Software Stack

| Component | Version |
|------|------|
| SGLang | 0.5.16 (commit `fdebc938`) |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| Python | 3.12.13 |
| CUDA | 13.0 |

## Model Preparation

The SM80 architecture does not support native FP8. Non-expert weights must be converted to BF16
while MoE expert weights stay in MXFP4.

**Option 1 — download the pre-converted weights (recommended)**

```bash
# HuggingFace
hf download yobo2u/DeepSeek-V4-Flash-0731-A100 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731-A100

# ModelScope
modelscope download yobo2u/DeepSeek-V4-Flash-0731-A100 \
  --local_dir /path/to/models/DeepSeek-V4-Flash-0731-A100
```

- **HuggingFace**: [yobo2u/DeepSeek-V4-Flash-0731-A100](https://huggingface.co/yobo2u/DeepSeek-V4-Flash-0731-A100)
- **ModelScope**: [yobo2u/DeepSeek-V4-Flash-0731-A100](https://modelscope.cn/models/yobo2u/DeepSeek-V4-Flash-0731-A100)

Converted size is about 173 GB across 48 shards: 47,927 tensors total, of which 35,328 expert
tensors remain in MXFP4. After downloading, verify that every shard referenced by
`model.safetensors.index.json` is present.

**Option 2 — convert from the original checkpoint yourself**

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731
```

Convert the non-expert weights from FP8 to BF16 and keep the MoE expert weights in MXFP4,
then confirm the output has 48 shards and that the tensor count matches the source checkpoint.

## Quick Start

**1. Install SGLang 0.5.16**

```bash
uv venv /path/to/venv-sglang0516 --python 3.12
uv pip install --python /path/to/venv-sglang0516/bin/python \
  "sglang[all] @ git+https://github.com/sgl-project/sglang.git@fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1#subdirectory=python"
```

**2. Install the A100 monkeypatch**

```bash
git clone https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516.git
cd deepseek-v4-a100-sglang-v0516
uv pip install --python /path/to/venv-sglang0516/bin/python -e . --no-deps
```

**3. Launch the server**

```bash
bash launch.sh
```

`launch.sh` ships with the recommended config C. To reproduce config A or B, use
`launch_mem085_chunk32k.sh` (identical to `launch.sh`) or `launch_mem090_chunk32k.sh`.

Or set the environment manually:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731-A100
export VENV=/path/to/venv-sglang0516
export PROJECT_ROOT=/path/to/deepseek-v4-a100-sglang-v0516
export PYTHONPATH="${PROJECT_ROOT}:${VENV}/lib/python3.12/site-packages"

# Monkeypatch env
export ENABLE_SGLANG_DSV4_A100_PATCH=1
export SGLANG_DSV4_A100_DIRECT_ATTENTION=1
export SGLANG_DSV4_A100_INT8_INDEXER=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_DSV4_MXFP4_MOE_BACKEND=mxfp4_int8
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TOPK_V2=0

exec "${VENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization fp8 \
  --moe-runner-backend marlin \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 \
  --port 8082 \
  --served-model-name deepseek-v4-flash-0731 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --speculative-algorithm DSPARK \
  --chunked-prefill-size 32768 \
  --max-running-requests 16 \
  --context-length 1048576
```

## Benchmarks (8×A800, DSpark)

Measured on the recommended **config C** (`mem-fraction-static 0.85` + `chunked-prefill-size 32768`):
5 context lengths × 3 concurrency levels × 2 repeats = 30 groups, 900 requests, 0 errors and 0 timeouts.

| Metric | Value |
|------|-----|
| Single-stream decode (C1, 1000/TPOT) | ~217 tok/s |
| Single-stream aggregate throughput (C1) | ~205 tok/s (166–222) |
| Aggregate throughput (C16) | ~1232 tok/s (peak 1334) |
| Accept Rate | ~60% (0.40–0.86, rises with context) |
| Accept Len | ~4.01 (3.01–5.30) |
| TTFT (C1) | ~296 ms |
| Peak VRAM / GPU | ~51.9 GB |
| Context | 1M tokens |

> Accept Rate and Accept Len rise substantially with context length: about 0.40 / 3.0 at 1K,
> and about 0.82 / 5.1 at 128K. The table reports means over all 30 groups — see the raw JSON
> under [benchmarks](benchmarks/) for per-group values.

## Parameter Ablation

Three configurations, 30 groups / 900 requests each:

| Config | mem-fraction | chunked-prefill | Mean throughput | Peak throughput | Mean AR | Peak VRAM/GPU | Best-in-group |
|------|---|---|---|---|---|---|---|
| A | 0.85 | 16384 | 746.2 tok/s | 1260.0 | 0.603 | 48.58 GiB | 2 / 15 |
| B | 0.90 | 32768 | 759.8 tok/s | 1528.3 | 0.594 | 53.80 GiB | 6 / 15 |
| **C (recommended)** | **0.85** | **32768** | **766.4 tok/s** | 1334.1 | 0.602 | 51.92 GiB | **7 / 15** |

**Factor decomposition** (C vs A isolates the chunk effect; C vs B isolates the mem effect):

- **The 128K long-context regression comes from `mem-fraction-static=0.90`, not from the 32K chunk.**
  With chunk fixed at 32768, dropping mem from 0.90 to 0.85 recovers **+11.55%** on average across
  the three 128K concurrency levels; with mem fixed at 0.85, raising chunk from 16K to 32K changes
  128K by only **+1.69%**.
- `mem-fraction 0.90` squeezes the available KV cache headroom, dropping the 128K Accept Rate from 0.80 to 0.69.
- Config C simultaneously achieves the highest mean throughput and the fewest high-variance groups
  (3 vs 6 for both A and B), while using less VRAM than B.

Full report: [three-way factor comparison](benchmarks/comparison-3way.html) · [PDF](benchmarks/comparison-3way.pdf)

## Key Parameters

| Parameter | Value | Notes |
|------|-----|------|
| `context-length` | 1048576 | 1M context |
| `tp-size` | 8 | 8-way tensor parallel |
| `chunked-prefill-size` | 32768 | Measured better than 16384 |
| `max-running-requests` | 16 | Max concurrent requests |
| `mem-fraction-static` | 0.85 | 85% of VRAM for KV cache (0.90 regresses at 128K) |
| `speculative-algorithm` | DSPARK | Speculative decoding |

## Repository Contents

| File | Purpose |
|---|---|
| `launch.sh` | Recommended launch script (config C) |
| `launch_mem085_chunk32k.sh` | Config C, identical to `launch.sh` |
| `launch_mem090_chunk32k.sh` | Config B, for reproducing the ablation |
| `benchmark.py` | Single-configuration benchmark |
| `benchmark_dspark_full.py` | Full 30-group DSpark matrix |
| `make_dspark_3way.py` | Three-way comparison report generator |
| `benchmarks/` | Reports and raw results |

## Credits

- [yaleyoou/deepseek-v4-a100-sglang-v0516](https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516) — A100 monkeypatch
- [Qeeweew/deepseek-v4-a100-sglang](https://github.com/Qeeweew/deepseek-v4-a100-sglang) — original patch
- [nudt-eddie](https://huggingface.co/nudt-eddie) — deployment validation

## License

MIT, following the upstream DeepSeek-V4-Flash-0731 repository.
