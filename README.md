# DeepSeek V4 Flash on A800 (SM80) Deployment

在 8× NVIDIA A800-SXM4-80GB (SM80) 上部署 DeepSeek-V4-Flash-0731 的完整方案。

## 硬件环境

- 8× A800-SXM4-80GB (SM80, Compute Capability 8.0)
- NVLink 8×25 GB/s 双向
- CUDA 13.0+ / Driver 595.71+

## 软件栈

| 组件 | 版本 |
|------|------|
| SGLang | 0.5.16 (commit fdebc938) |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| Python | 3.12.13 |
| CUDA | 13.0 |

## 模型准备

### 模型下载

**HuggingFace:**
```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731
```

**ModelScope:**
```bash
modelscope download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local_dir /path/to/models/DeepSeek-V4-Flash-0731
```

### 离线转换 (FP8 → BF16 + MXFP4)

A800/A100 (SM80) 不支持原生 FP8，需要将非 expert 权重转为 BF16：

```bash
# 先验证原始模型
python scripts/validate_dsv4_checkpoint.py original \
  /path/to/models/DeepSeek-V4-Flash-0731

# 转换
python scripts/convert_deepseek_v4_flash_moe_mxfp4_bf16.py \
  --input /path/to/models/DeepSeek-V4-Flash-0731 \
  --output /path/to/models/DeepSeek-V4-Flash-0731-BF16-MXFP4

# 验证转换后模型
python scripts/validate_dsv4_checkpoint.py converted \
  /path/to/models/DeepSeek-V4-Flash-0731-BF16-MXFP4 \
  --reference /path/to/models/DeepSeek-V4-Flash-0731
```

转换后模型约 162GB（原始 156GB）。47,927 tensors, 35,328 expert tensors 保留 MXFP4。

## 快速开始

### 1. 安装 SGLang 0.5.16

```bash
uv venv /path/to/venv-sglang0516 --python 3.12
uv pip install --python /path/to/venv-sglang0516/bin/python \
  "sglang[all] @ git+https://github.com/sgl-project/sglang.git@fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1#subdirectory=python"
```

### 2. 安装 A100 monkeypatch

```bash
git clone https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516.git
cd deepseek-v4-a100-sglang-v0516
uv pip install --python /path/to/venv-sglang0516/bin/python -e . --no-deps
```

### 3. 启动服务

```bash
bash launch_dsv4_a100.sh
```

或手动设置环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731-BF16-MXFP4
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

## 性能基准 (8×A800, DSpark)

推荐配置 **C**（`mem-fraction-static 0.85` + `chunked-prefill-size 32768`）实测，
5 档上下文 × 3 并发 × 2 重复 = 30 组，900 请求，Error / Timeout 均为 0：

| 指标 | 值 |
|------|-----|
| 单流解码 (C1, 1000/TPOT) | ~217 tok/s |
| 单流聚合吞吐 (C1) | ~205 tok/s（166–222） |
| 并发聚合吞吐 (C16) | ~1232 tok/s（峰值 1334） |
| Accept Rate | ~60%（0.40–0.86，随上下文增长） |
| Accept Len | ~4.01（3.01–5.30） |
| TTFT (C1) | ~296 ms |
| 峰值显存 / GPU | ~51.9 GB |
| 上下文 | 1M tokens |

> Accept Rate / Accept Len 随上下文长度显著上升：1K 档约 0.40 / 3.0，128K 档约 0.82 / 5.1。
> 上表为 30 组全局均值，单点数值请查阅 [benchmarks](benchmarks/) 下的原始 JSON。

### 参数消融对比

三配置完整对比（各 30 组 / 900 请求）：

| 配置 | mem-fraction | chunked-prefill | 全局均值吞吐 | 峰值吞吐 | AR 均值 | 峰值显存/GPU | 最优组合数 |
|------|---|---|---|---|---|---|---|
| A | 0.85 | 16384 | 746.2 tok/s | 1260.0 | 0.603 | 48.58 GiB | 2 / 15 |
| B | 0.90 | 32768 | 759.8 tok/s | 1528.3 | 0.594 | 53.80 GiB | 6 / 15 |
| **C（推荐）** | **0.85** | **32768** | **766.4 tok/s** | 1334.1 | 0.602 | 51.92 GiB | **7 / 15** |

**因子分解结论**（C vs A 隔离 chunk 效应，C vs B 隔离 mem 效应）：

- **128K 长上下文的性能回退源自 `mem-fraction-static=0.90`，而非 32K chunk。**
  固定 chunk=32768，仅将 mem 由 0.90 降到 0.85，128K 三档并发平均回升 **+11.55%**；
  固定 mem=0.85，将 chunk 由 16K 提到 32K，128K 档仅变化 **+1.69%**。
- `mem-fraction 0.90` 压缩了可用 KV cache 余量，128K 档 Accept Rate 由 0.80 跌至 0.69。
- 配置 C 同时取得最高全局均值吞吐、最少高波动组（3 组 vs A/B 各 6 组），显存低于 B。

完整报告：[三配置因子分解对比](benchmarks/comparison-3way.html) · [PDF](benchmarks/comparison-3way.pdf)

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| context-length | 1048576 | 1M 上下文 |
| tp-size | 8 | 8 路张量并行 |
| chunked-prefill-size | 32768 | 分块 prefill（实测优于 16384） |
| max-running-requests | 16 | 最大并发请求 |
| mem-fraction-static | 0.85 | 85% 显存给 KV cache（0.90 会在 128K 掉速） |
| speculative-algorithm | DSPARK | 投机解码 |

## 模型下载

转换后的模型（BF16 + MXFP4，162GB）托管在：

- **HuggingFace**: [yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4](https://huggingface.co/yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4)
- **ModelScope**: [yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4](https://modelscope.cn/models/yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4)

```bash
# HuggingFace
hf download yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4 \
  --local-dir /path/to/models/DeepSeek-V4-Flash-0731-BF16-MXFP4

# ModelScope
modelscope download yobo2u/DeepSeek-V4-Flash-0731-BF16-MXFP4 \
  --local_dir /path/to/models/DeepSeek-V4-Flash-0731-BF16-MXFP4
```

## 致谢

- [yaleyoou/deepseek-v4-a100-sglang-v0516](https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516) - A100 monkeypatch
- [Qeeweew/deepseek-v4-a100-sglang](https://github.com/Qeeweew/deepseek-v4-a100-sglang) - 原始 patch
- [nudt-eddie](https://huggingface.co/nudt-eddie) - 部署验证