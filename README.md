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
  --chunked-prefill-size 16384 \
  --max-running-requests 16 \
  --context-length 1048576
```

## 性能基准 (8×A800, DSpark)

| 指标 | 值 |
|------|-----|
| Decode 吞吐 | ~126 tok/s |
| Accept Rate | ~22% |
| Accept Len | ~2.08 |
| 显存 / GPU | ~47 GB |
| 上下文 | 1M tokens |

### 消融对比

| 配置 | 吞吐 | Accept Rate |
|------|------|-------------|
| 无 DSpark | ~73 tok/s | - |
| DSpark 默认 | ~100 tok/s | 17% |
| DSpark + chunked 16K | **~126 tok/s** | **22%** |

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| context-length | 1048576 | 1M 上下文 |
| tp-size | 8 | 8 路张量并行 |
| chunked-prefill-size | 16384 | 分块 prefill |
| max-running-requests | 16 | 最大并发请求 |
| mem-fraction-static | 0.85 | 85% 显存给 KV cache |
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