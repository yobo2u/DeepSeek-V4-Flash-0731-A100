#!/usr/bin/env python3
"""Reproducible DSpark performance benchmark with streaming latency metrics."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:8082/v1"
MODEL = "deepseek-v4-flash-0731"
MODEL_PATH = "/root/work/models/DeepSeek-V4-Flash-0731-BF16-MXFP4"
LOG_DIR = Path("/root/work/models/dsv4-runtime/logs")
CONTEXT_LENGTHS = [1024, 4096, 16384, 65536, 131072]
CONCURRENCIES = [1, 8, 16]
OUTPUT_TOKENS = 1024
REQUESTS_PER_GROUP = 30
WARMUP_COUNT = 5
REPEATS = 2
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 1
SEED = 42
REQUEST_TIMEOUT = 1800
GPU_SAMPLE_INTERVAL = 0.5

TASK_PROMPT = """请针对输入内容进行深入分析，并持续生成结构化结果，直到接近指定的最大输出长度。

要求：
* 不要提前结束回答
* 不要只做摘要
* 持续展开分析、推理、归纳和结论
* 输出结构清晰，使用分级标题
* 尽量保持连续生成
* 不调用外部工具
* 不进行联网搜索
* 不输出无意义重复内容
* 保证每次测试任务类型一致

具体任务：
“请完整分析以上上下文内容，提取其中的主要事实、技术参数、关键关系、潜在问题和可优化点。先给出整体理解，再逐项分析重要信息之间的关系，随后指出可能存在的性能瓶颈、配置冲突、风险因素以及优化方向。对于每个判断说明依据，并给出优先级排序。最后形成一份结构化技术评估，包括现状、问题、原因、影响、优化建议和结论。尽可能充分利用给定上下文信息展开分析。”"""

CONTEXT_BLOCK = """
## 技术运行记录
系统使用八张 NVIDIA A800-SXM4-80GB GPU，计算能力为 SM80，通过 NVLink 互联并采用八路张量并行。推理框架固定为 SGLang 0.5.16，模型为 DeepSeek-V4-Flash-0731。模型权重经过离线转换：非专家权重使用 BF16，MoE 专家权重保留 MXFP4，并通过专用 monkeypatch 在 SM80 上执行。注意力路径使用 BF16 KV cache 与 INT8 indexer，投机解码算法为 DSpark。系统需要同时考虑吞吐、首 token 延迟、逐 token 延迟、显存、功耗、并发扩展、长上下文衰减、投机接受率和运行稳定性。测试要求固定采样参数、固定随机种子、相同任务类型和相同输出长度。潜在风险包括 GPU 负载不均衡、长上下文 prefill 开销、KV cache 容量、CUDA graph 压力、分块预填充粒度、并发请求调度、DSpark 草稿预测精度以及日志采集遗漏。评估时应区分单请求 decode 性能与多请求聚合吞吐，并结合所有 GPU 的利用率、显存和功耗判断瓶颈。
""".strip()


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-html", required=True, type=Path)
    p.add_argument("--expected-mem-fraction", type=float, required=True)
    p.add_argument("--expected-chunked-prefill", type=int, required=True)
    p.add_argument("--expected-max-running", type=int, required=True)
    return p.parse_args()


def mean(values):
    vals = [x for x in values if x is not None]
    return statistics.mean(vals) if vals else None


def percentile(values, q):
    vals = sorted(x for x in values if x is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def latest_server_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("server_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def verify_server(args: argparse.Namespace) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
        models = json.load(r)
    out = subprocess.check_output(["pgrep", "-af", "sglang.launch_server"], text=True)
    line = next((x for x in out.splitlines() if "--port 8082" in x), "")
    expected = {
        "--mem-fraction-static": str(args.expected_mem_fraction),
        "--chunked-prefill-size": str(args.expected_chunked_prefill),
        "--max-running-requests": str(args.expected_max_running),
        "--speculative-algorithm": "DSPARK",
    }
    missing = [f"{k} {v}" for k, v in expected.items() if f"{k} {v}" not in line]
    if missing:
        raise RuntimeError(f"服务器参数不匹配: {missing}; process={line}")
    log = latest_server_log()
    if not log:
        raise RuntimeError("未找到 server_*.log，AR/AL 无法采集，拒绝运行")
    return {"models": models, "process": line, "server_log": str(log)}


def build_exact_prompts(tokenizer):
    task_ids = tokenizer.encode("\n\n" + TASK_PROMPT, add_special_tokens=False)
    block_ids = tokenizer.encode(CONTEXT_BLOCK + "\n", add_special_tokens=False)
    prompts = {}
    for target in CONTEXT_LENGTHS:
        budget = target - len(task_ids)
        if budget <= 0:
            raise RuntimeError(f"任务提示已超过目标长度 {target}")
        context_ids = (block_ids * (budget // len(block_ids) + 1))[:budget]
        text = tokenizer.decode(context_ids + task_ids, skip_special_tokens=False)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        # Round-trip tokenization can shift a few boundary tokens. Adjust using a stable token.
        for _ in range(64):
            if actual == target:
                break
            if actual > target:
                context_ids = context_ids[: -(actual - target)]
            else:
                context_ids += block_ids[: target-actual]
            text = tokenizer.decode(context_ids + task_ids, skip_special_tokens=False)
            actual = len(tokenizer.encode(text, add_special_tokens=False))
        if abs(actual - target) > 2:
            raise RuntimeError(f"无法构造 {target} token prompt，实际 {actual}")
        prompts[target] = {"text": text, "local_tokens": actual}
        print(f"[prompt] target={target} local_tokens={actual} chars={len(text)}", flush=True)
    return prompts


def make_request(prompt: str, request_id: int) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content = None
    last_content = None
    usage = {}
    chunk_times = []
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if first_content is None:
                        first_content = now
                    last_content = now
                    chunk_times.append(now)
        ended = time.perf_counter()
        completion_tokens = int(usage.get("completion_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        if first_content is None or completion_tokens <= 0:
            raise RuntimeError(f"流式响应缺少 token 或 usage: usage={usage}")
        ttft = first_content - started
        latency = ended - started
        tpot = (ended - first_content) / max(completion_tokens - 1, 1)
        token_normalized_itl = (last_content - first_content) / max(completion_tokens - 1, 1)
        # Mean observed inter-chunk latency; chunks may contain >1 token, so standardized ITL
        # is additionally represented by token-normalized TPOT above.
        observed_itl = mean([b-a for a, b in zip(chunk_times, chunk_times[1:])])
        return {
            "request_id": request_id,
            "success": True,
            "timeout": False,
            "latency_s": latency,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "itl_s": token_normalized_itl,
            "stream_chunk_itl_s": observed_itl,
            "stream_chunks": len(chunk_times),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        }
    except Exception as e:
        elapsed = time.perf_counter() - started
        timeout = isinstance(e, (TimeoutError, socket.timeout)) or "timed out" in str(e).lower()
        if isinstance(e, urllib.error.URLError) and isinstance(e.reason, socket.timeout):
            timeout = True
        return {
            "request_id": request_id,
            "success": False,
            "timeout": timeout,
            "latency_s": elapsed,
            "error": f"{type(e).__name__}: {e}",
        }


def read_gpu_sample():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    sample = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        sample[int(p[0])] = {
            "util_pct": float(p[1]),
            "memory_mib": float(p[2]),
            "power_w": float(p[3]),
        }
    return sample


def gpu_monitor(stop: threading.Event, samples: list):
    while not stop.is_set():
        try:
            samples.append({"time": time.time(), "gpus": read_gpu_sample()})
        except Exception as e:
            samples.append({"time": time.time(), "error": str(e), "gpus": {}})
        stop.wait(GPU_SAMPLE_INTERVAL)


def parse_accept_metrics(log: Path, offset: int):
    # Give logging/tee a short chance to flush after the request group.
    time.sleep(1.0)
    with log.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read()
    rows = []
    for line in chunk.splitlines():
        if "accept len:" not in line or "accept rate:" not in line:
            continue
        ml = re.search(r"accept len:\s*([0-9.]+)", line)
        mr = re.search(r"accept rate:\s*([0-9.]+)", line)
        if ml and mr:
            rows.append({"accept_len": float(ml.group(1)), "accept_rate": float(mr.group(1))})
    return rows


def aggregate_gpu(samples):
    result = {}
    for i in range(8):
        rows = [s["gpus"][i] for s in samples if i in s.get("gpus", {})]
        result[str(i)] = {
            "avg_util_pct": mean([r["util_pct"] for r in rows]),
            "avg_memory_mib": mean([r["memory_mib"] for r in rows]),
            "peak_memory_mib": max([r["memory_mib"] for r in rows], default=None),
            "avg_power_w": mean([r["power_w"] for r in rows]),
            "peak_power_w": max([r["power_w"] for r in rows], default=None),
            "samples": len(rows),
        }
    return result


def run_group(prompt_info, ctx, concurrency, repeat, log):
    print(f"[group] ctx={ctx} concurrency={concurrency} repeat={repeat} warmup={WARMUP_COUNT}", flush=True)
    warmup_errors = []
    for i in range(WARMUP_COUNT):
        r = make_request(prompt_info["text"], -(i+1))
        if not r["success"]:
            warmup_errors.append(r.get("error"))
    if warmup_errors:
        raise RuntimeError(f"warmup 失败: {warmup_errors}")

    offset = log.stat().st_size
    requests = []
    gpu_samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(stop, gpu_samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(make_request, prompt_info["text"], i) for i in range(REQUESTS_PER_GROUP)]
        for f in as_completed(futures):
            requests.append(f.result())
    wall = time.perf_counter() - started
    stop.set()
    monitor.join(timeout=10)
    try:
        gpu_samples.append({"time": time.time(), "gpus": read_gpu_sample()})
    except Exception:
        pass

    ok = [r for r in requests if r["success"]]
    errors = [r for r in requests if not r["success"]]
    completion = sum(r["completion_tokens"] for r in ok)
    accepts = parse_accept_metrics(log, offset)
    gpus = aggregate_gpu(gpu_samples)
    actual_prompt_tokens = sorted(set(r["prompt_tokens"] for r in ok))
    result = {
        "context_length_target": ctx,
        "context_tokens_local": prompt_info["local_tokens"],
        "prompt_tokens_server": actual_prompt_tokens,
        "concurrency": concurrency,
        "repeat": repeat,
        "requests": REQUESTS_PER_GROUP,
        "output_tokens_target": OUTPUT_TOKENS,
        "wall_time_s": wall,
        "completion_tokens": completion,
        "gen_throughput_tok_s": completion / wall if wall else None,
        "accept_rate": mean([x["accept_rate"] for x in accepts]),
        "accept_len": mean([x["accept_len"] for x in accepts]),
        "accept_log_samples": len(accepts),
        "ttft_mean_ms": (mean([r["ttft_s"] for r in ok]) or 0) * 1000 if ok else None,
        "ttft_p50_ms": (percentile([r["ttft_s"] for r in ok], .5) or 0) * 1000 if ok else None,
        "ttft_p95_ms": (percentile([r["ttft_s"] for r in ok], .95) or 0) * 1000 if ok else None,
        "tpot_mean_ms": (mean([r["tpot_s"] for r in ok]) or 0) * 1000 if ok else None,
        "itl_mean_ms": (mean([r["itl_s"] for r in ok]) or 0) * 1000 if ok else None,
        "stream_chunk_itl_mean_ms": (mean([r["stream_chunk_itl_s"] for r in ok]) or 0) * 1000 if ok else None,
        "latency_mean_s": mean([r["latency_s"] for r in ok]),
        "success_count": len(ok),
        "error_count": len(errors),
        "timeout_count": sum(bool(r.get("timeout")) for r in errors),
        "errors": [r.get("error") for r in errors],
        "gpu_metrics": gpus,
        "peak_vram_mib": max((g["peak_memory_mib"] or 0) for g in gpus.values()),
        "average_gpu_power_w": mean([g["avg_power_w"] for g in gpus.values()]),
        "request_metrics": requests,
    }
    print(
        f"[result] throughput={result['gen_throughput_tok_s']:.2f} AR={result['accept_rate']} "
        f"AL={result['accept_len']} TTFT={result['ttft_mean_ms']:.2f}ms "
        f"TPOT={result['tpot_mean_ms']:.3f}ms ok={len(ok)} err={len(errors)}",
        flush=True,
    )
    if not accepts:
        raise RuntimeError("本组未解析到 AR/AL；停止测试，避免生成不完整报告")
    return result


def fmt(v, digits=2):
    return "N/A" if v is None else f"{v:.{digits}f}"


def variation(a, b):
    if a is None or b is None:
        return None
    m = (abs(a) + abs(b)) / 2
    return abs(a-b) / m * 100 if m else 0.0


def generate_html(meta, results, output: Path):
    summaries = []
    for ctx in CONTEXT_LENGTHS:
        for c in CONCURRENCIES:
            rows = [r for r in results if r["context_length_target"] == ctx and r["concurrency"] == c]
            if not rows:
                continue
            summaries.append({
                "context": ctx,
                "concurrency": c,
                "n": len(rows),
                "throughput": mean([r["gen_throughput_tok_s"] for r in rows]),
                "accept_rate": mean([r["accept_rate"] for r in rows]),
                "accept_len": mean([r["accept_len"] for r in rows]),
                "ttft": mean([r["ttft_mean_ms"] for r in rows]),
                "tpot": mean([r["tpot_mean_ms"] for r in rows]),
                "itl": mean([r["itl_mean_ms"] for r in rows]),
                "peak_vram": max(r["peak_vram_mib"] for r in rows),
                "power": mean([r["average_gpu_power_w"] for r in rows]),
                "errors": sum(r["error_count"] for r in rows),
                "timeouts": sum(r["timeout_count"] for r in rows),
                "variation": variation(rows[0]["gen_throughput_tok_s"], rows[1]["gen_throughput_tok_s"]) if len(rows)==2 else None,
            })
    thr = [s["throughput"] for s in summaries if s["throughput"] is not None]
    mx, mn = (max(thr), min(thr)) if thr else (None, None)
    css = """body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e8eefc;margin:0;padding:28px}h1,h2{color:#fff}.card{background:#151d33;border:1px solid #273453;border-radius:12px;padding:18px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.metric{background:#10182b;padding:12px;border-radius:9px}.value{font-size:24px;font-weight:700;color:#6ee7ff}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #2a3755;text-align:right}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#1a2540}.bad{color:#ff8b8b;font-weight:700}.good{color:#79e2a5}.muted{color:#9fb0ce}.scroll{overflow:auto;max-height:70vh}code{color:#a7f3d0}a{color:#6ee7ff}"""
    summary_rows = "".join(
        f"<tr><td>{s['context']//1024}K</td><td>{s['concurrency']}</td><td>{s['n']}</td>"
        f"<td>{fmt(s['throughput'])}</td><td>{fmt(s['accept_rate'],3)}</td><td>{fmt(s['accept_len'],3)}</td>"
        f"<td>{fmt(s['ttft'])}</td><td>{fmt(s['tpot'],3)}</td><td>{fmt(s['itl'],3)}</td>"
        f"<td>{fmt(s['peak_vram'])}</td><td>{fmt(s['power'])}</td>"
        f"<td class={'bad' if s['variation'] is not None and s['variation']>5 else 'good'}>{fmt(s['variation'])}%</td>"
        f"<td>{s['errors']}/{s['timeouts']}</td></tr>" for s in summaries)
    detail_rows = ""
    for r in results:
        gpu = "<br>".join(
            f"GPU{i}: {fmt(r['gpu_metrics'][str(i)]['avg_util_pct'])}% / {fmt(r['gpu_metrics'][str(i)]['avg_memory_mib'])} MiB / {fmt(r['gpu_metrics'][str(i)]['avg_power_w'])} W"
            for i in range(8))
        detail_rows += (
            f"<tr><td>{r['context_length_target']//1024}K</td><td>{r['concurrency']}</td><td>{r['repeat']}</td>"
            f"<td>{fmt(r['gen_throughput_tok_s'])}</td><td>{fmt(r['accept_rate'],3)}</td><td>{fmt(r['accept_len'],3)}</td>"
            f"<td>{fmt(r['ttft_mean_ms'])}</td><td>{fmt(r['tpot_mean_ms'],3)}</td><td>{fmt(r['itl_mean_ms'],3)}</td>"
            f"<td>{gpu}</td><td>{fmt(r['peak_vram_mib'])}</td><td>{fmt(r['average_gpu_power_w'])}</td>"
            f"<td>{r['error_count']}/{r['timeout_count']}</td></tr>")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(meta['label'])}</title><style>{css}</style></head><body>
<h1>DeepSeek V4 Flash · DSpark 性能报告</h1><p class='muted'>{html.escape(meta['label'])} · 生成于 {html.escape(meta['updated_at'])}</p>
<div class='grid'><div class='metric'>完成测试组<div class='value'>{len(results)}/30</div></div><div class='metric'>最大吞吐<div class='value'>{fmt(mx)} tok/s</div></div><div class='metric'>最小吞吐<div class='value'>{fmt(mn)} tok/s</div></div><div class='metric'>总错误/超时<div class='value'>{sum(r['error_count'] for r in results)}/{sum(r['timeout_count'] for r in results)}</div></div></div>
<div class='card'><h2>测试配置</h2><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></div>
<div class='card'><h2>Context × Concurrency 平均汇总</h2><div class='scroll'><table><thead><tr><th>Context</th><th>C</th><th>Repeats</th><th>Gen tok/s</th><th>AR</th><th>AL</th><th>TTFT ms</th><th>TPOT ms</th><th>ITL ms</th><th>Peak VRAM MiB</th><th>Avg Power W/GPU</th><th>波动</th><th>Error/Timeout</th></tr></thead><tbody>{summary_rows}</tbody></table></div></div>
<div class='card'><h2>每组详细结果</h2><div class='scroll'><table><thead><tr><th>Context</th><th>C</th><th>Repeat</th><th>Gen tok/s</th><th>AR</th><th>AL</th><th>TTFT ms</th><th>TPOT ms</th><th>ITL ms</th><th>GPU0–7 Util/Mem/Power</th><th>Peak VRAM</th><th>Avg Power</th><th>Error/Timeout</th></tr></thead><tbody>{detail_rows}</tbody></table></div></div>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def save(args, meta, results):
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_html(meta, results, args.output_html)


def main():
    args = args_parser()
    server = verify_server(args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompts = build_exact_prompts(tokenizer)
    log = Path(server["server_log"])
    meta = {
        "label": args.label,
        "base_url": BASE_URL,
        "model": MODEL,
        "contexts": CONTEXT_LENGTHS,
        "concurrencies": CONCURRENCIES,
        "requests_per_group": REQUESTS_PER_GROUP,
        "warmup_per_group": WARMUP_COUNT,
        "repeats": REPEATS,
        "output_tokens": OUTPUT_TOKENS,
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed": SEED, "ignore_eos": True},
        "server": server,
        "expected_config": {
            "mem_fraction_static": args.expected_mem_fraction,
            "chunked_prefill_size": args.expected_chunked_prefill,
            "max_running_requests": args.expected_max_running,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    results = []
    save(args, meta, results)
    total = len(CONTEXT_LENGTHS) * len(CONCURRENCIES) * REPEATS
    for ctx in CONTEXT_LENGTHS:
        for concurrency in CONCURRENCIES:
            for repeat in range(1, REPEATS + 1):
                print(f"[progress] {len(results)+1}/{total}", flush=True)
                result = run_group(prompts[ctx], ctx, concurrency, repeat, log)
                results.append(result)
                save(args, meta, results)
    print(f"[PASS] complete groups={len(results)} json={args.output_json} html={args.output_html}", flush=True)


if __name__ == "__main__":
    main()
