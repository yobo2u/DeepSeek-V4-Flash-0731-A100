#!/usr/bin/env python3
"""DSV4 DSpark comprehensive benchmark."""

import json, time, os, sys, statistics, subprocess, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

BASE_URL = "http://127.0.0.1:8082/v1"
MODEL = "deepseek-v4-flash-0731"
OUTPUT_FILE = "/root/work/models/dsv4-runtime/benchmark_results.json"

CONTEXT_LENGTHS = [1024, 4096, 16384, 65536, 131072]
CONCURRENCIES = [1, 8, 16]
OUTPUT_TOKENS = 1024
REQUESTS_PER_GROUP = 30
WARMUP_COUNT = 5
REPEATS = 2

PROMPT_TEMPLATE = """请完整分析以上上下文内容，提取其中的主要事实、技术参数、关键关系、潜在问题和可优化点。先给出整体理解，再逐项分析重要信息之间的关系，随后指出可能存在的性能瓶颈、配置冲突、风险因素以及优化方向。对于每个判断说明依据，并给出优先级排序。最后形成一份结构化技术评估，包括现状、问题、原因、影响、优化建议和结论。尽可能充分利用给定上下文信息展开分析。"""

def generate_context_text(target_tokens):
    """Generate a context of approximately target_tokens."""
    # Use a repeated technical paragraph to fill context
    base = "DeepSeek V4 is a large language model with mixture-of-experts architecture. "
    base += "The model uses Multi-head Latent Attention (MLA) with FP8 KV cache compression. "
    base += "It employs DSpark speculative decoding with Markov head for draft token prediction. "
    base += "The A800 GPU platform uses SM80 compute capability with 80GB HBM2e memory. "
    base += "Tensor parallelism is configured across 8 GPUs with NVLink interconnect. "
    base += "Key parameters: context_length=1048576, tp_size=8, mem_fraction=0.85. "
    base += "MoE layers use MXFP4 quantization with INT8 activation for SM80 compatibility. "
    base += "The attention backend uses BF16 KV cache with INT8 indexer on A100/A800. "
    base += "top_k=6 experts per token, chunked_prefill_size=16384, max_running_requests=16. "
    return (base * (target_tokens // 4 + 1))[:target_tokens * 4]

def make_request(prompt_text, max_tokens):
    """Make a single API request and return timing."""
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False
    }).encode()
    
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    start = time.time()
    first_token_time = None
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        body = resp.read()
        ttft = None  # non-streaming, can't measure TTFT easily
        elapsed = time.time() - start
        result = json.loads(body)
        usage = result.get("usage", {})
        return {
            "success": True,
            "elapsed": elapsed,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "error": str(e), "elapsed": elapsed}

def get_gpu_metrics():
    """Get GPU metrics via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=5
        ).decode().strip().split("\n")
        gpus = {}
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            gpus[int(parts[0])] = {
                "util": float(parts[1]),
                "mem": float(parts[2]),
                "power": float(parts[3]) if parts[3] != "[Not Supported]" else 0
            }
        return gpus
    except:
        return {}

def get_decode_metrics_from_log():
    """Get the most recent decode metrics from SGLang log."""
    # Read last 200 lines of the log
    try:
        # Find the log file
        out = subprocess.check_output(
            ["bash", "-c", "ls -t /root/work/models/dsv4-runtime/logs/*.log 2>/dev/null | head -1"],
            timeout=5
        ).decode().strip()
        if not out:
            return None
        with open(out) as f:
            lines = f.readlines()
        lines = lines[-200:]
        metrics = []
        for line in lines:
            if "accept len:" in line and "gen throughput" in line:
                parts = line.split(",")
                m = {}
                for p in parts:
                    p = p.strip()
                    if "accept len:" in p:
                        m["accept_len"] = float(p.split(":")[-1].strip())
                    elif "accept rate:" in p:
                        m["accept_rate"] = float(p.split(":")[-1].strip())
                    elif "gen throughput" in p:
                        m["gen_throughput"] = float(p.split(":")[-1].strip().split()[0])
                if m:
                    metrics.append(m)
        return metrics
    except:
        return None

def run_benchmark_group(ctx_len, concurrency, repeat_num):
    """Run one benchmark group."""
    context = generate_context_text(ctx_len)
    prompt = f"{context}\n\n{PROMPT_TEMPLATE}"
    
    print(f"  [repeat {repeat_num}] warmup {WARMUP_COUNT}...")
    # Warmup
    for i in range(WARMUP_COUNT):
        make_request(prompt, OUTPUT_TOKENS)
        time.sleep(0.5)
    
    print(f"  [repeat {repeat_num}] running {REQUESTS_PER_GROUP} requests concurrency={concurrency}...")
    
    results = []
    gpu_samples = []
    
    def do_request(i):
        return make_request(prompt, OUTPUT_TOKENS)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(REQUESTS_PER_GROUP):
            futures.append(executor.submit(do_request, i))
            # Collect GPU metrics every 5 requests
            if i % 5 == 0 and i > 0:
                gpu_samples.append(get_gpu_metrics())
        
        for f in as_completed(futures):
            results.append(f.result())
    
    total_time = time.time() - start_time
    gpu_samples.append(get_gpu_metrics())
    
    # Aggregate
    successes = [r for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]
    
    completion_tokens = sum(r["completion_tokens"] for r in successes)
    gen_throughput = completion_tokens / total_time if total_time > 0 else 0
    
    # Get decode metrics from log
    decode_metrics = get_decode_metrics_from_log()
    
    avg_accept_rate = None
    avg_accept_len = None
    if decode_metrics:
        rates = [m["accept_rate"] for m in decode_metrics if "accept_rate" in m]
        lens = [m["accept_len"] for m in decode_metrics if "accept_len" in m]
        if rates:
            avg_accept_rate = statistics.mean(rates)
        if lens:
            avg_accept_len = statistics.mean(lens)
    
    # GPU aggregate
    avg_gpu = {}
    if gpu_samples:
        for gpu_id in range(8):
            utils = [s.get(gpu_id, {}).get("util", 0) for s in gpu_samples if gpu_id in s]
            mems = [s.get(gpu_id, {}).get("mem", 0) for s in gpu_samples if gpu_id in s]
            powers = [s.get(gpu_id, {}).get("power", 0) for s in gpu_samples if gpu_id in s]
            avg_gpu[gpu_id] = {
                "avg_util": statistics.mean(utils) if utils else 0,
                "avg_mem": statistics.mean(mems) if mems else 0,
                "avg_power": statistics.mean(powers) if powers else 0,
            }
    
    return {
        "context_length": ctx_len,
        "concurrency": concurrency,
        "repeat": repeat_num,
        "gen_throughput": gen_throughput,
        "accept_rate": avg_accept_rate,
        "accept_len": avg_accept_len,
        "total_time": total_time,
        "completion_tokens": completion_tokens,
        "success_count": len(successes),
        "error_count": len(errors),
        "gpu_metrics": avg_gpu,
        "peak_vram": max(s.get(i, {}).get("mem", 0) for s in gpu_samples for i in range(8)) if gpu_samples else 0,
    }

def main():
    print("=" * 60)
    print(f"DSV4 DSpark Benchmark - {datetime.now().isoformat()}")
    print(f"Server: {BASE_URL}  Model: {MODEL}")
    print("=" * 60)
    
    all_results = []
    
    # Progress tracking
    total_groups = len(CONTEXT_LENGTHS) * len(CONCURRENCIES) * REPEATS
    group_num = 0
    
    for ctx_len in CONTEXT_LENGTHS:
        ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        print(f"\n{'='*40}")
        print(f"Context: {ctx_label}")
        print(f"{'='*40}")
        
        for concurrency in CONCURRENCIES:
            print(f"  Concurrency: {concurrency}")
            
            for repeat in range(1, REPEATS + 1):
                group_num += 1
                print(f"  [{group_num}/{total_groups}] Context={ctx_label} Concurrency={concurrency} Repeat={repeat}")
                
                result = run_benchmark_group(ctx_len, concurrency, repeat)
                all_results.append(result)
                
                r = result
                print(f"    => Gen Throughput: {r['gen_throughput']:.1f} tok/s")
                if r['accept_rate'] is not None:
                    print(f"       Accept Rate: {r['accept_rate']:.3f}  Accept Len: {r['accept_len']:.2f}")
                print(f"       Success: {r['success_count']}  Errors: {r['error_count']}")
                
                # Save intermediate
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Benchmark complete! Results saved to:", OUTPUT_FILE)
    print("=" * 60)
    
    # Print summary
    print_summary(all_results)

def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    for ctx_len in CONTEXT_LENGTHS:
        ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        print(f"\n--- Context: {ctx_label} ---")
        print(f"{'Concurrency':<12} {'Repeat':<8} {'Throughput':<12} {'AcceptRate':<12} {'AcceptLen':<10} {'Errors':<8}")
        print("-" * 65)
        
        for concurrency in CONCURRENCIES:
            for repeat in range(1, REPEATS + 1):
                matches = [r for r in results 
                          if r["context_length"] == ctx_len 
                          and r["concurrency"] == concurrency 
                          and r["repeat"] == repeat]
                if matches:
                    r = matches[0]
                    ar = f"{r['accept_rate']:.3f}" if r['accept_rate'] else "N/A"
                    al = f"{r['accept_len']:.2f}" if r['accept_len'] else "N/A"
                    print(f"{concurrency:<12} {repeat:<8} {r['gen_throughput']:<12.1f} {ar:<12} {al:<10} {r['error_count']:<8}")

if __name__ == "__main__":
    main()
