"""Full GSM8K evaluation across 8 GPU servers in parallel."""
import argparse
import json
import os
import re
import time
import subprocess
import concurrent.futures
from openai import OpenAI


def download_gsm8k():
    import urllib.request
    path = "/tmp/gsm8k_test.jsonl"
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_answer(text):
    """Extract numeric answer from model output."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        val = boxed[-1].replace(',', '').replace('$', '').strip()
        nums = re.findall(r'-?\d+\.?\d*', val)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                pass
    m = re.findall(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if m:
        try:
            return float(m[-1].replace(',', ''))
        except ValueError:
            pass
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def extract_gold(answer_str):
    m = re.findall(r'####\s*(-?\d[\d,]*\.?\d*)', answer_str)
    if m:
        return float(m[-1].replace(',', ''))
    return None


def eval_one(base_url, model, question, max_tokens):
    client = OpenAI(base_url=base_url, api_key="none")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
    )
    return resp.choices[0].message.content


def start_servers(gpu_ids, model_path, config, base_port):
    """Start servers on specified GPUs. Returns list of (gpu_id, port) tuples."""
    servers = []
    for gpu_id in gpu_ids:
        port = base_port + gpu_id
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"PATH=/usr/local/cuda-12.9/bin:$PATH "
            f"CUDA_HOME=/usr/local/cuda-12.9 "
            f"SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            f"python -m sglang.launch_server "
            f"--model-path {model_path} "
            f"--trust-remote-code --tp-size 1 --mem-fraction-static 0.85 "
            f"--max-running-requests 64 --attention-backend flashinfer "
            f"--dllm-algorithm DreamShiftBlockN "
            f"--dllm-algorithm-config {config} "
            f"--dtype bfloat16 --port {port} --chunked-prefill-size 4096"
        )
        log_path = f"/tmp/dllm_gpu{gpu_id}.log"
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=open(log_path, 'w'),
            stderr=subprocess.STDOUT,
        )
        servers.append((gpu_id, port, proc))
        print(f"  GPU {gpu_id}: port {port}, pid {proc.pid}")
    return servers


def wait_servers_ready(servers, timeout=600):
    """Wait for all servers to be healthy."""
    import requests
    ready = set()
    t0 = time.time()
    while len(ready) < len(servers) and time.time() - t0 < timeout:
        for gpu_id, port, proc in servers:
            if port in ready:
                continue
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    ready.add(port)
                    print(f"  GPU {gpu_id} port {port}: READY ({time.time()-t0:.0f}s)")
            except Exception:
                pass
        if len(ready) < len(servers):
            time.sleep(5)
    if len(ready) < len(servers):
        missing = [f"GPU {g} port {p}" for g, p, _ in servers if p not in ready]
        print(f"WARNING: {len(missing)} servers not ready: {missing}")
    return [port for _, port, _ in servers if port in ready]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--model-path", default="/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont")
    parser.add_argument("--config", default="dreamshift_blockN3_greedy_standard.yaml")
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--parallel-per-gpu", type=int, default=8, help="Concurrent requests per GPU")
    parser.add_argument("--skip-server-start", action="store_true", help="Use existing servers")
    parser.add_argument("--existing-ports", default="", help="Comma-separated ports of existing servers")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    # Load GSM8K
    data = download_gsm8k()
    total = len(data)
    print(f"Loaded {total} GSM8K questions")

    if args.existing_ports:
        ports = [int(p) for p in args.existing_ports.split(",")]
        print(f"Using existing servers on ports: {ports}")
    elif args.skip_server_start:
        ports = [args.base_port + g for g in gpu_ids]
        print(f"Using existing servers on ports: {ports}")
    else:
        # Start servers
        print(f"\nStarting {num_gpus} servers...")
        servers = start_servers(gpu_ids, args.model_path, args.config, args.base_port)
        print(f"\nWaiting for servers to be ready...")
        ports = wait_servers_ready(servers)
        if not ports:
            print("ERROR: No servers ready!")
            return

    num_servers = len(ports)
    total_parallel = args.parallel_per_gpu * num_servers
    print(f"\n{num_servers} servers ready, {total_parallel} total concurrent requests")

    # Distribute questions across servers (round-robin)
    assignments = []  # (item_idx, port)
    for i, item in enumerate(data):
        port = ports[i % num_servers]
        assignments.append((i, item, port))

    # Run evaluation
    correct = 0
    invalid = 0
    wrong_items = []
    t0 = time.time()

    def process(args_tuple):
        idx, item, port = args_tuple
        q = item["question"]
        gold = extract_gold(item["answer"])
        base_url = f"http://localhost:{port}/v1"
        try:
            resp_text = eval_one(base_url, "default", q, args.max_tokens)
            pred = extract_answer(resp_text)
            return idx, gold, pred, None
        except Exception as e:
            return idx, gold, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_parallel) as ex:
        futures = [ex.submit(process, a) for a in assignments]
        done = 0
        for f in concurrent.futures.as_completed(futures):
            idx, gold, pred, err = f.result()
            done += 1
            if err:
                invalid += 1
                status = f"ERROR: {err[:50]}"
            elif pred is None:
                invalid += 1
                status = "INVALID"
            elif abs(pred - gold) < 0.01:
                correct += 1
                status = "CORRECT"
            else:
                status = f"WRONG (pred={pred}, gold={gold})"
                wrong_items.append((idx, pred, gold))
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"[{done}/{total}] {correct}/{done} correct ({correct/done:.1%}) elapsed={elapsed:.0f}s — last: {status}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"GSM8K Full Evaluation Results")
    print(f"{'='*60}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}/{total} = {correct/total:.1%}")
    print(f"Wrong: {len(wrong_items)}/{total}")
    print(f"Invalid/Error: {invalid}/{total}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Servers: {num_servers} GPUs, config: {args.config}")
    print(f"Max tokens: {args.max_tokens}")

    if wrong_items:
        print(f"\nWrong questions (first 20):")
        for idx, pred, gold in wrong_items[:20]:
            print(f"  Q{idx}: pred={pred}, gold={gold}")


if __name__ == "__main__":
    main()
