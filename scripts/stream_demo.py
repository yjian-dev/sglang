"""Stream demo for DreamShiftBlock2 serving."""
import argparse
import json
import sys
import time

import requests


def stream_chat(url, prompt, max_tokens=1024, temperature=1.0, top_k=50, top_p=0.95):
    t0 = time.time()
    first_token_time = None
    token_count = 0

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "sdar",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        content = delta.get("content", "")
        if content:
            if first_token_time is None:
                first_token_time = time.time()
            sys.stdout.write(content)
            sys.stdout.flush()
            token_count += len(content.split())  # approximate

    elapsed = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time else 0
    print(f"\n\n--- {elapsed:.1f}s total, TTFT {ttft:.2f}s ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompt = args.prompt or "请用中文详细解释什么是机器学习，包括其主要类型和应用场景。"
    print(f"Prompt: {prompt}\n{'='*60}\n")
    stream_chat(args.url, prompt, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
