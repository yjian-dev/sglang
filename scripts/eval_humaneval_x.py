"""
HumanEval-X evaluation across multiple sglang servers.

Tests code generation in 5 languages: Python, C++, Java, JavaScript, Go.
164 problems per language, 820 total.

Usage:
  python scripts/eval_humaneval_x.py                              # all 5 langs
  python scripts/eval_humaneval_x.py --langs python cpp           # specific langs
  python scripts/eval_humaneval_x.py --num-problems 10            # quick test
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import hf_hub_download


LANGS = ["python", "cpp", "java", "js", "go"]


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_code(text, lang):
    """Extract code from model output."""
    text = strip_thinking(text)
    # Find code blocks
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    # No code block: return raw text
    return text.strip()


def build_prompt(item, lang):
    """Build prompt for code completion."""
    prompt_code = item["prompt"]
    lang_names = {"python": "Python", "cpp": "C++", "java": "Java", "js": "JavaScript", "go": "Go"}
    lang_name = lang_names.get(lang, lang)
    return (
        f"Complete the following {lang_name} function. "
        f"Return ONLY the complete function implementation in a code block.\n\n"
        f"```{lang}\n{prompt_code}```"
    )


def run_python_test(prompt, completion, test_code):
    """Run Python test case."""
    # Build full code: prompt (function signature) + completion + test
    func_name = re.search(r"def\s+(\w+)", prompt)
    if not func_name:
        return False

    full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({func_name.group(1)})\n"
    try:
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def run_one(args):
    idx, prompt_text, item, lang, port, max_tokens, timeout, temperature, top_p, top_k = args
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["message"]["content"]
        comp = r["usage"]["completion_tokens"]
        code = extract_code(content, lang)
        return idx, code, item, lang, comp, None
    except Exception as e:
        return idx, "", item, lang, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, nargs="+", default=LANGS)
    parser.add_argument("--num-problems", type=int, default=0, help="0 = all 164")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ports = args.ports
    all_results = {}
    t0_total = time.time()

    for lang in args.langs:
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        print(f"{'='*60}")

        path = hf_hub_download("THUDM/humaneval-x", f"data/{lang}/data/humaneval.jsonl", repo_type="dataset")
        data = [json.loads(l) for l in open(path)]
        N = min(args.num_problems, len(data)) if args.num_problems > 0 else len(data)
        data = data[:N]

        tasks = []
        for i, item in enumerate(data):
            prompt_text = build_prompt(item, lang)
            tasks.append((i, prompt_text, item, lang, ports[i % len(ports)],
                         args.max_tokens, args.timeout, args.temperature, args.top_p, args.top_k))

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(run_one, t) for t in tasks]
            results = [f.result() for f in futures]
        elapsed = time.time() - t0

        # For Python, we can actually run the tests
        if lang == "python":
            passed = 0
            for idx, code, item, _, comp, err in sorted(results, key=lambda x: x[0]):
                if err:
                    continue
                if run_python_test(item["prompt"], code, item["test"]):
                    passed += 1
            acc = passed / N * 100
            print(f"  Python pass@1: {passed}/{N} ({acc:.1f}%)")
        else:
            # For other langs, just report generation stats
            total_tok = sum(comp for _, _, _, _, comp, err in results if not err)
            errors = sum(1 for _, _, _, _, _, err in results if err)
            print(f"  Generated {N} solutions, {errors} errors")
            print(f"  Total tokens: {total_tok:,}, Time: {elapsed:.1f}s")
            acc = None
            passed = None

        total_tok = sum(comp for _, _, _, _, comp, err in results if not err)
        all_results[lang] = {
            "total": N,
            "passed": passed,
            "accuracy": acc,
            "tokens": total_tok,
            "time": elapsed,
        }

        # Save generated code for manual inspection / external evaluation
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            samples = []
            for idx, code, item, _, comp, err in sorted(results, key=lambda x: x[0]):
                samples.append({
                    "task_id": item["task_id"],
                    "completion": code,
                })
            outpath = os.path.join(args.output_dir, f"humaneval_x_{lang}_samples.jsonl")
            with open(outpath, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            print(f"  Saved to {outpath}")

    elapsed_total = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"HumanEval-X Summary ({elapsed_total:.0f}s total)")
    print(f"{'='*60}")
    for lang, r in all_results.items():
        if r["accuracy"] is not None:
            print(f"  {lang:10s}: {r['passed']}/{r['total']} = {r['accuracy']:.1f}%")
        else:
            print(f"  {lang:10s}: {r['total']} generated ({r['tokens']:,} tokens)")


if __name__ == "__main__":
    main()
