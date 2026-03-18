"""
MMMLU (Multilingual MMLU) evaluation across multiple sglang servers.

Uses openai/MMMLU, samples from 6 languages: EN_US, ZH_CN, JA_JP, FR_FR, DE_DE, KO_KR.
~200 per language.

Usage:
  python scripts/eval_mmmlu.py                          # ~1200 problems, 8 GPUs
  python scripts/eval_mmmlu.py --per-language 100       # quick
"""
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


LANGUAGES = ["ZH_CN", "JA_JP", "FR_FR", "DE_DE", "KO_KR", "ES_LA"]


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_choice(text):
    """Extract single letter choice (A-D) from model output."""
    text = strip_thinking(text)
    m = re.search(r"[Aa]nswer is:?\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"答案[是为：:]\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-Da-d])\}", text)
    if m:
        return m.group(1).upper()
    m = re.findall(r'\b([A-Da-d])\b', text)
    if m:
        return m[-1].upper()
    return "?"


def run_one(args):
    idx, question, choices_str, gold, lang, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"{question}\n\n{choices_str}\n\n"
        "Think step by step, then give your answer as \"The answer is (X)\"."
    )
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["message"]["content"]
        finish = r["choices"][0].get("finish_reason", "")
        comp = r["usage"]["completion_tokens"]
        pred = extract_choice(content)
        return idx, pred, gold, lang, comp, finish, content, None
    except Exception as e:
        return idx, "?", gold, lang, 0, "", "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    LETTERS = "ABCD"

    all_problems = []
    for lang in LANGUAGES:
        print(f"Loading MMMLU {lang}...")
        try:
            ds = load_dataset("openai/MMMLU", lang, split="test")
        except Exception:
            ds = load_dataset("openai/MMMLU", lang, split="validation")

        n = min(args.per_language, len(ds))
        indices = random.sample(range(len(ds)), n)

        for i in indices:
            item = ds[i]
            q = item["Question"]
            choices_str = "\n".join(f"{LETTERS[j]}. {item[LETTERS[j]]}" for j in range(4))
            gold = item["Answer"]
            all_problems.append((q, choices_str, gold, lang))

    N = len(all_problems)
    ports = args.ports
    print(f"MMMLU eval: {N} problems ({len(LANGUAGES)} languages), {len(ports)} servers")

    tasks = [
        (i, q, c, g, lang, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, c, g, lang) in enumerate(all_problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0

    # Per-language stats
    lang_correct = {l: 0 for l in LANGUAGES}
    lang_total = {l: 0 for l in LANGUAGES}
    total_tok = errors = truncated = no_extract = wrong = 0
    wrong_examples = []

    for idx, pred, gold, lang, comp, finish, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        lang_total[lang] += 1
        if pred == gold:
            lang_correct[lang] += 1
        else:
            if finish == "length":
                truncated += 1
            elif pred == "?":
                no_extract += 1
            else:
                wrong += 1
            if len(wrong_examples) < 10:
                wrong_examples.append({
                    "idx": idx, "pred": pred, "gold": gold, "lang": lang,
                    "finish": finish, "output": content[:200]
                })

    correct = sum(lang_correct.values())
    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"MMMLU {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Overall Accuracy: {correct}/{N} ({acc:.1f}%)")
    print(f"\nPer-language:")
    for lang in LANGUAGES:
        lt = lang_total[lang]
        lc = lang_correct[lang]
        la = lc / lt * 100 if lt > 0 else 0
        print(f"  {lang}: {lc}/{lt} ({la:.1f}%)")
    print(f"\nTruncated:      {truncated}")
    print(f"No extraction:  {no_extract}")
    print(f"Wrong answer:   {wrong}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    if errors:
        print(f"Errors:         {errors}")

    if wrong_examples:
        print(f"\n--- Sample wrong answers ---")
        for ex in wrong_examples[:5]:
            print(f"  [{ex['idx']}] lang={ex['lang']} pred={ex['pred']} gold={ex['gold']} finish={ex['finish']}")
            print(f"    {ex['output'][:150]}...")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        per_lang = {l: {"correct": lang_correct[l], "total": lang_total[l],
                        "accuracy": lang_correct[l] / lang_total[l] * 100 if lang_total[l] > 0 else 0}
                    for l in LANGUAGES}
        with open(os.path.join(args.output_dir, "mmmlu_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "per_language": per_lang, "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
