"""
Run GSM-Plus evaluation (adversarial GSM8K variants).

Dataset: qintongli/GSM-Plus (10,552 test questions, 8 perturbation types)
Uses 0-shot chat format with \\boxed{} extraction (same as GSM8K 0-shot eval).

Usage:
  python3 -m sglang.test.eval_gsm_plus --host 127.0.0.1 --port 30000
  python3 -m sglang.test.eval_gsm_plus --num-questions 1000 --parallel 64

Multi-GPU sharding:
  python3 -m sglang.test.eval_gsm_plus --shard-id 0 --num-shards 8 --port 30000
"""

import argparse
import json
import os
import re
import time

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import dump_state_text, normalize_base_url


# ─── answer extraction (MATHEvaluator v2, same as few_shot_gsm8k.py) ───

def _extract_boxed_answer(text):
    """Extract answer from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    # Fallback: regex for malformed \boxed
    m = re.search(r"\\boxed\{([^}]*)", text)
    return m.group(1).strip() if m else None


def _strip_string_v2(s):
    s = str(s).strip()
    if s.endswith("\\"):
        s = s[:-1]
    for wrapper in ["\\text{", "\\mathrm{", "\\textbf{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper):-1]
    for ch in [",", "%", "$", '"']:
        s = s.replace(ch, "")
    s = s.replace("\\%", "").replace("\\$", "").replace("\\ ", "")
    return s.strip()


def _normalize_final_answer(s):
    s = _strip_string_v2(s)
    if s.startswith("-") and s[1:].replace(".", "", 1).isdigit():
        return s
    s = s.lstrip("0") or "0"
    return s


def _is_equiv(pred, gt):
    if pred is None:
        return False
    try:
        pred_n = _normalize_final_answer(pred)
        gt_n = _normalize_final_answer(gt)
        if pred_n == gt_n:
            return True
        pf, gf = float(pred_n), float(gt_n)
        return abs(pf - gf) < 1e-6
    except (ValueError, OverflowError):
        return _normalize_final_answer(pred) == _normalize_final_answer(gt)


def _math_postprocess_v2(text):
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return _strip_string_v2(boxed)
    # Fallback: last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.split("answer")[-1] if "answer" in text.lower() else text)
    return numbers[-1] if numbers else ""


# ─── dataset loading ───

def load_gsm_plus(split="test", num_questions=None):
    """Load GSM-Plus from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("qintongli/GSM-Plus", split=split)
    data = list(ds)
    if num_questions and num_questions < len(data):
        data = data[:num_questions]
    return data


# ─── eval logic ───

def run_eval(args):
    set_default_backend(RuntimeEndpoint(normalize_base_url(args.host, args.port)))

    if args.data_path:
        print(f"Loading from {args.data_path}...")
        with open(args.data_path) as f:
            data = [json.loads(l) for l in f]
        if args.num_questions and args.num_questions < len(data):
            data = data[:args.num_questions]
    else:
        print(f"Loading GSM-Plus ({args.split} split)...")
        data = load_gsm_plus(split=args.split, num_questions=args.num_questions)
    num_questions = len(data)
    print(f"Loaded {num_questions} questions")

    # Sharding
    indices = list(range(num_questions))
    if args.num_shards > 1:
        indices = [i for i in indices if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} questions")

    import sglang as sgl

    prompt_suffix = (
        "\nPlease reason step by step, "
        "and put your final answer within \\boxed{}."
    )

    @sgl.function
    def gsm_plus_0shot(s, question):
        s += sgl.user(question + prompt_suffix)
        s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_new_tokens))

    arguments = [{"question": data[i]["question"]} for i in indices]
    tic = time.perf_counter()
    states = gsm_plus_0shot.run_batch(
        arguments, temperature=args.temperature,
        num_threads=args.parallel, progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Evaluate
    correct = 0
    per_type_correct = {}
    per_type_total = {}
    details = []

    for idx_pos, qi in enumerate(indices):
        raw_output = states[idx_pos]["answer"]
        pred = _math_postprocess_v2(raw_output)
        gt = str(data[qi]["answer"]).strip()
        # Handle "None" answers (critical thinking: unanswerable questions)
        if gt.lower() == "none":
            # Model should indicate the problem is unsolvable.
            # Check if model output contains indicators of "cannot be determined".
            lower_output = raw_output.lower()
            is_correct = any(phrase in lower_output for phrase in [
                "cannot be determined", "not enough information",
                "cannot determine", "insufficient information",
                "impossible to determine", "not possible to determine",
                "can't be determined", "can not be determined",
                "don't know", "do not know", "no way to",
                "missing information", "not given", "not provided",
                "not specified", "unknown", "cannot solve",
                "can't solve", "unable to",
            ])
        else:
            is_correct = _is_equiv(pred, gt)
        if is_correct:
            correct += 1

        ptype = data[qi].get("perturbation_type", "unknown")
        per_type_total[ptype] = per_type_total.get(ptype, 0) + 1
        per_type_correct[ptype] = per_type_correct.get(ptype, 0) + (1 if is_correct else 0)

        details.append({
            "idx": qi,
            "question": data[qi]["question"],
            "gt_answer": gt,
            "pred_extracted": pred,
            "correct": is_correct,
            "perturbation_type": ptype,
            "model_output": raw_output,
        })

    acc = correct / len(indices) if indices else 0

    # Speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    shard_str = f" (shard {args.shard_id}/{args.num_shards})" if args.num_shards > 1 else ""
    print(f"\n{'='*50}")
    print(f"GSM-Plus Results{shard_str}")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{len(indices)})")
    print(f"Latency: {latency:.1f}s")
    print(f"Output throughput: {output_throughput:.1f} token/s")

    # Per perturbation type breakdown
    if per_type_total:
        print(f"\nPer perturbation type:")
        print(f"  {'Type':<40} {'Acc':>8} {'Correct':>8} {'Total':>6}")
        print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*6}")
        for ptype in sorted(per_type_total.keys()):
            tc = per_type_correct.get(ptype, 0)
            tt = per_type_total[ptype]
            print(f"  {ptype:<40} {tc/tt:>7.1%} {tc:>8} {tt:>6}")

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""

    output_file = os.path.join(output_dir, f"gsm_plus{shard_suffix}.jsonl")
    with open(output_file, "w") as f:
        for d in details:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    summary = {
        "accuracy": float(acc),
        "correct": correct,
        "total": len(indices),
        "latency": latency,
        "output_throughput": output_throughput,
        "per_type": {k: {"correct": per_type_correct.get(k, 0), "total": v,
                         "accuracy": per_type_correct.get(k, 0) / v}
                     for k, v in per_type_total.items()},
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    summary_file = os.path.join(output_dir, f"gsm_plus_summary{shard_suffix}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")

    dump_state_text(os.path.join(output_dir, f"gsm_plus_raw{shard_suffix}.txt"), states)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to local jsonl file (skip HF download)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split: test (10552) or testmini (2400)")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Max questions to eval (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    run_eval(args)
