"""
Run GSM-8K evaluation.

Modes:
  - Few-shot (default): N-shot with raw text prompts
  - 0-shot chat (--chat-0shot): OpenCompass gsm8k_0shot_v2 format with chat template,
    \\boxed{} answer extraction, and MATHEvaluator v2 comparison.

Usage:
  python3 -m sglang.test.few_shot_gsm8k --num-questions 200
  python3 -m sglang.test.few_shot_gsm8k --chat-0shot --num-questions 200 --max-new-tokens 8192

Multi-GPU sharding (0-shot only):
  python3 -m sglang.test.few_shot_gsm8k --chat-0shot --num-questions 1319 \\
      --max-new-tokens 8192 --shard-id 0 --num-shards 8 --port 30010
"""

import argparse
import ast
import json
import os
import re
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import (
    download_and_cache_file,
    dump_state_text,
    normalize_base_url,
    read_jsonl,
)

INVALID = -9999999


# ─── few-shot helpers ───

def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


# ─── OpenCompass MATHEvaluator v2 (inlined) ───

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = _last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = _remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match(r"^\{(.*)\}$", answer)
        if match:
            answer = match.group(1)
    return answer


def _normalize_final_answer(final_answer: str) -> str:
    SUBSTITUTIONS = [
        ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
        (r"\ ", ""), (" ", ""), ("mbox", "text"),
        (",\\text{and}", ","), ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"), ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square", "ways", "integers", "dollars", "mph", "inches", "ft",
        "hours", "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
        "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals",
        "edges", "students", "childrentickets", "multiples", "\\text{s}",
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f',
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = final_answer.replace("\n", "").replace("\r", "").replace("\f", "")
    if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]
    if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
        final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]
    if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]
    if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
        final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def _math_postprocess_v2(text: str) -> str:
    """Answer extraction: OpenCompass math_postprocess_v2 + robust regex fallback.

    Uses proper brace-matching first (OpenCompass behavior), then falls back
    to regex extraction for malformed \\boxed{} patterns that can occur in
    sglang's streamed/batched output.
    """
    # 1. OpenCompass: proper brace-matching extraction
    cand_ans = _extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans
    # 2. Regex fallback for malformed \boxed{} (e.g., \boxed{18 $$}, unclosed braces)
    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not boxed_matches:
        boxed_matches = re.findall(r"\\boxed\{([^}$]+)", text)
    if boxed_matches:
        ans = boxed_matches[-1].strip().replace(",", "").replace("$", "")
        nums = re.findall(r"-?\d+\.?\d*", ans)
        if nums:
            v = nums[0].rstrip(".")
            try:
                fv = float(v)
                if not (fv != fv or fv == float('inf') or fv == float('-inf')):
                    if fv == int(fv):
                        return str(int(fv))
            except (ValueError, OverflowError):
                pass
            return v
    # 3. "final answer" / "answer is" pattern
    for maybe_ans in text.split("."):
        if re.search("final answer|answer is", maybe_ans.lower()):
            return _normalize_final_answer(maybe_ans)
    # 4. Last number in text
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    if nums:
        return nums[-1]
    return _normalize_final_answer(text.split(".")[0])


def _strip_string_v2(string):
    string = str(string).strip().replace("\n", "").rstrip(".")
    string = string.replace("\\!", "").replace("\\ ", "")
    string = string.replace("\\\\", "\\").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\text", "").replace("x\\in", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("'", "").replace('"', "")
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    return string


def _is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string_v2(str1)
        ss2 = _strip_string_v2(str2)
        if ss1 == ss2:
            return True
        ss1 = _normalize_final_answer(ss1)
        ss2 = _normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass
    try:
        ss1 = _normalize_final_answer(str1)
        ss2 = _normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass
    return str1 == str2


def _strip_thinking(text):
    # Remove <think>...</think> block
    result = re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    # If <think> exists but no </think> (truncated), remove from <think> to end
    if "<think>" in result and "</think>" not in result:
        result = result[: result.index("<think>")]
    return result


def _gsm8k_dataset_postprocess(text):
    return text.split("#### ")[1].replace(",", "") if "#### " in text else text


# ─── eval logic ───

def run_eval(args):
    # Select backend
    set_default_backend(RuntimeEndpoint(normalize_base_url(args.host, args.port)))

    if args.data_path is None:
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path
    lines = list(read_jsonl(filename))
    num_questions = min(args.num_questions, len(lines))

    # Sharding support
    indices = list(range(num_questions))
    if args.num_shards > 1:
        indices = [i for i in indices if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} questions")

    import sglang as sgl

    if args.chat_0shot:
        # ── 0-shot chat mode (OpenCompass format) ──
        prompt_suffix = (
            "\nPlease reason step by step, "
            "and put your final answer within \\boxed{}."
        )

        @sgl.function
        def gsm8k_0shot_chat(s, question):
            s += sgl.user(question + prompt_suffix)
            s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_new_tokens))

        arguments = [{"question": lines[i]["question"]} for i in indices]
        tic = time.perf_counter()
        states = gsm8k_0shot_chat.run_batch(
            arguments, temperature=args.temperature,
            num_threads=args.parallel, progress_bar=True,
        )
        latency = time.perf_counter() - tic

        # Evaluate with MATHEvaluator v2 logic
        correct = 0
        details = []
        for idx_pos, qi in enumerate(indices):
            raw_output = states[idx_pos]["answer"]
            # Note: OpenCompass does NOT strip thinking before extraction.
            # _math_postprocess_v2 finds the last \boxed{} which is typically
            # in the final answer section after </think>.
            output = raw_output
            pred = _math_postprocess_v2(output)
            gt = _gsm8k_dataset_postprocess(lines[qi]["answer"])
            is_correct = _is_equiv(pred, gt)
            if is_correct:
                correct += 1
            details.append({
                "idx": qi,
                "question": lines[qi]["question"],
                "gt_answer": lines[qi]["answer"],
                "gt_extracted": gt,
                "pred_extracted": pred,
                "correct": is_correct,
                "model_output": raw_output,
            })
        acc = correct / len(indices) if indices else 0
        invalid = 0.0
    else:
        # ── original few-shot mode ──
        num_shots = args.num_shots
        few_shot_examples = get_few_shot_examples(lines, num_shots)
        arguments = [{"question": get_one_example(lines, i, False)} for i in indices]
        labels = [get_answer_value(lines[i]["answer"]) for i in indices]

        @sgl.function
        def few_shot_gsm8k(s, question):
            s += few_shot_examples + question
            s += sgl.gen("answer", max_tokens=args.max_new_tokens,
                         stop=["Question", "Assistant:", "<|separator|>"])

        tic = time.perf_counter()
        states = few_shot_gsm8k.run_batch(
            arguments, temperature=args.temperature,
            num_threads=args.parallel, progress_bar=True,
        )
        latency = time.perf_counter() - tic

        preds = [get_answer_value(s["answer"]) for s in states]
        acc = np.mean(np.array(preds) == np.array(labels))
        invalid = np.mean(np.array(preds) == INVALID)
        details = [
            {"idx": indices[i], "question": lines[indices[i]]["question"],
             "gt_answer": lines[indices[i]]["answer"], "pred": preds[i],
             "gt": labels[i], "correct": preds[i] == labels[i],
             "model_output": states[i]["answer"]}
            for i in range(len(indices))
        ]

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    mode = "0-shot chat" if args.chat_0shot else f"{args.num_shots}-shot"
    shard_str = f" (shard {args.shard_id}/{args.num_shards})" if args.num_shards > 1 else ""
    print(f"Mode: {mode}{shard_str}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Save detailed results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    output_file = os.path.join(output_dir, f"gsm8k_{mode.replace(' ', '_')}{shard_suffix}.jsonl")
    with open(output_file, "w") as f:
        for d in details:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
    summary = {
        "mode": mode, "accuracy": float(acc), "invalid": float(invalid),
        "latency": latency, "output_throughput": output_throughput,
        "num_questions": len(indices), "num_correct": sum(1 for d in details if d["correct"]),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
    }
    summary_file = os.path.join(output_dir, f"summary_{mode.replace(' ', '_')}{shard_suffix}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_file}")

    dump_state_text(os.path.join(output_dir, f"raw_states{shard_suffix}.txt"), states)

    return {"accuracy": acc, "invalid": invalid, "latency": latency,
            "output_throughput": output_throughput}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save detailed results.")
    parser.add_argument("--chat-0shot", action="store_true",
                        help="0-shot chat format (OpenCompass gsm8k_0shot_v2).")
    # Multi-GPU sharding
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for multi-GPU eval.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    args = parser.parse_args()
    run_eval(args)
