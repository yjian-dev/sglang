"""
CMMLU (Chinese MMLU) evaluation across multiple sglang servers.

Loads from haonan-li/cmmlu zip file (dataset scripts unsupported in new HF).

Usage:
  python scripts/eval_cmmlu.py                          # 2000 random problems, 8 GPUs
  python scripts/eval_cmmlu.py --num-problems 500       # quick
"""
import argparse
import csv
import json
import os
import random
import re
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import hf_hub_download


CMMLU_SUBJECT_MAPPING = {
    'agronomy': '农学',
    'anatomy': '解剖学',
    'ancient_chinese': '古汉语',
    'arts': '艺术学',
    'astronomy': '天文学',
    'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试',
    'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史',
    'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格',
    'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学',
    'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学',
    'college_law': '大学法律',
    'college_mathematics': '大学数学',
    'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学',
    'computer_science': '计算机科学',
    'computer_security': '计算机安全',
    'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理',
    'economics': '经济学',
    'education': '教育学',
    'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文',
    'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学',
    'ethnology': '民族学',
    'food_science': '食品科学',
    'genetics': '遗传学',
    'global_facts': '全球事实',
    'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学',
    'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为',
    'international_law': '国际法学',
    'journalism': '新闻学',
    'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础',
    'logical': '逻辑学',
    'machine_learning': '机器学习',
    'management': '管理学',
    'marketing': '市场营销',
    'marxist_theory': '马克思主义理论',
    'modern_chinese': '现代汉语',
    'nutrition': '营养学',
    'philosophy': '哲学',
    'professional_accounting': '专业会计',
    'professional_law': '专业法学',
    'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学',
    'public_relations': '公共关系',
    'security_study': '安全研究',
    'sociology': '社会学',
    'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药',
    'virology': '病毒学',
    'world_history': '世界历史',
    'world_religions': '世界宗教',
}


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_choice(text):
    """Extract single letter choice (A-D) from model output."""
    text = strip_thinking(text)
    # OC primary: 答案: X (with optional non-word chars around letter)
    m = re.search(r'(?i)答案\s*:\s*[\W]*([A-D])[\W]*', text)
    if m:
        return m.group(1).upper()
    m = re.search(r"[Aa]nswer is:?\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"答案[是为：:]\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"选\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-Da-d])\}", text)
    if m:
        return m.group(1).upper()
    m = re.findall(r'\b([A-Da-d])\b', text)
    if m:
        return m[-1].upper()
    return "?"


def load_cmmlu():
    """Load CMMLU from zip file."""
    path = hf_hub_download("haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset")
    tmpdir = tempfile.mkdtemp()
    items = []
    LETTERS = "ABCD"
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("test/") and name.endswith(".csv"):
                z.extract(name, tmpdir)
                filepath = os.path.join(tmpdir, name)
                subject = os.path.basename(name).replace(".csv", "")
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # skip header
                    for row in reader:
                        if len(row) >= 6:
                            # Format: idx, Question, A, B, C, D, Answer
                            q = row[1]
                            choices = [row[2], row[3], row[4], row[5]]
                            gold = row[6].strip() if len(row) > 6 else "?"
                            items.append({
                                "question": q,
                                "choices": choices,
                                "answer": gold,
                                "subject": subject,
                            })
    return items


def run_one(args):
    idx, question, choices_str, gold, subject_cn, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"请回答以下关于{subject_cn}的单项选择题, "
        "你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.\n\n"
        f"{question}\n\n{choices_str}"
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
        return idx, pred, gold, comp, finish, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0=full dataset")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("Loading CMMLU dataset from zip...")
    all_items = load_cmmlu()
    print(f"Total CMMLU test problems: {len(all_items)}")

    random.seed(42)
    if args.num_problems > 0 and args.num_problems < len(all_items):
        N = args.num_problems
        sampled = random.sample(all_items, N)
    else:
        N = len(all_items)
        sampled = all_items
    ports = args.ports

    LETTERS = "ABCD"
    problems = []
    for item in sampled:
        q = item["question"]
        choices_str = "\n".join(f"{LETTERS[i]}) {item['choices'][i]}" for i in range(4))
        gold = item["answer"]
        subject = item["subject"]
        subject_cn = CMMLU_SUBJECT_MAPPING.get(subject, subject)
        problems.append((q, choices_str, gold, subject_cn))

    print(f"CMMLU eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, q, c, g, scn, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, c, g, scn) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % max(N // 5, 1) == 0:
                print(f"  {done}/{N} done ({time.time() - t0:.0f}s)")
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    correct = total_tok = errors = truncated = no_extract = wrong = 0
    wrong_examples = []
    for idx, pred, gold, comp, finish, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        if pred == gold:
            correct += 1
        else:
            if finish == "length":
                truncated += 1
            elif pred == "?":
                no_extract += 1
            else:
                wrong += 1
            if len(wrong_examples) < 10:
                wrong_examples.append({
                    "idx": idx, "pred": pred, "gold": gold,
                    "finish": finish, "output": content[:200]
                })

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"CMMLU {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Truncated:      {truncated}")
    print(f"No extraction:  {no_extract}")
    print(f"Wrong answer:   {wrong}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if wrong_examples:
        print(f"\n--- Sample wrong answers ---")
        for ex in wrong_examples[:5]:
            print(f"  [{ex['idx']}] pred={ex['pred']} gold={ex['gold']} finish={ex['finish']}")
            print(f"    {ex['output'][:150]}...")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "cmmlu_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok, "truncated": truncated,
                       "no_extract": no_extract, "wrong": wrong}, f, indent=2)


if __name__ == "__main__":
    main()
