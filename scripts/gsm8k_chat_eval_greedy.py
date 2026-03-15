"""Quick GSM8K evaluation using chat API with greedy decoding."""
import argparse
import json
import re
import concurrent.futures
from openai import OpenAI

def download_gsm8k():
    import urllib.request
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    path = "/tmp/gsm8k_test.jsonl"
    urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        val = boxed[-1].replace(',', '').replace('$', '').strip()
        nums = re.findall(r'-?\d+\.?\d*', val)
        if nums:
            try: return float(nums[0])
            except ValueError: pass
    m = re.findall(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if m:
        try: return float(m[-1].replace(',', ''))
        except ValueError: pass
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if nums:
        try: return float(nums[-1].replace(',', ''))
        except ValueError: pass
    return None

def extract_gold(answer_str):
    m = re.findall(r'####\s*(-?\d[\d,]*\.?\d*)', answer_str)
    if m: return float(m[-1].replace(',', ''))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:30001/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    data = download_gsm8k()[:args.num_questions]
    client = OpenAI(base_url=args.base_url, api_key="none")

    correct = 0
    invalid = 0
    total = len(data)

    def process(item):
        q = item["question"]
        gold = extract_gold(item["answer"])
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": q}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = resp.choices[0].message.content
        pred = extract_answer(text)
        return gold, pred, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(process, item) for item in data]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            gold, pred, resp = f.result()
            if pred is None:
                invalid += 1
                status = "INVALID"
            elif abs(pred - gold) < 0.01:
                correct += 1
                status = "CORRECT"
            else:
                status = f"WRONG (pred={pred}, gold={gold})"
            print(f"[{i+1}/{total}] {status}")

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"Invalid: {invalid}/{total}")

if __name__ == "__main__":
    main()
