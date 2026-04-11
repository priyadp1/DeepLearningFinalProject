import os
import re
import json
import math
import argparse
from datasets import load_dataset, get_dataset_config_names
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- 2. NUMERIC EVALUATION ---
def get_numeric_metrics(pred_str, true_str, rel_tol=0.02, abs_tol=1e-5):
    def parse_value(s):
        if s is None: return None
        s = str(s).lower().replace(' ', '')
        s = re.sub(r'\\times10\^|x10\^|\*10\^', 'e', s) 
        match = re.search(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', s)
        try:
            return float(match.group()) if match else None
        except (ValueError, AttributeError):
            return None

    # Handle exact matches for Booleans/Strings (common in TheoremQA)
    if str(pred_str).lower().strip() == str(true_str).lower().strip():
        return True, 0.0

    val_pred = parse_value(pred_str)
    val_true = parse_value(true_str)

    if val_pred is None or val_true is None:
        return False, None

    if abs(val_true) > 1e-9:
        rel_diff = abs(val_pred - val_true) / abs(val_true)
    else:
        rel_diff = abs(val_pred - val_true)
    
    is_correct = math.isclose(val_pred, val_true, rel_tol=rel_tol, abs_tol=abs_tol)
    return is_correct, rel_diff

# --- 3. CORE PROCESSING FUNCTION ---
def run_benchmark(dataset_type):
    output_file = f"{dataset_type}_{f"llama" if "llama-3.3-70B" in MODEL_ID else "gpt-oss-120b"}_openrouter.jsonl"
    print(f"Starting benchmark: {dataset_type}")

    # Prepare data based on flag
    items = []
    if dataset_type == "scibench":
        subsets = get_dataset_config_names("xw27/scibench")
        for sub in subsets:
            ds = load_dataset("xw27/scibench", sub, split="train")
            for row in ds:
                items.append({
                    "problem_id": str(row.get('problem_id', row.get('problemid'))),
                    "source": row.get('source', 'n/a'),
                    "unit": row.get('unit', 'n/a'),
                    "problem_text": row['problem_text'],
                    "answer_number": str(row['answer_number']),
                    "subset": sub
                })
    else:  # TheoremQA
        ds = load_dataset("TIGER-Lab/TheoremQA", split="test")
        for i, row in enumerate(ds):
            items.append({
                "problem_id": str(row.get('id', i)),
                "source": "TheoremQA",
                "unit": row.get('Answer_type', 'n/a'),
                "problem_text": row['Question'],
                "answer_number": str(row['Answer']),
                "subset": "theoremqa"
            })

    for item in tqdm(items):
        prompt = (
            f"Problem (Subject: {item['subset']}): {item['problem_text']}\n\n"
            f"Note: Provide answer in {item['unit']}.\n"
            f"Solve step-by-step. End with: #### <number> for your final answer"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                # logprobs=True # Logits logic enabled
            )
            
            full_cot = response.choices[0].message.content
            token_count = response.usage.completion_tokens
            
            # --- LOGPROB EXTRACTION ---
            # avg_logprob = None
            # raw_logprobs = None
            # if response.choices[0].logprobs and response.choices[0].logprobs.content:
            #     raw_logprobs = [lp.logprob for lp in response.choices[0].logprobs.content]
            #     avg_logprob = sum(raw_logprobs) / len(raw_logprobs)
            
            # Match number or capture string for TheoremQA flexibility
            match = re.search(r'####\s*(.*)', full_cot)
            predicted_val_str = match.group(1).strip() if match else None

            is_correct, rel_diff = get_numeric_metrics(predicted_val_str, item['answer_number'])

            # SCHEMA: Matching your original requested fields
            record = {
                "problemid": item['problem_id'],
                "source": item['source'],
                "unit": item['unit'],
                "problem_text": item['problem_text'],
                "teacher_cot": full_cot,
                "predicted_answer": predicted_val_str,
                "actual_answer": item['answer_number'],
                "relative_difference": rel_diff,
                "output_tokens": token_count,
                "is_correct": is_correct,
                # "avg_logprob": avg_logprob,
                # "raw_logprobs": raw_logprobs,
            }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

        except Exception as e:
            print(f"Error processing {item['problem_id']}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scibench", "theoremqa"], default="scibench")
    args = parser.parse_args()
    run_benchmark(args.dataset)