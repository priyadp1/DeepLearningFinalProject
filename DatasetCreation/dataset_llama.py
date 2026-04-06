import os
import re
import sys
import json
import math
from datasets import load_dataset, get_dataset_config_names
from openai import OpenAI
from tqdm import tqdm
from dot_env import load_dotenv

# --- 1. CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = "meta-llama/llama-4-maverick"
OUTPUT_FILE = "scibench_results.jsonl"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- 2. ENHANCED COMPARISON FUNCTION ---
def get_numeric_metrics(pred_str, true_str, rel_tol=0.02, abs_tol=1e-5):
    """
    Returns (is_correct, relative_diff)
    Handles signs, scientific notation, and rounding errors.
    """
    def parse_value(s):
        if s is None: return None
        s = str(s).lower().replace(' ', '')
        s = re.sub(r'\\times10\^|x10\^|\*10\^', 'e', s) 
        match = re.search(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', s)
        try:
            return float(match.group()) if match else None
        except (ValueError, AttributeError):
            return None

    val_pred = parse_value(pred_str)
    val_true = parse_value(true_str)

    if val_pred is None or val_true is None:
        return False, None

    # Calculate Relative Difference
    if abs(val_true) > 1e-9:
        rel_diff = abs(val_pred - val_true) / abs(val_true)
    else:
        # If true answer is 0, we use absolute difference
        rel_diff = abs(val_pred - val_true)
    # distance <= max(rel_tol * max(abs(a),abs(b)),abs_tol)
    is_correct = math.isclose(val_pred, val_true, rel_tol=rel_tol, abs_tol=abs_tol)
    
    return is_correct, rel_diff

# --- 3. PROCESSING LOOP ---
subsets = get_dataset_config_names("xw27/scibench")
print(subsets)

for subset in subsets:
    print(f"\n Processing Subset: {subset}")
    try:
        ds = load_dataset("xw27/scibench", subset, split="train")
    except Exception as e:
        print(f"Skipping {subset}: {e}")
        continue
    
    for i in tqdm(range(len(ds))):
        row = ds[i]
        problem_id = str(row.get('problem_id', row.get('problemid', f"{subset}_{i}")))
        source = row.get('source', 'n/a')
        unit = row.get('unit', 'n/a')
        question = row['problem_text']
        true_answer = str(row['answer_number'])

        prompt = (
            f"Problem (Subject: {subset}): {question}\n\n"
            # f"Note: Provide answer in {unit}.\n"
            f"Solve step-by-step. End with: #### <number> for your final answer"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            full_cot = response.choices[0].message.content
            token_count = response.usage.completion_tokens
            
            # Regex for number extraction (including scientific notation)
            match = re.search(r'####\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', full_cot)
            predicted_val_str = match.group(1) if match else None

            # Calculate metrics
            is_correct, rel_diff = get_numeric_metrics(predicted_val_str, true_answer)

            # --- 4. CONSTRUCT RECORD ---
            record = {
                "problemid": problem_id,
                "source":source,
                "unit": unit,
                "problem_text": question,
                "answer_number": true_answer,
                "teacher_cot": full_cot,
                "predicted_answer": predicted_val_str,
                "relative_difference": rel_diff, # NEW COLUMN
                "output_tokens": token_count,
                "is_correct": is_correct
            }

            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

        except Exception as e:
            print(f"Error at {subset} index {i}: {e}")
            continue

print(f"\n Done! Data with error metrics saved to {OUTPUT_FILE}")