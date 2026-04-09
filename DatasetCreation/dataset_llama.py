import os
import re
import sys
import json
import math
from datasets import load_dataset, get_dataset_config_names
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from together import Together

# --- 1. CONFIGURATION ---
load_dotenv()
# Make sure you have TOGETHER_API_KEY in your .env file
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") 
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
OUTPUT_FILE = "scibench_results_together_llama3.3.jsonl"

# Pointing directly to Together AI
# client = OpenAI(
#     base_url="https://api.together.xyz/v1",
#     api_key=TOGETHER_API_KEY,
# )

client = Together()

# --- 2. ENHANCED COMPARISON FUNCTION ---
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

# --- 3. PROCESSING LOOP ---
subsets = get_dataset_config_names("xw27/scibench")

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
            f"Note: Provide answer in {unit}.\n"
            f"Solve step-by-step. End with: #### <number> for your final answer"
        )

        try:
            # response = client.chat.completions.create(
            #     model=MODEL_ID,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.1,
            #     logprobs=True,       
            #     # top_logprobs is required by Together when logprobs=True
            #     # If set to 1, it returns the logprob for the chosen token
            #     top_logprobs=1       
            # )


            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                # In Together SDK, logprobs=1 returns the logprob for the chosen token
                logprobs=1 
            )
            # print(completion)
            full_cot = completion.choices[0].message.content
            # --- TOGETHER SDK LOGPROB EXTRACTION ---
            # The structure differs slightly from the OpenAI response object
            raw_logprobs = None
            avg_logprob = None

            if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
                # Together returns a list of logprobs for each token
                token_logprobs = completion.choices[0].logprobs.token_ids
                # And the actual probability values are here:
                raw_logprobs = completion.choices[0].logprobs.token_logprobs
                
                if raw_logprobs:
                    avg_logprob = sum(raw_logprobs) / len(raw_logprobs)
            token_count = completion.usage.completion_tokens
            # full_cot = response.choices[0].message.content
            # token_count = response.usage.completion_tokens
            
            # # --- SAFE LOGPROB EXTRACTION ---
            # raw_logprobs = None
            # avg_logprob = None
            # print("LOGPROBS\n")
            # print(response)
            
            # # Extraction logic for Together's OpenAI-compatible response
            # if response.choices[0].logprobs and hasattr(response.choices[0].logprobs, 'content'):
            #     content_list = response.choices[0].logprobs.content
            #     if content_list:
            #         raw_logprobs = [lp.logprob for lp in content_list]
            #         avg_logprob = sum(raw_logprobs) / len(raw_logprobs)
            
            match = re.search(r'####\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', full_cot)
            predicted_val_str = match.group(1) if match else None

            is_correct, rel_diff = get_numeric_metrics(predicted_val_str, true_answer)

            record = {
                "problemid": problem_id,
                "source": source,
                "unit": unit,
                "problem_text": question,
                "teacher_cot": full_cot,
                "predicted_answer": predicted_val_str,
                "actual_answer": true_answer,
                "relative_difference": rel_diff,
                "output_tokens": token_count,
                "is_correct": is_correct,
                "avg_logprob": avg_logprob,
                "raw_logprobs": raw_logprobs,
            }

            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

        except Exception as e:
            print(f"Error at {subset} index {i}: {e}")
            continue

print(f"\n Done! Data saved to {OUTPUT_FILE}")