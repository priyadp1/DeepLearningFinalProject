import argparse
import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score


def load_model(path, device):
    print(f"  Loading {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(problem_text):
    return f"Solve the following problem step by step.\n\nProblem: {problem_text.strip()}"


def extract_answer(response):
    """Pull the answer after #### if present, otherwise grab the last number."""
    if "####" in response:
        raw = response.split("####")[-1].strip()
        # Strip trailing punctuation, units, whitespace
        raw = re.sub(r"[^\d.\-eE]+$", "", raw).strip()
        return raw

    # Fallback: last number in the response
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", response)
    return numbers[-1] if numbers else ""


def answers_match(predicted, actual, tolerance=0.01):
    """Compare numerically with tolerance, fall back to string match."""
    predicted = predicted.strip().rstrip(".")
    actual = str(actual).strip().rstrip(".")

    if predicted == actual:
        return True

    try:
        p = float(predicted)
        a = float(actual)
        if a == 0:
            return abs(p) < tolerance
        return abs(p - a) / abs(a) < tolerance
    except (ValueError, ZeroDivisionError):
        return predicted.lower() == actual.lower()

def extract_reasoning(response):
    """Extract the reasoning steps from the response, if present."""
    response = response.strip()
    if "####" in response:
        cleaned_response = response.split("####")[0].strip()
        return cleaned_response

def reasoning_match(predicted, actual):
    if not actual or not predicted:
        return {"bert_p": 0.0, "bert_r": 0.0, "bert_f1": 0.0, "reasoning_correct": False}
    P, R, F1 = score([predicted], [actual], lang="en", verbose=False)
    f1 = F1.item()
    return {
        "bert_p": round(P.item(), 4),
        "bert_r": round(R.item(), 4),
        "bert_f1": round(f1, 4),
        "reasoning_correct": f1 > 0.75,
    }

def evaluate_model(model, tokenizer, records, device, max_new_tokens):
    results = []
    correct = 0

    for i, r in enumerate(records):
        prompt = format_prompt(r["problem_text"])
        tokens = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start

        response = tokenizer.decode(out[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        predicted_answer = extract_answer(response)
        actual_answer = str(r["actual_answer"]).strip()
        match = answers_match(predicted_answer, actual_answer)
        actual_reasoning = r.get("teacher_cot", "").strip()
        predicted_reasoning = extract_reasoning(response)
        bert_scores = reasoning_match(predicted_reasoning, actual_reasoning)
        correct += int(match)

        results.append({
            "problemid": r.get("problemid", f"problem_{i}"),
            "source": r.get("source", ""),
            "predicted": predicted_answer,
            "actual": actual_answer,
            "correct": match,
            **bert_scores,
            "generation_time": round(elapsed, 2),
            "response": response,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"    {i+1}/{len(records)} — accuracy: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")

    return results, correct


def print_comparison(base_results, tuned_results):
    """Side-by-side comparison table."""
    print("\n" + "=" * 80)
    print(f"{'Problem':<20} {'Actual':>10} {'Base':>10} {'Tuned':>10} {'Base':>6} {'Tuned':>6}")
    print(f"{'':<20} {'':>10} {'Pred':>10} {'Pred':>10} {'✓/✗':>6} {'✓/✗':>6}")
    print("-" * 80)

    base_only = 0
    tuned_only = 0
    both_right = 0
    both_wrong = 0

    for b, t in zip(base_results, tuned_results):
        pid = b["problemid"][:18]
        actual = b["actual"][:10]
        b_pred = b["predicted"][:10] if b["predicted"] else "—"
        t_pred = t["predicted"][:10] if t["predicted"] else "—"
        b_mark = "  ✓" if b["correct"] else "  ✗"
        t_mark = "  ✓" if t["correct"] else "  ✗"

        print(f"{pid:<20} {actual:>10} {b_pred:>10} {t_pred:>10} {b_mark:>6} {t_mark:>6}")

        if b["correct"] and t["correct"]:
            both_right += 1
        elif b["correct"] and not t["correct"]:
            base_only += 1
        elif not b["correct"] and t["correct"]:
            tuned_only += 1
        else:
            both_wrong += 1

    n = len(base_results)
    b_correct = sum(1 for r in base_results if r["correct"])
    t_correct = sum(1 for r in tuned_results if r["correct"])
    b_time = sum(r["generation_time"] for r in base_results)
    t_time = sum(r["generation_time"] for r in tuned_results)

    b_avg_f1 = sum(r["bert_f1"] for r in base_results) / n
    t_avg_f1 = sum(r["bert_f1"] for r in tuned_results) / n
    b_avg_r = sum(r["bert_r"] for r in base_results) / n
    t_avg_r = sum(r["bert_r"] for r in tuned_results) / n
    b_avg_p = sum(r["bert_p"] for r in base_results) / n
    t_avg_p = sum(r["bert_p"] for r in tuned_results) / n
    b_reasoning = sum(1 for r in base_results if r["reasoning_correct"])
    t_reasoning = sum(1 for r in tuned_results if r["reasoning_correct"])

    print("=" * 80)
    print(f"\n{'METRIC':<30} {'BASE':>15} {'TUNED':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<30} {b_correct}/{n} ({100*b_correct/n:.1f}%){'':<3} {t_correct}/{n} ({100*t_correct/n:.1f}%)")
    print(f"{'Reasoning Correct':<30} {b_reasoning}/{n} ({100*b_reasoning/n:.1f}%){'':<3} {t_reasoning}/{n} ({100*t_reasoning/n:.1f}%)")
    print(f"{'Avg BERTScore F1':<30} {b_avg_f1:>15.4f} {t_avg_f1:>15.4f}")
    print(f"{'Avg BERTScore Recall':<30} {b_avg_r:>15.4f} {t_avg_r:>15.4f}")
    print(f"{'Avg BERTScore Precision':<30} {b_avg_p:>15.4f} {t_avg_p:>15.4f}")
    print(f"{'Total gen time':<30} {b_time:>12.1f}s {t_time:>12.1f}s")
    print(f"{'Avg time per problem':<30} {b_time/n:>12.2f}s {t_time/n:>12.2f}s")
    print()
    print(f"  Both correct:     {both_right}")
    print(f"  Both wrong:       {both_wrong}")
    print(f"  Base only correct: {base_only}")
    print(f"  Tuned only correct: {tuned_only}")

    if t_correct > b_correct:
        print(f"\n  → Tuned model improved by +{t_correct - b_correct} problems ({100*(t_correct-b_correct)/n:.1f}%)")
    elif b_correct > t_correct:
        print(f"\n  → Tuned model regressed by -{b_correct - t_correct} problems ({100*(b_correct-t_correct)/n:.1f}%)")
    else:
        print(f"\n  → Same accuracy — check per-problem breakdown for differences")

    # Breakdown by source if available
    sources = set(r.get("source", "") for r in base_results)
    if len(sources) > 1:
        print(f"\n{'SOURCE':<20} {'BASE':>15} {'TUNED':>15}")
        print("-" * 50)
        for src in sorted(sources):
            b_src = [r for r in base_results if r["source"] == src]
            t_src = [r for r in tuned_results if r["source"] == src]
            b_c = sum(1 for r in b_src if r["correct"])
            t_c = sum(1 for r in t_src if r["correct"])
            n_s = len(b_src)
            print(f"{src:<20} {b_c}/{n_s} ({100*b_c/n_s:.1f}%){'':<3} {t_c}/{n_s} ({100*t_c/n_s:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tuned_model", type=str, default="./kd_scibench/final")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N problems (for quick testing)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = "cpu"
    print(f"Device: {device}\n")

    # Load test data
    records = []
    with open(args.test_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Test set: {len(records)} problems\n")

    # Evaluate base model
    print("─── Base model ───")
    base_model, base_tok = load_model(args.base_model, device)
    base_results, base_correct = evaluate_model(base_model, base_tok, records, device, args.max_new_tokens)
    del base_model
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()

    # Evaluate tuned model
    print("\n─── Tuned model ───")
    tuned_model, tuned_tok = load_model(args.tuned_model, device)
    tuned_results, tuned_correct = evaluate_model(tuned_model, tuned_tok, records, device, args.max_new_tokens)
    del tuned_model
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()

    # Print comparison
    print_comparison(base_results, tuned_results)

    # Save detailed results
    length = len(records)
    output = {
        "base_model": args.base_model,
        "tuned_model": args.tuned_model,
        "test_size": length,
        "base_accuracy": base_correct / length,
        "tuned_accuracy": tuned_correct / length,
        "base_reasoning_correct": sum(1 for r in base_results if r["reasoning_correct"]) / length,
        "tuned_reasoning_correct": sum(1 for r in tuned_results if r["reasoning_correct"]) / length,
        "base_avg_bert_f1": sum(r["bert_f1"] for r in base_results) / length,
        "tuned_avg_bert_f1": sum(r["bert_f1"] for r in tuned_results) / length,
        "base_avg_bert_r": sum(r["bert_r"] for r in base_results) / length,
        "tuned_avg_bert_r": sum(r["bert_r"] for r in tuned_results) / length,
        "base_avg_bert_p": sum(r["bert_p"] for r in base_results) / length,
        "tuned_avg_bert_p": sum(r["bert_p"] for r in tuned_results) / length,
        "base_results": base_results,
        "tuned_results": tuned_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
