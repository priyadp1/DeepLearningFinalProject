import argparse
import json
import os
import re
import statistics
import torch
from bert_score import score as bert_score_fn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import load_jsonl
from model import get_device, get_dtype


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n,.]+)", re.IGNORECASE
)
_EQUALS_RE = re.compile(r"=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", re.MULTILINE)
_LAST_NUMBER_RE = re.compile(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_answer(text: str) -> str:
    """Best-effort extraction of the final numerical answer from generated text."""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")

    m = _EQUALS_RE.search(text)
    if m:
        return m.group(1).strip()

    nums = _LAST_NUMBER_RE.findall(text)
    if nums:
        return nums[-1]

    return text.strip().split("\n")[-1].strip()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _try_float(s: str):
    s = s.strip().lstrip("+").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def check_correct(predicted: str, actual: str, tol: float = 0.05):
    """Return (is_correct, relative_difference).

    Tries numeric comparison first (with *tol* relative tolerance).  Falls back
    to exact string equality.
    """
    pf = _try_float(predicted)
    af = _try_float(actual)
    if pf is not None and af is not None:
        if af == 0:
            rd = abs(pf)
            return rd < 1e-6, rd
        rd = abs(pf - af) / abs(af)
        return rd <= tol, rd
    return predicted.strip() == actual.strip(), 0.0 if predicted.strip() == actual.strip() else 1.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_response(model, tokenizer, problem_text, device, max_new_tokens=512):
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# BERTScore wrapper
# ---------------------------------------------------------------------------

def compute_bert_scores(candidates, references, device):
    """Return dict with per-item P, R, F1 lists and their means/stds."""
    str_device = "mps" if device.type == "mps" else str(device)

    # BERTScore may not natively support MPS; fall back to CPU in that case.
    try:
        P, R, F1 = bert_score_fn(
            candidates, references,
            lang="en",
            device=str_device,
            verbose=False,
        )
    except Exception:
        P, R, F1 = bert_score_fn(
            candidates, references,
            lang="en",
            device="cpu",
            verbose=False,
        )

    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()

    def _stats(vals):
        return {
            "mean": round(statistics.mean(vals), 4) if vals else 0.0,
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4) if vals else 0.0,
            "max": round(max(vals), 4) if vals else 0.0,
        }

    return {
        "precision": _stats(p_list),
        "recall": _stats(r_list),
        "f1": _stats(f1_list),
        "per_item": {
            "precision": [round(v, 4) for v in p_list],
            "recall": [round(v, 4) for v in r_list],
            "f1": [round(v, 4) for v in f1_list],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-path", required=True,
                    help="Path to the saved LoRA adapter directory")
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--teacher-dir", required=True,
                    choices=["gpt_teacher_data", "llama_teacher_data"],
                    help="Which teacher's test files to load and compare against")
    p.add_argument("--output-dir", default="./eval_results")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--tolerance", type=float, default=0.05,
                    help="Relative tolerance for numeric answer comparison")
    p.add_argument("--eval-student-model", action="store_true",
                    help="Also evaluate the student model (no LoRA) as a no-KD baseline")
    return p.parse_args()


def evaluate_dataset(model, tokenizer, test_data, device, max_new_tokens, tolerance):
    """Run generation + answer extraction on a list of test records.

    Returns (response_records, student_cots, teacher_cots).
    """
    records = []
    student_cots = []
    teacher_cots = []

    for i, row in enumerate(test_data):
        problem = row["problem_text"]
        actual = str(row["actual_answer"]).strip()
        teacher_cot = row.get("teacher_cot", "")

        print(f"    [{i+1}/{len(test_data)}] {row.get('problemid', '')[:40]}", end=" … ", flush=True)
        student_cot = generate_response(model, tokenizer, problem, device, max_new_tokens)

        predicted = extract_answer(student_cot)
        is_correct, rel_diff = check_correct(predicted, actual, tol=tolerance)
        print(f"pred={predicted!r}  actual={actual!r}  {'✓' if is_correct else '✗'}")

        rec = {
            "problemid": row.get("problemid", ""),
            "source": row.get("source", ""),
            "unit": row.get("unit", ""),
            "problem_text": problem,
            "student_cot": student_cot,
            "teacher_cot": teacher_cot,
            "predicted_answer": predicted,
            "actual_answer": actual,
            "relative_difference": round(rel_diff, 6),
            "output_tokens": len(tokenizer.encode(student_cot)),
            "is_correct": is_correct,
        }
        records.append(rec)
        student_cots.append(student_cot)
        teacher_cots.append(teacher_cot)

    return records, student_cots, teacher_cots

def evaluate_student_model(student_model_name, datasets, dtype, device, max_new_tokens, tolerance):
    print(f"  Loading student model: {student_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=dtype).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    all_records, all_s_cots, all_t_cots = [], [], []
    dataset_metrics = {}

    for ds_name, test_data in datasets.items():
        print(f"\n  [{ds_name}] ({len(test_data)} problems)")
        records, s_cots, t_cots = evaluate_dataset(
            model, tokenizer, test_data, device, max_new_tokens, tolerance
        )
        all_records.extend(records)
        all_s_cots.extend(s_cots)
        all_t_cots.extend(t_cots)

        correct = sum(1 for r in records if r["is_correct"])
        total = len(records)
        accuracy = correct / total if total else 0.0
        print(f"  {ds_name} accuracy: {correct}/{total} = {accuracy:.3f}")
        print(f"  Computing BERTScores for {ds_name} …")
        bs = compute_bert_scores(s_cots, t_cots, device)

        dataset_metrics[ds_name] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "bert_score": {k: v for k, v in bs.items() if k != "per_item"},
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return all_records, all_s_cots, all_t_cots, dataset_metrics


def main():
    args = parse_args()
    device = get_device()
    dtype = get_dtype(device)
    print(f"Device: {device}  dtype: {dtype}")

    # ---- Load model -------------------------------------------------------
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
    ).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # ---- Load test data ---------------------------------------------------
    datasets = {}
    for name in ("scibench", "theoremqa"):
        path = os.path.join(args.teacher_dir, f"{name}_test_clean.jsonl")
        if os.path.exists(path):
            datasets[name] = load_jsonl(path)
            print(f"  Loaded {name} test: {len(datasets[name])} problems")
        else:
            print(f"  Warning: {path} not found, skipping {name}")

    if not datasets:
        print("No test data found. Exiting.")
        return
    

    # ---- Generate + evaluate per dataset ----------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    all_records = []
    all_student_cots = []
    all_teacher_cots = []
    dataset_metrics = {}

    for ds_name, test_data in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {ds_name} ({len(test_data)} problems)")
        print(f"{'='*60}")

        records, s_cots, t_cots = evaluate_dataset(
            model, tokenizer, test_data, device,
            args.max_new_tokens, args.tolerance,
        )
        all_records.extend(records)
        all_student_cots.extend(s_cots)
        all_teacher_cots.extend(t_cots)

        correct = sum(1 for r in records if r["is_correct"])
        total = len(records)
        accuracy = correct / total if total else 0.0

        print(f"\n  {ds_name} accuracy: {correct}/{total} = {accuracy:.3f}")
        print(f"  Computing BERTScores for {ds_name} …")
        bs = compute_bert_scores(s_cots, t_cots, device)

        dataset_metrics[ds_name] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "bert_score": {k: v for k, v in bs.items() if k != "per_item"},
            "bert_score_per_item": bs["per_item"],
        }

    # ---- Overall metrics --------------------------------------------------
    total_all = sum(m["total"] for m in dataset_metrics.values())
    correct_all = sum(m["correct"] for m in dataset_metrics.values())
    overall_acc = correct_all / total_all if total_all else 0.0

    print(f"\n  Computing overall BERTScores …")
    overall_bs = compute_bert_scores(all_student_cots, all_teacher_cots, device)

    metrics = {
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "teacher_dir": args.teacher_dir,
        "tolerance": args.tolerance,
        "overall": {
            "total": total_all,
            "correct": correct_all,
            "accuracy": round(overall_acc, 4),
            "bert_score": {k: v for k, v in overall_bs.items() if k != "per_item"},
        },
        "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "bert_score_per_item"}
                      for k, v in dataset_metrics.items()},
        "datasets_detailed": dataset_metrics,
    }

    # ---- Optional: student model (no-KD) baseline ----------------------------
    if args.eval_student_model:
        # Free the LoRA model before loading a second copy to avoid OOM
        del model, base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        print(f"\n{'='*60}")
        print(f"  Evaluating student model (no-KD baseline): {args.base_model}")
        print(f"{'='*60}")

        all_student_records, all_student_s_cots, all_student_t_cots, student_dataset_metrics = \
            evaluate_student_model(args.base_model, datasets, dtype, device, args.max_new_tokens, args.tolerance)

        student_total_all = sum(m["total"] for m in student_dataset_metrics.values())
        student_correct_all = sum(m["correct"] for m in student_dataset_metrics.values())
        student_acc = student_correct_all / student_total_all if student_total_all else 0.0

        print(f"\n  Computing overall BERTScores for student model …")
        student_overall_bs = compute_bert_scores(all_student_s_cots, all_student_t_cots, device)

        metrics["student_model_baseline"] = {
            "overall": {
                "total": student_total_all,
                "correct": student_correct_all,
                "accuracy": round(student_acc, 4),
                "bert_score": {k: v for k, v in student_overall_bs.items() if k != "per_item"},
            },
            "datasets": student_dataset_metrics,
        }

        student_responses_path = os.path.join(args.output_dir, "student_model_responses.jsonl")
        with open(student_responses_path, "w") as f:
            for rec in all_student_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Student model responses written to: {student_responses_path}")

    # ---- Write outputs ----------------------------------------------------
    responses_path = os.path.join(args.output_dir, "responses.jsonl")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    with open(responses_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results written to:")
    print(f"    {responses_path}")
    print(f"    {metrics_path}")
    print(f"\n  Overall accuracy: {correct_all}/{total_all} = {overall_acc:.3f}")
    print(f"  Overall BERTScore F1: {overall_bs['f1']['mean']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
