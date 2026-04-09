import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from kd_trainer import KDTrainer
from model import load_model
import json

student_model_name = "mistralai/Mistral-7B-v0.3"
print(f"Loading student model: {student_model_name}...")
tokenizer, student_model = load_model(student_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
teacher_model_name = "openai/gpt-oss-120b"
print(f"Loading teacher model: {teacher_model_name}...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)
print("Models loaded.")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_scibench_data(file_path):
    return load_jsonl(file_path)

def tokenize_scibench_dataset(tokenizer, data, seq_length=512):
    questions = [d["problem_text"] for d in data]
    answers = [d["teacher_cot"] for d in data]
    full_texts = [q + " " + a for q, a in zip(questions, answers)]
    question_encodings = tokenizer(
        list(questions),
        truncation=True,
        padding="max_length",
        max_length=seq_length,
        return_tensors="pt"
    )
    encodings = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=seq_length,
        return_tensors="pt"
    )
    labels = encodings["input_ids"].clone()
    # Mask padding tokens
    labels[encodings["attention_mask"] == 0] = -100
    # Mask question tokens so loss is only computed on the answer
    q_lengths = question_encodings["attention_mask"].sum(dim=1)
    for i, q_len in enumerate(q_lengths):
        labels[i, :q_len] = -100
    encodings["labels"] = labels
    return encodings

raw_dataset = tokenize_scibench_dataset(tokenizer, read_scibench_data("DatasetCreation/scibench_results_gpt-oss-120b_train.jsonl"))

class Wrapper(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

wrapped_dataset = Wrapper(raw_dataset)
print(f"Dataset wrapped: {len(wrapped_dataset)} samples.")

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results_gpt_sequence",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    warmup_steps=10,
    logging_steps=1,
    fp16=False
)

print("Initializing KDTrainer...")
trainer = KDTrainer(
    model=student_model,
    args=training_args,
    train_dataset=wrapped_dataset,
    teacher_model=teacher_model,
    alpha=0.5,
    beta=0.0,
    use_sequence_distill=True,
    seq_kd=0.1,
    temperature=2.0
)

print("Starting training...")
trainer.train()
trainer.save_model("./gpt_oss_scibench_sequence_kd_model")


