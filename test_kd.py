import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from kd_trainer import KDTrainer
from model import load_model

model_name = "gpt2"

print(f"Loading student model: {model_name}...")
tokenizer, student_model = load_model(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Student model loaded.")

print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
print("Teacher model loaded.")


def create_dummy_dataset(tokenizer, num_samples=100, seq_length=20):
    texts = ["What is the capital of France?"] * num_samples

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_length,
        return_tensors="pt"
    )

    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

print("Creating dummy dataset...")
raw_dataset = create_dummy_dataset(tokenizer)
print(f"Dataset created with {len(raw_dataset['input_ids'])} samples.")


class Wrapper(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dummy_dataset = Wrapper(raw_dataset)
print(f"Dataset wrapped: {len(dummy_dataset)} samples.")

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    max_steps=5,
    logging_steps=1,
    fp16=False
)

print("Initializing KDTrainer...")
trainer = KDTrainer(
    model=student_model,
    args=training_args,
    train_dataset=dummy_dataset,
    teacher_model=teacher_model,
    alpha=0.5,
    beta=0.3,
    use_sequence_distill=False,
    temperature=2.0
)
print("Trainer ready.")

print("Starting training...")
trainer.train()
print("Training completed!")