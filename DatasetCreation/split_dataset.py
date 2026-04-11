import json
import random
import os

def split_distillation_data(input_file, train_ratio=0.8):
    # 1. Load and Filter
    valid_records = []
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # Only keep records where the teacher was actually correct
            if record.get('is_correct') is True:
                valid_records.append(record)

    print(f"Total valid records found: {len(valid_records)}")

    # 2. Shuffle and Split
    random.seed(42) # For reproducibility
    random.shuffle(valid_records)
    
    split_index = int(len(valid_records) * train_ratio)
    train_data = valid_records[:split_index]
    test_data = valid_records[split_index:]

    # 3. Write to new JSONL files
    base_name = input_file.replace('.jsonl', '')
    train_file = f"{base_name}_train.jsonl"
    test_file = f"{base_name}_test.jsonl"

    def write_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')

    write_jsonl(train_data, train_file)
    write_jsonl(test_data, test_file)

    print(f"Successfully created:")
    print(f" - {train_file} ({len(train_data)} records)")
    print(f" - {test_file} ({len(test_data)} records)")

# Run the split
split_distillation_data("theoremqa_gpt-oss-120b_openrouter.jsonl")