import pandas as pd
import os

def get_test_dataset(original_path, training_path, output_path):
    # Check if files exist before trying to read them
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Could not find original file at: {os.path.abspath(original_path)}")
    if not os.path.exists(training_path):
        raise FileNotFoundError(f"Could not find training file at: {os.path.abspath(training_path)}")

    # Load JSONL files directly into DataFrames
    df_original = pd.read_json(original_path, lines=True)
    df_training = pd.read_json(training_path, lines=True)

    # Filter: Keep rows in original NOT in training
    test_df = df_original[~df_original['problemid'].isin(df_training['problemid'])]

    # Save to a new JSONL file
    test_df.to_json(output_path, orient='records', lines=True)
    
    print(f"Extraction complete. {len(test_df)} entries saved to {output_path}.")
    return test_df

# Since your script is IN the DatasetCreation folder, just use the filenames:
get_test_dataset(
    'theoremqa_llama3.3_openrouter.jsonl', 
    'theoremqa_llama3.3_openrouter_train.jsonl', 
    'theoremqa_llama3.3_openrouter_FULL_TEST.jsonl'
)
