import json
import os
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from utils import dataset_version_hash, prepare_data_with_safety

def clean_and_split_data(input_jsonl, tokenizer_name, system_prompt=None, val_ratio=0.1, seed=42):
    """
    Clean, filter, apply chat template, store prompt & response separately.
    Returns dataset with columns: prompt (user side), response (assistant side), text (full).
    """
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load raw
    raw_data = []
    with open(input_jsonl) as f:
        for line in f:
            ex = json.loads(line)
            raw_data.append(ex)

    # Safety filter
    raw_data = prepare_data_with_safety(raw_data)

    # Deduplicate (instruction + output)
    seen = set()
    unique = []
    for ex in raw_data:
        key = (ex["instruction"], ex["output"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    # Length filtering
    unique = [ex for ex in unique if 5 <= len(ex["instruction"]) <= 1024 and len(ex["output"]) >= 10]

    # Train/val split
    random.shuffle(unique)
    split_idx = int(len(unique) * (1 - val_ratio))
    train_data = unique[:split_idx]
    val_data = unique[split_idx:]

    def build_prompt_and_response(example):
        # Build user prompt
        if example["input"]:
            user_prompt = f"{example['instruction']}\n\nInput: {example['input']}"
        else:
            user_prompt = example["instruction"]
        # Apply chat template for the full text
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": example["output"]})
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"prompt": user_prompt, "response": example["output"], "text": full_text}

    train_dataset = Dataset.from_list(train_data).map(build_prompt_and_response)
    val_dataset = Dataset.from_list(val_data).map(build_prompt_and_response)

    version = dataset_version_hash(input_jsonl)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # Store metadata separately
    metadata = {
        "version": version,
        "seed": seed,
        "system_prompt": system_prompt
    }

    return dataset_dict, tokenizer, metadata
if __name__ == "__main__":
    ds, tok, metadata = clean_and_split_data(
        "data/raw_data.jsonl",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    ds.save_to_disk("data/processed")

    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
    print(f"Version hash: {metadata['version']}")