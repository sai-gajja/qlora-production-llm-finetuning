import torch
import hashlib
import json
from transformers import AutoTokenizer

def find_target_modules(model):
    """Architecture‑aware LoRA target module detection."""
    target_modules = set()
    # Known patterns for popular architectures
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Common projection names
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj",
                                       "query_key_value", "dense"]):
                target_modules.add(name.split(".")[-1])
    # Fallback: if nothing found, add all Linear modules (rare)
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name.split(".")[-1])
    return list(target_modules)

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def dataset_version_hash(file_path):
    """Compute SHA256 hash of raw dataset for versioning."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

def safety_filter(text):
    """Basic toxicity / jailbreak filter (placeholder – expand with real classifiers)."""
    banned = ["ignore instructions", "hack", "illegal", "violence"]  # example
    return not any(b in text.lower() for b in banned)

def prepare_data_with_safety(data_list):
    """Filter out unsafe examples."""
    return [ex for ex in data_list if safety_filter(ex["instruction"]) and safety_filter(ex["output"])]