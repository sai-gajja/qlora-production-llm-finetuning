# src/evaluate.py
import torch
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from evaluate import load as load_metric
import numpy as np

def compute_metrics(predictions, references):
    rouge = load_metric("rouge")
    bleu = load_metric("bleu")
    bertscore = load_metric("bertscore")
    rouge_res = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")
    return {
        "rougeL": rouge_res["rougeL"],
        "bleu": bleu_res["bleu"],
        "bertscore_f1": np.mean(bert_res["f1"])
    }

def generate_responses(model, tokenizer, prompts, max_new_tokens=128):
    model.eval()
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_path", default="outputs/finetuned")
    parser.add_argument("--val_data", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    # Load validation dataset (FIXED: avoid referencing dataset before assignment)
    val_dataset = load_from_disk(args.val_data)["validation"]
    val_dataset = val_dataset.select(range(min(args.max_samples, len(val_dataset))))

    # Extract prompts and responses using stored columns (no parsing)
    prompts = [ex["prompt"] for ex in val_dataset]
    references = [ex["response"] for ex in val_dataset]

    # Build full user prompt with chat template (to be consistent with training)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    # Recreate the same user message format used during training
    # Since we stored only the user content in "prompt", we need to wrap it in chat template
    full_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        full_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    # Fine‑tuned model
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    # Generate
    base_preds = generate_responses(base_model, tokenizer, full_prompts)
    ft_preds = generate_responses(ft_model, tokenizer, full_prompts)

    base_metrics = compute_metrics(base_preds, references)
    ft_metrics = compute_metrics(ft_preds, references)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Base  - ROUGE-L: {base_metrics['rougeL']:.3f}, BLEU: {base_metrics['bleu']:.3f}, BERTScore F1: {base_metrics['bertscore_f1']:.3f}")
    print(f"FT    - ROUGE-L: {ft_metrics['rougeL']:.3f}, BLEU: {ft_metrics['bleu']:.3f}, BERTScore F1: {ft_metrics['bertscore_f1']:.3f}")
    print(f"Delta - ROUGE-L: +{(ft_metrics['rougeL']-base_metrics['rougeL'])*100:.1f}%, BERTScore: +{(ft_metrics['bertscore_f1']-base_metrics['bertscore_f1'])*100:.1f}%")
    print("========================================\n")

if __name__ == "__main__":
    main()