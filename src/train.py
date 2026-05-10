import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, set_seed, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk
from utils import find_target_modules, count_parameters
import wandb
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/finetuned")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--use_bf16", action="store_true", default=False)  # newer GPUs
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--packing_train", action="store_true", default=True)
    parser.add_argument("--packing_eval", action="store_true", default=False)  # separate
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.use_wandb:
        wandb.init(project="llm-finetune-prod", config=vars(args))

    # Quantization config
    bnb_config = None
    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA
    target_modules = find_target_modules(model)
    print(f"Target modules: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    if args.use_wandb:
        wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # Dataset
    dataset = load_from_disk(args.data_path)

    # Training arguments with scheduler, clipping, early stopping
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        fp16=not args.use_bf16,
        bf16=args.use_bf16,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "tensorboard",
        seed=args.seed,
        dataloader_num_workers=4,
    )

    # Custom data collator for token masking (mask non‑assistant tokens)
    # We need to use DataCollatorForCompletionOnlyLM – but SFTTrainer can handle it via `data_collator`
    # However, SFTTrainer already does instruction masking if we use `response_template`.
    # We'll use the `response_template` parameter.
    response_template = tokenizer.encode("\n### Response:\n", add_special_tokens=False)[2:]  # heuristic; better to use chat template
    # For simplicity, we rely on SFTTrainer's built‑in masking when we pass `formatting_func` that returns only assistant part? Actually,
    # the proper way: use `DataCollatorForCompletionOnlyLM` from trl.
    from trl import DataCollatorForCompletionOnlyLM
    # Example: we assume assistant token is "<|assistant|>" in the chat template.
    # We'll just use the generic approach: use the response_template as the separator.
    # But because our dataset already has full text, we can use the same collator.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        formatting_func=lambda x: x["text"],
        packing=args.packing_train,
        data_collator=collator,   # applies token masking
    )

    # Early stopping
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Start training with resume support
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()