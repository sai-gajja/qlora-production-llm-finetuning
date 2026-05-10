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
import psutil

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
    parser.add_argument("--use_bf16", action="store_true", default=False,
                        help="Use bfloat16 for model weights (no automatic mixed precision)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--packing_train", action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.use_wandb:
        wandb.init(project="llm-finetune-prod", config=vars(args))

    # ---------- BFloat16 capability check ----------
    if args.use_bf16:
        if not torch.cuda.is_available():
            print("⚠️  --use_bf16 requires CUDA. Falling back to float16.")
            args.use_bf16 = False
        else:
            major, minor = torch.cuda.get_device_capability()
            if major < 8:
                print(f"⚠️  GPU compute capability {major}.{minor} does not support BFloat16. Falling back to float16.")
                args.use_bf16 = False

    # ---------- Quantization config ----------
    # Compute dtype: only used for 4-bit compute, not for AMP
    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # ---------- Load model ----------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=compute_dtype if not args.use_4bit else None,  # for non-quantized case
    )
    model.config.use_cache = False
    # Enable gradient checkpointing without reentrancy warning
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        # Older PyTorch versions
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------- LoRA ----------
    target_modules = find_target_modules(model)
    print(f"Target modules: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    if args.use_wandb:
        wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # ---------- Dataset (keep only "text" column) ----------
    dataset = load_from_disk(args.data_path)
    for split in ["train", "validation"]:
        if split not in dataset:
            continue
        cols = dataset[split].column_names
        if "text" not in cols:
            raise ValueError(f"Dataset {split} has no 'text' column. Found: {cols}")
        to_remove = [c for c in cols if c != "text"]
        if to_remove:
            print(f"Removing columns from {split}: {to_remove}")
            dataset[split] = dataset[split].remove_columns(to_remove)

    # ---------- Training arguments ----------
    total_train_samples = len(dataset["train"])
    effective_batch_size = args.batch_size * args.grad_accum
    steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.03 * total_steps)

    cpu_count = psutil.cpu_count(logical=True)
    num_workers = min(4, cpu_count // 2 if cpu_count else 2)

    # CRITICAL: Disable automatic mixed precision to avoid gradient scaler with BFloat16
    # We set both fp16 and bf16 to False. The model already uses bf16 (if requested) natively.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        fp16=False,                     # No FP16 AMP
        bf16=False,                     # No BFloat16 AMP (model is already bf16)
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "tensorboard",
        seed=args.seed,
        dataloader_num_workers=num_workers,
    )

    # ---------- Create SFTTrainer (modern single-column text mode) ----------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # ---------- Train ----------
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()