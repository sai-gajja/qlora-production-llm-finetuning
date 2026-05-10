import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@torch.no_grad()
def benchmark(model, tokenizer, prompt="Explain quantum computing.", num_runs=10, max_new_tokens=128):
    device = model.device
    is_cuda = device.type == "cuda"

    # Warmup
    for _ in range(2):
        _ = model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_new_tokens=10)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated(device)

    latencies = []
    total_tokens = 0
    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if is_cuda:
            torch.cuda.synchronize()
        latencies.append(time.time() - start)
        total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]

    avg_latency = sum(latencies) / num_runs
    throughput = total_tokens / sum(latencies)

    print(f"Avg latency: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) - start_mem
        print(f"Peak VRAM increase: {peak_mem / 1024**2:.2f} MB")
    else:
        print("CPU: memory measurement not performed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--is_adapter", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token

    if args.is_adapter:
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    benchmark(model, tokenizer)