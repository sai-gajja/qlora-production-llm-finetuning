import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_upload(base_model_name, adapter_path, merge_output_path, hub_repo_id=None, push_to_hub=False):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merge_output_path)
    tokenizer.save_pretrained(merge_output_path)
    if push_to_hub and hub_repo_id:
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        print(f"Pushed to https://huggingface.co/{hub_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--merge_path", default="./merged_model")
    parser.add_argument("--hub_repo_id", type=str)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    merge_and_upload(args.base_model, args.adapter_path, args.merge_path, args.hub_repo_id, args.push)