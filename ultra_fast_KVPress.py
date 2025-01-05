import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from kvpress import StreamingLLMPress
import time

model_path = "/leonardo_scratch/large/userexternal/tahmed00/model/Llama-3.2-3B"
output_folder = "/leonardo_scratch/large/userexternal/tahmed00/dock-exp"
result_file = os.path.join(output_folder, "result.txt")
error_file = os.path.join(output_folder, "error.txt")
token_decode_file = os.path.join(output_folder, "token-decode.txt")
prompt_file = "/leonardo_scratch/large/userexternal/tahmed00/scripts/prompt.txt"

os.makedirs(output_folder, exist_ok=True)

for file_path in [result_file, error_file, token_decode_file]:
    if os.path.exists(file_path):
        os.remove(file_path)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda"
    )

    press = StreamingLLMPress(compression_ratio=0.7)

    with open(prompt_file, "r") as f:
        prompt = f.read().strip()

    prompts = [prompt] * 20

    batch_size = 20
    all_results = []
    total_tokens_generated = 0
    total_time_taken = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=11000).to("cuda")

        start_time = time.time()
        with press(model):
            generated_ids = model.generate(**model_inputs)
        end_time = time.time()

        result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_results.extend(result)

        tokens_generated = sum(len(ids) for ids in generated_ids)
        total_tokens_generated += tokens_generated
        total_time_taken += (end_time - start_time)

    throughput = total_tokens_generated / total_time_taken
    latency = total_time_taken / len(prompts)

    with open(result_file, "w") as f:
        for res in all_results:
            f.write(res + "\n")

    with open(token_decode_file, "w") as f:
        f.write(f"Throughput (tokens/sec): {throughput:.4f}\n")
        f.write(f"Latency (seconds): {latency:.4f}\n")

except Exception as e:
    with open(error_file, "w") as f:
        f.write(str(e))
