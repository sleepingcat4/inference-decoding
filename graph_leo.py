from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

model_path = "/leonardo_scratch/large/userexternal/<username>/model/SmolLM-1.7B"
output_folder = "/leonardo_scratch/large/userexternal/<username>/dock-exp"
result_file = os.path.join(output_folder, "result.txt")
error_file = os.path.join(output_folder, "error.txt")
time_file = os.path.join(output_folder, "time-bench.txt")

os.makedirs(output_folder, exist_ok=True)

for file_path in [result_file, error_file, time_file]:
    if os.path.exists(file_path):
        os.remove(file_path)

try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda"
    )

    model.generation_config.cache_implementation = "static"
    model.generation_config.max_batch_size = 8
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    input_text = "Alice and Bob "
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    start_time = time.time()
    outputs = model.generate(**input_ids, max_length=100, use_cache=True)
    end_time = time.time()

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    with open(result_file, "w") as f:
        f.write("\n".join(result))

    time_taken = end_time - start_time
    with open(time_file, "w") as f:
        f.write(f"Time taken for generation: {time_taken:.4f} seconds\n")

except Exception as e:
    with open(error_file, "w") as f:
        f.write(str(e))
