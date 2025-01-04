import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/leonardo_scratch/large/userexternal/<username>/model/SmolLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

result_file = "/leonardo_scratch/large/userexternal/<username>/dock-exp/result.txt"
time_file = "/leonardo_scratch/large/userexternal/<username>/dock-exp/time-bench.txt"
error_file = "/leonardo_scratch/large/userexternal/<username>/dock-exp/error.txt"

for file in [result_file, time_file, error_file]:
    if os.path.exists(file):
        os.remove(file)

def make_cached_inference(prompt, cache):
    if prompt in cache:
        return cache[prompt], 0
    start_time = time.time()
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs["input_ids"], max_length=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        cache[prompt] = generated_text
        return generated_text, end_time - start_time
    except Exception as e:
        with open(error_file, "w") as ef:
            ef.write(str(e))
        return None, 0

cache = {}
prompt = "Alice and Bob"
result, latency = make_cached_inference(prompt, cache)

if result:
    with open(result_file, "w") as rf:
        rf.write(f"Generated Text: {result}")
    with open(time_file, "w") as tf:
        tf.write(f"Latency: {latency:.2f} seconds")
