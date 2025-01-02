# importing the libaries and modules
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from kvpress import StreamingLLMPress

# loading the module in HF pipeline
# for now KVPress can only be used through high-level abstraction pipeline. 
# I didn't test it on model.generate method yet
# feel free to try

checkpoint = "HuggingFaceTB/SmolLM-1.7B"

pipe = pipeline(
    "kv-press-text-generation",
    model=checkpoint,
    torch_dtype="auto",
    model_kwargs={
        "max_length": 100,
        "use_cache": True,
    }
)

# Move the model to GPU
pipe.model.to('cuda')

# found 0.7 does better but better understanding will be a benchmark
# benchmark coming soon

context = "Alice and Bob were walking in the park when suddenly they encountered a mysterious figure."
question = "What happened next?"

press = StreamingLLMPress(compression_ratio=0.7)

# generating the results

answer = pipe(context, question=question, press=press)["answer"]

print(answer)
