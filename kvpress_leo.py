from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from kvpress import StreamingLLMPress

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

pipe.model.to('cuda')

context = "Alice and Bob were walking in the park when suddenly they encountered a mysterious figure."
question = "What happened next?"

press = StreamingLLMPress(compression_ratio=0.7)

answer = pipe(context, question=question, press=press)["answer"]

result_file = "/leonardo_scratch/large/userexternal/tahmed00/dock-exp/result.txt"
with open(result_file, "w") as f:
    f.write(answer)
