from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from kvpress import StreamingLLMPress

model_dir = "/leonardo_scratch/large/userexternal/<username>/model/SmolLM-1.7B"
checkpoint = model_dir

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pipe = pipeline(
    "kv-press-text-generation",
    model=model,
    tokenizer=tokenizer,
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

result_file = "/leonardo_scratch/large/userexternal/<username>/dock-exp/result.txt"
with open(result_file, "w") as f:
    f.write(answer)
