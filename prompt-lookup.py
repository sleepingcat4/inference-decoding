from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
inputs = tokenizer("Alice and Bob went to the park and they were walking like drunk humans like being obscurely moronic way but", return_tensors="pt").to("cuda")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B", torch_dtype="auto", quantization_config=bnb_config)

outputs = model.generate(**inputs, prompt_lookup_num_tokens=100, use_cache=True, max_length=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# check https://colab.research.google.com/drive/17U4lj2YLNH0GdxR9iovBnHdONB4QEh_a?usp=sharing

# On colab it was the fastest trick I could find but on Leonardo it was slower than KVPress. 
