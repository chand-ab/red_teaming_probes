import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "achand45/first-llama"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

user_text = input("Enter text:")
messages = [
    {"role": "user", "content": user_text}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

assistant_tokens = outputs[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(assistant_tokens, skip_special_tokens=True))
