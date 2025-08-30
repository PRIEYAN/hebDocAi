from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model and tokenizer
model_path = "./herbal-medllama2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval().to("cpu")


# Define your prompt
instruction = input("Enter your question or instruction: ")
prompt = f"<s>[INST] {instruction} [/INST]"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
)

# Decode and print result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model Response ===\n")
print(result)
