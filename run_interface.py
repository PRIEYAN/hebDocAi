from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model & tokenizer once (so it doesn't reload on every request)
model_path = "./herbal-medllama2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval().to("cpu")  # change to "cuda" if you have GPU support

def generate_response(instruction: str) -> str:
    prompt = f"<s>[INST] {instruction} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Flask route
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "instruction" not in data:
        return jsonify({"error": "Request body must contain 'instruction'"}), 400

    instruction = data["instruction"]
    result = generate_response(instruction)

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
