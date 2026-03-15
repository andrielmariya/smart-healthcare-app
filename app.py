import os
from flask import Flask, jsonify, request, render_template
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Initialize the client with your custom model path
# Replace 'Andriel/medical-llama3-model' with your actual path
client = InferenceClient(
    model="Andriel/medical-llama3-model",
    token=os.getenv("HF_TOKEN")
)

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    data = request.json
    symptoms = data.get("symptoms", "")
    
    # Format the prompt exactly like we did in training
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{symptoms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    try:
        response = client.text_generation(prompt, max_new_tokens=256)
        # Assuming the model returns the JSON structure you trained it for
        return jsonify({"analysis": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500