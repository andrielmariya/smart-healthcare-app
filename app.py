import os  # <--- THIS IS THE MISSING PIECE!
from flask import Flask, jsonify, request, render_template
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv() # This helps load variables if you're testing locally

app = Flask(__name__)

# Now 'os' will be recognized here:
client = InferenceClient(
    model="andrielmariya/medical-llama3-model", 
    token=os.getenv("HF_TOKEN")
)
from huggingface_hub import InferenceClient

# Use your actual HF token from your settings
client = InferenceClient(model="Andriel/medical-llama3-model", token=os.getenv("HF_TOKEN"))

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    data = request.json
    symptoms = data.get("symptoms", "")
    
    # This must match the prompt format you used during training!
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{symptoms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    try:
        response = client.text_generation(prompt, max_new_tokens=256, stop_sequences=["<|eot_id|>"])
        
        # If your model was trained to output JSON, you might need to parse it
        # For now, let's send the raw medical advice
        return jsonify({
            "possible_condition": "Analysis based on Custom Model",
            "confidence_level": "Calculated by Fine-Tuning",
            "recommended_actions": [response],
            "seek_medical_help_if": ["Symptoms worsen", "Emergency signs appear"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500