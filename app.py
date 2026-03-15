import os
from flask import Flask, jsonify, request, render_template
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Correct lowercase username from your profile
client = InferenceClient(
    model="andrielmariya/medical-llama3-model", 
    token=os.getenv("HF_TOKEN")
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    data = request.json
    symptoms = data.get("symptoms", "")
    
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{symptoms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    try:
        # Changed 'stop_sequences' to 'stop' to fix the 500 error
        response = client.text_generation(prompt, max_new_tokens=256, stop=["<|eot_id|>"])
        
        return jsonify({
            "possible_condition": "Analysis based on Custom Model",
            "confidence_level": "High",
            "recommended_actions": [response],
            "seek_medical_help_if": ["Symptoms worsen", "Emergency signs appear"]
        })
    except Exception as e:
        print(f"Error: {e}") # This will show up in Render logs if it fails
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # This specific port logic is what Render's scanner is looking for
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)