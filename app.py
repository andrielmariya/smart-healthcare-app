import os
import json
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from groq import Groq

# Load the secret key
load_dotenv()

app = Flask(__name__)

# Initialize the Groq Client
my_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=my_api_key)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    data = request.json
    user_symptoms = data.get("symptoms", "")
    
    prompt = f"""
    You are a medical triage assistant. A user has reported the following symptoms: "{user_symptoms}".
    Analyze the symptoms and provide a response strictly in JSON format with exactly these keys:
    - "possible_condition": (string) The most likely general condition.
    - "confidence_level": (string) "Low", "Medium", or "High".
    - "recommended_actions": (array of strings) 3 short, basic home remedies.
    - "seek_medical_help_if": (array of strings) 2 severe red flags to watch out for.
    - "disclaimer": (string) Must be exactly: "This system provides AI-generated suggestions and is not a substitute for professional medical advice."
    """
    
    try:
        # Call Groq's Llama 3.1 model and force JSON output
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant that outputs strictly in JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        
        # Read the AI's response
        ai_response_text = chat_completion.choices[0].message.content
        ai_data = json.loads(ai_response_text)
        
        return jsonify(ai_data)
        
    except Exception as e:
        print("Error connecting to Groq:", e)
        return jsonify({"error": "Failed to generate AI response"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)