from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F
import os
from huggingface_hub import InferenceClient
import json
import re
import pandas as pd

app = Flask(__name__)

# Load model and tokenizer
# model_path = ".\saved_model\merged_model"\

# Construct the relative path
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "saved_model", "merged_model")



tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["Cancer", "Non-Cancer"]

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Disease Classification API!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No input text provided."}), 400

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get logits and apply softmax
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

    # Format output
    confidence_scores = {label: float(score) for label, score in zip(labels, probs)}
    predicted_label = labels[torch.argmax(torch.tensor(probs))]

    result = {
        "predicted_labels": [predicted_label],
        "confidence_scores": confidence_scores
    }

    return jsonify(result)


@app.route("/ExtractDisease", methods=["POST"])
def extract_disease():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No input text provided."}), 400
    
    os.environ["HF_TOKEN"]="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"
    
    client = InferenceClient(
        api_key="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",
        provider="nebius",
    )

    completion = client.chat.completions.create(
        model="microsoft/phi-4",
        messages=[
            {"role": "system",
             "content": '''You would be provided with research paper having Id and Abstract. You would need to identify the disease mentioned in abstract. The response should be returned in below format as an example:"
                {
                "abstract_id": "12345",
                "extracted_diseases": ["Lung Cancer", "Breast Cancer"]
                }'''},
            {"role": "user",
             "content": text}
        ],
    )

    response_content = completion.choices[0].message["content"]
    cleaned_string = re.sub(r"\`\`\`json\s*", "", response_content.strip(), flags=re.IGNORECASE)
    cleaned_string = re.sub(r"\`\`\`", "", cleaned_string.strip(), flags=re.IGNORECASE)

    try:
        parsed_json = json.loads(cleaned_string)
        return jsonify(parsed_json)
    except json.JSONDecodeError as e:
        return jsonify({"error": "Invalid JSON format.", "details": str(e)}), 400


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=7860)
    app.run(host="127.0.0.1", port=8080)
