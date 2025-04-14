from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F
import os


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

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=7860)
    app.run(host="127.0.0.1", port=8080)
