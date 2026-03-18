import torch
from transformers import BertTokenizer, BertForSequenceClassification

print("Loading BERT model...")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model for classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

print("Model loaded successfully")

# Save model in project model folder
model.save_pretrained("../model")
tokenizer.save_pretrained("../model")

print("Model saved successfully in model folder")