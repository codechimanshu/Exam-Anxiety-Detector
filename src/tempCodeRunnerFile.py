import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "../model/anxiety_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

label_map = {
    0: "Low Anxiety",
    1: "Moderate Anxiety",
    2: "High Anxiety"
}

def predict_anxiety(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return label_map[prediction]


text = "I feel very stressed about my exams."

print(predict_anxiety(text))

test_sentences = [

    "I feel calm and prepared for my exam tomorrow.",

    "I am a little nervous about the exam but I think I can manage.",

    "I am extremely worried and stressed about failing my exam.",

    "My heart is racing and I feel panic thinking about the exam.",

    "I studied well and feel confident."

]