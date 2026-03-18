from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

model_path = "../model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

label_map = {
    0: "Low Anxiety",
    1: "Moderate Anxiety",
    2: "High Anxiety"
}

class TextInput(BaseModel):
    text: str

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


@app.get("/")
def home():
    return {"message": "Exam Anxiety Detector API running"}


@app.post("/predict")
def predict(data: TextInput):

    result = predict_anxiety(data.text)

    return {
        "input_text": data.text,
        "predicted_anxiety": result
    }