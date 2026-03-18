import streamlit as st
import requests

st.title("Exam Anxiety Detector")

text = st.text_input("Enter your text")

if st.button("Predict"):

    url = "http://127.0.0.1:8000/predict"

    response = requests.post(url, json={"text": text})

    result = response.json()

    st.write("Input Text:", result["input_text"])
    st.write("Prediction:", result["predicted_anxiety"])