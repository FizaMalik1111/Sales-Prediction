# app.py

import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 Sales Prediction App")

# User inputs
tv = st.number_input("TV Advertising Budget", min_value=0.0, step=0.1)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, step=0.1)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Sales"):
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Sales: {prediction:.2f} units")
