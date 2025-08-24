import streamlit as st
import pickle
import numpy as np
import os

# Load scaler and model
BASE_DIR = os.path.dirname(__file__)
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

scaler = pickle.load(open(scaler_path, 'rb'))

ridge_path = os.path.join(BASE_DIR, 'models', 'ridge.pkl')
ridge_model = pickle.load(open(ridge_path, 'rb'))

st.set_page_config(page_title="FWI Prediction", page_icon="🔥", layout="centered")

st.title("🔥 Fire Weather Index (FWI) Prediction")

# Create form layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Temperature = st.number_input("🌡️ Temperature", value=20.0)
        RH = st.number_input("💧 Relative Humidity", value=50.0)
        Ws = st.number_input("🌬️ Wind Speed", value=10.0)
        Rain = st.number_input("☔ Rainfall", value=0.0)
        FFMC = st.number_input("🔥 FFMC", value=85.0)

    with col2:
        DMC = st.number_input("🌲 DMC", value=20.0)
        ISI = st.number_input("⚡ ISI", value=5.0)
        Classes = st.number_input("📊 Classes (numeric)", value=0.0)
        Region = st.number_input("🌍 Region (numeric)", value=0.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Scale and predict
    features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    new_data_scaled = scaler.transform(features)
    prediction = ridge_model.predict(new_data_scaled)[0]

    st.success(f"🔥 Predicted FWI = {prediction:.2f}")