import streamlit as st
import pickle

# Load Ridge model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔥 Fire Weather Index (FWI) Prediction")

# Input fields
Temperature = st.number_input("Temperature", value=20.0)
RH = st.number_input("Relative Humidity", value=50.0)
Ws = st.number_input("Wind Speed", value=10.0)
Rain = st.number_input("Rainfall", value=0.0)
FFMC = st.number_input("FFMC", value=85.0)
DMC = st.number_input("DMC", value=20.0)
ISI = st.number_input("ISI", value=5.0)
Classes = st.text_input("Classes", "not-used")
Region = st.text_input("Region", "not-used")

if st.button("Predict"):
    # Prepare features (ignoring text fields for now)
    features = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]]
    result = model.predict(features)[0]
    st.success(f"🔥 Predicted FWI = {result:.2f}")
