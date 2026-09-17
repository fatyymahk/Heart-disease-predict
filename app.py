import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your saved files from GitHub
model = joblib.load('Heartdisease_prediction.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('HD_columns.pkl')

st.title("🫀 Heart Disease Prediction App")
st.write("Welcome! Adjust the settings below to check your prediction.")

# Inputs for your model
age = st.number_input("Age", min_value=1, max_value=105, value=50)
sex = st.selectbox("Sex", ['Male', 'Female'])
RestingBP = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])    
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["up", "Flat", "down"])

if st.button("Predict"):
    # Create an input dataframe matching your training data structure
    input_data = pd.DataFrame([[
        age, sex, RestingBP, Cholesterol, FastingBS, 
        max_hr, Oldpeak, chest_pain, resting_ecg, exercise_angina, st_slope
    ]], columns=columns)
    
    try:
        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        if prediction[0] == 1:
            st.error("The model predicts a **high risk** of heart disease.")
        else:
            st.success("The model predicts a **low risk** of heart disease.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")