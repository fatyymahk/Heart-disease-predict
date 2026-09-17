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

# User inputs matching your original data format
Age = st.number_input("Age", min_value=1, max_value=105, value=50)
Sex = st.selectbox("Sex", ['M', 'F'])
RestingBP = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
MaxHR = st.slider("Max Heart Rate", 60, 220, 150)
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
ChestPainType = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
RestingECG = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])    
ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
ST_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    try:
        # 1. Create a dataframe with the raw inputs
        input_dict = {
            'Age': [Age],
            'Sex': [Sex],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBS],
            'MaxHR': [MaxHR],
            'Oldpeak': [Oldpeak],
            'ChestPainType': [ChestPainType],
            'RestingECG': [RestingECG],
            'ExerciseAngina': [ExerciseAngina],
            'ST_slope': [ST_slope]
        }
        df_input = pd.DataFrame(input_dict)
        
        # 2. Apply get_dummies just like you did during training
        df_encoded = pd.get_dummies(df_input)
        
        # 3. Align columns to match the exact columns expected by your model/scaler (HD_columns.pkl)
        # This fills any missing dummy columns with 0 and drops any extras
        df_final = df_encoded.reindex(columns=columns, fill_value=0)
        
        # 4. Scale and predict
        scaled_data = scaler.transform(df_final)
        prediction = model.predict(scaled_data)
        
        if prediction[0] == 1:
            st.error("⚠️ The model predicts a **high risk** of heart disease.")
        else:
            st.success("✅ The model predicts a **low risk** of heart disease.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")