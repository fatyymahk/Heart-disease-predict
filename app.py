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
sex = st.selectbox("Sex", ['M', 'F'])
RestingBP = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])    
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    try:
        # Create a dataframe initialized with 0 for all expected columns
        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0  
        
        # Assign numerical/basic inputs if they exist in columns
        if 'age' in input_data.columns: input_data['age'] = age
        if 'RestingBP' in input_data.columns: input_data['RestingBP'] = RestingBP
        if 'Cholesterol' in input_data.columns: input_data['Cholesterol'] = Cholesterol
        if 'FastingBS' in input_data.columns: input_data['FastingBS'] = FastingBS
        if 'MaxHR' in input_data.columns: input_data['MaxHR'] = max_hr
        if 'Oldpeak' in input_data.columns: input_data['Oldpeak'] = Oldpeak
        
        # Assign Sex
        if 'Sex_M' in input_data.columns:
            input_data['Sex_M'] = 1 if sex == 'M' else 0
            
        # Assign Chest Pain Dummies
        if f'ChestPainType_{chest_pain}' in input_data.columns:
            input_data[f'ChestPainType_{chest_pain}'] = 1
            
        # Assign Resting ECG Dummies
        if f'RestingECG_{resting_ecg}' in input_data.columns:
            input_data[f'RestingECG_{resting_ecg}'] = 1
            
        # Assign Exercise Angina
        if f'ExerciseAngina_{exercise_angina}' in input_data.columns:
            input_data[f'ExerciseAngina_{exercise_angina}'] = 1
            
        # Assign ST Slope
        if f'ST_Slope_{st_slope}' in input_data.columns:
            input_data[f'ST_Slope_{st_slope}'] = 1

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        if prediction[0] == 1:
            st.error("⚠️ The model predicts a **high risk** of heart disease.")
        else:
            st.success("✅ The model predicts a **low risk** of heart disease.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")