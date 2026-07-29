import streamlit as st
import joblib
import pandas as pd

model = joblib.load("logistic_regression_model.sav")

st.title("Diabetes Prediction")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    df = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                      columns=[
                          'Pregnancies', 'Glucose', 'BloodPressure',
                          'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'
                      ])

    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
