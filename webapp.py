import pandas as pd
import numpy as np
import streamlit as st
import joblib


st.title('Cross sell Prediction')
df = pd.read_csv("preprocessed_train_data.csv")


# Gender                         1
# Age                           44
# Driving_License                1
# Region_Code                 28.0
# Previously_Insured             0
# Vehicle_Damage                 1
# Annual_Premium           40454.0
# Policy_Sales_Channel        26.0
# Vintage                      217
# Response                     1.0
# Vehicle_Age_< 1 Year       False
# Vehicle_Age_> 2 Years       True

Gender = st.selectbox("Gender",pd.unique(df['Gender']))
Age = st.number_input("Age")
Driving_License = st.selectbox("Driving_License",pd.unique(df['Driving_License']))
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.selectbox("Previously_Insured",pd.unique(df['Previously_Insured']))
Vehicle_Damage = st.selectbox("Vehicle_Damage",pd.unique(df['Vehicle_Damage']))
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")
Vehicle_Age_1_Year = st.selectbox("Vehicle_Age_1_Year",pd.unique(df['Vehicle_Age_1_Year']))
Vehicle_Age_2_Years = st.selectbox("Vehicle_Age_2_Years",pd.unique(df['Vehicle_Age_2_Years']))


inputs = {
    'Gender':Gender,
    'Age':Age,
    'Driving_License':Driving_License,
    'Region_Code':Region_Code,
    'Previously_Insured':Previously_Insured,
   'Vehicle_Damage': Vehicle_Damage,
    'Annual_Premium':Annual_Premium,
    'Policy_Sales_Channel':Policy_Sales_Channel,
    'Vintage':Vintage,
    'Vehicle_Age_1_Year': Vehicle_Age_1_Year,  # Renamed for clarity
    'Vehicle_Age_2_Years': Vehicle_Age_2_Years
}

if st.button('Predict'):
    model = joblib.load("Final_predication_SoftVoting.pkl")

    x_input = pd.DataFrame(inputs , index=[0])
    prediction  = model.predict(x_input)
    st.write(prediction)