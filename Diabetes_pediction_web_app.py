# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:13:37 2025

@author: Hagar
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st 

## loading the saved model
loaded_model = pickle.load(open("C:/Users/Hagar/coursePython/Exercises with me & Youtube/trained_model.csv", 'rb'))  
scaler = StandardScaler()

## creating a function for prediction 
def diabetes_prediction(input_data):
    input_data = (8,133,72,0,0,32.9,0.27,39)

    #Changing the input data to numpy array
    input_data_asarray = np.asarray(input_data)

    ## Reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_asarray.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)     
    print(prediction)

    if(prediction[0] == 0):
        return"The person doesn't have diabetes"     
    else:
        return"The person has diabetes"
        
        
def main():
    #giving a title 
    st.title("Diabetes Prediction web App")
    
    # getting the input data from the user
    
    Pregnancies = st.selectbox("Number of Pregnancies", np.arange(0,18,1))
    Glucose = st.selectbox("Number of Glucose", np.arange(85, 190, 1))
    BloodPressure = st.selectbox("BloodPressure value", np.arange(66, 114, 1))
    SkinThickness = st.selectbox("SkinThickness value", np.arange(0, 99, 1))
    Insulin = st.selectbox("Insulin level", np.arange(0, 255, 1))
    BMI = st.selectbox("BMI value", np.arange(0, 46))
    DiabetesPedigreeFunction = st.selectbox("DiabetesPedigreeFunction value", 0.351)
    Age = st.selectbox("Age of the person", np.arange(21, 70, 1))
    
    # code for prediction
    diagnoses = ''
    
    # creating a button for prediction
    
    if st.button ("Diabetes Test Result"):
        diagnoses = diabetes_prediction([Pregnancies, Glucose, BloodPressure,
                  SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age ])

    st.success(diagnoses)
    
    
    
if __name__ == '__main__':
    main()