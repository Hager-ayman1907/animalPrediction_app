# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle


@st.cashe_data
## loading the saved model
loaded_model = pickle.load(open("C:/Users/Hagar/coursePython/Exercises with me & Youtube/trained_model.csv", 'rb'))  


input_data = (8,133,72,0,0,32.9,0.27,39)

#Changing the input data to numpy array
input_data_asarray = np.asarray(input_data)

## Reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_asarray.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The person doesn't have diabetes")
else:
    print("The person has diabetes")