import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('C:/Users/ADMIN/Desktop/vs code/streamlit/project4/diabetes_model.sav','rb'))

def diabetes_prediction(input_data):

    data_numeric = np.array(input_data).astype(float)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():

    st.title('Diabetes Prediction App')

    Pregnancies = st.number_input('Number of pregnancies : ')
    Glucose = st.number_input('Glucose Level : ')
    BloodPressure = st.number_input('Blood Pressure Value : ')
    SkinThickness = st.number_input('Skin Thickness Value : ')
    Insulin = st.number_input('Insulin Value : ')
    bmi = st.number_input('BMI Value : ')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value  : ')
    Age = st.number_input('Enter your Age : ')

    # code for prediction

    diagnosis = ' '

    if st.button('PREDICT'):
        diagnosis = diabetes_prediction([Pregnancies , Glucose , BloodPressure , SkinThickness , Insulin , bmi , DiabetesPedigreeFunction , Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    