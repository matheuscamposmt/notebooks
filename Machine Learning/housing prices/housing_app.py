import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("California Housing Prices Prediction")

data = pd.read_csv('housing.csv')

max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

print(max_values)

with st.container() as container:
    st.header("Enter the attributes of the housing.")

    housing_median_age = st.number_input(
        "Median Age (in years)",
        min_value=int(min_values['housing_median_age']), 
        max_value=int(max_values['housing_median_age']), 
        step=1)
    total_rooms = st.number_input(
        "Total Rooms", 
        min_value=int(min_values['total_rooms']), 
        max_value=int(max_values['total_rooms']), 
        step=5)
    total_bedrooms = st.number_input(
        "Total Bedrooms", 
        min_value=int(min_values['total_bedrooms']), 
        max_value=int(max_values['total_bedrooms']), 
        step=5)
    
    population = st.number_input(
        "Population of the Locality", 
        min_value=int(min_values['population']), 
        max_value=int(max_values['population']), step=5)
    households = st.number_input(
        "Number of Households in the Locality", 
        min_value=int(min_values['households']), 
        max_value=int(max_values['households']), step=5)
    
    median_income = st.number_input(
        "Median Income of the Locality", 
        min_value=min_values['median_income'], 
        max_value=max_values['median_income'], step=0.5)

input_data = {"housing_median_age": housing_median_age,
              "total_rooms": total_rooms,
              "total_bedrooms": total_bedrooms,
              "population": population,
              "households": households,
              "median_income": median_income}

input_df = pd.DataFrame([input_data])

loaded_model = pickle.load(open('reg.sav', 'rb'))

prediction = loaded_model.predict(input_df)

st.write(f"Prediction: {prediction:.2f}")
