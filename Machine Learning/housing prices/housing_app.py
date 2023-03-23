import streamlit as st
import pandas as pd
import pickle

st.title("California Housing Prices Prediction")

st.sidebar.header("Enter the details of your property")

housing_median_age = st.sidebar.slider("Median Age of the House (in years)", 1, 100, 50)
total_rooms = st.sidebar.slider("Total Rooms in the House", 1, 10000, 5000)
total_bedrooms = st.sidebar.slider("Total Bedrooms in the House", 1, 10000, 5000)
population = st.sidebar.slider("Population of the Locality", 1, 100000, 50000)
households = st.sidebar.slider("Number of Households in the Locality", 1, 100000, 50000)
median_income = st.sidebar.slider("Median Income of the Locality (in USD)", 1000, 200000, 50000)

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
