import streamlit as st
import pandas as pd
import numpy as np
import pickle
from combiner import CombinedAttributesAdder
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

st.title("California Housing Prices Prediction")
st.header("Enter the attributes of the housing.")

data = pd.read_csv('housing.csv')

max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

@st.cache_resource
def load_model(filepath: str):
    return pickle.load(open(filepath, 'rb'))

loaded_model = load_model('reg.sav')

def get_coords(address: str, geolocator):
    return geolocator.geocode(address)

def create_marker(m: folium.Map, coords, address=None):
    location = [coords.latitude, coords.longitude]
    marker = folium.Marker(location=location, popup=address, icon=folium.Icon(color='red'))
    return marker

def clear_markers():
    st.session_state['markers'] = []

def create_map():
    # Define the boundaries of California
    min_lat, min_lon = 32.5295, -124.4820
    max_lat, max_lon = 42.0095, -114.1315

    # Create a map centered on California
    map_ca = folium.Map(location=[37.7749, -122.4194], zoom_start=6, min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon, no_wrap=True, max_bounds=True)

    return map_ca

if 'markers' not in st.session_state:
    st.session_state['markers'] = []

map_ca = create_map()
fg = folium.FeatureGroup(name="markers")

for marker in st.session_state["markers"]:
    fg.add_child(marker)

geolocator = Nominatim(user_agent="app")
col1, col2 = st.columns(2)

with col1:
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

with col2:
    households = st.number_input(
        "Number of Households in the Locality", 
        min_value=int(min_values['households']), 
        max_value=int(max_values['households']), step=5)
    
    median_income = st.number_input(
        "Median Income of the Locality", 
        min_value=min_values['median_income'], 
        max_value=max_values['median_income'], step=0.5)
    
    ocean_proximity = st.selectbox(
    'Ocean Proximity:',
    ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))

address = st.text_input("Address")
st.caption("Press enter to mark the address in the map.")

button = st.button("Predict")
if address:
    coords = get_coords(address, geolocator)
    loc = np.array([coords.longitude, coords.latitude])

    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Latitude", value=f"{coords.latitude:.2f}")
    with c2:
        st.metric(label="Longitude", value=f"{coords.longitude:.2f}")
    
    marker = create_marker(map_ca, coords, address=address)
    st.session_state['markers'].append(marker)

if button:
    input_data = {
    "lon": coords.longitude,
    "lat": coords.latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
    }

    input_df = pd.DataFrame([input_data])
    prediction = loaded_model.predict(input_df).squeeze()
    st.metric(label="Prediction", value=f"$ {prediction:.2f}")

# Add the map to st_data
st_data = st_folium(map_ca, width=800, feature_group_to_add=fg)

st.button("Clear markers")
