import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
# ==========================
# Load Trained Model
# ==========================

model = joblib.load("models/linear_regression_model.pkl")

# ==========================
# Title
# ==========================
st.title("🏠 California House Price Prediction System")

st.markdown("""
Predict the price of a California house using a trained Machine Learning model.

Fill in the details below and click **Predict Price**.
""")

st.divider()

# ==========================
# Sidebar
# ==========================
st.sidebar.title("About")

st.sidebar.info("""
This project predicts California house prices using:

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Linear Regression
""")

st.sidebar.success("Created by Tanisha")

# ==========================
# Input Form
# ==========================

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "Longitude",
        value=-122.23
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88
    )

    housing_age = st.number_input(
        "Housing Median Age",
        value=41
    )

    total_rooms = st.number_input(
        "Total Rooms",
        value=880
    )

with col2:

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=129
    )

    population = st.number_input(
        "Population",
        value=322
    )

    households = st.number_input(
        "Households",
        value=126
    )

    median_income = st.number_input(
        "Median Income",
        value=8.3252
    )

st.divider()

ocean = st.selectbox(
    "Ocean Proximity",
    [
        "NEAR BAY",
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "ISLAND"
    ]
)

st.divider()

predict_button = st.button(
    "🏠 Predict Price"
)

if predict_button:

    # Feature Engineering
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    # Create Input DataFrame
    input_data = pd.DataFrame({

        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],

        "rooms_per_household": [rooms_per_household],
        "bedrooms_per_room": [bedrooms_per_room],
        "population_per_household": [population_per_household],

        "ocean_proximity_<1H OCEAN": [ocean == "<1H OCEAN"],
        "ocean_proximity_INLAND": [ocean == "INLAND"],
        "ocean_proximity_ISLAND": [ocean == "ISLAND"],
        "ocean_proximity_NEAR BAY": [ocean == "NEAR BAY"],
        "ocean_proximity_NEAR OCEAN": [ocean == "NEAR OCEAN"]
    })

    prediction = model.predict(input_data)

    st.success("Prediction Completed!")

    st.metric(
        "🏠 Predicted House Price",
        f"${prediction[0]:,.2f}"
    )