import joblib
import pandas as pd

# ==========================
# Load Trained Model
# ==========================
model = joblib.load("models/linear_regression_model.pkl")

print("✅ Model Loaded Successfully!")

# ==========================
# Sample House Details
# ==========================
sample_house = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [41],
    "total_rooms": [880],
    "total_bedrooms": [129],
    "population": [322],
    "households": [126],
    "median_income": [8.3252],

    # Engineered Features
    "rooms_per_household": [880 / 126],
    "bedrooms_per_room": [129 / 880],
    "population_per_household": [322 / 126],

    # One-Hot Encoded Features
    "ocean_proximity_<1H OCEAN": [False],
    "ocean_proximity_INLAND": [False],
    "ocean_proximity_ISLAND": [False],
    "ocean_proximity_NEAR BAY": [True],
    "ocean_proximity_NEAR OCEAN": [False]
})

# ==========================
# Predict House Price
# ==========================
prediction = model.predict(sample_house)

print("\n🏠 Predicted House Price:")
print(f"${prediction[0]:,.2f}")