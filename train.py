import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# Missing values
median_value = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_value)

# Feature Engineering
df["rooms_per_household"] = (
    df["total_rooms"] /
    df["households"]
)

df["bedrooms_per_room"] = (
    df["total_bedrooms"] /
    df["total_rooms"]
)

df["population_per_household"] = (
    df["population"] /
    df["households"]
)

# One Hot Encoding
df = pd.get_dummies(
    df,
    columns=["ocean_proximity"]
)

# Features & Target
X = df.drop(
    "median_house_value",
    axis=1
)

y = df["median_house_value"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Training
model.fit(
    X_train,
    y_train
)

print("Model Trained Successfully!")

# Prediction
predictions = model.predict(
    X_test
)

print("\nFirst 10 Predictions:")

print(predictions[:10])

# ==========================

# Model Evaluation

# ==========================



mae = mean_absolute_error(

    y_test,

    predictions

)



mse = mean_squared_error(

    y_test,

    predictions

)



rmse = mse ** 0.5



r2 = r2_score(

    y_test,

    predictions

)



print("\n=========================")

print("Model Evaluation")

print("=========================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}") 
# ==========================
# Save the Trained Model
# ==========================

joblib.dump(
    model,
    "models/linear_regression_model.pkl"
)

print("\n✅ Model saved successfully!")
