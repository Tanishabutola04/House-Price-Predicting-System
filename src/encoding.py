import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# Fill missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

# Create engineered features
df["rooms_per_household"] = (
    df["total_rooms"] / df["households"]
)

df["bedrooms_per_room"] = (
    df["total_bedrooms"] / df["total_rooms"]
)

df["population_per_household"] = (
    df["population"] / df["households"]
)

encoded_df = pd.get_dummies(
    df,
    columns=["ocean_proximity"]
)

print(encoded_df.head())