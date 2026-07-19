import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# Fill missing values
median_value = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_value)

# Feature Engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]

df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

df["population_per_household"] = (
    df["population"] / df["households"]
)

print("\nNew Dataset Shape:")
print(df.shape)

print("\nLast 3 Columns:")
print(df[[
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]].head())