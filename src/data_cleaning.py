import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# Check missing values
print("========== Missing Values Before Cleaning ==========\n")
print(df.isnull().sum())

# Find median
median_value = df["total_bedrooms"].median()

print("\nMedian of total_bedrooms:", median_value)

# Fill missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_value)

# Verify
print("\n========== Missing Values After Cleaning ==========\n")
print(df.isnull().sum())