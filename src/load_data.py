import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# First 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Last 5 rows
print("\n========== LAST 5 ROWS ==========")
print(df.tail())

# Shape
print("\n========== SHAPE ==========")
print(df.shape)

# Column Names
print("\n========== COLUMN NAMES ==========")
print(df.columns)

# Dataset Information
print("\n========== INFO ==========")
df.info()

# Summary Statistics
print("\n========== SUMMARY ==========")
print(df.describe())