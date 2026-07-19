import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/housing.csv")

print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nOcean Proximity Categories:")
print(df["ocean_proximity"].unique())