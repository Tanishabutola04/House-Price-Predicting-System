import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

print("Average House Price:")
print(df["median_house_value"].mean())

print("\nHighest House Price:")
print(df["median_house_value"].max())

print("\nLowest House Price:")
print(df["median_house_value"].min())

print("\nAverage Income:")
print(df["median_income"].mean())

print("\nMost Expensive House:")
print(df.loc[df["median_house_value"].idxmax()])