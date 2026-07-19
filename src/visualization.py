import pandas as pd
import matplotlib.pyplot as plt                     
df = pd.read_csv("data/raw/housing.csv")
plt.figure(figsize=(10,6))

plt.scatter(df["median_income"], df["median_house_value"])

plt.title("Income vs House Price")

plt.xlabel("Median Income")

plt.ylabel("House Price")

plt.show()