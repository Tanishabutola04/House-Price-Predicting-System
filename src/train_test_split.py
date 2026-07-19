import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("data/raw/housing.csv")

# ===========================
# Handle Missing Values
# ===========================
median_value = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_value)

# ===========================
# Feature Engineering
# ===========================

# Average number of rooms per household
df["rooms_per_household"] = (
    df["total_rooms"] / df["households"]
)

# Percentage of bedrooms among total rooms
df["bedrooms_per_room"] = (
    df["total_bedrooms"] / df["total_rooms"]
)

# Average population per household
df["population_per_household"] = (
    df["population"] / df["households"]
)

# ===========================
# Convert Categorical Data
# ===========================

df = pd.get_dummies(
    df,
    columns=["ocean_proximity"]
)

# ===========================
# Define Features (X) and Target (y)
# ===========================

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ===========================
# Split Dataset
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Print Results
# ===========================

print("=" * 50)
print("Training and Testing Data")
print("=" * 50)

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Labels Shape:")
print(y_train.shape)

print("\nTesting Labels Shape:")
print(y_test.shape)

print("\nFirst 5 Rows of Training Data:")
print(X_train.head())

print("\nFirst 5 Training Labels:")
print(y_train.head())