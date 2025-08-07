from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Save to CSV (optional, for DVC tracking)
df.to_csv("data/raw/california_housing.csv", index=False)
