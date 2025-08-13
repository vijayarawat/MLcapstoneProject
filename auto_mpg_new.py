import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("./auto-mpg.csv")

# Apply drift transformations
df["mpg"] = df["mpg"] * 5.5
df["weight"] = df["weight"] * 2.5
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce") * 5.2

# Replace ~10% of car names with new ones
new_names = ["tesla model s", "hyundai i20", "tata nano", "mahindra thar", "toyota fortuner"]
mask = np.random.rand(len(df)) < 0.1
df.loc[mask, "car name"] = np.random.choice(new_names, mask.sum())

# Save drifted dataset
df.to_csv("auto_mpg_drifted.csv", index=False)

print("Drifted dataset saved as auto_mpg_drifted.csv")
