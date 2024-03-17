# Download and load the dataset into Pandas DataFrames
import pandas as pd

# Download and load the dataset into Pandas DataFrames
titanic = pd.read_csv(r"titanic/train.csv")

# Example of storing these DataFrames locally
titanic.to_csv('titanic_dataset.csv', index=False)

print("Datasets downloaded and stored.")
