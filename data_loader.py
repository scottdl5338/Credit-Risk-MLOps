import pandas as pd

# Load the data
df = pd.read_csv('data/german_credit_data.csv', index_col=0)

print("First 5 rows of dataset:")
print(df.head())

# Corrected: shape gives you (rows, columns)
print("\nDataset Shape (Rows, Cols):")
print(df.shape) 

# Prints a Series with columns as indexes highlighting how many null are in each column
print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumn Info:")
df.info() # Note: .info() prints automatically, so no need for print(df.info())