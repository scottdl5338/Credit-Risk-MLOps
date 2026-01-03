import pandas as pd
#A tool used to convert text into numbers for Sex and Housing
from sklearn.preprocessing import LabelEncoder

# <<1>>. Load the data
df = pd.read_csv('data/german_credit_data.csv')

# <<2>>. Handle Missing (Nan) Values
# We fill NaN with 'unknown' because 'missing' is a risk signal in banking
# AkA imputation other common strategies are KNN and MICE

print("--- BEFORE FILLING ---")
print("Missing values in Saving accounts:", df['Saving accounts'].isnull().sum())
print("Missing values in Checking account:", df['Checking account'].isnull().sum())

df['Saving accounts']= df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("\n--- AFTER FILLING ---")
print("Missing values in Saving accounts:", df['Saving accounts'].isnull().sum())
print("Missing values in Checking account:", df['Checking account'].isnull().sum())
print("\n--- SAMPLE OF FILLED DATA ---\n", df[['Saving accounts', 'Checking account']].head(10))

# <<3>>. Transform Text into Numbers aka Encoding
# For 'Sex' and 'Housing', we use Label Encoding (0, 1, 2...) 

# Convert text categories into numbers 
# Original Data: ["male", "female", "female", "male"] 
# ---> 
# After Encoding: [1, 0, 0, 1] (where "female" = 0 and "male" = 1)
# Sorts them alphabetically then assigsn num
le = LabelEncoder()

print("\n--- SAMPLE OF FILLED DATA BEFORE ENCODING ---\n")
print(df.head())

#.fit() - scans the entire column to find all unique categories (e.g., it "learns" that the only options are "male" and "female").
#.transform() It actually replaces those words with their assigned numbers throughout your dataset.
df['Sex'] = le.fit_transform(df['Sex'])
df['Housing'] = le.fit_transform(df['Housing'])

print("\n--- SAMPLE OF FILLED DATA AFTER ENCODING ---\n")
print(df.head())

# <<4>> One-Hot Encoding for 'Purpose' and others
# One-Hot Encoding bascially used for when you have more than just two options 1 or 0
# Ex. Sex: Male = 0 Female = 1 Non-binary = 2 Trans = 3

# .get_dummies - This creates new columns for each category (e.g., Purpose_car, Purpose_radio)
# drop_first = True - Removes first column since if all other columns are False the process of elminination the only thing left must be True avoids Multicollinearity
# Always use for linear or small/effiecnt models
df = pd.get_dummies(df, columns=['Purpose', 'Saving accounts', 'Checking account','Risk'], drop_first=True)


# <<5>> Result
print("\n--- SAMPLE OF FILLED DATA AFTER HOT-ENCODING ---\n")
print("New Shape (More columns due to encoding):", df.shape)
print(df.head())


# <<6>> Save the cleaned data for Day 3 (Modeling)
df.to_csv('data/cleaned_credit_data.csv', index=True)
print("\nCleaned data saved to data/cleaned_credit_data.csv")

print("\n--- DATA INFO ---\n")
df.info()