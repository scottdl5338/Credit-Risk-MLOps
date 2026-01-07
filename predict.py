import pandas as pd
import joblib

#1. Load the model
model = joblib.load('credit_model.pkl')

#2. To get by adding like 50 different hot encoded cloumns we can use a template 
# as our sample 

template_df = pd.read_csv('data/cleaned_credit_data.csv')
# Removes the answer key aka Risk col
template_df = template_df.drop('Risk_good', axis = 1)


# 3. Now time to create a single "Fake" Customer using template
# This creates a row of zeros with the correct column names
sample_customer = template_df.head(1).copy()
for col in sample_customer.columns:
    sample_customer[col] = 0

# 4. Fill in the specific data for our "Banker Interview"
sample_customer['Age'] = 20
sample_customer['Credit amount'] = 2096
sample_customer['Duration'] = 12
sample_customer['Sex'] = 0 # Male
# Set one 'Purpose' to 1 (e.g., Purpose_car if it exists in your columns)
if 'Purpose_car' in sample_customer.columns:
    sample_customer['Purpose_education'] = 1


# DEBUG CHECK the # cols in model input == # col for input from smaple
print("--- PREDICTION CHECK ---")
print(f"Model expects {len(model.feature_names_in_)} features")
print(f"You are providing {len(sample_customer.columns)} features")

#5. Making a prediction
prediction = model.predict(sample_customer)
# .predict returns and array of all teh customers predictions
probability = model.predict_proba(sample_customer)
#.predirct_proba - returns a 2D arrary giving a number of how confident it is in its prediction

# 6. Output the Result 
result = "GOOD" if prediction[0] == 1 else "BAD"
print(f"Prediction: This loan is {result}")
print(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")

# Big Error: If you ever change your preprocess.py 
# (like adding a new column), your credit_model.pkl will become obsolete
# The Fix: If you change the data, you must rerun train_model.py to generate a new .pkl