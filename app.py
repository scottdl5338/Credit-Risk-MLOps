from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Load the "Brain" 
model = joblib.load('credit_model.pkl')

# 2. Initialize the App
app = FastAPI(title="IVC Credit Risk Service")

# 3. Define what the user sends us (The Waiter's Notepad)
class Borrower(BaseModel):
    Age: int
    Sex: int
    Job: int
    Housing: int
    Credit_amount: int
    Duration: int
    Purpose_car: int = 0
    Purpose_domestic_appliances: int = 0
    Purpose_education: int = 0
    Purpose_furniture_equipment: int = 0
    Purpose_radio_TV: int = 0
    Purpose_repairs: int = 0
    Purpose_vacation_others: int = 0
    Saving_accounts_moderate: int = 0
    Saving_accounts_quite_rich: int = 0
    Saving_accounts_rich: int = 0
    Saving_accounts_unknown: int = 0
    Checking_account_moderate: int = 0
    Checking_account_rich: int = 0
    Checking_account_unknown: int = 0

@app.get("/")
def home():
    return {"message": "Credit Risk API is Live! Visit /docs for the UI."}

@app.post("/predict")
def predict_risk(data: Borrower):
    # 1. Convert the validated data into a DataFrame
    # Note: .model_dump() is the modern version of .dict()
    input_df = pd.DataFrame([data.model_dump()])

    # 2. Rename 'Credit_amount' to 'Credit amount' (with a space) to match the model
    input_df = input_df.rename(columns={'Credit_amount': 'Credit amount'})

    # 3. The "Magic" Step: Get the exact list of columns the model expects
    model_features = model.feature_names_in_

    # 4. Reindex: This automatically adds missing columns as 0 
    # and puts them in the EXACT order the model was trained on.
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # 5. Prediction logic
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    status = "Good" if prediction[0] == 1 else "Bad"
    confidence = float(probability[0][prediction[0]])

    return {
        "prediction": status,
        "confidence": f"{confidence*100:.2f}%",
        "internship_demo": "Summer 2026 Candidate"
    }