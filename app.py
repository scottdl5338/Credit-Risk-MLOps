from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Load the "Brain" 
model = joblib.load('credit_model.pkl')

# 2. Initialize the App
app = FastAPI(title="Credit Risk Service")

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
    # 1. Convert and Align Features (The magic you did earlier)
    input_df = pd.DataFrame([data.model_dump()])
    input_df = input_df.rename(columns={'Credit_amount': 'Credit amount'})
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # 2. Get the Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # 3. Get Feature Importance (The Explainability part)
    # We map the feature names to their importance scores
    importances = dict(zip(model_features, model.feature_importances_))
    
    # Sort them to find the "Top 3" drivers of this model
    top_drivers = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
    explanation = {feature: f"{round(float(score) * 100, 2)}%" for feature, score in top_drivers}

    status = "Good" if prediction[0] == 1 else "Bad"
    confidence = float(probability[0][prediction[0]])

    return {
        "decision": status,
        "confidence": f"{confidence*100:.2f}%",
        "top_decision_drivers": explanation,
        "note": "This identifies which factors most influenced the model globally."
    }
#How Decorators Work in FastAPHow Decorators Work in FastAPI
# In your credit risk project, the decorators act as address labels for your web server.

# The @ Symbol: Tells Python, "I am applying a special rule to the function immediately below this."

# The app part: Refers to the FastAPI instance you created at the top of your script (app = FastAPI()).

# The .get or .post part: This defines the "Action." It tells the server to listen for a specific type of internet request.

# The ("/") or ("/predict") part: This defines the "Path." It’s the specific URL extension the user must visit to trigger that function.