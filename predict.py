from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app = FastAPI()

class Customer(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.dict()])

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    prediction = model.predict(data)[0]

    return {
        "Prediction": "Churn" if prediction == 1 else "No Churn"
    }