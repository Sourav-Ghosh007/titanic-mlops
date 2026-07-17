# app.py
# FastAPI service that loads model.pkl and serves survival predictions

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# -------------------------------
# Load the trained model ONCE when the app starts
# (not on every request - that would be slow)
# -------------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="Titanic Survival Predictor")

# -------------------------------
# Define what a valid request looks like
# Pydantic validates incoming data automatically -
# if someone sends a string instead of a number, FastAPI rejects it before it reaches your code
# -------------------------------
class Passenger(BaseModel):
    pclass: int      # 1, 2, or 3
    sex: int         # 0 = male, 1 = female
    age: float
    fare: float
    sibsp: int        # siblings/spouses aboard
    parch: int        # parents/children aboard

# -------------------------------
# Health check endpoint - useful to confirm the API is alive
# (you'll use endpoints like this in monitoring later)
# -------------------------------
@app.get("/")
def read_root():
    return {"status": "Titanic Predictor API is running"}

# -------------------------------
# The actual prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert the incoming request into the same shape the model expects
    input_data = pd.DataFrame([{
        "pclass": passenger.pclass,
        "sex": passenger.sex,
        "age": passenger.age,
        "fare": passenger.fare,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of survival

    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 4)
    }