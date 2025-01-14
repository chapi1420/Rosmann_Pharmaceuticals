from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="Sales Prediction API", description="API for real-time sales predictions", version="1.0")


try:
    model = joblib.load("models/sales_model_2025-01-09-01-27-25.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")


class SalesInput(BaseModel):
    CompetitionDistance: float
    Promo: int
    Weekday: int
    DaysToHoliday: int
    StoreType: str
    Assortment: str
    MonthPhase: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sales Prediction API is running"}


@app.post("/predict/")
def predict(sales_input: SalesInput):
  
    try:
        input_data = pd.DataFrame([sales_input.dict()])
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")