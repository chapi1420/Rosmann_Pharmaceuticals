from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


class SalesPredictionAPI:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.app = FastAPI(title="Sales Prediction API", 
                           description="API for real-time sales predictions", 
                           version="1.0")
        self.setup_routes()

    def load_model(self, model_path: str):
        try:
            return joblib.load(model_path)
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

    def setup_routes(self):
        @self.app.get("/")
        def health_check():
            return {"status": "ok", "message": "Sales Prediction API is running"}

        @self.app.post("/predict/")
        def predict(sales_input: self.SalesInput):
            try:
                input_data = pd.DataFrame([sales_input.dict()])
                prediction = self.model.predict(input_data)
                return {"prediction": prediction[0]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# To run the API
if __name__ == "__main__":
    sales_prediction_api = SalesPredictionAPI("C:\\Users\\nadew\\10x\\week4\\notebooks\\sales_model_2025-01-14-17-04-56.pkl")
    import uvicorn
    uvicorn.run(sales_prediction_api.app, host="0.0.0.0", port=8000)
