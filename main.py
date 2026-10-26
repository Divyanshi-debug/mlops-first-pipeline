from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Random Forest API")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def ready_to_serve():
    return {"status": "Ready to serve"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

#Endpoint to make predictions with Iris Random Forest model
@app.post("/predict")
def predict(data: IrisInput):
    features=[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    return {"prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]
            }