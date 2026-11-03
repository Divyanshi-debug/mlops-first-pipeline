from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import os
from contextlib import asynccontextmanager
from typing import Callable


#MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7001")
MODEL_URI="models:/iris-classifier@champion"

def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model=mlflow.sklearn.load_model(MODEL_URI)
    return model

def create_app(model_loader: Callable=load_mlflow_model):
    @asynccontextmanager
    def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load model: {ex}"
            )
        client=MlflowClient()
        champion_version=client.get_model_version_by_alias(
            name="iris-classifier",
            alias="champion"
            )
        app.state.model_version=champion_version.version
        print(f"Loaded model: iris-classifier version: {model_version} from MLFlow tracking server: {MLFLOW_TRACKING_URI}")
        yield

    app = FastAPI(
        title="Iris Classifiaction API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan
    )

        


#app = FastAPI(title="Iris Random Forest API")

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
    return {"status": "healthy",
            "model_name": "iris-classifier",
            "model_version": model_version,
            }

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