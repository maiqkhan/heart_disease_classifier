from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class PatientInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    Age: int = Field(..., ge=0)
    Sex: Literal["F", "M"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: int = Field(..., ge=0)
    Cholesterol: int = Field(..., ge=0)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR:  int = Field(..., ge=0)
    ExerciseAngina: Literal["N", "Y"]
    Oldpeak: float = Field(...)
    ST_Slope: Literal["Up", "Flat", "Down"]

class HeartDiseaseResponse(BaseModel):
    heart_disease_indicator: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model 
    model = joblib.load("log_regression.joblib")
    yield 
    model = None 


app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=HeartDiseaseResponse)
def predict_heart_disease(patient: PatientInput):

    patient_df = pd.DataFrame([patient.model_dump()])
    
    prediction = model.predict(patient_df)[0]
    
    return {"heart_disease_indicator": int(prediction)}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=9696, reload=True)