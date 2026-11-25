
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.responses import FileResponse

app = FastAPI()

# Load the trained ML model
model = joblib.load("model.pkl")


# Input schema
class Job(BaseModel):
    title: str
    company_profile: str = ""
    description: str = ""
    requirements: str = ""

    employment_type: str = "unknown"
    required_experience: str = "unknown"
    required_education: str = "unknown"

    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


# Home page
@app.get("/")
def home():
    return FileResponse("index.html")


# Prediction endpoint
@app.post("/predict")
def predict(job: Job):

    # Combine text columns
    text = (
        f"{job.title} "
        f"{job.company_profile} "
        f"{job.description} "
        f"{job.requirements}"
    )

    # Create DataFrame
    row = pd.DataFrame([{
        "text": text,
        "employment_type": job.employment_type,
        "required_experience": job.required_experience,
        "required_education": job.required_education,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions
    }])

    # Predict probability
    prob = float(model.predict_proba(row)[0][1])

    # Return result
    return {
        "fraudulent": int(prob > 0.5),
        "probability": round(prob, 2)
    }
