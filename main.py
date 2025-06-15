from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pickle
import numpy as np

# Load model and encoders
with open('model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('model/feature_encoders.pkl', 'rb') as f:
    feature_encoders = pickle.load(f)

# Define FastAPI app
app = FastAPI()

# Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected input schema
class PatientData(BaseModel):
    Age: int
    Gender: str
    BMI: float
    BP: float
    Sugar: float
    Cholesterol: float
    Smoking: str
    FamilyHistory: str

@app.post("/predict")
async def predict(data: PatientData):
    # Convert input to dict
    input_dict = data.dict()

    # Encode categorical inputs using saved encoders
    for col, encoder in feature_encoders.items():
        if col in input_dict:
            input_dict[col] = int(encoder.transform([input_dict[col]])[0])

    # Prepare input for model
    input_array = np.array([[input_dict[col] for col in ['Age', 'Gender', 'BMI', 'BP', 'Sugar', 'Cholesterol', 'Smoking', 'FamilyHistory']]])

    # Make prediction
    prediction = model.predict(input_array)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Return prediction as label
    return {"prediction": predicted_label}

# Root route for base URL
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>✅ FastAPI Backend is Running!</h2>
    <p>Go to <a href='/docs'>/docs</a> to use Swagger UI.</p>
    """
