
import re
import html
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create FastAPI application
app = FastAPI(
    title="Phishing Email Risk Classification API",
    description="AI-powered email phishing detection system",
    version="1.0"
)

# Load trained model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Email request structure
class EmailRequest(BaseModel):
    email_text: str = Field(min_length=1, max_length=20000)


# Text preprocessing
def clean_text(text):

    text = text.lower()
    text = html.unescape(text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL_TOKEN ",
        text
    )

    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Phishing Email Classification API is running!"
    }


# Email prediction endpoint
@app.post("/predict")
def predict_email(request: EmailRequest):

    email = request.email_text.strip()

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email text cannot be empty."
        )

    # Clean email
    cleaned_email = clean_text(email)

    # Convert email into numerical features
    email_vector = vectorizer.transform([cleaned_email])

    # Model prediction
    prediction = int(model.predict(email_vector)[0])

    probabilities = model.predict_proba(email_vector)[0]

    phishing_probability = float(probabilities[1] * 100)

    # Risk classification
    if phishing_probability >= 70:
        risk_level = "High Risk"

    elif phishing_probability >= 40:
        risk_level = "Medium Risk"

    else:
        risk_level = "Low Risk"

    # Model-based explanation
    contributions = (
        email_vector.toarray()[0] * model.coef_[0]
    )

    feature_names = vectorizer.get_feature_names_out()

    positive_indices = np.argsort(contributions)[::-1]

    indicators = []

    for i in positive_indices:
        if contributions[i] > 0 and len(indicators) < 5:

            indicators.append({
                "word": str(feature_names[i]),
                "contribution": round(
                    float(contributions[i]), 4
                )
            })

    # Return results
    return {
        "prediction": (
            "Phishing" if prediction == 1 else "Legitimate"
        ),
        "risk_level": risk_level,
        "phishing_probability": round(
            phishing_probability, 2
        ),
        "confidence": round(
            float(probabilities[prediction] * 100), 2
        ),
        "suspicious_indicators": indicators
    }
