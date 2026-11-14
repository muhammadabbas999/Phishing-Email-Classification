
import joblib
import re
import html
import numpy as np

# Load trained model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Clean email text
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


# Explain email prediction
def explain_email(email):

    # Preprocess email
    cleaned_email = clean_text(email)

    # Convert text into TF-IDF features
    email_vector = vectorizer.transform([cleaned_email])

    # Predict email
    prediction = model.predict(email_vector)[0]

    # Get model probability
    probabilities = model.predict_proba(email_vector)[0]

    # Get feature names and model coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Calculate individual feature contributions
    contributions = (
        email_vector.toarray()[0] * coefficients
    )

    # Find positive and negative contributions
    positive_indices = np.argsort(contributions)[::-1]
    negative_indices = np.argsort(contributions)

    positive_words = [
        (feature_names[i], contributions[i])
        for i in positive_indices
        if contributions[i] > 0
    ][:5]

    negative_words = [
        (feature_names[i], contributions[i])
        for i in negative_indices
        if contributions[i] < 0
    ][:5]

    # Display report
    print("\n===== EXPLAINABLE AI REPORT =====")

    print(
        "Prediction:",
        "Phishing" if prediction == 1 else "Legitimate"
    )

    print(
        "Phishing Probability:",
        round(probabilities[1] * 100, 2), "%"
    )

    print("\nWords Supporting Phishing:")

    if positive_words:
        for word, score in positive_words:
            print(f"{word}: +{score:.4f}")
    else:
        print("No positive feature contributions found.")

    print("\nWords Supporting Legitimate:")

    if negative_words:
        for word, score in negative_words:
            print(f"{word}: {score:.4f}")
    else:
        print("No negative feature contributions found.")

    print("\n=================================")


# User input
email = input("Enter email text: ")

explain_email(email)
