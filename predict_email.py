
import joblib

# Load trained model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Email prediction function
def predict_email(email):

    # Convert email into numerical features
    email_vector = vectorizer.transform([email])

    # Predict email category
    prediction = model.predict(email_vector)[0]

    # Get model probability
    probabilities = model.predict_proba(email_vector)[0]

    phishing_probability = probabilities[1] * 100

    # Display prediction
    if prediction == 1:
        result = "Phishing"
    else:
        result = "Legitimate"

    # Risk category based on phishing probability
    if phishing_probability >= 70:
        risk = "High Risk"
    elif phishing_probability >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    print("\n===== EMAIL SECURITY REPORT =====")
    print("Prediction:", result)
    print("Phishing Probability:", round(phishing_probability, 2), "%")
    print("Risk Level:", risk)
    print("=================================")


# Take email input from user
email = input("Enter email text: ")

# Run prediction
predict_email(email)
