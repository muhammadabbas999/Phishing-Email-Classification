
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load training and testing data
X_train = load_npz("models/X_train.npz")
X_test = load_npz("models/X_test.npz")

y_train = pd.read_csv("models/y_train.csv").values.ravel()
y_test = pd.read_csv("models/y_test.csv").values.ravel()

# Create machine learning model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model
print("Training model...")

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save trained model
joblib.dump(model, "models/phishing_model.pkl")

print("\nModel saved successfully!")
