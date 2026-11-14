
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

# Load processed dataset
df = pd.read_csv("dataset/processed_emails.csv")

# Input features and target labels
X = df["cleaned_text"].fillna("")
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

# Learn vocabulary from training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform testing data
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save numerical features
save_npz("models/X_train.npz", X_train_tfidf)
save_npz("models/X_test.npz", X_test_tfidf)

# Save labels
y_train.to_csv("models/y_train.csv", index=False)
y_test.to_csv("models/y_test.csv", index=False)

# Display results
print("TF-IDF Vectorization Completed!")

print("\nTraining Data Shape:", X_train_tfidf.shape)
print("Testing Data Shape:", X_test_tfidf.shape)

print("\nSample Vocabulary:")
print(vectorizer.get_feature_names_out()[:15])

print("\nAll files saved successfully!")
