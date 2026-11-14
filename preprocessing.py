
import pandas as pd
import re
import html

# Load dataset
df = pd.read_csv("dataset/emails.csv")

# Text cleaning function
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Decode HTML entities
    text = html.unescape(text)

    # Preserve URLs as a useful security indicator
    text = re.sub(r"https?://\S+|www\.\S+", " URL_TOKEN ", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply preprocessing
df["cleaned_text"] = df["email_text"].apply(clean_text)

# Display original and cleaned emails
print("Original and Cleaned Emails:\n")

for index, row in df.head(5).iterrows():
    print("Original:", row["email_text"])
    print("Cleaned:", row["cleaned_text"])
    print("-" * 50)


# Save processed dataset
df.to_csv("dataset/processed_emails.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Processed dataset saved.")
