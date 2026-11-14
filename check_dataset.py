
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/emails.csv")

# Display dataset
print(df)

# Show dataset information
print("\nDataset Information:")
print(df.info())

# Count email categories
print("\nEmail Categories:")
print(df["category"].value_counts())
