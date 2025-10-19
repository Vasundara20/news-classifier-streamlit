import pandas as pd

# Load JSON dataset
df = pd.read_json('news_category_dataset_v3.json', lines=True)
print("✅ Dataset Loaded Successfully!")

# Show first few rows
print(df.head())

# Check category distribution
print("\nCategory Counts:")
print(df['category'].value_counts())
