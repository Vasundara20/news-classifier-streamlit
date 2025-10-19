import zipfile
import os
import pandas as pd

# Path to your zip file
zip_path = r"C:\Users\Admin\Desktop\News_Classification_Project\archive.zip"

# Folder where you’ll extract it
extract_folder = r"C:\Users\Admin\Desktop\News_Classification_Project\extracted_data"

# Step 1️⃣ — Extract the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print("✅ Zip extracted successfully!")

# Step 2️⃣ — Find the 'bbc' folder inside the extracted files
bbc_folder = os.path.join(extract_folder, "bbc-fulltext (document classification)", "bbc")

# Step 3️⃣ — Read files and make CSV
texts = []
categories = []

for category in os.listdir(bbc_folder):
    category_path = os.path.join(bbc_folder, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                categories.append(category)

# Step 4️⃣ — Create DataFrame and save to your existing CSV
df = pd.DataFrame({'text': texts, 'category': categories})
csv_path = r"C:\Users\Admin\Desktop\News_Classification_Project\bbc-text.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')

print("✅ CSV created successfully at:", csv_path)
print("Total articles:", len(df))
print(df['category'].value_counts())
