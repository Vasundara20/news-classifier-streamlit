import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1️⃣ — Load your CSV
df = pd.read_csv('bbc-text.csv')
print("✅ Data loaded successfully!")
print(df.head())

# Step 2️⃣ — Split text & labels
X = df['text']
y = df['category']

# Step 3️⃣ — Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4️⃣ — TF-IDF vectorization
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,   # keep top 5000 most informative words
    ngram_range=(1,2)    # consider single words + bigrams
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("✅ TF-IDF matrix created!")
print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)
