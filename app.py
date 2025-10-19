import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import pickle

# ---- Load and train model once ----
@st.cache_resource
def load_model():
    df = pd.read_csv('bbc-text.csv')
    X = df['text']
    y = df['category']

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # PCA for dimensionality reduction
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Train model
    model = LinearSVC()
    model.fit(X_pca, y)

    return tfidf, pca, model

tfidf, pca, model = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")

st.title("📰 News Category Classifier")
st.write("Enter a news article and find out which category it belongs to!")

user_input = st.text_area("✍️ Paste your news article here:")

if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a news article to classify.")
    else:
        # Transform input
        X_new_tfidf = tfidf.transform([user_input])
        X_new_pca = pca.transform(X_new_tfidf.toarray())
        pred = model.predict(X_new_pca)[0]
        st.success(f"🧠 Predicted Category: **{pred.upper()}**")

