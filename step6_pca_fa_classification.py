# 📘 News Article Classification using PCA vs Factor Analysis
# Author: Vas
# Project: BBC News Classification Comparison
# Saves results and plots automatically

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Create output folder for plots and CSV
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Step 1️⃣ — Load data
df = pd.read_csv('bbc-text.csv')
print("✅ Data Loaded Successfully!")
print(df.head())

X = df['text']
y = df['category']

# Step 2️⃣ — TF-IDF vectorization
print("\n🔹 Creating TF-IDF matrix...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)
print("✅ TF-IDF matrix created!")
print("Shape:", X_tfidf.shape)

# Step 3️⃣ — Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data split completed!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Step 4️⃣ — Apply PCA & Factor Analysis
components = [10, 25, 50, 100]
results = []

for n in components:
    print(f"\n🔹 Testing with {n} components...")

    # PCA
    pca = PCA(n_components=n, random_state=42)
    start = time.time()
    X_train_pca = pca.fit_transform(X_train.toarray())
    X_test_pca = pca.transform(X_test.toarray())
    pca_time = time.time() - start

    # Factor Analysis
    fa = FactorAnalysis(n_components=n, random_state=42)
    start = time.time()
    X_train_fa = fa.fit_transform(X_train.toarray())
    X_test_fa = fa.transform(X_test.toarray())
    fa_time = time.time() - start

    # Models
    models = {
        'SVM': LinearSVC(),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        # PCA
        model.fit(X_train_pca, y_train)
        y_pred_pca = model.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, y_pred_pca) * 100  # convert to percentage

        # FA
        model.fit(X_train_fa, y_train)
        y_pred_fa = model.predict(X_test_fa)
        acc_fa = accuracy_score(y_test, y_pred_fa) * 100  # convert to percentage

        results.append({
            'Model': name,
            'Components': n,
            'PCA Accuracy (%)': round(acc_pca, 2),
            'FA Accuracy (%)': round(acc_fa, 2),
            'PCA Time (s)': round(pca_time, 2),
            'FA Time (s)': round(fa_time, 2)
        })

# Step 5️⃣ — Results DataFrame
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(output_dir, "results_summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n✅ Results Summary saved as {results_csv_path}")
print(results_df)

# Step 6️⃣ — Accuracy Comparison Plot
plt.figure(figsize=(10,6))
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    plt.plot(subset['Components'], subset['PCA Accuracy (%)'], marker='o', label=f'{model} - PCA')
    plt.plot(subset['Components'], subset['FA Accuracy (%)'], marker='x', linestyle='--', label=f'{model} - FA')

plt.title('PCA vs Factor Analysis - Accuracy Comparison')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
acc_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
plt.savefig(acc_plot_path)
plt.show()
print(f"✅ Accuracy plot saved as {acc_plot_path}")

# Step 7️⃣ — Final Visualization & Insights

print("\n📊 Step 7: Visualizing Explained Variance and Summarizing Results...\n")

# Explained variance for PCA
pca_full = PCA(n_components=100, random_state=42)
pca_full.fit(X_train.toarray())

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_) * 100, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA – Explained Variance Curve')
plt.grid(True)
variance_plot_path = os.path.join(output_dir, "explained_variance.png")
plt.savefig(variance_plot_path)
plt.show()
print(f"✅ Explained variance plot saved as {variance_plot_path}")

# Accuracy comparison again (bar style)
plt.figure(figsize=(10,6))
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    plt.plot(subset['Components'], subset['PCA Accuracy (%)'], marker='o', label=f'{model} (PCA)')
    plt.plot(subset['Components'], subset['FA Accuracy (%)'], marker='x', linestyle='--', label=f'{model} (FA)')

plt.title('Accuracy Comparison: PCA vs Factor Analysis')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
bar_plot_path = os.path.join(output_dir, "accuracy_bar_comparison.png")
plt.savefig(bar_plot_path)
plt.show()
print(f"✅ Bar-style accuracy comparison saved as {bar_plot_path}")

# Auto summary
best_pca = results_df.loc[results_df['PCA Accuracy (%)'].idxmax()]
best_fa = results_df.loc[results_df['FA Accuracy (%)'].idxmax()]

print("🧠 SUMMARY INSIGHTS:")
print(f"• Best PCA → {best_pca['Model']} | {best_pca['Components']} comps | Acc = {best_pca['PCA Accuracy (%)']:.4f}%")
print(f"• Best FA  → {best_fa['Model']} | {best_fa['Components']} comps | Acc = {best_fa['FA Accuracy (%)']:.4f}%")

if best_pca['PCA Accuracy (%)'] > best_fa['FA Accuracy (%)']:
    print("\n✅ PCA performed slightly better overall in accuracy and efficiency.")
else:
    print("\n✅ Factor Analysis performed comparably or slightly better in some models.")

print("\n💡 Notes:")
print("• PCA tends to preserve more variance and works better for sparse text features.")
print("• FA identifies latent semantic patterns but is slower.")
print("• Around 50 components usually balance performance and accuracy.")
