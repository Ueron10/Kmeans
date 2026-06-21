import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'data', 'chronic_kidney_disease_normalized.csv'))

selected_features = ['serum_creatinine', 'blood_urea', 'hemoglobin']
X = df[selected_features]
y = df['class']

print("=== K-MEANS CLUSTERING ===")
print("Dataset shape:", X.shape)

with open(os.path.join(base_dir, 'data', 'optimal_k.txt'), 'r') as f:
    optimal_k = int(f.read().strip())

print(f"Using k={optimal_k} based on elbow method")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X)

clusters = kmeans.predict(X)

joblib.dump(kmeans, os.path.join(base_dir, 'models', 'kmeans_model.pkl'))
print("\nKMeans model saved to: models/kmeans_model.pkl")

results = X.copy()
results['actual_class'] = y.values.flatten()
results['cluster'] = clusters
results.to_csv(os.path.join(base_dir, 'results', 'clusters.csv'), index=False)

print("Cluster assignments saved to: results/clusters.csv")
print(f"Cluster distribution:")
print(pd.Series(clusters).value_counts().sort_index())

print("\nK-Means clustering complete!")
