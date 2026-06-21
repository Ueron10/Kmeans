import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'data', 'chronic_kidney_disease_normalized.csv'))

selected_features = ['serum_creatinine', 'blood_urea', 'hemoglobin']
X = df[selected_features]

print("=== DETERMINE K (ELBOW METHOD) ===")
print("Dataset shape:", X.shape)

inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f'k={k}: Inertia={kmeans.inertia_:.2f}')

inertia_drops = []
for i in range(1, len(inertias)):
    drop = inertias[i-1] - inertias[i]
    inertia_drops.append(drop)
    print(f'Drop from k={i+1} to k={i+2}: {drop:.2f}')

optimal_k = k_range[np.argmax(inertia_drops) + 1]
print(f'\nOptimal k based on elbow method: {optimal_k}')

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.savefig(os.path.join(base_dir, 'results', 'elbow_method.png'))
print("\nElbow method plot saved to: results/elbow_method.png")

with open(os.path.join(base_dir, 'data', 'optimal_k.txt'), 'w') as f:
    f.write(str(optimal_k))

print(f"Optimal k saved to: data/optimal_k.txt")
