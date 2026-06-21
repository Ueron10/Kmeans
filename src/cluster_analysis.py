import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'results', 'clusters.csv'))

print("=== CLUSTER ANALYSIS ===")
print("Dataset shape:", df.shape)

X = df[['serum_creatinine', 'blood_urea', 'hemoglobin']]
y = df['actual_class']
clusters = df['cluster']

ari = adjusted_rand_score(y.values.flatten(), clusters)
nmi = normalized_mutual_info_score(y.values.flatten(), clusters)

print(f"\nAdjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")

print("\nCluster Statistics:")
for cluster in sorted(clusters.unique()):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(cluster_data)}")
    print(f"  Avg Serum Creatinine: {cluster_data['serum_creatinine'].mean():.4f}")
    print(f"  Avg Blood Urea: {cluster_data['blood_urea'].mean():.4f}")
    print(f"  Avg Hemoglobin: {cluster_data['hemoglobin'].mean():.4f}")

cluster_interpretations = {
    0: "Moderate Kidney Function - Patients with relatively normal kidney function indicators (low creatinine and urea, normal hemoglobin)",
    1: "Severe Kidney Disease - Patients with significantly elevated creatinine and urea, indicating advanced kidney dysfunction",
    2: "Mild Kidney Disease - Patients with slightly elevated creatinine and urea, and low hemoglobin indicating early-stage kidney issues"
}

print("\nCluster Interpretations:")
for cluster, interpretation in cluster_interpretations.items():
    print(f"Cluster {cluster}: {interpretation}")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], 
                       c=y.values.flatten(), cmap='viridis', alpha=0.6)
ax1.set_xlabel('serum_creatinine')
ax1.set_ylabel('blood_urea')
ax1.set_zlabel('hemoglobin')
ax1.set_title('Data (Actual Labels)')
plt.colorbar(scatter1, ax=ax1)

ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], 
                       c=clusters, cmap='viridis', alpha=0.6)
ax2.set_xlabel('serum_creatinine')
ax2.set_ylabel('blood_urea')
ax2.set_zlabel('hemoglobin')
ax2.set_title('Data (KMeans Clusters)')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'results', 'clusters_3d.png'))
print("\n3D cluster visualization saved to: results/clusters_3d.png")

print("\nCluster analysis complete!")
