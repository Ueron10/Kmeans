import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from scipy.spatial.distance import euclidean
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'results', 'clusters.csv'))
X = df[['serum_creatinine', 'blood_urea', 'hemoglobin']].values
clusters = df['cluster'].values
actual = df['actual_class'].values

silhouette = silhouette_score(X, clusters)
db = davies_bouldin_score(X, clusters)
ch = calinski_harabasz_score(X, clusters)
cm = confusion_matrix(actual, clusters)
precision = precision_score(actual, clusters, average='macro', zero_division=0)
recall = recall_score(actual, clusters, average='macro', zero_division=0)
f1 = f1_score(actual, clusters, average='macro', zero_division=0)

centroids = {}
for cluster in np.unique(clusters):
    cluster_data = X[clusters == cluster]
    centroids[cluster] = cluster_data.mean(axis=0)

wcss = 0
for cluster in np.unique(clusters):
    cluster_data = X[clusters == cluster]
    centroid = centroids[cluster]
    wcss += np.sum((cluster_data - centroid) ** 2)

global_centroid = X.mean(axis=0)
bcss = 0
for cluster in np.unique(clusters):
    cluster_data = X[clusters == cluster]
    cluster_size = len(cluster_data)
    bcss += cluster_size * np.sum((centroids[cluster] - global_centroid) ** 2)

tss = wcss + bcss
ratio = bcss / tss
