from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score

app = Flask(__name__)

def load_model_and_data():
    try:
        kmeans = joblib.load('models/kmeans_model.pkl')
        df = pd.read_csv('data/chronic_kidney_disease_normalized.csv')
        scaler = joblib.load('data/scaler_3features.pkl')
        le = joblib.load('data/label_encoder.pkl')
        clusters = pd.read_csv('results/clusters.csv')
        return kmeans, df, scaler, le, clusters
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None, None, None

kmeans, df, scaler, le, clusters = load_model_and_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        serum_creatinine = float(request.form['serum_creatinine'])
        blood_urea = float(request.form['blood_urea'])
        hemoglobin = float(request.form['hemoglobin'])
        input_data = np.array([[serum_creatinine, blood_urea, hemoglobin]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        cluster_data = clusters[clusters['cluster'] == cluster]
        cluster_interpretations = {
            0: "Fungsi Ginjal Sedang - Pasien dengan indikator fungsi ginjal relatif normal (kreatinin dan urea rendah, hemoglobin normal)",
            1: "Penyakit Ginjal Parah - Pasien dengan kreatinin dan urea yang sangat tinggi, menunjukkan disfungsi ginjal lanjut",
            2: "Penyakit Ginjal Ringan - Pasien dengan kreatinin dan urea yang sedikit tinggi, dan hemoglobin rendah yang menunjukkan masalah ginjal tahap awal"
        }
        stats = {
            'cluster': int(cluster),
            'serum_creatinine_mean': float(cluster_data['serum_creatinine'].mean()),
            'blood_urea_mean': float(cluster_data['blood_urea'].mean()),
            'hemoglobin_mean': float(cluster_data['hemoglobin'].mean()),
            'cluster_size': len(cluster_data),
            'interpretation': cluster_interpretations.get(int(cluster), "Cluster tidak diketahui")
        }
        return jsonify({'success': True, 'result': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cluster_info')
def cluster_info():
    try:
        if clusters is None:
            return jsonify({'success': False, 'error': 'Data cluster tidak dimuat'})
        cluster_info = []
        cluster_interpretations = {
            0: "Fungsi Ginjal Sedang - Pasien dengan indikator fungsi ginjal relatif normal (kreatinin dan urea rendah, hemoglobin normal)",
            1: "Penyakit Ginjal Parah - Pasien dengan kreatinin dan urea yang sangat tinggi, menunjukkan disfungsi ginjal lanjut",
            2: "Penyakit Ginjal Ringan - Pasien dengan kreatinin dan urea yang sedikit tinggi, dan hemoglobin rendah yang menunjukkan masalah ginjal tahap awal"
        }
        for cluster in sorted(clusters['cluster'].unique()):
            cluster_data = clusters[clusters['cluster'] == cluster]
            info = {
                'cluster': int(cluster),
                'size': len(cluster_data),
                'serum_creatinine_mean': float(cluster_data['serum_creatinine'].mean()),
                'blood_urea_mean': float(cluster_data['blood_urea'].mean()),
                'hemoglobin_mean': float(cluster_data['hemoglobin'].mean()),
                'interpretation': cluster_interpretations.get(int(cluster), "Cluster tidak diketahui")
            }
            cluster_info.append(info)
        return jsonify({'success': True, 'clusters': cluster_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_evaluation')
def model_evaluation():
    try:
        if clusters is None:
            return jsonify({'success': False, 'error': 'Data cluster tidak dimuat'})
        X = clusters[['serum_creatinine', 'blood_urea', 'hemoglobin']].values
        cluster_labels = clusters['cluster'].values
        actual_labels = clusters['actual_class'].values
        silhouette = silhouette_score(X, cluster_labels)
        db_index = davies_bouldin_score(X, cluster_labels)
        ch_index = calinski_harabasz_score(X, cluster_labels)
        ari = adjusted_rand_score(actual_labels, cluster_labels)
        nmi = normalized_mutual_info_score(actual_labels, cluster_labels)
        centroids = {}
        wcss = 0
        for cluster in np.unique(cluster_labels):
            cluster_data = X[cluster_labels == cluster]
            centroid = cluster_data.mean(axis=0)
            centroids[cluster] = centroid
            wcss += np.sum((cluster_data - centroid) ** 2)
        global_centroid = X.mean(axis=0)
        bcss = 0
        for cluster in np.unique(cluster_labels):
            cluster_data = X[cluster_labels == cluster]
            cluster_size = len(cluster_data)
            bcss += cluster_size * np.sum((centroids[cluster] - global_centroid) ** 2)
        tss = wcss + bcss
        bcss_ratio = bcss / tss if tss > 0 else 0
        from scipy.spatial.distance import euclidean
        inter_cluster_distances = []
        for i in centroids:
            for j in centroids:
                if i < j:
                    dist = euclidean(centroids[i], centroids[j])
                    inter_cluster_distances.append({
                        'pair': f'{i}-{j}',
                        'distance': float(dist)
                    })
        evaluation = {
            'internal_metrics': {
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(db_index),
                'calinski_harabasz_index': float(ch_index)
            },
            'external_metrics': {
                'adjusted_rand_index': float(ari),
                'normalized_mutual_info': float(nmi)
            },
            'variance_analysis': {
                'wcss': float(wcss),
                'bcss': float(bcss),
                'tss': float(tss),
                'bcss_ratio': float(bcss_ratio)
            },
            'inter_cluster_distances': inter_cluster_distances,
            'centroids': {
                str(int(k)): [float(v) for v in centroids[k]] for k in centroids
            }
        }
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/results/<filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
