import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'data', 'chronic_kidney_disease_preprocessed.csv'))

print("=== NORMALIZATION ===")
print("Dataset shape:", df.shape)

numerical_cols = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
                  'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
                  'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 
                  'red_blood_cell_count']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.to_csv(os.path.join(base_dir, 'data', 'chronic_kidney_disease_normalized.csv'), index=False)

joblib.dump(scaler, os.path.join(base_dir, 'data', 'scaler.pkl'))

feature_cols = ['serum_creatinine', 'blood_urea', 'hemoglobin']
scaler_3features = StandardScaler()
scaler_3features.fit(df[feature_cols])
joblib.dump(scaler_3features, os.path.join(base_dir, 'data', 'scaler_3features.pkl'))

print("\nNormalization complete!")
print(f"Normalized dataset shape: {df.shape}")
print("\nFiles saved:")
print("- chronic_kidney_disease_normalized.csv")
print("- scaler.pkl (full dataset)")
print("- scaler_3features.pkl (3 features for KMeans)")
