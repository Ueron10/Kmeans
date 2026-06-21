import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(base_dir, 'docs', 'dataset', 'chronic_kidney_disease_full.csv'))

print("=== PREPROCESSING ===")
print("Original dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

numerical_cols = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
                  'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
                  'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 
                  'red_blood_cell_count']

for col in numerical_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

categorical_cols = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                    'appetite', 'pedal_edema', 'anemia']

for col in categorical_cols:
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

le_target = LabelEncoder()
df['class'] = le_target.fit_transform(df['class'].astype(str))
print("\nClass encoding:")
print(dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

df.to_csv(os.path.join(base_dir, 'data', 'chronic_kidney_disease_preprocessed.csv'), index=False)

joblib.dump(le_target, os.path.join(base_dir, 'data', 'label_encoder.pkl'))
joblib.dump(le_dict, os.path.join(base_dir, 'data', 'label_encoders.pkl'))

print("\nPreprocessing complete!")
print(f"Preprocessed dataset shape: {df.shape}")
print("\nFiles saved:")
print("- chronic_kidney_disease_preprocessed.csv")
print("- label_encoder.pkl")
print("- label_encoders.pkl")
