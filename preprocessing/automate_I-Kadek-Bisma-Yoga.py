import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' tidak ditemukan!")
        return

    df = pd.read_csv(input_path)
    
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    if 'Provinsi' in df.columns:
        df['Provinsi'] = le.fit_transform(df['Provinsi'])

    features = ['Provinsi', 'Tahun', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    available_features = [f for f in features if f in df.columns]
    
    scaler = StandardScaler()
    df[available_features] = scaler.fit_transform(df[available_features])

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_path, index=False)
    print(f"Berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    input_file = 'Data_Tanaman_Padi_Sumatera_version_1.csv' 
    output_file = 'preprocessing/padi_preprocessing.csv'
    preprocess_data(input_file, output_file)