import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    df = pd.read_csv(input_path)
    print("Dataset berhasil dimuat.")

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Provinsi'] = le.fit_transform(df['Provinsi'])

    features = ['Provinsi', 'Tahun', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    scaler = StandardScaler()
    
    # Melakukan scaling pada fitur
    df[features] = scaler.fit_transform(df[features])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing selesai. File disimpan di: {output_path}")
    print(f"Daftar kolom yang diproses: {df.columns.tolist()}")

if __name__ == "__main__":
    input_file = 'Data_Tanaman_Padi_Sumatera_version_1.csv' 
    output_file = 'preprocessing\padi_preprocessing.csv'
    
    preprocess_data(input_file, output_file)