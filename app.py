import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
model = joblib.load('mlp_sklearn_model.pkl')

# Judul aplikasi
st.title("🌱 Aplikasi Klasifikasi Jenis Tanaman Pangan")
st.write("Masukkan data karakteristik lahan untuk memprediksi jenis tanaman yang optimal.")

# Input form
ph = st.slider("pH Tanah", 4.5, 8.5, 6.5)
nitrogen = st.slider("Kandungan Nitrogen (ppm)", 30, 60, 45)
phosphor = st.slider("Kandungan Fosfor (ppm)", 20, 40, 30)
potassium = st.slider("Kandungan Kalium (ppm)", 20, 35, 28)
tekstur_tanah = st.selectbox("Tekstur Tanah", ['berpasir', 'lempung', 'liat'])
suhu = st.slider("Suhu Rata-rata (°C)", 25.0, 32.0, 28.0)
curah_hujan = st.slider("Curah Hujan (mm)", 900, 1700, 1200)
ketinggian = st.slider("Ketinggian Wilayah (mdpl)", 50, 500, 100)
intensitas_cahaya = st.slider("Intensitas Cahaya (jam/hari)", 80, 200, 120)

# Buat DataFrame untuk input
input_data = pd.DataFrame([{
    'ph': ph,
    'nitrogen': nitrogen,
    'phosphor': phosphor,
    'potassium': potassium,
    'tekstur_tanah': tekstur_tanah,
    'suhu': suhu,
    'curah_hujan': curah_hujan,
    'ketinggian': ketinggian,
    'intensitas_cahaya': intensitas_cahaya
}])

# Tombol prediksi
if st.button("🔍 Prediksi Jenis Tanaman"):
    prediction = model.predict(input_data)
    predicted_label = prediction[0]
    st.success(f"🌾 Jenis tanaman yang direkomendasikan: **{predicted_label}**")

    # Tambah hasil prediksi ke DataFrame input
    input_data['hasil_prediksi'] = predicted_label

    # Simpan ke riwayat file
    riwayat_file = 'riwayat_prediksi.csv'
    if os.path.exists(riwayat_file):
        riwayat_df = pd.read_csv(riwayat_file)
        riwayat_df = pd.concat([riwayat_df, input_data], ignore_index=True)
    else:
        riwayat_df = input_data

    riwayat_df.to_csv(riwayat_file, index=False)
    st.success("✅ Riwayat prediksi berhasil disimpan ke *riwayat_prediksi.csv*!")

    # Tampilkan riwayat terakhir
    st.subheader("📜 Riwayat Prediksi Terbaru")
    st.dataframe(riwayat_df.tail(5))
