import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
with open('model.pkl', 'rb') as file:
    models = pickle.load(file)

model = models['rf_model']  # Menggunakan Random Forest sebagai contoh

st.title('Prediksi Kepuasan Pelanggan')

# CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #333;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.title('Prediksi Kepuasan Pelanggan')

st.markdown("""
    <div class="main">
    <h3>Masukkan Data Pelanggan</h3>
    </div>
""", unsafe_allow_html=True)

# Input pengguna
age = st.number_input('Umur', min_value=0, max_value=100, step=1)
gender = st.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
income = st.number_input('Pendapatan Tahunan', min_value=0)
score = st.number_input('Skor Pengeluaran', min_value=0, max_value=100, step=1)

# Konversi input ke dalam bentuk yang sesuai untuk model
gender_encoded = 1 if gender == 'Pria' else 0
features = np.array([[age, gender_encoded, income, score]])

if st.button('Prediksi'):
    prediction = model.predict(features)
    output = 'Puas' if prediction[0] == 1 else 'Tidak Puas'
    st.write(f'Prediksi Kepuasan Pelanggan: {output}')

st.markdown("""
    <h3>Output Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
