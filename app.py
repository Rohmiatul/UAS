import streamlit as st
import pandas as pd
import joblib # Atau import joblib jika model Anda disimpan dengan joblib

# --- Memuat Model KNN ---
try:
    # Ganti 'knn_model.pkl' dengan nama file model KNN Anda
    with open('knn_model.pkl', 'rb') as file:
        knn = joblib.load(file)
except FileNotFoundError:
    st.error("File 'knn_model.pkl' tidak ditemukan. Pastikan model berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Judul Aplikasi Streamlit ---
st.title('Aplikasi Prediksi Personality')
st.write('Aplikasi ini memprediksi Personality (Extrovert/Introvert) berdasarkan data aktivitas sosial.')

# --- Input dari Pengguna ---
st.sidebar.header('Input Data Baru')
new_Social_event_attendance = st.sidebar.slider(
    'Jumlah Kegiatan Sosial (Social Event Attendance)',
    min_value=0, max_value=20, value=5, step=1
)
new_Going_outside = st.sidebar.slider(
    'Jumlah Kegiatan Bermain di Luar (Going Outside)',
    min_value=0, max_value=20, value=5, step=1
)
new_Friends_circle_size = st.sidebar.slider(
    'Jumlah Pertemanan (Friends Circle Size)',
    min_value=0, max_value=50, value=10, step=1
)

# --- Menampilkan Input yang Diterima ---
st.write("**Data yang dimasukkan:**")
st.write(f"- Kegiatan Sosial: {new_Social_event_attendance}")
st.write(f"- Kegiatan di Luar: {new_Going_outside}")
st.write(f"- Jumlah Pertemanan: {new_Friends_circle_size}")

# --- Tombol untuk Melakukan Prediksi ---
if st.button('Prediksi Personality'):
    try:
        # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
        new_data_df = pd.DataFrame(
            [[new_Social_event_attendance, new_Going_outside, new_Friends_circle_size]],
            columns=['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        )

        # Lakukan prediksi
        predicted_code = knn.predict(new_data_df)[0] # hasilnya 0 atau 1

        # Konversi hasil prediksi ke label asli
        label_mapping = {1: 'Extrovert', 0: 'Introvert'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"**Prediksi Personality adalah: {predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")