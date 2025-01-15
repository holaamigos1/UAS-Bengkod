import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Judul Aplikasi
st.title("Prediksi Harga Rumah")

# Upload Dataset
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel('DATA RUMAH.xlsx')
    st.write("Data yang diunggah:", data.head())

    # Pilih fitur dan target
    X = data[['LB', 'LT', 'KT', 'KM', 'GRS']]
    y = data['HARGA']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Inisialisasi model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Tampilkan hasil evaluasi
    st.write("**Evaluasi Model:**")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
