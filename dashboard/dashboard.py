import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression

# Judul dan deskripsi
st.title("Proyek Analisis Data: Bike Sharing Dataset")
st.write("Nama: Rosema Dina Apriliani")
st.write("Email: m686b4kx3967@bangkit.academy")
st.write("ID Dicoding: m686b4kx3967")

# Mengambil data
df = pd.read_csv('day.csv')
st.write(df.head())

# Analisis distribusi jumlah penyewaan sepeda
st.subheader('Analisis Distribusi Jumlah Penyewaan Sepeda')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['cnt'], bins=30, kde=True, ax=ax)
ax.set_title('Distribusi Penyewaan Sepeda Harian')
st.pyplot(fig)

# Menghapus kolom yang tidak relevan untuk analisis korelasi
df_numeric = df.select_dtypes(include=[np.number])  # Mengambil kolom numerik

# Analisis korelasi
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriks Korelasi Variabel')
st.pyplot(fig)

st.markdown("""
**Insight:**
- Terdapat hubungan positif yang kuat antara jumlah penyewaan sepeda dengan variabel cuaca dan musim.
- Kelembaban dan kecepatan angin tampak memiliki pengaruh negatif terhadap jumlah penyewaan.
""")

# Visualization & Explanatory Analysis
st.subheader('Visualization & Explanatory Analysis')

# Visualisasi pengaruh musim terhadap jumlah penyewaan sepeda
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='season', y='cnt', data=df, ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Musim')
ax.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
st.pyplot(fig)

# Visualisasi pengaruh cuaca terhadap jumlah penyewaan sepeda
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=df, ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Cuaca')
ax.set_xticklabels(['Clear', 'Mist', 'Light Rain/Snow', 'Heavy Rain/Snow'])
st.pyplot(fig)

# Visualisasi pengaruh hari kerja terhadap penyewaan sepeda
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='workingday', y='cnt', data=df, ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Hari Kerja vs Akhir Pekan')
ax.set_xticklabels(['Akhir Pekan/Hari Libur', 'Hari Kerja'])
st.pyplot(fig)

st.markdown("""
**Insight:**
- Penyewaan sepeda tertinggi terjadi pada musim panas dan terendah di musim dingin.
- Cuaca cerah mendorong penyewaan lebih tinggi, sementara cuaca buruk seperti hujan atau salju menurunkan penyewaan.
- Jumlah penyewaan sepeda lebih tinggi pada akhir pekan dibandingkan dengan hari kerja.
""")

# Analisis Lanjutan (Opsional)
st.subheader('Analisis Lanjutan (Opsional)')

# Model sederhana untuk memprediksi jumlah penyewaan sepeda
X = df[['temp', 'hum', 'windspeed']]
y = df['cnt']

model = LinearRegression()
model.fit(X, y)

# Prediksi dan visualisasi hasil
predictions = model.predict(X)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y, predictions)
ax.set_xlabel('Actual Rentals')
ax.set_ylabel('Predicted Rentals')
ax.set_title('Prediksi vs Realitas Penyewaan Sepeda')
st.pyplot(fig)

st.markdown("""
## Conclusion
- Conclusion pertanyaan 1: Cuaca cerah dan musim panas cenderung meningkatkan jumlah penyewaan sepeda. Sebaliknya, cuaca buruk dan musim dingin menurunkan penyewaan.
- Conclusion pertanyaan 2: Penyewaan sepeda lebih tinggi pada akhir pekan dan hari libur dibandingkan hari kerja, menunjukkan bahwa orang lebih sering menggunakan sepeda untuk rekreasi saat waktu senggang.
""")
