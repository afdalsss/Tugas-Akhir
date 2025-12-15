import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL & SCALER
# ===============================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler_kmeans.pkl")

st.set_page_config(page_title="Clustering Pelanggan", layout="centered")

st.title("📊 Aplikasi Clustering Pelanggan Wholesale")
st.write(
    """
    Aplikasi ini digunakan untuk mengelompokkan pelanggan berdasarkan
    pola pembelian produk menggunakan metode **K-Means Clustering**.
    """
)

# ===============================
# DEFINISI CLUSTER
# ===============================
cluster_desc = {
    0: "Pelanggan dengan tingkat pembelian rendah",
    1: "Pelanggan dengan tingkat pembelian menengah",
    2: "Pelanggan dengan tingkat pembelian tinggi"
}

# ===============================
# INPUT USER
# ===============================
st.sidebar.header("🔢 Input Data Pembelian Pelanggan")

fresh = st.sidebar.number_input("Fresh", min_value=0, value=5000)
milk = st.sidebar.number_input("Milk", min_value=0, value=3000)
grocery = st.sidebar.number_input("Grocery", min_value=0, value=4000)
frozen = st.sidebar.number_input("Frozen", min_value=0, value=2000)
detergents = st.sidebar.number_input("Detergents_Paper", min_value=0, value=1000)
delicassen = st.sidebar.number_input("Delicassen", min_value=0, value=800)

# ===============================
# PREDIKSI
# ===============================
if st.button("🔍 Prediksi Cluster"):
    input_data = pd.DataFrame([{
        "Fresh": fresh,
        "Milk": milk,
        "Grocery": grocery,
        "Frozen": frozen,
        "Detergents_Paper": detergents,
        "Delicassen": delicassen
    }])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi cluster
    cluster = kmeans.predict(input_scaled)[0]

    # ===============================
    # OUTPUT
    # ===============================
    st.success(f"📌 Hasil Prediksi Cluster: **Cluster {cluster}**")
    st.info(f"🧠 Karakteristik Cluster: {cluster_desc.get(cluster)}")

    st.subheader("📌 Kesimpulan")
    st.write(
        f"""
        Berdasarkan data pembelian yang dimasukkan,
        pelanggan ini termasuk dalam **{cluster_desc.get(cluster)}**.
        Informasi ini dapat digunakan sebagai dasar untuk
        **strategi pemasaran dan pengelolaan pelanggan**.
        """
    )

    st.subheader("📊 Data Input")
    st.dataframe(input_data)
