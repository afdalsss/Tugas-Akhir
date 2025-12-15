import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===============================
# LOAD MODEL, SCALER, DATASET
# ===============================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler_kmeans.pkl")

df = pd.read_csv("Wholesale_customers.csv")

features = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"
]

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="Clustering Pelanggan", layout="wide")

st.title("📊 Clustering Pelanggan Wholesale")
st.write(
    """
    Aplikasi ini mengelompokkan pelanggan berdasarkan pola pembelian
    menggunakan metode **K-Means Clustering**.
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
# SIDEBAR INPUT
# ===============================
st.sidebar.header("🔢 Input Data Pelanggan")

fresh = st.sidebar.number_input("Fresh", min_value=0, value=5000)
milk = st.sidebar.number_input("Milk", min_value=0, value=3000)
grocery = st.sidebar.number_input("Grocery", min_value=0, value=4000)
frozen = st.sidebar.number_input("Frozen", min_value=0, value=2000)
detergents = st.sidebar.number_input("Detergents_Paper", min_value=0, value=1000)
delicassen = st.sidebar.number_input("Delicassen", min_value=0, value=800)

# ===============================
# PREDIKSI
# ===============================
if st.sidebar.button("🔍 Prediksi Cluster"):
    input_data = pd.DataFrame([[
        fresh, milk, grocery, frozen, detergents, delicassen
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    # ===============================
    # OUTPUT TEKS
    # ===============================
    st.success(f"📌 Hasil Prediksi: **Cluster {cluster}**")
    st.info(f"🧠 Karakteristik: {cluster_desc.get(cluster)}")

    st.subheader("📌 Kesimpulan")
    st.write(
        f"""
        Berdasarkan nilai pembelian yang dimasukkan,
        pelanggan ini termasuk dalam **{cluster_desc.get(cluster)}**.
        Hasil ini dapat digunakan sebagai dasar
        pengambilan keputusan strategi pemasaran.
        """
    )

    # ===============================
    # VISUALISASI CLUSTER (PCA)
    # ===============================
    st.subheader("📈 Visualisasi Clustering (PCA 2D)")

    # Scaling seluruh dataset
    X_scaled = scaler.transform(df[features])

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # PCA input user
    input_pca = pca.transform(input_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans.labels_,
        cmap="viridis",
        alpha=0.6
    )

    ax.scatter(
        input_pca[:, 0],
        input_pca[:, 1],
        c="red",
        s=200,
        marker="X",
        label="Input Pelanggan"
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Visualisasi Cluster Pelanggan")
    ax.legend()

    st.pyplot(fig)

    # ===============================
    # TAMPILKAN DATA INPUT
    # ===============================
    st.subheader("📊 Data Input Pelanggan")
    st.dataframe(input_data)
