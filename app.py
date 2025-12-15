import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL & SCALER
# ======================
kmeans = joblib.load("kmeans_model.pkl")
scaler_kmeans = joblib.load("scaler_kmeans.pkl")
reg_model = joblib.load("reg_model.pkl")
scaler_reg = joblib.load("scaler_reg.pkl")

st.title("📊 Aplikasi Data Mining")
st.subheader("Clustering & Regresi Pelanggan Wholesale")

# ======================
# INPUT USER
# ======================
st.sidebar.header("Input Data Pelanggan")

milk = st.sidebar.number_input("Milk", 0, 200000, 1000)
grocery = st.sidebar.number_input("Grocery", 0, 200000, 1000)
frozen = st.sidebar.number_input("Frozen", 0, 100000, 500)
detergents = st.sidebar.number_input("Detergents_Paper", 0, 100000, 500)
delicassen = st.sidebar.number_input("Delicassen", 0, 100000, 300)

# ======================
# REGRESI
# ======================
input_reg = pd.DataFrame([{
    "Milk": milk,
    "Grocery": grocery,
    "Frozen": frozen,
    "Detergents_Paper": detergents,
    "Delicassen": delicassen
}])

input_reg_scaled = scaler_reg.transform(input_reg)
pred_fresh = reg_model.predict(input_reg_scaled)[0]

st.success(f"🧮 Prediksi Fresh: {pred_fresh:.2f}")

# ======================
# CLUSTERING
# ======================
input_cluster = pd.DataFrame([{
    "Fresh": pred_fresh,
    "Milk": milk,
    "Grocery": grocery,
    "Frozen": frozen,
    "Detergents_Paper": detergents,
    "Delicassen": delicassen
}])

cluster_scaled = scaler_kmeans.transform(input_cluster)
cluster = kmeans.predict(cluster_scaled)[0]

st.info(f"📌 Pelanggan termasuk dalam Cluster: {cluster}")

# ======================
# VISUALISASI REGRESI
# ======================
st.subheader("📈 Visualisasi Regresi")

fig, ax = plt.subplots()
ax.scatter(range(len(reg_model.coef_)), reg_model.coef_)
ax.set_xticks(range(len(reg_model.coef_)))
ax.set_xticklabels(input_reg.columns, rotation=45)
ax.set_title("Koefisien Regresi Linear")

st.pyplot(fig)
