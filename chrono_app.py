import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from chrono_main import chrono_sim, FRT

st.set_page_config(layout="wide")
st.title("Simulador de Cronoamperometría (MH vs BV)")

with st.sidebar:
    E_appl = st.number_input("Potencial aplicado (V)", value=-0.5)
    t_total = st.number_input("Duración total (s)", value=5.0)
    dt = st.number_input("Paso de tiempo (s)", value=0.01, format="%.3f")
    lambda1 = st.slider("λ (eV)", 0.1, 2.0, 0.5, 0.1)
    k01 = st.number_input("k01 (s⁻¹)", 0.1)
    k02 = st.number_input("k02 (s⁻¹)", 0.1)
    E02 = st.number_input("E02 (V)", -0.25)
    alpha = st.slider("α", 0.1, 1.0, 0.5)

res = chrono_sim(E_appl, t_total, dt, k01, k02, E02, alpha, lambda1 * FRT)

st.subheader("Corriente vs Tiempo")
fig, ax = plt.subplots()
ax.plot(res.time, res.I_MH, label="MH")
ax.plot(res.time, res.I_BV, "--", label="BV")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Corriente (Ψ)")
ax.legend()
st.pyplot(fig)

st.subheader("Excesos superficiales (fO, fR, fI)")
fig, ax = plt.subplots()
ax.plot(res.time, res.fO_MH, label="fO (MH)")
ax.plot(res.time, res.fR_MH, label="fR (MH)")
ax.plot(res.time, res.fI_MH, label="fI (MH)")
ax.plot(res.time, res.fO_BV, "--", label="fO (BV)")
ax.plot(res.time, res.fR_BV, "--", label="fR (BV)")
ax.plot(res.time, res.fI_BV, "--", label="fI (BV)")
ax.legend()
st.pyplot(fig)

def download_txt(label, filename, header, data):
    buf = io.StringIO()
    np.savetxt(buf, data, header=header)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/plain")

st.subheader("📥 Descargar datos en .txt")

download_txt(
    "Descargar corriente (MH)",
    "corriente_MH.txt",
    "tiempo (s)	Ψ_MH",
    np.column_stack((res.time, res.I_MH)),
)

download_txt(
    "Descargar corriente (BV)",
    "corriente_BV.txt",
    "tiempo (s)	Ψ_BV",
    np.column_stack((res.time, res.I_BV)),
)

download_txt(
    "Descargar excesos superficiales (MH)",
    "superficie_MH.txt",
    "tiempo (s)	fO	fR	fI",
    np.column_stack((res.time, res.fO_MH, res.fR_MH, res.fI_MH)),
)

download_txt(
    "Descargar excesos superficiales (BV)",
    "superficie_BV.txt",
    "tiempo (s)	fO	fR	fI",
    np.column_stack((res.time, res.fO_BV, res.fR_BV, res.fI_BV)),
)