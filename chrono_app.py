import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chrono_main import simulate_EirreEirre, simulate_EquasiEquasi

st.set_page_config(layout="wide")
st.title("📉 Chronoamperometry: EirreEirre vs EquasiEquasi")

# Sidebar inputs
with st.sidebar:
    E_appl = st.number_input("Applied potential (V)", value=-0.3, step=0.001, format="%.3f")
    duration = st.number_input("Duration (s)", value=2.0, step=0.1)
    dt = st.number_input("Time step (s)", value=0.01, step=0.001, format="%.3f")
    lambda1 = st.number_input("λ (eV)", value=0.5, step=0.01)
    k01 = st.number_input("k01 (s⁻¹)", value=0.1, step=0.01)
    k02 = st.number_input("k02 (s⁻¹)", value=0.1, step=0.01)
    E02 = st.number_input("E02 (V)", value=-0.2, step=0.001, format="%.3f")
    alpha = st.number_input("α", value=0.5, step=0.01)

# Run simulations
res1 = simulate_EirreEirre(E_appl, duration, dt, k01, k02, E02, alpha, lambda1 * 38.923074)
res2 = simulate_EquasiEquasi(E_appl, duration, dt, k01, k02, E02, lambda1 * 38.923074)

# Plot adimensional current Psi
st.subheader("Ψ(t*) - Adimensional current response")
fig, ax = plt.subplots()
ax.plot(res1.time_star, res1.psi, label="EirreEirre")
ax.plot(res2.time_star, res2.psi, "--", label="EquasiEquasi")
ax.set_xlabel("t / duration")
ax.set_ylabel("Ψ")
ax.legend()
st.pyplot(fig)

# Plot surface coverages
st.subheader("Surface excesses (EirreEirre)")
fig, ax = plt.subplots()
ax.plot(res1.time_star, res1.fO, label="fO")
ax.plot(res1.time_star, res1.fR, label="fR")
ax.plot(res1.time_star, res1.fI, label="fI")
ax.set_xlabel("t / duration")
ax.set_ylabel("Fractional coverage")
ax.legend()
st.pyplot(fig)

# Regression analysis of ln|Ψ|
st.subheader("Regression: ln|Ψ| vs t*")
fig, ax = plt.subplots()
mask1 = ~np.isnan(res1.ln_psi)
mask2 = ~np.isnan(res2.ln_psi)
ax.plot(res1.time_star[mask1], res1.ln_psi[mask1], label="EirreEirre")
ax.plot(res2.time_star[mask2], res2.ln_psi[mask2], "--", label="EquasiEquasi")

# Draw regression lines
line1 = res1.linreg["slope"] * res1.time_star + res1.linreg["intercept"]
line2 = res2.linreg["slope"] * res2.time_star + res2.linreg["intercept"]
ax.plot(res1.time_star, line1, "k:", alpha=0.3)
ax.plot(res2.time_star, line2, "k--", alpha=0.3)

ax.set_xlabel("t / duration")
ax.set_ylabel("ln(Ψ)")
ax.legend()
st.pyplot(fig)

# Regression results table
st.markdown("### Regression summary")
df = pd.DataFrame([
    {
        "Model": "EirreEirre",
        "Slope": res1.linreg["slope"],
        "Intercept": res1.linreg["intercept"],
        "R": res1.linreg["r"]
    },
    {
        "Model": "EquasiEquasi",
        "Slope": res2.linreg["slope"],
        "Intercept": res2.linreg["intercept"],
        "R": res2.linreg["r"]
    }
])
st.dataframe(df)

# Downloads
st.subheader("📥 Download Ψ vs t* data")

def download_txt(label, filename, header, data):
    import io
    buf = io.StringIO()
    np.savetxt(buf, data, header=header)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/plain")

download_txt("Download EirreEirre Ψ", "EirreEirre_Psi.txt", "t*\tPsi", np.column_stack((res1.time_star, res1.psi)))
download_txt("Download EquasiEquasi Ψ", "EquasiEquasi_Psi.txt", "t*\tPsi", np.column_stack((res2.time_star, res2.psi)))