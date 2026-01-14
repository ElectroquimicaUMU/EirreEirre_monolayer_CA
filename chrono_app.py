
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from chrono_main_with_regression import chrono_sim_with_regression

st.set_page_config(layout="wide", page_title="Monolayer Chronoamperometry")

st.title("⏱️ Chronoamperometry Simulation — Monolayer System")

# ---- Input parameters ----
st.sidebar.header("🔧 Simulation Parameters")

E_appl = st.sidebar.number_input("Applied Potential (V)", value=-0.3, format="%.3f")
duration = st.sidebar.number_input("Total Time (s)", value=10.0)
dt = st.sidebar.number_input("Time Step (s)", value=0.01)
k01 = st.sidebar.number_input("k₀₁ (s⁻¹)", value=0.1, format="%.3e")
k02 = st.sidebar.number_input("k₀₂ (s⁻¹)", value=0.1, format="%.3e")
E02 = st.sidebar.number_input("E₀₂ (V)", value=-0.25, format="%.3f")
lambda1 = st.sidebar.number_input("λ (Marcus parameter, eV)", value=0.5)
alpha = st.sidebar.slider("α (BV only)", min_value=0.0, max_value=1.0, value=0.5)

st.sidebar.markdown("---")

theta_fit_min = st.sidebar.slider("θ min (regression)", 0.0, 1.0, 0.1)
theta_fit_max = st.sidebar.slider("θ max (regression)", 0.0, 1.0, 1.0)

# ---- Run simulation ----
res = chrono_sim_with_regression(
    E_appl, duration, dt, k01, k02, E02, alpha,
    lambda1 * 38.923074, theta_fit_min=theta_fit_min, theta_fit_max=theta_fit_max
)

st.subheader("📈 Current Response (Ψ vs θ)")

fig1, ax1 = plt.subplots()
ax1.plot(res.theta, res.I_MH * dt * 38.923074, label="MH")
ax1.plot(res.theta, res.I_BV * dt * 38.923074, "--", label="BV")
ax1.set_xlabel("θ")
ax1.set_ylabel("Ψ")
ax1.legend()
st.pyplot(fig1)

st.subheader("📉 ln|Ψ| vs θ with Regression")

fig2, ax2 = plt.subplots()
ax2.plot(res.theta, res.ln_psi_MH, label="MH")
ax2.plot(res.theta, res.ln_psi_BV, "--", label="BV")
ax2.plot(res.theta, res.reg_MH.intercept + res.reg_MH.slope * res.theta, "k:", label=f"MH fit: slope={res.reg_MH.slope:.3f}")
ax2.plot(res.theta, res.reg_BV.intercept + res.reg_BV.slope * res.theta, "r:", label=f"BV fit: slope={res.reg_BV.slope:.3f}")
ax2.set_xlabel("θ")
ax2.set_ylabel("ln|Ψ|")
ax2.legend()
st.pyplot(fig2)

st.subheader("📊 Surface Excesses fO, fI, fR (MH and BV)")

fig3, ax3 = plt.subplots()
ax3.plot(res.theta, res.fO_MH, label="fO (MH)")
ax3.plot(res.theta, res.fI_MH, label="fI (MH)")
ax3.plot(res.theta, res.fR_MH, label="fR (MH)")
ax3.plot(res.theta, res.fO_BV, "--", label="fO (BV)")
ax3.plot(res.theta, res.fI_BV, "--", label="fI (BV)")
ax3.plot(res.theta, res.fR_BV, "--", label="fR (BV)")
ax3.set_xlabel("θ")
ax3.set_ylabel("Surface Excess")
ax3.legend()
st.pyplot(fig3)

# ---- Download section ----
def download_txt(label, filename, header, data):
    from io import StringIO
    buffer = StringIO()
    np.savetxt(buffer, data, header=header, delimiter="\t")
    st.download_button(label, data=buffer.getvalue(), file_name=filename, mime="text/plain")

st.subheader("📥 Download Results as TXT")

download_txt("Download Ψ (MH)", "psi_mh.txt", "theta\tPsi", np.column_stack((res.theta, res.I_MH * dt * 38.923074)))
download_txt("Download Ψ (BV)", "psi_bv.txt", "theta\tPsi", np.column_stack((res.theta, res.I_BV * dt * 38.923074)))

download_txt("Download Excesses (MH)", "excess_mh.txt", "theta\tfO\tfI\tfR", np.column_stack((res.theta, res.fO_MH, res.fI_MH, res.fR_MH)))
download_txt("Download Excesses (BV)", "excess_bv.txt", "theta\tfO\tfI\tfR", np.column_stack((res.theta, res.fO_BV, res.fI_BV, res.fR_BV)))
