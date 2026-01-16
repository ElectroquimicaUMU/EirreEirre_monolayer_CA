import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chrono_main import simulate_EirreEirre, simulate_EquasiEquasi

dpi = 400

st.set_page_config(layout="wide")
st.title("📉 Chronoamperometry: EirreEirre vs EquasiEquasi")

# Sidebar inputs
with st.sidebar:
    E_appl = st.number_input("Applied potential (V)", value=-0.3, step=0.001, format="%.3f")
    duration = st.number_input("Duration (s)", value=2.0, step=0.1)
    dt = st.number_input("Time step (s)", value=0.01, step=0.001, format="%.3f")
    lambda1 = st.number_input("λ1 (eV)", value=0.5, step=0.01)
    lambda2 = st.number_input("λ2 (eV)", value=0.5, step=0.01)
    k01 = st.number_input("k01 (s⁻¹)", value=0.1, step=0.01)
    k02 = st.number_input("k02 (s⁻¹)", value=0.1, step=0.01)
    E02 = st.number_input("E02 (V)", value=-0.2, step=0.001, format="%.3f")
    st.markdown("### Linear regression range (θ)")
    theta_min = st.slider("θ min", 0.0, 1.0, 0.0, step=0.01)
    theta_max = st.slider("θ max", 0.0, 1.0, 1.0, step=0.01)

# Run simulations
res1 = simulate_EirreEirre(E_appl, duration, dt, k01, k02, E02, lambda1 * 38.923074, lambda2 * 38.923074)
res2 = simulate_EquasiEquasi(E_appl, duration, dt, k01, k02, E02, lambda1 * 38.923074, lambda2 * 38.923074)

# Apply regression in selected theta range
def custom_regression(t_star, ln_psi, theta_min, theta_max):
    mask = (t_star >= theta_min) & (t_star <= theta_max) & (~np.isnan(ln_psi))
    from scipy.stats import linregress
    slope, intercept, r, _, _ = linregress(t_star[mask], ln_psi[mask])
    return dict(slope=slope, intercept=intercept, r=r)

res1_reg = custom_regression(res1.time_star, res1.ln_psi, theta_min, theta_max)
res2_reg = custom_regression(res2.time_star, res2.ln_psi, theta_min, theta_max)

# Plot adimensional current Psi
st.subheader("Ψ(θ) - Adimensional current response")
fig, ax = plt.subplots(dpi=dpi)
ax.plot(res1.time_star, res1.psi, label="EirreEirre")
ax.plot(res2.time_star, res2.psi, "--", label="EquasiEquasi")
ax.set_xlabel("θ = t / duration")
ax.set_ylabel("Ψ")
ax.legend()
st.pyplot(fig)

# Plot surface coverages
st.subheader("Surface excesses")
fig, ax = plt.subplots(dpi=dpi)
ax.plot(res1.time_star, res1.fO, label="fO (EirreEirre)")
ax.plot(res1.time_star, res1.fR, label="fR (EirreEirre)")
ax.plot(res1.time_star, res1.fI, label="fI (EirreEirre)")
ax.plot(res2.time_star, res2.fO, "--", label="fO (EquasiEquasi)")
ax.plot(res2.time_star, res2.fR, "--", label="fR (EquasiEquasi)")
ax.plot(res2.time_star, res2.fI, "--", label="fI (EquasiEquasi)")
ax.set_xlabel("θ = t / duration")
ax.set_ylabel("Fractional coverage")
ax.legend()
st.pyplot(fig)

# Regression analysis of ln|Ψ|
st.subheader("Regression: ln|Ψ| vs θ")
fig, ax = plt.subplots(dpi=dpi)
mask1 = ~np.isnan(res1.ln_psi)
mask2 = ~np.isnan(res2.ln_psi)
ax.plot(res1.time_star[mask1], res1.ln_psi[mask1], label="EirreEirre")
ax.plot(res2.time_star[mask2], res2.ln_psi[mask2], "--", label="EquasiEquasi")

# Draw regression lines in selected range
line1 = res1_reg["slope"] * res1.time_star + res1_reg["intercept"]
line2 = res2_reg["slope"] * res2.time_star + res2_reg["intercept"]
ax.plot(res1.time_star, line1, "k:", alpha=0.3)
ax.plot(res2.time_star, line2, "k--", alpha=0.3)

ax.set_xlabel("θ = t / duration")
ax.set_ylabel("ln(Ψ)")
ax.legend()
st.pyplot(fig)

# Regression results table
st.markdown("### Regression summary")
df = pd.DataFrame([
    {
        "Model": "EirreEirre",
        "Slope": res1_reg["slope"],
        "Intercept": res1_reg["intercept"],
        "R^2": res1_reg["r"]*res1_reg["r"]
    },
    {
        "Model": "EquasiEquasi",
        "Slope": res2_reg["slope"],
        "Intercept": res2_reg["intercept"],
        "R^2": res2_reg["r"]*res2_reg["r"]
    }
])
st.dataframe(df)

# Downloads
st.subheader("📥 Download Ψ vs θ data")

def download_txt(label, filename, header, data):
    import io
    buf = io.StringIO()
    np.savetxt(buf, data, header=header)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/plain")

download_txt("Download EirreEirre Ψ", "EirreEirre_Psi.txt", "θ	Ψ", np.column_stack((res1.time_star, res1.psi)))
download_txt("Download EquasiEquasi Ψ", "EquasiEquasi_Psi.txt", "θ	Ψ", np.column_stack((res2.time_star, res2.psi)))

st.subheader("📥 Download surface excesses")

download_txt("Download EirreEirre surface excesses", "EirreEirre_excesses.txt", "θ\tfO\tfR\tfI", np.column_stack((res1.time_star, res1.fO, res1.fR, res1.fI)))
download_txt("Download EquasiEquasi surface excesses", "EquasiEquasi_excesses.txt", "θ\tfO\tfR\tfI", np.column_stack((res2.time_star, res2.fO, res2.fR, res2.fI)))
