
import numpy as np
import scipy.integrate as integrate
from collections import namedtuple
from scipy.stats import linregress

FRT = 38.923074

ChronoResult = namedtuple("ChronoResult", [
    "time", "theta",
    "I_MH", "I_BV",
    "ln_psi_MH", "ln_psi_BV",
    "reg_MH", "reg_BV",
    "fO_MH", "fR_MH", "fI_MH",
    "fO_BV", "fR_BV", "fI_BV"
])

def chrono_sim_with_regression(E_appl, t_total, dt, k01, k02, E02, alpha, lambda1, model="MH", theta_fit_min=0.1, theta_fit_max=1.0):
    lambda2 = lambda1
    tau = dt
    n_steps = int(t_total / dt) + 1

    time = np.linspace(0, t_total, n_steps)
    theta = time / t_total

    I_MH = np.zeros(n_steps)
    I_BV = np.zeros(n_steps)

    fO_MH = np.zeros(n_steps)
    fI_MH = np.zeros(n_steps)
    fR_MH = np.zeros(n_steps)

    fO_BV = np.zeros(n_steps)
    fI_BV = np.zeros(n_steps)
    fR_BV = np.zeros(n_steps)

    fO_MH[0], fO_BV[0] = 1.0, 1.0

    nu1 = FRT * E_appl
    nu2 = FRT * (E_appl - E02)

    S01 = integrate.quad(lambda x: np.exp(-lambda1/4*(1 + x/lambda1)**2) / (1 + np.exp(-x)), -50, 50)[0]
    S02 = integrate.quad(lambda x: np.exp(-lambda2/4*(1 + x/lambda2)**2) / (1 + np.exp(-x)), -50, 50)[0]

    for i in range(1, n_steps):
        MH1 = integrate.quad(lambda x: np.exp(-lambda1/4*(1 + (nu1 + x)/lambda1)**2) / (1 + np.exp(-x)), -50, 50)[0]
        MH2 = integrate.quad(lambda x: np.exp(-lambda2/4*(1 + (nu2 + x)/lambda2)**2) / (1 + np.exp(-x)), -50, 50)[0]
        kMH1red = k01 * tau * MH1 / S01
        kMH2red = k02 * tau * MH2 / S02

        fO_MH[i] = fO_MH[i-1] * np.exp(-kMH1red)
        den = kMH1red - kMH2red if abs(kMH1red - kMH2red) > 1e-15 else 1e-15
        fR_MH[i] = (
            1 + (fR_MH[i-1] - 1) * np.exp(-kMH2red)
            + fO_MH[i-1] * (np.exp(-kMH1red) - np.exp(-kMH2red)) * kMH2red / den
        )
        fI_MH[i] = 1 - fO_MH[i] - fR_MH[i]
        I_MH[i] = (fO_MH[i] * kMH1red + fI_MH[i] * kMH2red) / dt / FRT

        kBV1red = k01 * tau * np.exp(-alpha * nu1)
        kBV2red = k02 * tau * np.exp(-alpha * nu2)

        fO_BV[i] = fO_BV[i-1] * np.exp(-kBV1red)
        denBV = kBV1red - kBV2red if abs(kBV1red - kBV2red) > 1e-15 else 1e-15
        fR_BV[i] = (
            1 + (fR_BV[i-1] - 1) * np.exp(-kBV2red)
            + fO_BV[i-1] * (np.exp(-kBV1red) - np.exp(-kBV2red)) * kBV2red / denBV
        )
        fI_BV[i] = 1 - fO_BV[i] - fR_BV[i]
        I_BV[i] = (fO_BV[i] * kBV1red + fI_BV[i] * kBV2red) / dt / FRT

    psi_MH = I_MH * dt * FRT
    psi_BV = I_BV * dt * FRT

    ln_psi_MH = np.log(np.abs(psi_MH))
    ln_psi_BV = np.log(np.abs(psi_BV))

    mask = (theta >= theta_fit_min) & (theta <= theta_fit_max)

    reg_MH = linregress(theta[mask], ln_psi_MH[mask])
    reg_BV = linregress(theta[mask], ln_psi_BV[mask])

    return ChronoResult(
        time[1:], theta[1:],
        I_MH[1:], I_BV[1:],
        ln_psi_MH[1:], ln_psi_BV[1:],
        reg_MH, reg_BV,
        fO_MH[1:], fR_MH[1:], fI_MH[1:],
        fO_BV[1:], fR_BV[1:], fI_BV[1:]
    )
