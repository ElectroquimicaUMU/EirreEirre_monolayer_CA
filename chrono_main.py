import numpy as np
import scipy.integrate as integrate
from collections import namedtuple

FRT = 38.923074

ChronoResult = namedtuple("ChronoResult", [
    "time", "I_MH", "I_BV",
    "fO_MH", "fR_MH", "fI_MH",
    "fO_BV", "fR_BV", "fI_BV"
])

def chrono_sim(E_appl, t_total, dt, k01, k02, E02, alpha, lambda1, model="MH"):
    lambda2 = lambda1
    tau = dt
    n_steps = int(t_total / dt)

    time = np.linspace(0, t_total, n_steps)

    kMH1red = np.zeros(n_steps)
    kMH2red = np.zeros(n_steps)
    kBV1red = np.zeros(n_steps)
    kBV2red = np.zeros(n_steps)
    I_MH = np.zeros(n_steps)
    I_BV = np.zeros(n_steps)

    fO_MH, fI_MH, fR_MH = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    fO_BV, fI_BV, fR_BV = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    fO_MH[0], fO_BV[0] = 1.0, 1.0

    nu1 = FRT * E_appl
    nu2 = FRT * (E_appl - E02)

    S01 = integrate.quad(lambda x: np.exp(-lambda1 / 4 * (1 + x / lambda1)**2) / (1 + np.exp(-x)), -50, 50)[0]
    S02 = integrate.quad(lambda x: np.exp(-lambda2 / 4 * (1 + x / lambda2)**2) / (1 + np.exp(-x)), -50, 50)[0]

    for i in range(1, n_steps):
        MH1 = integrate.quad(lambda x: np.exp(-lambda1 / 4 * (1 + (nu1 + x) / lambda1)**2) / (1 + np.exp(-x)), -50, 50)[0]
        MH2 = integrate.quad(lambda x: np.exp(-lambda2 / 4 * (1 + (nu2 + x) / lambda2)**2) / (1 + np.exp(-x)), -50, 50)[0]

        kMH1red[i] = k01 * tau * MH1 / S01
        kMH2red[i] = k02 * tau * MH2 / S02

        fO_MH[i] = fO_MH[i - 1] * np.exp(-kMH1red[i])
        den = kMH1red[i] - kMH2red[i] if abs(kMH1red[i] - kMH2red[i]) > 1e-15 else 1e-15
        fR_MH[i] = (
            1
            + (fR_MH[i - 1] - 1) * np.exp(-kMH2red[i])
            + fO_MH[i - 1]
            * (np.exp(-kMH1red[i]) - np.exp(-kMH2red[i]))
            * kMH2red[i]
            / den
        )
        fI_MH[i] = 1 - fO_MH[i] - fR_MH[i]
        I_MH[i] = (fO_MH[i] * kMH1red[i] + fI_MH[i] * kMH2red[i]) / dt / FRT

        kBV1red[i] = k01 * tau * np.exp(-alpha * nu1)
        kBV2red[i] = k02 * tau * np.exp(-alpha * nu2)

        fO_BV[i] = fO_BV[i - 1] * np.exp(-kBV1red[i])
        denBV = kBV1red[i] - kBV2red[i] if abs(kBV1red[i] - kBV2red[i]) > 1e-15 else 1e-15
        fR_BV[i] = (
            1
            + (fR_BV[i - 1] - 1) * np.exp(-kBV2red[i])
            + fO_BV[i - 1]
            * (np.exp(-kBV1red[i]) - np.exp(-kBV2red[i]))
            * kBV2red[i]
            / denBV
        )
        fI_BV[i] = 1 - fO_BV[i] - fR_BV[i]
        I_BV[i] = (fO_BV[i] * kBV1red[i] + fI_BV[i] * kBV2red[i]) / dt / FRT

    return ChronoResult(
        time, I_MH, I_BV,
        fO_MH, fR_MH, fI_MH,
        fO_BV, fR_BV, fI_BV
    )