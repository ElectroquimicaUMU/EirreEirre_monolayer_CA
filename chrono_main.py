import numpy as np
from scipy.integrate import quad, solve_ivp
from collections import namedtuple
from scipy.stats import linregress

FRT = 38.923074

ChronoResult = namedtuple("ChronoResult", [
    "time", "time_star", "psi", "fO", "fR", "fI", "ln_psi", "linreg"
])

def simulate_EirreEirre(E_appl, duration, dt, k01, k02, E02, lambda1):
    import numpy as np

    FRT = 38.923074
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)
    theta = time / duration

    # Calculamos el potencial reducido
    nu1 = FRT * (E_appl - 0)
    nu2 = FRT * (E_appl - E02)

    # Constantes de tasa
    k1 = k01 * np.exp(-lambda1 / 4 * (1 + nu1 / lambda1)**2)
    k2 = k02 * np.exp(-lambda1 / 4 * (1 + nu2 / lambda1)**2)

    # fO(t)
    fO = np.exp(-k1 * theta)

    # fI(t)
    den = k2 - k1 if abs(k2 - k1) > 1e-15 else 1e-15
    fI = (k1 / den) * (np.exp(-k1 * theta) - np.exp(-k2 * theta))

    # fR(t)
    fR = 1 - fO - fI

    # Psi
    psi = fO * k1 + fI * k2

    # Resultados como objeto simple
    class Result:
        pass

    res = Result()
    res.time = time
    res.time_star = theta
    res.fO = fO
    res.fI = fI
    res.fR = fR
    res.psi = psi
    return res

    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    t_star = t / duration

    nu1 = FRT * E_appl
    nu2 = FRT * (E_appl - E02)

    S01 = quad(lambda x: np.exp(-lambda1/4*(1 + x/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
    S02 = quad(lambda x: np.exp(-lambda2/4*(1 + x/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

    MH1 = quad(lambda x: np.exp(-lambda1/4*(1 + (nu1 + x)/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
    MH2 = quad(lambda x: np.exp(-lambda2/4*(1 + (nu2 + x)/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

    k1 = k01 * MH1 / S01
    k2 = k02 * MH2 / S02

    fO = np.zeros(n_steps)
    fR = np.zeros(n_steps)
    fI = np.zeros(n_steps)
    psi = np.zeros(n_steps)

    fO[0] = 1.0
    fR[0] = 0.0
    fI[0] = 0.0
    psi[0] = k1 * fO[0] + k2 * fI[0]

    for i in range(1, n_steps):
        fO[i] = fO[i-1] * np.exp(-k1*dt)
        den = k1 - k2 if abs(k1 - k2) > 1e-15 else 1e-15
        fR[i] = 1 + (fR[i-1] - 1) * np.exp(-k2*dt) + fO[i-1] * (np.exp(-k1*dt) - np.exp(-k2*dt)) * k2 / den
        fI[i] = 1 - fO[i] - fR[i]
        psi[i] = k1 * fO[i] + k2 * fI[i]

    ln_psi = np.where(psi > 0, np.log(psi), np.nan)

    mask = ~np.isnan(ln_psi)
    slope, intercept, r, _, _ = linregress(t_star[mask], ln_psi[mask])
    linreg = dict(slope=slope, intercept=intercept, r=r)

    return ChronoResult(t, t_star, psi, fO, fR, fI, ln_psi, linreg)

def simulate_EquasiEquasi(E_appl, duration, dt, k01, k02, E02, lambda1):
    lambda2 = lambda1
    t = np.linspace(0, duration, int(duration/dt))
    t_star = t / duration

    nu1 = FRT * E_appl
    nu2 = FRT * (E_appl - E02)

    S01 = quad(lambda x: np.exp(-lambda1/4*(1 + x/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
    S02 = quad(lambda x: np.exp(-lambda2/4*(1 + x/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

    MH1_red = quad(lambda x: np.exp(-lambda1/4*(1 + (nu1 + x)/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
    MH2_red = quad(lambda x: np.exp(-lambda2/4*(1 + (nu2 + x)/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

    k1 = k01 * MH1_red / S01
    k3 = k02 * MH2_red / S02
    k2 = k1 * np.exp(nu1)
    k4 = k3 * np.exp(nu2)

    def ODEivp(t, x):
        fO, fI = x
        return np.array([
            -k1 * fO + k2 * fI,
            (k1 - k4) * fO - (k2 + k3 + k4) * fI + k4
        ])

    sol = solve_ivp(ODEivp, [0, duration], [1, 0], t_eval=t)
    fO = sol.y[0]
    fR = sol.y[1]
    fI = 1 - fO - fR
    psi = k1*fO + k3*fI - k2*fI - k4*fR
    ln_psi = np.where(psi > 0, np.log(psi), np.nan)

    mask = ~np.isnan(ln_psi)
    slope, intercept, r, _, _ = linregress(t_star[mask], ln_psi[mask])
    linreg = dict(slope=slope, intercept=intercept, r=r)

    return ChronoResult(t, t_star, psi, fO, fR, fI, ln_psi, linreg)