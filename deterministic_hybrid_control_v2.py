"""
Deterministic, non-collapsing hybrid control reference implementation.
(Versione tweakata per eventi real-time e no overshoot)
"""

import numpy as np
from scipy.integrate import solve_ivp

# Parametri invariati dal tuo codice

# Dinamiche invariati

# Eventi invariati

# Next_state invariati

# Simulate modificata per gestire traiettoria completa senza overshoot
def simulate(Tf=50.0, verbose=True):
    x = np.array([0.4, 1.8, 0.0, 1.0, 0.0, 1.0])
    state = NOMINAL
    t = 0.0
    last_switch = -1e9

    trajectory_t = [t]
    trajectory_x = [x.copy()]
    trajectory_state = [state]

    if verbose:
        print("=== Inizio simulazione ===")
        print(f"Stato iniziale: {STATE_NAME[state]} | x = {x}")

    while t < Tf:
        def ode_func(tt, yy): return dynamics(tt, yy, state)

        sol = solve_ivp(
            ode_func,
            [t, Tf],
            x,
            events=EVENTS,
            max_step=0.1,
            rtol=1e-6,
            atol=1e-8,
        )

        # Aggiungi traiettoria completa (non solo finale)
        trajectory_t.extend(sol.t[1:])  # salta t[0] duplicato
        trajectory_x.extend(sol.y.T[1:])
        trajectory_state.extend([state] * (len(sol.t) - 1))

        t = sol.t[-1]
        x = sol.y[:, -1]

        if sol.status != 1:
            if verbose:
                print(f"t = {t:.2f} | Nessun evento ulteriore.")
            break

        active = [ev(t, x) >= 0 for ev in EVENTS]
        new_state = next_state(state, active)

        if (t - last_switch) >= DWELL_TIME and new_state != state:
            if verbose:
                print(f"t = {t:.2f} | Eventi attivi: {active} (g_T, g_dT, g_P, g_B, g_r)")
                print(f"Transizione: {STATE_NAME[state]} → {STATE_NAME[new_state]}")
            state = new_state
            last_switch = t

            trajectory_state[-1] = state  # aggiorna ultimo stato

            if state == SAFE:
                x[0] = 0.0
                if verbose:
                    print("SAFE raggiunto. P forzato a 0. Simulazione termina.")
                break
        else:
            if verbose:
                print(f"t = {t:.2f} | Dwell non scaduto o no cambio stato → continua")
            continue

    if verbose:
        print("=== Fine simulazione ===")
        print(f"Stato finale: {STATE_NAME[state]} | t = {t:.2f}")
        print(f"x finale = {x}")

    return trajectory_t, trajectory_x, trajectory_state, state, x

if __name__ == "__main__":
    t_list, x_list, s_list, final_state, final_x = simulate(Tf=1000.0)
    # Qui potresti plottare: import matplotlib.pyplot as plt; plt.plot(t_list, [xi[1] for xi in x_list], label='T(t)')
