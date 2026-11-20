"""
Avrami (JMAK) crystallization kinetics and field-assisted models for TiO2 thin films.

Provides:
- Analytical JMAK X(t) = 1 - exp(-(k*t)^n)
- Arrhenius temperature-dependent reaction rate k(T)
- Time-Temperature-Transformation (TTT) curves for arbitrary crystalline fractions (5%, 50%, 95%)
- Field-assisted barrier reduction k(T, E) with Linear and Poole-Frenkel models
- Plotting utilities for Wooding et al. validation and thermal poling experimental matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from src.params import (
    K_B_EV,
    WOODING_TABLE_2,
    WOODING_ARRHENIUS,
    WOODING_GLOBAL_N,
    WOODING_GLOBAL_EA,
    POLING_EXPERIMENTS,
    D_EFF_FIELD,
    FIELD_BASE_EA,
    FIELD_BASE_K0,
    FIELD_BASE_N,
    U_DEPLETION_THRESHOLD_V,
    BETA_PF_EFFECTIVE,
    ALPHA_LINEAR,
)


# ==============================================================================
# 1. Analytical JMAK Core Functions
# ==============================================================================
def jmak_fraction(t_s, k, n):
    """
    Compute transformed crystalline fraction X(t) according to the JMAK equation:
        X(t) = 1 - exp(-(k * t)^n)

    Parameters
    ----------
    t_s : float or np.ndarray
        Time in seconds.
    k : float
        Reaction rate constant [s^-1].
    n : float
        Avrami exponent (dimensionless).

    Returns
    -------
    X : float or np.ndarray
        Transformed fraction (clamped between 0 and 1).
    """
    t_arr = np.asarray(t_s, dtype=float)
    # Ensure non-negative time
    t_safe = np.maximum(0.0, t_arr)
    kt_n = (k * t_safe) ** n
    # Avoid overflow in exp for extremely large kt_n
    kt_n_clipped = np.clip(kt_n, 0.0, 700.0)
    X = 1.0 - np.exp(-kt_n_clipped)
    return np.clip(X, 0.0, 1.0)


def arrhenius_k(T_C, k0, Ea_eV):
    """
    Compute reaction rate k(T) using the Arrhenius relationship:
        k(T) = k0 * exp(-Ea / (k_B * T))

    Parameters
    ----------
    T_C : float or np.ndarray
        Temperature in Celsius.
    k0 : float
        Pre-exponential frequency factor [s^-1].
    Ea_eV : float
        Activation energy in eV/atom.

    Returns
    -------
    k : float or np.ndarray
        Reaction rate constant [s^-1].
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    exponent = -Ea_eV / (K_B_EV * T_K)
    return k0 * np.exp(np.clip(exponent, -700.0, 0.0))


def ttt_time_seconds(X_target, T_C, k0, Ea_eV, n):
    """
    Calculate isothermal time to reach a given transformed fraction X_target:
        t_X = (1 / k(T)) * [-ln(1 - X_target)]^(1/n)

    Parameters
    ----------
    X_target : float
        Target transformation fraction (e.g., 0.05, 0.50, 0.95).
    T_C : float or np.ndarray
        Temperature in Celsius.
    k0 : float
        Pre-exponential frequency factor [s^-1].
    Ea_eV : float
        Activation energy [eV/atom].
    n : float
        Avrami exponent.

    Returns
    -------
    t_s : float or np.ndarray
        Time in seconds.
    """
    k = arrhenius_k(T_C, k0, Ea_eV)
    prefactor = (-np.log(1.0 - X_target)) ** (1.0 / n)
    return prefactor / k


# ==============================================================================
# 2. Field-Assisted Crystallization Kinetics
# ==============================================================================
def barrier_reduction(E_field_V_m, model="poole_frenkel", alpha=ALPHA_LINEAR, beta=BETA_PF_EFFECTIVE):
    """
    Compute activation energy barrier reduction dE(E) under an electric field.

    Parameters
    ----------
    E_field_V_m : float or np.ndarray
        Electric field magnitude in V/m.
    model : str
        'poole_frenkel' (dE = beta * sqrt(E)) or 'linear' (dE = alpha * E).
    alpha : float
        Linear coupling parameter [eV / (V/m)].
    beta : float
        Poole-Frenkel coupling parameter [eV * (m/V)^(1/2)].

    Returns
    -------
    dE_eV : float or np.ndarray
        Barrier reduction in eV.
    """
    E_safe = np.maximum(0.0, np.asarray(E_field_V_m, dtype=float))
    if model.lower() in ("poole_frenkel", "pf"):
        return beta * np.sqrt(E_safe)
    elif model.lower() in ("linear", "lin"):
        return alpha * E_safe
    else:
        raise ValueError(f"Unknown barrier reduction model: {model}. Choose 'poole_frenkel' or 'linear'.")


def field_assisted_k(T_C, E_field_V_m, k0=FIELD_BASE_K0, Ea_eV=FIELD_BASE_EA,
                     model="poole_frenkel", alpha=ALPHA_LINEAR, beta=BETA_PF_EFFECTIVE,
                     min_barrier_eV=0.15):
    """
    Compute field-assisted rate constant k(T, E):
        k(T, E) = k0 * exp(-(Ea - dE(E)) / (k_B * T))

    Parameters
    ----------
    T_C : float or np.ndarray
        Temperature in Celsius.
    E_field_V_m : float or np.ndarray
        Electric field in V/m.
    k0 : float
        Pre-exponential factor [s^-1].
    Ea_eV : float
        Base thermal activation energy [eV/atom].
    model : str
        'poole_frenkel' or 'linear'.
    alpha, beta : float
        Coupling coefficients.
    min_barrier_eV : float
        Minimum barrier to prevent unphysical negative values.

    Returns
    -------
    k : float or np.ndarray
        Reaction rate constant [s^-1].
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    dE = barrier_reduction(E_field_V_m, model=model, alpha=alpha, beta=beta)
    E_eff = np.maximum(min_barrier_eV, Ea_eV - dE)
    exponent = -E_eff / (K_B_EV * T_K)
    return k0 * np.exp(np.clip(exponent, -700.0, 0.0))


def field_assisted_fraction(t_s, T_C, U_V, d_eff_m=D_EFF_FIELD, u_th_V=U_DEPLETION_THRESHOLD_V,
                            k0=FIELD_BASE_K0, Ea_eV=FIELD_BASE_EA, n=FIELD_BASE_N,
                            model="poole_frenkel", alpha=ALPHA_LINEAR, beta=BETA_PF_EFFECTIVE):
    """
    Compute transformed fraction X(t) under thermal poling (temperature T_C and voltage U_V).
    Accounts for the threshold voltage u_th_V required to form the sub-anodic ion-depleted layer.
    """
    U_arr = np.asarray(U_V, dtype=float)
    U_eff = np.maximum(0.0, U_arr - u_th_V)
    E_field = U_eff / d_eff_m
    k = field_assisted_k(T_C, E_field, k0=k0, Ea_eV=Ea_eV, model=model, alpha=alpha, beta=beta)
    return jmak_fraction(t_s, k, n)



# ==============================================================================
# 3. Plotting & Validation Functions
# ==============================================================================
def plot_wooding_scurves(save_path="outputs/plot1_wooding_scurves.png"):
    """
    Reproduce Wooding et al. Fig. 4: Amorphous TiO2 to anatase phase transformation S-curves
    for 120 °C, 140 °C, and 160 °C ALD films annealed at 160 - 220 °C.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    dep_configs = [
        (120, axes[0], [180, 200, 220], {"180": "#1f77b4", "200": "#d62728", "220": "#000000"}),
        (140, axes[1], [160, 180, 200], {"160": "#e377c2", "180": "#1f77b4", "200": "#d62728"}),
        (160, axes[2], [160, 180, 200], {"160": "#e377c2", "180": "#1f77b4", "200": "#d62728"}),
    ]

    t_s = np.logspace(2, 7.2, 500)  # 100 s to ~1.5e7 s

    for dep_T, ax, pda_temps, color_map in dep_configs:
        params_dep = WOODING_ARRHENIUS[dep_T]
        k0 = params_dep["k0"]
        Ea = params_dep["Ea_avg"]

        for pda_T in pda_temps:
            key = (dep_T, pda_T)
            meas = WOODING_TABLE_2[key]
            n_meas = meas["n"]
            k_meas = meas["k"]

            # Model curve using measured Table 2 (n, k)
            X_meas_curve = jmak_fraction(t_s, k_meas, n_meas)
            c = color_map[str(pda_T)]
            ax.plot(t_s, X_meas_curve, color=c, lw=2.2, label=f"PDA {pda_T} °C (k={k_meas:.1e})")

            # Mark 50% transformation point
            t_50 = (np.log(2.0)) ** (1.0 / n_meas) / k_meas
            ax.scatter([t_50], [0.5], color=c, s=40, zorder=4)

        ax.set_xscale("log")
        ax.set_xlim(1e2, 2e7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Annealing Time (s)", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.45)
        ax.set_title(f"ALD Deposition: {dep_T} °C\n($k_0={k0:.1e}$ s$^{{-1}}$, $E_a={Ea}$ eV)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    axes[0].set_ylabel("Crystalline Fraction, X(t)", fontsize=11, fontweight="bold")
    plt.suptitle("Validation against Wooding et al. (2023) Fig. 4: Isothermal JMAK S-Curves",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Wooding S-curves plot to: {save_path}")


def plot_wooding_ttt(save_path="outputs/plot2_wooding_ttt.png"):
    """
    Reproduce Wooding et al. Fig. 9: TTT diagrams for 5%, 50%, and 95% anatase transformation
    at ALD deposition temperatures: 120 °C, 140 °C, and 160 °C.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    dep_temps = [120, 140, 160]
    pda_range_C = np.linspace(130, 240, 200)

    fractions = [
        (0.05, "5% (Start)", "-", 2.0),
        (0.50, "50% (Half)", "--", 2.2),
        (0.95, "95% (Complete)", ":", 2.4),
    ]

    for idx, dep_T in enumerate(dep_temps):
        ax = axes[idx]
        params = WOODING_ARRHENIUS[dep_T]
        k0 = params["k0"]
        Ea = params["Ea_avg"]
        n = params["n_avg"]

        colors = ["#2ca02c", "#1f77b4", "#d62728"]
        for (X_val, lbl, ls, lw), c in zip(fractions, colors):
            t_s = ttt_time_seconds(X_val, pda_range_C, k0, Ea, n)
            log10_t = np.log10(t_s)
            ax.plot(log10_t, pda_range_C, color=c, ls=ls, lw=lw, label=lbl)

        # Plot discrete experimental t_50 data points from Table 2
        for (d_t, pda_t), data in WOODING_TABLE_2.items():
            if d_t == dep_T:
                k_exp = data["k"]
                n_exp = data["n"]
                t50_exp = (np.log(2.0)) ** (1.0 / n_exp) / k_exp
                ax.scatter(np.log10(t50_exp), pda_t, color="#1f77b4", s=55, edgecolors="black", zorder=5)

        ax.set_xlim(2.0, 8.5)
        ax.set_ylim(120, 250)
        ax.set_xlabel(r"log$_{10}$(Time [s])", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_title(f"Deposition T = {dep_T} °C\n($n \\approx {n:.2f}$)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, frameon=True)

        # Shading regions
        t5_s = ttt_time_seconds(0.05, pda_range_C, k0, Ea, n)
        t95_s = ttt_time_seconds(0.95, pda_range_C, k0, Ea, n)
        ax.text(2.3, 140, "Amorphous", fontsize=10, color="gray", fontweight="bold")
        ax.text(6.8, 220, "Anatase\nCrystalline", fontsize=10, color="darkred", fontweight="bold")

    axes[0].set_ylabel("PDA Temperature (°C)", fontsize=11, fontweight="bold")
    plt.suptitle("Validation against Wooding et al. (2023) Fig. 9: TTT Diagrams for Plate-like Anatase",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Wooding TTT diagrams to: {save_path}")


def plot_field_assisted_kinetics(save_path="outputs/plot3_field_assisted_kinetics.png"):
    """
    Field-assisted crystallization kinetics:
    1. Crystallinity vs. Voltage (U = 0 to 1250 V) at T = 300 °C, t = 1 h.
    2. Crystallinity vs. Temperature (T = 150 to 300 °C) at U = 500 V, t = 1 h.
    3. Crystallization kinetics X(t) at 300 °C, 500 V over t = 0 to 2 h.
    Comparing Linear and Poole-Frenkel models with experimental data points from Report.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: X vs Voltage at 300 °C, 1 h
    U_arr = np.linspace(0, 1300, 300)
    t_1h = 3600.0
    T_300 = 300.0

    X_lin_U = [field_assisted_fraction(t_1h, T_300, u, model="linear") for u in U_arr]
    X_pf_U = [field_assisted_fraction(t_1h, T_300, u, model="poole_frenkel") for u in U_arr]

    axes[0].plot(U_arr, X_pf_U, color="#d62728", lw=2.4, label="Poole-Frenkel Model")
    axes[0].plot(U_arr, X_lin_U, color="#1f77b4", lw=2.2, ls="--", label="Linear Model")

    # Overlay Report voltage data at 300 °C, 1 h:
    # C1 (250V, X~0), C20 (500V, X~1), C19 (750V, X~1), C4 (1000V, X~1), C5 (1250V, X~1)
    exp_U = [250, 500, 750, 1000, 1250]
    exp_X_U = [0.02, 0.98, 1.0, 1.0, 1.0]
    exp_labels_U = ["C1", "C20", "C19", "C4", "C5"]
    axes[0].scatter(exp_U, exp_X_U, color="black", s=65, zorder=5, label="Report (Raman)")
    for x, y, lbl in zip(exp_U, exp_X_U, exp_labels_U):
        axes[0].annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                         fontsize=9, fontweight="bold")

    axes[0].set_xlabel("Applied Voltage U (V)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Crystalline Fraction X(1 h)", fontsize=11, fontweight="bold")
    axes[0].set_title(r"Voltage Response at 300 °C (1 h anneal)", fontsize=12, fontweight="bold")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend(loc="center left", fontsize=9)
    axes[0].set_ylim(-0.05, 1.1)

    # Panel B: X vs Temperature at 500 V, 1 h
    T_arr = np.linspace(140, 320, 300)
    U_500 = 500.0

    X_lin_T = [field_assisted_fraction(t_1h, t, U_500, model="linear") for t in T_arr]
    X_pf_T = [field_assisted_fraction(t_1h, t, U_500, model="poole_frenkel") for t in T_arr]

    axes[1].plot(T_arr, X_pf_T, color="#d62728", lw=2.4, label="Poole-Frenkel Model")
    axes[1].plot(T_arr, X_lin_T, color="#1f77b4", lw=2.2, ls="--", label="Linear Model")

    # Thermal alone without field at 300 °C
    X_thermal = [jmak_fraction(t_1h, arrhenius_k(t, FIELD_BASE_K0, FIELD_BASE_EA), FIELD_BASE_N) for t in T_arr]
    axes[1].plot(T_arr, X_thermal, color="gray", lw=1.8, ls=":", label="Thermal Alone (0 V)")

    # Overlay Report temperature data at 500 V, 1 h:
    # C10 (150 °C, X~0), C9 (200 °C, X~0), C8 (250 °C, X~0.25 seed transition), C20 (300 °C, X~1)
    exp_T = [150, 200, 250, 300]
    exp_X_T = [0.01, 0.02, 0.22, 0.98]
    exp_labels_T = ["C10", "C9", "C8 (seeds)", "C20"]
    axes[1].scatter(exp_T, exp_X_T, color="black", s=65, zorder=5, label="Report (Raman)")
    for x, y, lbl in zip(exp_T, exp_X_T, exp_labels_T):
        axes[1].annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                         fontsize=9, fontweight="bold")

    axes[1].set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[1].set_title(r"Temperature Response at 500 V (1 h anneal)", fontsize=12, fontweight="bold")
    axes[1].grid(True, ls="--", alpha=0.5)
    axes[1].legend(loc="center left", fontsize=9)
    axes[1].set_ylim(-0.05, 1.1)

    # Panel C: X vs Time at 300 °C, 500 V (t = 0 to 1.8 h)
    t_hours = np.linspace(0.01, 1.8, 300)
    t_seconds = t_hours * 3600.0

    X_lin_t = field_assisted_fraction(t_seconds, T_300, U_500, model="linear")
    X_pf_t = field_assisted_fraction(t_seconds, T_300, U_500, model="poole_frenkel")
    X_thermal_t = jmak_fraction(t_seconds, arrhenius_k(T_300, FIELD_BASE_K0, FIELD_BASE_EA), FIELD_BASE_N)

    axes[2].plot(t_hours, X_pf_t, color="#d62728", lw=2.4, label="Poole-Frenkel (500 V)")
    axes[2].plot(t_hours, X_lin_t, color="#1f77b4", lw=2.2, ls="--", label="Linear (500 V)")
    axes[2].plot(t_hours, X_thermal_t, color="gray", lw=1.8, ls=":", label="Thermal Alone (0 V)")

    # Overlay Report time dependence data at 300 °C, 500 V:
    # C13 (0.25 h, X~0.85 surface crystallized), C12 (0.5 h, X~0.95), C20 (1.0 h, X~1.0), C11 (1.5 h, X~1.0)
    exp_t = [0.25, 0.5, 1.0, 1.5]
    exp_X_t = [0.88, 0.97, 0.99, 1.0]
    exp_labels_t = ["C13 (15 min)", "C12 (30 min)", "C20 (1 h)", "C11 (1.5 h)"]
    axes[2].scatter(exp_t, exp_X_t, color="black", s=65, zorder=5, label="Report (Raman)")
    for x, y, lbl in zip(exp_t, exp_X_t, exp_labels_t):
        axes[2].annotate(lbl, (x, y), textcoords="offset points", xytext=(0, -14), ha="center",
                         fontsize=8.5, fontweight="bold")

    axes[2].set_xlabel("Poling Time (hours)", fontsize=11, fontweight="bold")
    axes[2].set_title(r"Kinetic Evolution at 300 °C, 500 V", fontsize=12, fontweight="bold")
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].set_ylim(-0.05, 1.1)

    plt.suptitle("Field-Assisted Crystallization Kinetics (Report & Barrier Reduction Models)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved field-assisted kinetics plot to: {save_path}")


def plot_crystallization_regime_map(save_path="outputs/plot4_crystallization_regime_map.png"):
    """
    2D Contour Map in (Temperature, Voltage) space showing predicted degree of crystallization
    at t = 1 hour, overlaid with experimental data points C1 - C20 from Report.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T_grid = np.linspace(140, 320, 150)
    U_grid = np.linspace(100, 1300, 150)
    TT, UU = np.meshgrid(T_grid, U_grid)

    t_1h = 3600.0
    XX = np.zeros_like(TT)

    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            XX[i, j] = field_assisted_fraction(t_1h, TT[i, j], UU[i, j], model="poole_frenkel")

    fig, ax = plt.subplots(figsize=(9, 6.8))

    # Contour filled
    levels = np.linspace(0, 1, 21)
    cp = ax.contourf(TT, UU, XX, levels=levels, cmap="RdYlBu_r", alpha=0.88)
    cbar = fig.colorbar(cp, ax=ax, label="Crystalline Anatase Fraction X (t = 1 h)")

    # Contour lines for thresholds: 5% (onset) and 90% (complete)
    c_lines = ax.contour(TT, UU, XX, levels=[0.05, 0.50, 0.90], colors=["blue", "black", "white"],
                         linewidths=[1.8, 1.5, 2.0], linestyles=["--", "-", "-."])
    ax.clabel(c_lines, inline=True, fmt={0.05: "5% Onset", 0.50: "50%", 0.90: "90% Crystalline"},
              fontsize=9)

    # Overlay experimental samples
    # For a 1-hour crystallization regime map, display the 1 h benchmark samples
    # (C1, C10, C9, C8, C20, C19, C4, C5), ensuring C20 is correctly labeled at (300 °C, 500 V, 1 h)
    samples_1h = [exp for exp in POLING_EXPERIMENTS if exp.get("t_h") == 1.0]
    for exp in samples_1h:
        t_exp = exp["T_C"]
        u_exp = exp["U_V"]
        s_code = exp["state_code"]
        name = exp["sample"]

        if s_code == 0:
            # Amorphous
            m = "o"
            color = "#1f77b4"
            edge = "white"
        elif s_code == 1:
            # Transition
            m = "s"
            color = "#ff7f0e"
            edge = "black"
        else:
            # Crystalline
            m = "^"
            color = "#2ca02c"
            edge = "black"

        ax.scatter(t_exp, u_exp, marker=m, s=110, c=color, edgecolors=edge, lw=1.5, zorder=6)
        
        xy_offset = (8, -4)
        name_label = name
        if name == "C20":
            name_label = "C20 (1 h)"
            xy_offset = (8, 6)

        ax.annotate(name_label, (t_exp, u_exp), textcoords="offset points", xytext=xy_offset,
                    fontsize=9, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))


    # Custom legend for markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='white',
                   markersize=10, label='Report: Amorphous (No Cryst.)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markeredgecolor='black',
                   markersize=10, label='Report: Transition Phase (Seeds)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c', markeredgecolor='black',
                   markersize=10, label='Report: Complete Anatase'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.92, fontsize=10)

    ax.set_xlabel("Temperature T (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Applied Voltage U (V)", fontsize=12, fontweight="bold")
    ax.set_title("TiO2 Crystallization Phase Regime Map (t = 1 h)\nPoole-Frenkel Model vs. Experimental Report",
                 fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved crystallization regime map to: {save_path}")
