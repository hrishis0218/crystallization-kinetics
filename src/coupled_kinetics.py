"""
Coupled Interfacial Electric Field and Dynamic JMAK TiO2 Crystallization Solver
================================================================================
Stage 3 of the simulation suite: Couples the time-varying interfacial electric
field E(x=0, t) computed in Stage 2 (from 1D multi-ion Poisson-coupled poling)
into the generalized non-isothermal / time-varying Avrami (JMAK) crystallization
model for amorphous-to-anatase TiO2 thin films.

Governing Equations:
--------------------
1. Generalized Dynamic JMAK Integral:
   X(t) = 1 - exp( - [ integral_0^t k(T, E(t')) dt' ]^n )

2. Field-Assisted Reaction Rate:
   k(T, E(t)) = k_0 * exp(-(E_a - dE(E(t))) / (k_B * T))

3. Barrier Lowering Mechanisms:
   - Poole-Frenkel Model: dE_PF(t)  = beta_PF * sqrt(E(t))
   - Linear Model:         dE_lin(t) = alpha_lin * E(t)

4. Effective Activation Energy:
   E_eff(t) = max(E_min, E_a - dE(E(t)))

References:
-----------
- Wooding et al., Atomic Layer Deposition 1: 1-18 (2023), DOI: 10.3897/aldj.1.101276
- Report on TiO2 crystallization (Bordeaux & IRB experimental EFAD study)
- Pervan et al., Modelling20accepted.pdf (multi-ion glass thermal poling)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Relative / standalone import support
try:
    from .params import (
        K_B_EV,
        FIELD_BASE_EA,
        FIELD_BASE_K0,
        FIELD_BASE_N,
        POLING_EXPERIMENTS,
        GLASS_THICKNESS_W,
        POLING_VOLTAGE_U,
        safe_savefig,
    )
except ImportError:
    from params import (
        K_B_EV,
        FIELD_BASE_EA,
        FIELD_BASE_K0,
        FIELD_BASE_N,
        POLING_EXPERIMENTS,
        GLASS_THICKNESS_W,
        POLING_VOLTAGE_U,
        safe_savefig,
    )


class CoupledJMAKSolver:
    """
    Couples Stage 2 dynamic space-charge interfacial electric field E(x=0, t)
    with Stage 1 generalized Avrami (JMAK) transformation kinetics.
    """

    def __init__(
        self,
        npz_path=None,
        Ea_eV=FIELD_BASE_EA,
        k0=FIELD_BASE_K0,
        n=FIELD_BASE_N,
        T_C=300.0,
        beta_PF=7.00e-6,       # [eV * (m/V)^(1/2)] calibrated for C13 / C12
        alpha_lin=1.65e-10,     # [eV / (V/m)] calibrated for C13 / C12
        min_barrier_eV=0.15,
    ):
        """
        Initialize the coupled JMAK crystallization solver.

        Parameters
        ----------
        npz_path : str, optional
            Path to outputs/interfacial_field_300C_500V.npz.
        Ea_eV : float
            Base thermal activation energy in eV/atom (default: 1.55 eV).
        k0 : float
            Pre-exponential frequency factor in s^-1 (default: 2.0e8 s^-1).
        n : float
            Avrami exponent (default: 2.75).
        T_C : float
            Poling temperature in Celsius (default: 300.0 °C).
        beta_PF : float
            Calibrated Poole-Frenkel coupling constant [eV * (m/V)^(1/2)].
        alpha_lin : float
            Calibrated Linear coupling constant [eV / (V/m)].
        min_barrier_eV : float
            Minimum allowable effective barrier in eV to avoid unphysical negative values.
        """
        self.Ea_eV = float(Ea_eV)
        self.k0 = float(k0)
        self.n = float(n)
        self.T_C = float(T_C)
        self.T_K = self.T_C + 273.15
        self.beta_PF = float(beta_PF)
        self.alpha_lin = float(alpha_lin)
        self.min_barrier_eV = float(min_barrier_eV)

        # Default npz path resolution
        if npz_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            npz_path = os.path.join(base_dir, "outputs", "interfacial_field_300C_500V.npz")
        self.npz_path = npz_path

        # Load interfacial field time series
        self.load_field_profile(self.npz_path)

    def load_field_profile(self, npz_path):
        """
        Load the time-series array of interfacial electric field E(0, t) from Stage 2.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Stage 2 field output not found at: {npz_path}. "
                "Please run 'python run_stage2.py' first to generate interfacial_field_300C_500V.npz."
            )
        data = np.load(npz_path)
        self.time_s = np.asarray(data["time_s"], dtype=float)
        self.time_min = self.time_s / 60.0
        self.time_h = self.time_s / 3600.0
        self.E_interfacial_V_m = np.asarray(data["E_interfacial_V_m"], dtype=float)
        self.E_interfacial_MV_cm = self.E_interfacial_V_m * 1e-8
        self.depletion_depth_nm = np.asarray(data["depletion_depth_nm"], dtype=float)

        # Peak electric field metrics
        self.idx_peak = int(np.argmax(self.E_interfacial_V_m))
        self.t_peak_s = self.time_s[self.idx_peak]
        self.t_peak_min = self.time_min[self.idx_peak]
        self.E_peak_V_m = self.E_interfacial_V_m[self.idx_peak]
        self.E_peak_MV_cm = self.E_interfacial_MV_cm[self.idx_peak]

    def compute_dynamic_rate(self, model="poole_frenkel"):
        """
        Compute the instantaneous reaction rate k(t), barrier reduction dE(t),
        and effective activation energy E_eff(t) across the poling timeline.

        Parameters
        ----------
        model : str
            'poole_frenkel' or 'linear'.

        Returns
        -------
        k_t : np.ndarray
            Reaction rate array [s^-1].
        dE_t : np.ndarray
            Barrier lowering array [eV].
        E_eff_t : np.ndarray
            Effective activation energy array [eV].
        """
        if model.lower() in ("poole_frenkel", "pf"):
            dE_t = self.beta_PF * np.sqrt(np.maximum(0.0, self.E_interfacial_V_m))
        elif model.lower() in ("linear", "lin"):
            dE_t = self.alpha_lin * np.maximum(0.0, self.E_interfacial_V_m)
        else:
            raise ValueError(f"Unknown barrier model: '{model}'. Choose 'poole_frenkel' or 'linear'.")

        E_eff_t = np.maximum(self.min_barrier_eV, self.Ea_eV - dE_t)
        exponent = -E_eff_t / (K_B_EV * self.T_K)
        k_t = self.k0 * np.exp(np.clip(exponent, -700.0, 0.0))
        return k_t, dE_t, E_eff_t

    def solve_cumulative_jmak(self, model="poole_frenkel"):
        """
        Evaluate the generalized cumulative JMAK integral:
            X(t) = 1 - exp( - [ integral_0^t k(t') dt' ]^n )

        Parameters
        ----------
        model : str
            'poole_frenkel' or 'linear'.

        Returns
        -------
        dict
            Dictionary containing X_t, int_k, k_t, dE_t, E_eff_t.
        """
        k_t, dE_t, E_eff_t = self.compute_dynamic_rate(model=model)

        # Cumulative trapezoidal integration of k(t) dt
        int_k = np.zeros_like(self.time_s)
        for i in range(1, len(self.time_s)):
            int_k[i] = np.trapezoid(k_t[: i + 1], self.time_s[: i + 1])

        # Generalized JMAK transformed fraction
        kt_n = np.clip(int_k**self.n, 0.0, 700.0)
        X_t = np.clip(1.0 - np.exp(-kt_n), 0.0, 1.0)

        return {
            "model": model,
            "X_t": X_t,
            "int_k": int_k,
            "k_t": k_t,
            "dE_t": dE_t,
            "E_eff_t": E_eff_t,
        }

    def solve_static_baselines(self):
        """
        Compute baseline transformation curves for comparison:
        1. Static uniform field: E_uniform = U / W = 500 V / 1 mm = 0.5 MV/m
        2. Pure thermal annealing: E = 0 V/m
        """
        # 1. Uniform field across full 1 mm slab without space-charge amplification
        E_uniform = POLING_VOLTAGE_U / GLASS_THICKNESS_W  # 5.0e5 V/m = 0.5 MV/m
        dE_unif_pf = self.beta_PF * np.sqrt(E_uniform)
        E_eff_unif = self.Ea_eV - dE_unif_pf
        k_unif = self.k0 * np.exp(-E_eff_unif / (K_B_EV * self.T_K))
        X_uniform = np.clip(1.0 - np.exp(-np.clip((k_unif * self.time_s)**self.n, 0.0, 700.0)), 0.0, 1.0)

        # 2. Thermal only (E = 0)
        k_thermal = self.k0 * np.exp(-self.Ea_eV / (K_B_EV * self.T_K))
        X_thermal = np.clip(1.0 - np.exp(-np.clip((k_thermal * self.time_s)**self.n, 0.0, 700.0)), 0.0, 1.0)

        return {
            "E_uniform_V_m": E_uniform,
            "dE_uniform_eV": dE_unif_pf,
            "E_eff_uniform_eV": E_eff_unif,
            "k_uniform": k_unif,
            "X_uniform": X_uniform,
            "k_thermal": k_thermal,
            "X_thermal": X_thermal,
        }

    @staticmethod
    def get_experimental_samples():
        """
        Return experimental samples from Bordeaux / IRB report for 300 °C, 500 V:
        - C13: 15 min (0.25 h) -> Complete crystallization
        - C12: 30 min (0.50 h) -> Complete crystallization
        - C20: 60 min (1.00 h) -> Complete crystallization
        - C11: 90 min (1.50 h) -> Complete crystallization
        """
        target_names = ["C13", "C12", "C20", "C11"]
        matched = []
        for exp in POLING_EXPERIMENTS:
            if exp["sample"] in target_names and exp["T_C"] == 300 and exp["U_V"] == 500:
                matched.append(exp)
        return matched

    def plot_coupled_crystallization(self, save_path="outputs/plot9_coupled_avrami_vs_field.png"):
        """
        Generate Plot 9: 2-panel stacked visualization showing:
        (a) Stage 2 space-charge electric field E(0, t) driving crystallization.
        (b) Resulting dynamic transformed fraction X(t) with experimental markers
            and static-field / thermal-only baselines.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        res_pf = self.solve_cumulative_jmak(model="poole_frenkel")
        res_lin = self.solve_cumulative_jmak(model="linear")
        baselines = self.solve_static_baselines()
        exp_samples = self.get_experimental_samples()

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax_field, ax_jmak) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        # ----------------------------------------------------------------------
        # Top Panel (a): Driving Interfacial Electric Field E(0, t)
        # ----------------------------------------------------------------------
        # Shaded active poling acceleration window (10 to 30 min)
        ax_field.axvspan(10, 30, color="#fff3e0", alpha=0.75, label="Rapid transformation window (10–30 min)")

        # Main field curve
        ax_field.plot(
            self.time_min,
            self.E_interfacial_V_m * 1e-9,
            "-",
            color="#0d47a1",
            lw=2.8,
            label="Dynamic interfacial field $E(x=0, t)$",
        )

        # Static uniform field baseline
        ax_field.axhline(
            baselines["E_uniform_V_m"] * 1e-9,
            color="#d32f2f",
            linestyle="--",
            lw=1.8,
            alpha=0.85,
            label=f"Uniform field baseline $E_{{unif}} = U/W = 0.5\\text{{ MV/m}}$ ($5\\times 10^{{-4}}\\text{{ GV/m}}$)",
        )

        # Annotate peak electric field
        ax_field.scatter(
            [self.t_peak_min],
            [self.E_peak_V_m * 1e-9],
            color="#b71c1c",
            s=90,
            zorder=5,
        )
        ax_field.annotate(
            f"Space-charge peak $E(0) = {self.E_peak_V_m*1e-9:.2f}\\text{{ GV/m}}$\n"
            f"({self.E_peak_MV_cm:.1f} MV/cm at {self.t_peak_min:.1f} min)\n"
            f"$\\approx 4,000\\times$ amplification over $E_{{unif}}$",
            xy=(self.t_peak_min, self.E_peak_V_m * 1e-9),
            xytext=(self.t_peak_min + 10, self.E_peak_V_m * 1e-9 - 0.25),
            arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.6),
            fontsize=9.5,
            fontweight="bold",
            color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#ef9a9a", alpha=0.9),
        )

        ax_field.set_ylabel("Electric Field $E(0, t)$ [GV/m]", fontsize=12, fontweight="bold")
        ax_field.set_ylim(-0.05, 2.45)
        ax_field.set_title(
            "(a) Driving Sub-Anodic Interfacial Electric Field $E(x=0, t)$ vs. Time\n"
            "Derived from Stage 2 Poisson-Coupled Multi-Ion Poling Solver ($300^\\circ\\text{C}, 500\\text{ V}, 1\\text{ mm}$ Glass)",
            fontsize=12,
            fontweight="bold",
        )
        ax_field.legend(loc="upper right", fontsize=9.5, frameon=True)
        ax_field.grid(True, linestyle="--", alpha=0.6)

        # Secondary y-axis in MV/cm
        ax_field_r = ax_field.twinx()
        ax_field_r.set_ylim(-0.5, 24.5)
        ax_field_r.set_ylabel("Field [MV/cm]", fontsize=12, fontweight="bold")
        ax_field_r.grid(False)

        # ----------------------------------------------------------------------
        # Bottom Panel (b): Coupled Transformed Fraction X(t)
        # ----------------------------------------------------------------------
        ax_jmak.axvspan(10, 30, color="#fff3e0", alpha=0.75)

        # Dynamic JMAK curves
        ax_jmak.plot(
            self.time_min,
            res_pf["X_t"],
            "-",
            color="#1b5e20",
            lw=3.0,
            label=f"Dynamic Poole-Frenkel Model ($\\beta_{{PF}} = {self.beta_PF:.2e}$)",
        )
        ax_jmak.plot(
            self.time_min,
            res_lin["X_t"],
            "-.",
            color="#00838f",
            lw=2.4,
            label=f"Dynamic Linear Model ($\\alpha_{{lin}} = {self.alpha_lin:.2e}$)",
        )

        # Static Baselines
        ax_jmak.plot(
            self.time_min,
            baselines["X_uniform"],
            "--",
            color="#d32f2f",
            lw=2.0,
            label="Uniform field baseline ($E = 0.5\\text{ MV/m}$, $X < 10^{-4}$)",
        )
        ax_jmak.plot(
            self.time_min,
            baselines["X_thermal"],
            ":",
            color="#616161",
            lw=2.0,
            label="Thermal-only baseline ($E = 0$, $X < 10^{-4}$)",
        )

        # Overlay Experimental Markers (Report C13, C12, C20, C11)
        colors_exp = ["#e65100", "#c2185b", "#6a1b9a", "#283593"]
        for idx, exp in enumerate(exp_samples):
            t_exp_min = exp["t_h"] * 60.0
            # Target Raman state is complete crystallization (X = 1.0)
            ax_jmak.scatter(
                [t_exp_min],
                [1.0],
                color=colors_exp[idx],
                edgecolor="black",
                s=110,
                zorder=6,
                label=f"Sample {exp['sample']} ({exp['t_h']} h): {exp['state']}",
            )

        # 95% crystallization milestone line
        ax_jmak.axhline(0.95, color="gray", linestyle=":", lw=1.2, alpha=0.8)
        ax_jmak.text(75, 0.93, "95% Transformation Threshold", color="gray", fontsize=9.5, fontstyle="italic")

        ax_jmak.set_xlim(0, 120)
        ax_jmak.set_ylim(-0.02, 1.08)
        ax_jmak.set_xlabel("Poling Time $t$ [min]", fontsize=12, fontweight="bold")
        ax_jmak.set_ylabel("Crystalline Anatase Fraction $X(t)$", fontsize=12, fontweight="bold")
        ax_jmak.set_title(
            "(b) Generalized Dynamic JMAK Crystallization Kinetics $X(t)$\n"
            "Coupled Interfacial Field vs. Uniform Field & Thermal Baselines",
            fontsize=12,
            fontweight="bold",
        )
        ax_jmak.legend(loc="center right", fontsize=9.0, frameon=True, ncol=1)
        ax_jmak.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        safe_savefig(fig, save_path, dpi=300)
        plt.close(fig)
        print(f"[OK] Saved Plot 9 to: {save_path}")

    def plot_barrier_reduction_evolution(self, save_path="outputs/plot10_barrier_reduction_evolution.png"):
        """
        Generate Plot 10: Evolution of effective activation energy E_eff(t)
        over time, illustrating how the space-charge field lowers the barrier
        during the first 30 minutes of poling.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        res_pf = self.solve_cumulative_jmak(model="poole_frenkel")
        res_lin = self.solve_cumulative_jmak(model="linear")
        baselines = self.solve_static_baselines()

        min_E_eff_pf = np.min(res_pf["E_eff_t"])
        min_E_eff_lin = np.min(res_lin["E_eff_t"])
        delta_E_max_pf = self.Ea_eV - min_E_eff_pf

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(11, 6))

        # Highlight first 30 minutes
        ax.axvspan(0, 30, color="#e8f5e9", alpha=0.6, label="Primary barrier suppression window ($0–30\\text{ min}$)")

        # Base thermal activation energy line
        ax.axhline(
            self.Ea_eV,
            color="#212121",
            linestyle="--",
            lw=2.0,
            label=f"Base thermal barrier $E_a = {self.Ea_eV:.2f}\\text{{ eV/atom}}$ ($E=0$)",
        )

        # Uniform field barrier line
        ax.axhline(
            baselines["E_eff_uniform_eV"],
            color="#d32f2f",
            linestyle=":",
            lw=1.8,
            label=f"Uniform field barrier $E_{{eff}} = {baselines['E_eff_uniform_eV']:.4f}\\text{{ eV}}$ ($0.5\\text{{ MV/m}}$)",
        )

        # Dynamic Poole-Frenkel barrier curve
        ax.plot(
            self.time_min,
            res_pf["E_eff_t"],
            "-",
            color="#1b5e20",
            lw=2.8,
            label=f"Poole-Frenkel $E_{{eff}}(t) = E_a - \\beta_{{PF}}\\sqrt{{E(t)}}$",
        )

        # Dynamic Linear barrier curve
        ax.plot(
            self.time_min,
            res_lin["E_eff_t"],
            "-.",
            color="#00838f",
            lw=2.2,
            label=f"Linear $E_{{eff}}(t) = E_a - \\alpha_{{lin}} E(t)$",
        )

        # Minimum barrier annotation at t = 18.5 min
        ax.scatter([self.t_peak_min], [min_E_eff_pf], color="#b71c1c", s=90, zorder=5)
        ax.annotate(
            f"Minimum barrier $E_{{eff}} = {min_E_eff_pf:.3f}\\text{{ eV}}$\n"
            f"($\\Delta E = {delta_E_max_pf*1e3:.0f}\\text{{ meV}}$ at $t = {self.t_peak_min:.1f}\\text{{ min}}$)",
            xy=(self.t_peak_min, min_E_eff_pf),
            xytext=(self.t_peak_min + 12, min_E_eff_pf - 0.07),
            arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.6),
            fontsize=10,
            fontweight="bold",
            color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#ef9a9a", alpha=0.9),
        )

        # Secondary y-axis showing barrier reduction dE in meV
        ax_r = ax.twinx()
        ax_r.set_ylim((self.Ea_eV - 1.05) * 1e3, (self.Ea_eV - 1.62) * 1e3)
        ax_r.set_ylabel("Barrier Lowering $\\Delta E(t)$ [meV]", fontsize=11, fontweight="bold")
        ax_r.grid(False)

        ax.set_xlim(0, 120)
        ax.set_ylim(1.05, 1.62)
        ax.set_xlabel("Poling Time $t$ [min]", fontsize=12, fontweight="bold")
        ax.set_ylabel("Effective Activation Energy $E_{eff}(t)$ [eV/atom]", fontsize=12, fontweight="bold")
        ax.set_title(
            "Temporal Evolution of Effective Activation Energy $E_{eff}(t)$\n"
            "Demonstrating Space-Charge Field-Assisted Barrier Reduction in TiO$_2$",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        safe_savefig(fig, save_path, dpi=300)
        plt.close(fig)
        print(f"[OK] Saved Plot 10 to: {save_path}")
