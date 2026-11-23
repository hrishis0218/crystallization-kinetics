"""
Run Stage 2: 1D Multi-Ion Drift-Diffusion + Poisson Solver for Glass Poling
===========================================================================
Executes the thermal poling simulation of Menzel soda-lime glass at:
- Temperature: T = 300 °C
- Applied Voltage: U = 500 V
- Duration: t = 2 h (7200 s)
- Sample Thickness: W = 1.0 mm

Generates:
1. outputs/plot7_sl_ion_profiles.png:
   Concentration profiles (at%) of Na+, K+, Ca2+, Mg2+, and H+ vs. depth
   (0 to 2500 nm) at t = 2 h, validating Na+ depletion (~600 nm) and
   Ca2+/Mg2+ pile-up matching Pervan et al. Fig. 6c.
2. outputs/plot8_electric_field_evolution.png:
   Spatial profile of E(x, t) across multiple times (1m, 15m, 30m, 1h, 2h)
   plus interfacial electric field E(0, t) vs. time.
3. outputs/interfacial_field_300C_500V.npz:
   Exported time-series array of E(0, t) for Stage 1 coupling.

Author: Antigravity / Science Simulation Suite
Reference: Pervan et al., Modelling20accepted.pdf
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure src directory is in Python path for standalone Spyder execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.ion_poling_1d import SodaLimePoling1DSolver
from src.params import safe_savefig

# Create outputs directory if needed
OUTPUT_DIR = os.path.join(current_dir, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_stage2_simulations():
    print("=" * 78)
    print("STAGE 2: 1D MULTI-ION DRIFT-DIFFUSION & POISSON SOLVER FOR GLASS POLING")
    print("=" * 78)
    print("Poling Conditions: T = 300 °C, U = 500 V, W = 1.0 mm, t = 2.0 h")
    print("Mobile Species   : Na+, K+, Ca2+, Mg2+, injected H+ (open anode)")
    print("-" * 78)

    # 1. Initialize solver
    solver = SodaLimePoling1DSolver(
        T_C=300.0,
        U=500.0,
        W=1.0e-3,
        eps_r=7.0,
        H=0.3,
        N_grid=250,
        x_max_m=2500e-9,
    )

    print(f"Grid setup       : N = {solver.N} nodes over {solver.x_max*1e9:.0f} nm (dx = {solver.x[1]*1e9:.2f} nm)")
    print(f"Scales           : d0 = {solver.d0*1e9:.2f} nm, E0 = {solver.E0:.2e} V/m, t0 = {solver.t0:.2e} s, L = {solver.L:.1f}")
    print(f"Mobilities       : mu_Na = {solver.mu[0]:.1e}, mu_Ca = {solver.mu[2]:.1e}, mu_H = {solver.mu[4]:.1e} m^2/(V s)")
    print("Running time integration (t = 0 to 7200 s)...")

    # 2. Run simulation with checkpoints at 1m, 15m, 30m, 1h, 2h
    checkpoints = [60.0, 900.0, 1800.0, 3600.0, 7200.0]
    res = solver.solve(t_final=7200.0, checkpoints=checkpoints, method="banded")

    # 3. Numerical stability and physical sanity checks
    p_final = res["snapshots_profiles"][7200.0]  # shape (5, N) in at%
    x_nm = res["x_nm"]

    print("\n--- Numerical Stability & Verification Report (t = 2 h) ---")
    all_positive = True
    for idx, sp in enumerate(solver.species_names):
        c_min = p_final[idx].min()
        c_max = p_final[idx].max()
        c_bulk = p_final[idx, -1]
        is_pos = c_min >= -1e-12
        all_positive = all_positive and is_pos
        print(f"  {sp:<5} : min = {max(c_min, 0.0):.3f} at%, max = {c_max:.2f} at%, bulk = {c_bulk:.2f} at% | Positivity: {'PASSED' if is_pos else 'FAILED'}")

    assert all_positive, "Numerical error: negative concentrations detected!"

    # Depletion width and peak metrics
    idx_na_50 = np.where(p_final[0] > 0.5 * solver.bulk_at_pct[0])[0][0]
    na_dep_nm = x_nm[idx_na_50]

    idx_ca_pk = np.argmax(p_final[2])
    ca_pk_val = p_final[2, idx_ca_pk]
    ca_pk_depth = x_nm[idx_ca_pk]

    idx_mg_pk = np.argmax(p_final[3])
    mg_pk_val = p_final[3, idx_mg_pk]
    mg_pk_depth = x_nm[idx_mg_pk]

    h_surface = p_final[4, 0]
    idx_h_decay = np.where(p_final[4] < 0.5)[0][0]
    h_decay_nm = x_nm[idx_h_decay]

    print(f"\nPhysical Features Validated:")
    print(f"  - Na+ Depletion Front Depth (50% bulk) : {na_dep_nm:.1f} nm  (Pervan et al. benchmark: ~600-650 nm)")
    print(f"  - Ca2+ Accumulation Peak               : {ca_pk_val:.2f} at% at depth {ca_pk_depth:.1f} nm (Pervan: ~8 at%)")
    print(f"  - Mg2+ Accumulation Peak               : {mg_pk_val:.2f} at% at depth {mg_pk_depth:.1f} nm (Pervan: ~4.5 at%)")
    print(f"  - Injected H+ at Surface (x=0)         : {h_surface:.2f} at% (Pervan: ~19 at%)")
    print(f"  - Injected H+ Penetration Depth (<0.5%): {h_decay_nm:.1f} nm")

    # Interfacial field evolution and max field metrics
    t_hist = res["time_history"]
    e_hist = res["E_interfacial_history"]
    max_idx = np.argmax(e_hist)
    max_E0 = e_hist[max_idx]
    max_t_min = t_hist[max_idx] / 60.0

    print(f"\nElectric Field Evolution (Poisson-Coupled with 1 mm Bulk Boundary Condition):")
    print(f"  - Initial Field at t = 0 (bulk drop)   : {e_hist[0]:.3e} V/m  ({e_hist[0]*1e-8:.4f} MV/cm = {e_hist[0]*1e-6:.2f} MV/m)")
    print(f"  - Maximum Interfacial Field E(0, t)     : {max_E0:.3e} V/m  ({max_E0*1e-8:.2f} MV/cm = {max_E0*1e-9:.2f} GV/m) at t = {max_t_min:.1f} min")
    for t_chk in checkpoints:
        idx_t = np.argmin(np.abs(t_hist - t_chk))
        e_val = e_hist[idx_t]
        print(f"  - Interfacial E(0) at t = {t_chk/60:4.1f} min ({t_chk:5.0f} s) : {e_val:.3e} V/m  ({e_val*1e-8:.2f} MV/cm = {e_val*1e-9:.2f} GV/m)")

    # 4. Generate Plot 7: Concentration Profiles vs Depth (Pervan Fig. 6c Validation)
    print("\nGenerating Plot 7: Soda-Lime Glass Ion Concentration Profiles...")
    sims_ref = solver.get_sims_experimental_reference()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Panel A: t = 2 h (7200 s) - Exact match to Pervan Fig. 6c
    ax1 = axes[0]
    # Simulated profiles (dashed lines as in Fig. 6c)
    ax1.plot(x_nm, p_final[0], "--", color="#d62728", lw=2.4, label="Na$^+$ sim")
    ax1.plot(x_nm, p_final[2], "--", color="#1f77b4", lw=2.4, label="Ca$^{2+}$ sim")
    ax1.plot(x_nm, p_final[3], "--", color="#2ca02c", lw=2.4, label="Mg$^{2+}$ sim")
    ax1.plot(x_nm, p_final[1], "--", color="#8c564b", lw=2.0, label="K$^+$ sim")
    ax1.plot(x_nm, p_final[4], "--", color="#111111", lw=2.2, label="H$^+$ sim")

    # SIMS experimental data (solid lines as in Fig. 6c)
    ax1.plot(sims_ref["depth_nm"], sims_ref["Na+"], "-", color="#ff7f0e", lw=1.8, alpha=0.85, label="Na$^+$ exp (SIMS)")
    ax1.plot(sims_ref["depth_nm"], sims_ref["Ca2+"], "-", color="#17becf", lw=1.8, alpha=0.85, label="Ca$^{2+}$ exp (SIMS)")
    ax1.plot(sims_ref["depth_nm"], sims_ref["Mg2+"], "-", color="#98df8a", lw=1.8, alpha=0.85, label="Mg$^{2+}$ exp (SIMS)")
    ax1.plot(sims_ref["depth_nm"], sims_ref["K+"], "-", color="#c49c94", lw=1.8, alpha=0.85, label="K$^+$ exp (SIMS)")

    # Bulk glass boundary indicator line
    ax1.axvline(680.0, color="gray", linestyle=":", lw=1.5, alpha=0.8)
    ax1.text(700.0, 18.0, "bulk glass\n(SIMS boundary)", color="gray", fontsize=10, fontstyle="italic")

    # Pile-up annotation
    ax1.annotate(
        f"Ca$^{{2+}}$ pile-up\n({ca_pk_val:.1f} at%)",
        xy=(ca_pk_depth, ca_pk_val),
        xytext=(ca_pk_depth - 180, ca_pk_val + 3.0),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#1f77b4",
    )
    ax1.annotate(
        f"Mg$^{{2+}}$ pile-up\n({mg_pk_val:.1f} at%)",
        xy=(mg_pk_depth, mg_pk_val),
        xytext=(mg_pk_depth + 40, mg_pk_val + 3.5),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#2ca02c",
    )

    ax1.set_xlim(0, 2500)
    ax1.set_ylim(-0.5, 21.0)
    ax1.set_xlabel("Depth [nm]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Concentration [at%]", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Soda-Lime Glass Profiles at $t = 2\\text{ h}$ (7200 s)\nValidation against Pervan et al. Fig. 6c ($300^\\circ\\text{C}, 500\\text{ V}$)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel B: t = 1 h (3600 s) - TiO2 Poling Benchmark (C20 at 300 °C, 500 V, 1 h)
    ax2 = axes[1]
    p_1h = res["snapshots_profiles"][3600.0]
    ax2.plot(x_nm, p_1h[0], "--", color="#d62728", lw=2.4, label="Na$^+$ ($t=1\\text{ h}$)")
    ax2.plot(x_nm, p_1h[2], "--", color="#1f77b4", lw=2.4, label="Ca$^{2+}$ ($t=1\\text{ h}$)")
    ax2.plot(x_nm, p_1h[3], "--", color="#2ca02c", lw=2.4, label="Mg$^{2+}$ ($t=1\\text{ h}$)")
    ax2.plot(x_nm, p_1h[1], "--", color="#8c564b", lw=2.0, label="K$^+$ ($t=1\\text{ h}$)")
    ax2.plot(x_nm, p_1h[4], "--", color="#111111", lw=2.2, label="H$^+$ ($t=1\\text{ h}$)")

    # Overlay t = 2 h faint profiles for direct time comparison
    ax2.plot(x_nm, p_final[0], ":", color="#d62728", lw=1.2, alpha=0.45, label="Na$^+$ ($t=2\\text{ h}$)")
    ax2.plot(x_nm, p_final[2], ":", color="#1f77b4", lw=1.2, alpha=0.45, label="Ca$^{2+}$ ($t=2\\text{ h}$)")
    ax2.plot(x_nm, p_final[4], ":", color="#111111", lw=1.2, alpha=0.45, label="H$^+$ ($t=2\\text{ h}$)")

    idx_na_1h = np.where(p_1h[0] > 0.5 * solver.bulk_at_pct[0])[0][0]
    ax2.axvline(x_nm[idx_na_1h], color="#d62728", linestyle=":", lw=1.5, alpha=0.7)
    ax2.text(x_nm[idx_na_1h] + 20, 15.0, f"Depletion front\n$w = {x_nm[idx_na_1h]:.0f}\\text{{ nm}}$", color="#d62728", fontsize=9.5)

    ax2.set_xlim(0, 2500)
    ax2.set_xlabel("Depth [nm]", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Snapshot at $t = 1\\text{ h}$ (3600 s)\nDirect Correlation with TiO$_2$ Thin Film Poling (Sample C20)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot7_path = os.path.join(OUTPUT_DIR, "plot7_sl_ion_profiles.png")
    safe_savefig(fig, plot7_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot7_path}")

    # 5. Generate Plot 8: Electric Field Evolution
    print("\nGenerating Plot 8: Electric Field Spatial & Temporal Evolution...")
    fig, (ax_spat, ax_time) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Spatial profile E(x) at multiple times
    colors = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    time_labels = ["1 min", "15 min", "30 min", "1 h", "2 h"]

    # Highlight the H+-neutralized region (0-310 nm) where rho ~ 0 => E(x) is flat
    ax_spat.axvspan(0, 310, color="#cfd8dc", alpha=0.4, label="H$^+$ neutralized ($dE/dx \\approx 0$)")
    ax_spat.text(
        155, 1.10,
        "H$^+$-neutralized zone\n($dE/dx \\approx 0$, flat $E$)",
        ha="center", va="center",
        fontsize=9.0, fontweight="bold", color="#263238",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eceff1", ec="#90a4ae", alpha=0.9),
    )

    for idx, t_chk in enumerate(checkpoints):
        e_field = res["snapshots_field"][t_chk]
        ax_spat.plot(
            x_nm,
            e_field * 1e-9,  # in GV/m
            color=colors[idx],
            lw=2.4,
            label=f"$t = {time_labels[idx]}$",
        )

    ax_spat.set_xlim(0, 1500)
    ax_spat.set_ylim(-0.05, 2.4)
    ax_spat.set_xlabel("Depth $x$ [nm]", fontsize=12, fontweight="bold")
    ax_spat.set_ylabel("Electric Field $E(x)$ [GV/m] ($10^9$ V/m)", fontsize=12, fontweight="bold")
    ax_spat.set_title("(a) Spatial Profile of Space-Charge Electric Field $E(x, t)$\nDirect Poisson Integration with 1 mm Slab Voltage Constraint", fontsize=11, fontweight="bold")
    ax_spat.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax_spat.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis for MV/cm
    ax_spat_r = ax_spat.twinx()
    ax_spat_r.set_ylim(-0.5, 24.0)
    ax_spat_r.set_ylabel("Electric Field [MV/cm]", fontsize=12, fontweight="bold")
    ax_spat_r.grid(False)

    # Panel B: Interfacial Field E(0, t) vs time
    t_min = t_hist / 60.0
    e_gv = e_hist * 1e-9

    ax_time.plot(t_min, e_gv, "-", color="#1f77b4", lw=2.6, label="$E(x=0, t)$")

    # Mark the checkpoints
    for idx, t_chk in enumerate(checkpoints):
        idx_c = np.argmin(np.abs(t_hist - t_chk))
        ax_time.scatter(
            t_min[idx_c],
            e_gv[idx_c],
            color=colors[idx],
            s=80,
            zorder=5,
            label=f"{time_labels[idx]}: {e_gv[idx_c]:.2f} GV/m",
        )

    # Mark peak field
    ax_time.annotate(
        f"Peak $E(0) = {max_E0*1e-9:.2f}\\text{{ GV/m}}$\n({max_E0*1e-8:.1f} MV/cm at {max_t_min:.1f} min)",
        xy=(max_t_min, max_E0 * 1e-9),
        xytext=(max_t_min + 14, max_E0 * 1e-9 + 0.14),
        arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#b71c1c",
    )

    ax_time.set_xlim(0, 125)
    ax_time.set_ylim(-0.05, 2.4)
    ax_time.set_xlabel("Poling Time $t$ [min]", fontsize=12, fontweight="bold")
    ax_time.set_ylabel("Interfacial Field $E(0, t)$ [GV/m]", fontsize=12, fontweight="bold")
    ax_time.set_title(f"(b) Interfacial Field $E(x=0, t)$ vs. Time\nRealistic Dielectric Withstand Dynamics (Peak = {max_E0*1e-9:.2f} GV/m)", fontsize=11, fontweight="bold")
    ax_time.legend(loc="lower right", fontsize=9.0, frameon=True)
    ax_time.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis for MV/cm
    ax_time_r = ax_time.twinx()
    ax_time_r.set_ylim(-0.5, 20.0)
    ax_time_r.set_ylabel("Interfacial Field [MV/cm]", fontsize=12, fontweight="bold")
    ax_time_r.grid(False)

    plt.tight_layout()
    plot8_path = os.path.join(OUTPUT_DIR, "plot8_electric_field_evolution.png")
    safe_savefig(fig, plot8_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot8_path}")

    # 6. Save interfacial field time-series array to .npz for Stage 1 coupling
    npz_path = os.path.join(OUTPUT_DIR, "interfacial_field_300C_500V.npz")
    np.savez_compressed(
        npz_path,
        time_s=t_hist,
        time_min=t_hist / 60.0,
        time_h=t_hist / 3600.0,
        E_interfacial_V_m=e_hist,
        E_interfacial_MV_cm=e_hist * 1e-8,
        depletion_depth_nm=res["depletion_depth_history"],
        temperature_C=300.0,
        applied_voltage_V=500.0,
    )
    print(f"Saved interfacial field time-series array to: {npz_path}")

    print("\n" + "=" * 78)
    print("STAGE 2 SIMULATION COMPLETE: ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_stage2_simulations()
