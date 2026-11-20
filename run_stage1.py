#!/usr/bin/env python3
"""
Main Execution Script - Stage 1: TiO2 Crystallization Kinetics & Field-Assisted Poling.

Executes all Stage 1 simulation modules and generates publication-grade plots:
1. Wooding et al. (2023) JMAK isothermal S-curves (Fig. 4 reproduction).
2. Wooding et al. (2023) TTT diagrams for 5%, 50%, 95% crystallization (Fig. 9 reproduction).
3. Field-assisted barrier reduction kinetics (Linear vs. Poole-Frenkel models).
4. Phase crystallization regime map in (T, U) space overlaid with experimental Report samples C1-C20.
5. 2D spatial nucleation-and-growth microstructure simulation (4-panel evolution snapshots).
6. Quantitative verification of 2D simulated area fraction vs. analytical Avrami JMAK (n = 3).

Spyder / IDE friendly: uses relative paths and self-contained execution.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in Python module search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.params import (
    WOODING_ARRHENIUS,
    WOODING_TABLE_2,
    POLING_EXPERIMENTS,
    FIELD_BASE_EA,
    FIELD_BASE_K0,
    FIELD_BASE_N,
)
from src.avrami_jmak import (
    plot_wooding_scurves,
    plot_wooding_ttt,
    plot_field_assisted_kinetics,
    plot_crystallization_regime_map,
    field_assisted_fraction,
)
from src.avrami_2d import (
    Avrami2DSimulator,
    plot_avrami_2d_snapshots,
    plot_avrami_2d_verification,
)


def print_banner(text):
    line = "=" * 76
    print(f"\n{line}\n  {text}\n{line}")


def main():
    start_time = time.time()
    print_banner("TiO2 Crystallization & Thermal Poling Suite - Stage 1 Runner")

    # Ensure output directory exists
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Project root directory: {PROJECT_ROOT}")
    print(f"[*] Output directory:       {output_dir}")

    # ==========================================================================
    # Task 1: Wooding et al. Analytical JMAK Validation
    # ==========================================================================
    print_banner("1. Validating JMAK Kinetics against Wooding et al. (2023)")
    print(f"[*] Published parameters:")
    print(f"    - Consensus Activation Energy: {1.48:.2f} eV/atom")
    print(f"    - Mean Avrami Exponent:        n ~ 2.75 (range 2.31 - 3.39)")
    for dep_T, data in WOODING_ARRHENIUS.items():
        print(f"    - Deposition {dep_T} °C: k0 = {data['k0']:.1e} s^-1, series Ea = {data['Ea_series']:.2f} eV")

    plot1_path = os.path.join(output_dir, "plot1_wooding_scurves.png")
    plot_wooding_scurves(save_path=plot1_path)

    plot2_path = os.path.join(output_dir, "plot2_wooding_ttt.png")
    plot_wooding_ttt(save_path=plot2_path)

    # ==========================================================================
    # Task 2: Field-Assisted Crystallization Extension (Report Calibration)
    # ==========================================================================
    print_banner("2. Evaluating Field-Assisted Barrier Reduction Models")
    print(f"[*] Base thermal barrier (0 V): Ea = {FIELD_BASE_EA:.2f} eV, k0 = {FIELD_BASE_K0:.1e} s^-1")
    print(f"[*] Model evaluation across Report experimental matrix:")

    table_header = f"{'Sample':<8} {'T [°C]':<8} {'U [V]':<8} {'t [h]':<8} {'Report State':<26} {'Pred. X(PF)':<12} {'Pred. X(Lin)':<12}"
    print("-" * len(table_header))
    print(table_header)
    print("-" * len(table_header))

    for exp in POLING_EXPERIMENTS:
        t_sec = exp["t_h"] * 3600.0
        T_C = exp["T_C"]
        U_V = exp["U_V"]
        x_pf = field_assisted_fraction(t_sec, T_C, U_V, model="poole_frenkel")
        x_lin = field_assisted_fraction(t_sec, T_C, U_V, model="linear")
        print(f"{exp['sample']:<8} {T_C:<8} {U_V:<8} {exp['t_h']:<8.2f} {exp['state']:<26} {x_pf:<12.3f} {x_lin:<12.3f}")

    print("-" * len(table_header))

    plot3_path = os.path.join(output_dir, "plot3_field_assisted_kinetics.png")
    plot_field_assisted_kinetics(save_path=plot3_path)

    plot4_path = os.path.join(output_dir, "plot4_crystallization_regime_map.png")
    plot_crystallization_regime_map(save_path=plot4_path)

    # ==========================================================================
    # Task 3: 2D Spatial Nucleation and Growth Simulation
    # ==========================================================================
    print_banner("3. Running 2D Spatial Nucleation-and-Growth Simulation")
    print(f"[*] Initializing 2D domain: 350x350 grid, 10 um x 10 um, periodic boundaries...")
    sim2d = Avrami2DSimulator(grid_size=350, domain_length=10.0e-6, seed=123)
    
    t_sim_start = time.time()
    n_grains = sim2d.run(total_time_s=3600.0)
    t_sim_dur = time.time() - t_sim_start
    print(f"[OK] 2D Simulation completed in {t_sim_dur:.2f} s. Total nucleated grains: {n_grains}")

    # Verify Avrami exponent and rate
    t_eval = [600, 1200, 1800, 2400]
    x_sim_vals, x_ana_vals = sim2d.compute_fraction_series(t_eval)
    for t_v, xs, xa in zip(t_eval, x_sim_vals, x_ana_vals):
        print(f"    - t = {t_v:4d} s: Simulated X = {xs:.3f} | Analytical JMAK (n=3) = {xa:.3f} | Diff = {xs - xa:+.3f}")

    k_fit, n_fit = sim2d.fit_avrami(np.linspace(100, 2400, 60),
                                   sim2d.compute_fraction_series(np.linspace(100, 2400, 60))[0])
    print(f"[+] Fitted Avrami parameters from 2D simulation: n = {n_fit:.2f} (theoretical n = 3.00), k = {k_fit:.2e} s^-1")

    plot5_path = os.path.join(output_dir, "plot5_avrami_2d_snapshots.png")
    plot_avrami_2d_snapshots(sim2d, save_path=plot5_path)

    plot6_path = os.path.join(output_dir, "plot6_avrami_2d_verification.png")
    plot_avrami_2d_verification(sim2d, save_path=plot6_path)

    # ==========================================================================
    # Completion Summary
    # ==========================================================================
    total_elapsed = time.time() - start_time
    print_banner(f"Stage 1 Execution Successfully Finished in {total_elapsed:.2f} s")
    print("[*] All generated artifacts in outputs/:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        fsize_kb = os.path.getsize(fpath) / 1024.0
        print(f"    -> {f:<38} ({fsize_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
