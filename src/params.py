"""
Physical constants, experimental parameters, and configuration settings
for TiO2 crystallization kinetics and field-assisted thermal poling.

References:
- Wooding et al., Atomic Layer Deposition 1: 1-18 (2023), DOI: 10.3897/aldj.1.101276
  (ALD TiO2 JMAK parameters, n ~ 2.75-3.0, Ea ~ 1.48 eV/atom, k0 for 120/140/160 C)
- Report on TiO2 crystallization (Bordeaux & IRB experiments on EFAD of TiO2 thin films)
- Pervan et al., Modelling20accepted.pdf (Stage 2 multi-ion glass poling parameters)
"""

import numpy as np

# ==============================================================================
# 1. Fundamental Physical Constants (SI & eV)
# ==============================================================================
K_B_EV = 8.617333262e-5       # Boltzmann constant [eV/K]
K_B_J = 1.380649e-23          # Boltzmann constant [J/K]
E_CHARGE = 1.602176634e-19    # Elementary electric charge [C]
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
NA = 6.02214076e23            # Avogadro constant [1/mol]

# Material properties
EPS_R_TIO2 = 48.0             # Relative dielectric permittivity of TiO2 thin film
EPS_R_GLASS = 7.0             # Relative dielectric permittivity of soda-lime glass (Pervan et al.)
TIO2_FILM_THICKNESS = 350e-9  # Nominal TiO2 film thickness [m] (350 nm from Report)
TIO2_DENSITY_G_CM3 = 3.9      # Density of amorphous/anatase TiO2 [g/cm^3]
TIO2_MOLAR_MASS_G_MOL = 79.87 # Molar mass of TiO2 [g/mol]


# ==============================================================================
# 2. Wooding et al. (2023) Experimental Dataset (ALD TiO2 Post-Deposition Annealing)
# ==============================================================================
# Table 2: Measured Avrami exponents n and reaction rates k [s^-1]
# Format: (Dep_T_C, PDA_T_C): (n, n_err, k_s_inv, k_err_s_inv)
WOODING_TABLE_2 = {
    # Deposition at 160 °C
    (160, 200): {"n": 2.51, "n_err": 0.04, "k": 9.6e-5, "k_err": 0.05e-5},
    (160, 180): {"n": 3.39, "n_err": 0.08, "k": 2.2e-5, "k_err": 0.007e-5},
    (160, 160): {"n": 2.85, "n_err": 0.08, "k": 4.0e-6, "k_err": 0.04e-6},

    # Deposition at 140 °C
    (140, 200): {"n": 2.80, "n_err": 0.02, "k": 4.6e-5, "k_err": 0.004e-5},
    (140, 180): {"n": 2.82, "n_err": 0.09, "k": 8.8e-6, "k_err": 0.02e-6},
    (140, 160): {"n": 3.25, "n_err": 0.04, "k": 1.3e-6, "k_err": 0.005e-6},

    # Deposition at 120 °C
    (120, 220): {"n": 2.34, "n_err": 0.02, "k": 6.8e-6, "k_err": 0.04e-6},
    (120, 200): {"n": 2.31, "n_err": 0.02, "k": 2.2e-6, "k_err": 0.009e-6},
    (120, 180): {"n": 2.49, "n_err": 0.04, "k": 4.1e-7, "k_err": 0.05e-7},
}

# Table 3 & text: Arrhenius parameters evaluated at average Ea = 1.48 eV/atom
# k(T) = k0 * exp(-Ea / (k_B * T))
WOODING_ARRHENIUS = {
    120: {
        "k0": 1.1e10,         # Pre-exponential frequency factor [s^-1]
        "k0_err": 0.04e10,
        "Ea_series": 1.46,     # Series fitted Ea [eV/atom]
        "Ea_avg": 1.48,        # Consensus average Ea [eV/atom]
        "delta_G_nuc": 1.41,   # Critical Gibbs energy for nucleation [eV/atom]
        "n_avg": 2.38,
    },
    140: {
        "k0": 2.5e11,
        "k0_err": 0.2e11,
        "Ea_series": 1.58,
        "Ea_avg": 1.48,
        "delta_G_nuc": 1.35,
        "n_avg": 2.96,
    },
    160: {
        "k0": 6.2e11,
        "k0_err": 0.3e11,
        "Ea_series": 1.40,
        "Ea_avg": 1.48,
        "delta_G_nuc": 1.32,
        "n_avg": 2.92,
    },
}

# Global consensus Avrami exponent across all Wooding et al. series
WOODING_GLOBAL_N = 2.75
WOODING_GLOBAL_EA = 1.48  # eV/atom


# ==============================================================================
# 3. Experimental Poling (EFAD) Dataset from Report
# ==============================================================================
# TiO2 thin films (~350 nm) deposited on Menzel soda-lime glass
# Annealed / thermally poled across T = 150 - 300 °C, U = 250 - 1250 V, t = 0.25 - 1.5 h
POLING_EXPERIMENTS = [
    {"sample": "C1",  "T_C": 300, "U_V": 250,  "t_h": 1.0,  "state": "No crystallization", "state_code": 0},
    {"sample": "C10", "T_C": 150, "U_V": 500,  "t_h": 1.0,  "state": "No crystallization", "state_code": 0},
    {"sample": "C9",  "T_C": 200, "U_V": 500,  "t_h": 1.0,  "state": "No crystallization", "state_code": 0},
    {"sample": "C8",  "T_C": 250, "U_V": 500,  "t_h": 1.0,  "state": "Transition phase",    "state_code": 1},
    {"sample": "C13", "T_C": 300, "U_V": 500,  "t_h": 0.25, "state": "Complete crystallization", "state_code": 2},
    {"sample": "C12", "T_C": 300, "U_V": 500,  "t_h": 0.5,  "state": "Complete crystallization", "state_code": 2},
    {"sample": "C20", "T_C": 300, "U_V": 500,  "t_h": 1.0,  "state": "Complete crystallization", "state_code": 2},
    {"sample": "C11", "T_C": 300, "U_V": 500,  "t_h": 1.5,  "state": "Complete crystallization", "state_code": 2},
    {"sample": "C19", "T_C": 300, "U_V": 750,  "t_h": 1.0,  "state": "Complete crystallization", "state_code": 2},
    {"sample": "C4",  "T_C": 300, "U_V": 1000, "t_h": 1.0,  "state": "Complete crystallization", "state_code": 2},
    {"sample": "C5",  "T_C": 300, "U_V": 1250, "t_h": 1.0,  "state": "Complete crystallization", "state_code": 2},
]


# ==============================================================================
# 4. Field-Assisted Crystallization Model Parameters
# ==============================================================================
# Effective layer thickness across which the applied potential drops.
# During thermal poling of glass, cations deplete in a subanodic region d_eff ~ 350 - 500 nm.
# Nominal electric field E = U / D_EFF_FIELD
D_EFF_FIELD = 350e-9  # [m]

# Base thermal activation energy and prefactor for high-stability amorphous TiO2
# Pure thermal annealing at 300 °C for 1 h shows negligible crystallization (X < 0.001)
FIELD_BASE_EA = 1.55    # [eV/atom]
FIELD_BASE_K0 = 2.0e8   # [s^-1]
FIELD_BASE_N = 2.75

# Threshold voltage for sub-anodic space-charge layer formation in soda-lime glass.
# Below U_th (~200-250 V), ion depletion is incomplete and the field is screened.
# Above U_th, the field across the ~350 nm subanodic TiO2 film is E = (U - U_th) / d_eff.
U_DEPLETION_THRESHOLD_V = 220.0  # [V]

# Poole-Frenkel barrier lowering: dE_PF = beta_PF * sqrt(E)
# Theoretical beta_PF = sqrt(e^3 / (pi * epsilon_0 * eps_r)) in J*(m/V)^1/2 -> converted to eV*(m/V)^1/2
BETA_PF_THEORETICAL = np.sqrt(E_CHARGE**3 / (np.pi * EPSILON_0 * EPS_R_TIO2)) / E_CHARGE  # [eV*(m/V)^1/2]
# Calibrated Poole-Frenkel parameter matching C8 transition at 250 °C and C1 threshold at 250 V
BETA_PF_EFFECTIVE = 1.06e-5  # [eV*(m/V)^1/2]

# Linear barrier lowering: dE_lin = alpha_lin * E
# alpha = q * a_hop (effective dipole length / ionic hopping distance ~ 2 - 3 A)
ALPHA_LINEAR = 3.75e-10  # [eV / (V/m)]




# ==============================================================================
# 5. 2D Microstructural Simulation Parameters
# ==============================================================================
SIM_2D_GRID_SIZE = 350          # 350 x 350 cells
SIM_2D_PHYSICAL_L = 10.0e-6     # 10 micrometers x 10 micrometers domain
SIM_2D_TIME_STEPS = 120         # Time discretization steps
SIM_2D_TOTAL_TIME_S = 3600.0    # 1 hour simulation window
SIM_2D_NUCLEATION_RATE = 2.5e8  # Nucleation rate [nuclei / (m^2 * s)]
SIM_2D_GROWTH_VELOCITY = 1.2e-9 # Radial crystal growth rate [m/s]


# ==============================================================================
# 6. Soda-Lime Glass Multi-Ion Poling Parameters (Pervan et al., Tables 2 & 3)
# ==============================================================================
# Menzel microscope slide glass composition in atomic percent (at%)
GLASS_MENZEL_COMPOSITION_AT_PCT = {
    "Na": 9.5,
    "K": 0.5,
    "Ca": 2.4,
    "Mg": 2.2,
    "H": 0.0,  # 0 at% in bulk, injected at anode interface during poling
    "Si": 24.7,
    "O": 60.0,
    "Al": 0.6,
}

# Mobile ionic species valences q_i
GLASS_ION_VALENCES = {
    "Na": 1,
    "K": 1,
    "Ca": 2,
    "Mg": 2,
    "H": 1,
}

# Effective ionic mobilities mu_i [m^2 / (V s)] at 300 °C from Pervan et al. Table 3
GLASS_MOBILITIES_300C = {
    "Na": 5.4e-15,  # Fastest alkali cation (reference species i=1)
    "K": 2.0e-15,   # Alkali cation
    "Ca": 9.0e-20,  # Alkaline earth cation
    "Mg": 3.0e-20,  # Alkaline earth cation
    "H": 2.0e-20,   # Injected proton / hydronium ion
}

# Physical and poling parameters
GLASS_THICKNESS_W = 1.0e-3        # Physical glass substrate thickness [m] (1 mm)
GLASS_STATIC_EPS_R = 7.0          # Static dielectric constant of soda-lime glass
GLASS_HAVEN_RATIO = 0.3           # Haven ratio H ~ 0.3 (Fitzgerald / Pervan et al.)
GLASS_DENSITY_G_CM3 = 2.50        # Soda-lime glass density [g/cm^3]
GLASS_AVG_MOLAR_MASS = 20.575     # Average molar mass per glass atom [g/mol]
POLING_VOLTAGE_U = 500.0          # Applied poling voltage [V]
POLING_TEMPERATURE_C = 300.0      # Poling temperature [°C]
POLING_TEMPERATURE_K = 573.15     # Poling temperature [K]
POLING_DURATION_S = 7200.0        # Total poling duration [s] (2 hours)


# ==============================================================================
# 7. Robust Figure Saving Helper (Windows File Lock Tolerant)
# ==============================================================================
def safe_savefig(fig_or_plt, save_path, **kwargs):
    """
    Safely save a matplotlib figure, retrying if transient Windows file locks
    (such as indexer, thumbnail generator, or antivirus) occur.
    """
    import time
    for attempt in range(5):
        try:
            fig_or_plt.savefig(save_path, **kwargs)
            return
        except OSError:
            time.sleep(0.3)
    fig_or_plt.savefig(save_path, **kwargs)


