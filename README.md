# TiO2 Crystallization Kinetics & Thermal Poling Simulation Suite

A modular Python simulation suite for modeling phase transformations in titanium dioxide ($\text{TiO}_2$) thin films and field-assisted thermal poling of silicate glasses.

---

## Project Structure

```
tio2 simulations/
├── src/
│   ├── __init__.py
│   ├── params.py             # Physical constants, Wooding et al. dataset, poling experimental matrix
│   ├── avrami_jmak.py        # Analytical JMAK X(t), Arrhenius k(T), TTT diagram, field-assisted k(T, E)
│   ├── avrami_2d.py          # 2D spatial nucleation-and-growth simulator on a grid
│   ├── ion_poling_1d.py      # Stage 2: 1D multi-ion drift-diffusion + Poisson solver for glass poling
│   └── stage2_placeholder.py # Stage 2 module aliases and backwards-compatibility interface
├── outputs/                  # High-resolution publication-quality PNG figures
│   ├── plot1_wooding_scurves.png
│   ├── plot2_wooding_ttt.png
│   ├── plot3_field_assisted_kinetics.png
│   ├── plot4_crystallization_regime_map.png
│   ├── plot5_avrami_2d_snapshots.png
│   ├── plot6_avrami_2d_verification.png
│   ├── plot7_sl_ion_profiles.png           # Stage 2: at% profiles vs depth (Pervan Fig. 6c validation)
│   ├── plot8_electric_field_evolution.png  # Stage 2: E(x, t) spatial profiles and E(0, t) time series
│   └── interfacial_field_300C_500V.npz     # Stage 2: Exported interfacial field array
├── refs/                     # Reference papers and reports
│   ├── ALD-01-001_article-101276_en_1.pdf       (Wooding et al., ALDJ 2023)
│   ├── Report on Tio2 crystallization.pdf       (Bordeaux & IRB experimental study)
│   └── Modelling20accepted.pdf                  (Pervan et al., multi-ion glass poling)
├── run_stage1.py             # Main script to run all Stage 1 simulations and save plots
├── run_stage2.py             # Main script to run Stage 2 glass poling and save plots
├── requirements.txt          # Python dependencies (numpy, scipy, matplotlib)
├── .gitignore                # Git exclusions (__pycache__, *.pyc, .vscode, refs/*.pdf)
└── README.md                 # Project documentation and guide
```

---

## Physics & Mathematical Foundations

### 1. Johnson-Mehl-Avrami-Kolmogorov (JMAK) Kinetics
The isothermal transformed volume/area fraction $X(t)$ from amorphous to anatase $\text{TiO}_2$ follows:
$$X(t) = 1 - \exp\left(-(k \cdot t)^n\right)$$
where:
- $k$ is the combined reaction rate constant ($\text{s}^{-1}$),
- $n$ is the Avrami exponent characterizing transformation dimensionality and nucleation mechanism ($n \approx 2.75\text{--}3.0$ for 2D plate-like growth with continuous nucleation).

The temperature dependence of $k(T)$ is governed by the Arrhenius relation:
$$k(T) = k_0 \exp\left(-\frac{E_a}{k_B T}\right)$$
- Consensus activation energy: $E_a \approx 1.48\text{ eV/atom}$ (Wooding et al., 2023)
- Pre-exponential frequency factor: $k_0 \in [1.1 \times 10^{10}, 6.2 \times 10^{11}]\text{ s}^{-1}$ (dependent on ALD deposition temperature).

The isothermal Time-Temperature-Transformation (TTT) curve for any target fraction $X$ is given analytically by:
$$t_X(T) = \frac{1}{k(T)} \left[-\ln(1 - X)\right]^{1/n}$$

### 2. Field-Assisted Crystallization Extension (Thermal Poling)
During thermal poling of $\text{TiO}_2$ thin films on soda-lime glass, mobile sodium ions drift toward the cathode, producing a sub-anodic space-charge depletion layer ($d_0 \approx 350\text{ nm}$) across which the external potential $U$ drops. This induces a high local electric field $E \sim U / d_0 \sim 10^8\text{--}10^9\text{ V/m}$.

The electric field assists crystallization by lowering the activation energy barrier $\Delta E(E)$:
$$k(T, E) = k_0 \exp\left(-\frac{E_a - \Delta E(E)}{k_B T}\right)$$

Two barrier reduction models are implemented and compared:
1. **Poole-Frenkel Model**:
   $$\Delta E_{PF}(E) = \beta \sqrt{E}$$
   where theoretical $\beta_{PF} = \sqrt{\frac{e^3}{\pi \varepsilon_0 \varepsilon_r}}$ reflects field-enhanced thermal ionization and barrier suppression.
2. **Linear Model**:
   $$\Delta E_{lin}(E) = \alpha \cdot E$$
   reflecting direct dipole orientation / electro-migration work ($q \cdot a$).

### 3. 2D Spatial Microstructural Simulation
Simulates continuous Poisson nucleation and isotropic radial expansion on a discrete $N \times N$ domain:
- Untransformed area fraction $1 - X(t)$ dictates real nucleation events at rate $\dot{N}$.
- Grains grow radially at speed $v$.
- Grain impingement occurs naturally where advancing fronts meet.
- In 2D, this verifies the theoretical Avrami exponent $n = 3$:
  $$X_{analytical}(t) = 1 - \exp\left(-\frac{\pi}{3} \dot{N} v^2 t^3\right)$$

### 4. Stage 2: 1D Multi-Ion Drift-Diffusion + Poisson Solver for Glass Poling
Based on Petar Pervan et al., *Modelling of concentration profiles in thermally poled glasses and correlation with refractive index profiles* (`refs/Modelling20accepted.pdf`):
- **Nernst-Planck Drift-Diffusion**:
  $$\frac{\partial C_i}{\partial t} = -\mu_i \frac{\partial (E C_i)}{\partial x} + D_i \frac{\partial^2 C_i}{\partial x^2}$$
  for 5 mobile species in Menzel soda-lime glass: $\text{Na}^+, \text{K}^+, \text{Ca}^{2+}, \text{Mg}^{2+}, \text{H}^+$ (open anode).
- **Modified Nernst-Einstein Relation**:
  $$D_i = \frac{\mu_i H k_B T}{q_i e} \quad (H = 0.3, \ \varepsilon_r = 7.0)$$
- **Poisson Equation & Constant Voltage Constraint**:
  $$\frac{\partial E}{\partial x} = \frac{e}{\varepsilon_0 \varepsilon_r} \sum_{i=1}^{n_s} q_i (C_i - C_{i0}), \quad \int_0^W E(x, t)\, dx = U \ (500\text{ V}, \ W = 1.0\text{ mm})$$
- **Stiff Integration Strategy**:
  Dimensionless Method of Lines with an implicit tridiagonal M-matrix solver using upwind differencing for hyperbolic drift and central differencing for diffusion. Guarantees unconditional numerical positivity ($C_i \ge 0$) and stability across 5 orders of magnitude in mobilities ($\mu_{Na} = 5.4 \times 10^{-15}$ to $\mu_{Ca} = 9.0 \times 10^{-20}\text{ m}^2/(\text{V s})$).

---

## Installation & Requirements

Ensure Python 3.10+ is installed with `pip`:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `matplotlib >= 3.7.0`

---

## How to Run

### Command Line
Navigate to the project root directory and execute:
```bash
# Run Stage 1: TiO2 crystallization kinetics, TTT diagrams, and 2D spatial impingement
python run_stage1.py

# Run Stage 2: 1D multi-ion drift-diffusion and Poisson solver for soda-lime glass poling
python run_stage2.py
```

### In Spyder or VS Code
Open the project directory in Spyder, open either `run_stage1.py` or `run_stage2.py`, and press **Run (F5)**. All paths are relative to the project root and do not require hard-coded directories.

---

## Generated Outputs (`outputs/`)

| File | Stage | Description |
|---|---|---|
| `plot1_wooding_scurves.png` | Stage 1 | JMAK S-curves $X(t)$ across PDA temperatures for ALD films deposited at 120, 140, and 160 °C (Wooding Fig. 4 reproduction). |
| `plot2_wooding_ttt.png` | Stage 1 | TTT diagrams (5%, 50%, 95% anatase transformation) on $T_{PDA}$ vs. $\log_{10}(t)$ axes (Wooding Fig. 9 reproduction). |
| `plot3_field_assisted_kinetics.png` | Stage 1 | Field-assisted kinetics comparing Linear vs. Poole-Frenkel barrier lowering across Voltage ($0\text{--}1250\text{ V}$), Temperature ($150\text{--}300\text{ }^\circ\text{C}$), and Time ($0\text{--}1.8\text{ h}$). |
| `plot4_crystallization_regime_map.png` | Stage 1 | 2D phase regime contour map in $(T, U)$ space overlaid with experimental samples C1–C20 categorized by Raman crystalline state. |
| `plot5_avrami_2d_snapshots.png` | Stage 1 | 4-panel microstructural evolution showing early nucleation, circular grain growth, impingement, and polycrystalline anatase film. |
| `plot6_avrami_2d_verification.png` | Stage 1 | Quantitative verification of simulated area fraction $X_{sim}(t)$ against analytical Avrami ($n = 3.00$) and linearized Avrami fit. |
| `plot7_sl_ion_profiles.png` | Stage 2 | Concentration profiles (at%) of $\text{Na}^+, \text{K}^+, \text{Ca}^{2+}, \text{Mg}^{2+}, \text{H}^+$ vs depth ($0\text{--}2500\text{ nm}$) at $t = 2\text{ h}$ validating $\text{Na}^+$ depletion (~600 nm) and $\text{Ca}^{2+}/\text{Mg}^{2+}$ pile-up against Pervan et al. Fig. 6c, plus $t = 1\text{ h}$ snapshot for TiO$_2$ correlation. |
| `plot8_electric_field_evolution.png` | Stage 2 | Spatial profiles of $E(x, t)$ across multiple times ($1\text{ min}, 15\text{ min}, 30\text{ min}, 1\text{ h}, 2\text{ h}$) and peak interfacial field $E(x=0, t)$ vs time ($0\text{--}2\text{ h}$) on both linear and log scales. |
| `interfacial_field_300C_500V.npz` | Stage 2 | Exported interfacial electric field $E(0, t)$ and depletion front $w(t)$ time-series array for Stage 1 coupling. |

