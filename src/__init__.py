"""
TiO2 Crystallization Kinetics & Multi-Ion Glass Poling Simulation Suite.
Stage 1: Avrami (JMAK) crystallization kinetics and field-assisted models.
Stage 2: 1D multi-ion drift-diffusion + Poisson solver for glass poling.
Stage 3: Dynamic interfacial field coupling into generalized JMAK kinetics.
"""

from .ion_poling_1d import SodaLimePoling1DSolver
from .coupled_kinetics import CoupledJMAKSolver

__version__ = "3.0.0"
__all__ = ["SodaLimePoling1DSolver", "CoupledJMAKSolver"]

