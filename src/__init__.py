"""
TiO2 Crystallization Kinetics & Multi-Ion Glass Poling Simulation Suite.
Stage 1: Avrami (JMAK) crystallization kinetics and field-assisted models.
Stage 2: 1D multi-ion drift-diffusion + Poisson solver for glass poling.
"""

from .ion_poling_1d import SodaLimePoling1DSolver

__version__ = "2.0.0"
__all__ = ["SodaLimePoling1DSolver"]
