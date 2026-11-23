"""
Stage 2 Module: 1D Multi-Ion Drift-Diffusion + Poisson Solver for Glass Poling.
Now fully implemented via `src.ion_poling_1d.SodaLimePoling1DSolver`.

References:
Pervan et al., 'Modelling of concentration profiles in thermally poled glasses
and correlation with refractive index profiles' (refs/Modelling20accepted.pdf).
"""

from .ion_poling_1d import SodaLimePoling1DSolver

# Backward-compatibility alias
GlassPoling1DSolver = SodaLimePoling1DSolver

__all__ = ["SodaLimePoling1DSolver", "GlassPoling1DSolver"]
