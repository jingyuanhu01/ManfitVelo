"""ManfitVelo: velocity-aware manifold fitting."""
from .velocity_manifold_fitter import VelocityManifoldFitter
from .scalar_potential_manfit import fit_scalar_gradient_manfit
from .potential_from_gradient import potential_from_gradient

__version__ = "0.1.0"
__all__ = ["VelocityManifoldFitter", "fit_scalar_gradient_manfit", "potential_from_gradient"]
