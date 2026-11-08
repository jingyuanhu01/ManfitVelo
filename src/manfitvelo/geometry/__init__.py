"""Intrinsic Jacobian estimation and downstream geometry."""
from .intrinsic_jacobian import estimate_jacobian_field
from .connection_smoothing import connection_aware_smooth

__all__ = ["estimate_jacobian_field", "connection_aware_smooth"]
