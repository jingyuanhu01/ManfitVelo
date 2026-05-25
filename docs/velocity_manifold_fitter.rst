Velocity Manifold Fitter
========================

.. automodule:: scripts.velocity_manifold_fitter
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Priority
------------------

High-priority tuning parameters
   ``d_mode``, ``adaptive_variance_threshold``, ``adaptive_d_min``,
   ``adaptive_d_max``, ``k``, ``T``, ``eta_g``, and ``theta``.

Default modes
   ``fit(update_mode="normal_only")`` and ``bandwidth_mode="variable"``.
   These are the recommended defaults for most runs.

Lower-priority parameters
   ``global_d``, ``use_PCA``, ``PCA_dim``, ``gamma``, ``beta``, ``kappa``,
   ``cv``, ``max_step_frac``, ``h``, cosine sign conventions, and neighbor
   recomputation controls.
