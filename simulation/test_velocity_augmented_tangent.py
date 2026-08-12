import numpy as np

from scripts.velocity_manifold_fitter import VelocityManifoldFitter as AugmentedFitter


def _data(seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, 80)
    clean = np.c_[t, 0.3 * t**2, np.zeros(len(t))]
    velocity = np.c_[np.ones(len(t)), 0.6 * t, np.zeros(len(t))]
    noisy = clean + rng.normal(scale=0.03, size=(len(t), 1)) * np.array([[0.0, 0.0, 1.0]])
    observed_velocity = velocity + rng.normal(scale=0.05, size=velocity.shape)
    return noisy, observed_velocity


def _fit(cls, **extra):
    x, v = _data()
    fitter = cls(
        x, v, d_mode="global", global_d=1, k=16, T=2, eta_g=0.35,
        theta=0.1, kappa=1.0, use_PCA=False, random_state=9, **extra,
    )
    return fitter.fit(return_dict=True)


def test_lambda_zero_diagnostic_path_is_exact_special_case():
    historical_path = _fit(AugmentedFitter)
    augmented = _fit(
        AugmentedFitter, lambda_v=0.0, record_tangent_diagnostics=True,
        return_tangent_diagnostics=True,
    )
    for key in ("X", "V", "P", "weights"):
        np.testing.assert_array_equal(augmented[key], historical_path[key])
    np.testing.assert_array_equal(augmented["neighbors"], historical_path["neighbors"])


def test_trace_normalization_matches_position_trace():
    result = _fit(AugmentedFitter, lambda_v=0.5, record_tangent_diagnostics=True)
    diagnostics = result["tangent_diagnostics"]
    position_trace = np.trace(diagnostics["position_covariance"], axis1=1, axis2=2)
    velocity_trace = np.trace(diagnostics["velocity_covariance_scaled"], axis1=1, axis2=2)
    valid = position_trace > 1e-12
    np.testing.assert_allclose(velocity_trace[valid], position_trace[valid], rtol=1e-12, atol=1e-12)
    assert len(result["tangent_diagnostics_history"]) == 3


def test_covariance_plus_mean_equals_uncentered_second_moment():
    uncentered = _fit(AugmentedFitter, lambda_v=0.5, velocity_covariance_mode="uncentered")
    plus_mean = _fit(AugmentedFitter, lambda_v=0.5, velocity_covariance_mode="covariance_plus_mean")
    np.testing.assert_allclose(uncentered["X"], plus_mean["X"], atol=1e-14)
    np.testing.assert_allclose(uncentered["P"], plus_mean["P"], atol=1e-14)


def test_velocity_tangent_weight_zero_is_exact_no_op():
    """velocity_tangent_weight is an independent term reconciled in from the
    upstream "Add velocity-augmented tangent fitting" commit (Jingyuan Hu) --
    its default (0.0, matching upstream's own default) must reproduce
    lambda_v-only behavior exactly, whether or not lambda_v itself is used."""
    for lambda_v in (0.0, 0.5):
        baseline = _fit(AugmentedFitter, lambda_v=lambda_v)
        explicit_zero = _fit(AugmentedFitter, lambda_v=lambda_v, velocity_tangent_weight=0.0)
        np.testing.assert_array_equal(explicit_zero["X"], baseline["X"])
        np.testing.assert_array_equal(explicit_zero["V"], baseline["V"])


def test_velocity_tangent_weight_is_additive_and_independent_of_lambda_v():
    """A positive velocity_tangent_weight must actually change the fit (not
    be silently ignored), and combining it with lambda_v>0 must not raise --
    the two mechanisms coexist rather than being mutually exclusive."""
    lambda_only = _fit(AugmentedFitter, lambda_v=0.5, velocity_tangent_weight=0.0)
    tangent_only = _fit(AugmentedFitter, lambda_v=0.0, velocity_tangent_weight=0.7)
    both = _fit(AugmentedFitter, lambda_v=0.5, velocity_tangent_weight=0.7)
    assert not np.allclose(tangent_only["X"], lambda_only["X"])
    assert np.all(np.isfinite(both["X"])) and np.all(np.isfinite(both["V"]))


def test_velocity_tangent_weight_rejects_negative():
    import pytest

    with pytest.raises(ValueError):
        _fit(AugmentedFitter, velocity_tangent_weight=-0.1)


def test_settings_are_serializable_scalars():
    settings = _fit(AugmentedFitter, lambda_v=0.25)["algorithm_settings"]
    assert settings == {
        "lambda_v": 0.25,
        # lambda_v_confidence_scaling/_power added 2026-08-12 (current_plan.md P4.1
        # follow-up): per-point lambda_v discounting by velocity_confidence.
        # "none"/1.0 here reproduce the old unconditional-scalar behavior exactly.
        "lambda_v_confidence_scaling": "none",
        "lambda_v_confidence_power": 1.0,
        # lambda_v_relative_error_mean added 2026-08-12 same day (the
        # "inverse_error" redesign): mean of lambda_v_relative_error, which
        # defaults to all zeros (no discount) when the caller does not supply
        # it, so 0.0 here regardless of lambda_v_confidence_scaling="none".
        "lambda_v_relative_error_mean": 0.0,
        "velocity_covariance_mode": "centered",
        "velocity_trace_normalization": "match_position_trace",
        # velocity_tangent_weight added when reconciling the upstream
        # "Add velocity-augmented tangent fitting" commit (Jingyuan Hu) --
        # an independent, additive mechanism kept alongside lambda_v rather
        # than merged into it. 0.0 (its default, matching upstream's own
        # default) is a no-op, so this doesn't affect the values below.
        "velocity_tangent_weight": 0.0,
        "d_mode": "global",
        "global_d": 1,
        "adaptive_variance_threshold": None,
        "k": 16,
        "T": 2,
    }
