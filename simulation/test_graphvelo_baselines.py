import numpy as np

from scripts.graphvelo_official_adapter import (
    GRAPHVELO_CONFIG,
    graphvelo_velocity,
    graphvelo_velocity_standardized,
    noisy_standardization_scales,
    official_notebook_call,
    official_neighbors,
)
from scripts.simulation_baselines import cosine_kernel_projection, restore_noisy_speed, shared_knn_graph


def geometric_case(n=40):
    theta=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.c_[np.cos(theta),np.sin(theta),np.zeros(n)],np.c_[-np.sin(theta),np.cos(theta),np.zeros(n)]


def test_official_wrapper_matches_direct_notebook_path():
    X,V=geometric_case();direct,direct_info=official_notebook_call(X,V);wrapped,wrapped_info=graphvelo_velocity(X,V)
    assert np.allclose(wrapped,direct,rtol=1e-12,atol=1e-12)
    assert wrapped_info["neighbor_graph_hash"]==direct_info["neighbor_graph_hash"]
    assert GRAPHVELO_CONFIG|{} == {"n_neighbors":15,"a":1.0,"b":0.0,"r":1.0,"loss_func":"linear","softmax_adjusted":False,"approx":False,"preprocessing":"raw continuous coordinates; no log/log1p, normalization, or PCA"}


def test_official_neighbors_match_notebook_including_self():
    X,_=geometric_case();neighbors=official_neighbors(X)
    assert neighbors.shape==(len(X),15)
    assert all(i in row for i,row in enumerate(neighbors))


def test_cosine_speed_restoration_is_finite():
    X,V=geometric_case();graph=shared_knn_graph(X,8);direction,_=cosine_kernel_projection(X,V,graph);estimate,info=restore_noisy_speed(direction,V)
    assert estimate.shape==V.shape and np.all(np.isfinite(estimate))
    assert np.allclose(np.linalg.norm(estimate,axis=1),np.linalg.norm(V,axis=1))
    assert 0<=info["speed_restoration_fallback_fraction"]<=1


def test_standardized_graphvelo_is_equivariant_to_position_and_velocity_units():
    X,V=geometric_case();base,info=graphvelo_velocity_standardized(X,V)
    transformed,other=graphvelo_velocity_standardized(7.5*X+np.array([4.,-2.,3.]),.35*V)
    assert np.allclose(transformed,.35*base,rtol=2e-5,atol=2e-7)
    assert np.isclose(other["position_scale"],7.5*info["position_scale"])
    assert np.isclose(other["velocity_scale"],.35*info["velocity_scale"])
    assert info["truth_free_standardization"] and not info["selected_by_performance"]


def test_standardization_scales_use_noisy_inputs_only():
    X,V=geometric_case();center,sx,sv=noisy_standardization_scales(X,V)
    assert center.shape==(3,) and sx>0 and sv>0
    assert "truth" not in noisy_standardization_scales.__code__.co_varnames
