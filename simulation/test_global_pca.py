import numpy as np

from scripts.pca_denoisers import global_pca_denoise, project_vectors_with_pca_info
from scripts.simulation_baselines import global_pca_state


def test_global_pca_projector_properties_and_old_helper_equivalence():
    rng=np.random.default_rng(7);X=rng.normal(size=(50,6));V=rng.normal(size=(50,6))
    Xhat,Vhat,info=global_pca_state(X,V,3);P=info["projector"]
    old_X,old_info=global_pca_denoise(X,3,center=True,return_info=True);old_V=project_vectors_with_pca_info(V,old_info)
    assert np.allclose(P,P.T) and np.allclose(P@P,P)
    assert np.linalg.matrix_rank(P,tol=1e-10)==3
    assert Xhat.shape==X.shape and Vhat.shape==V.shape
    assert np.allclose(Xhat,old_X) and np.allclose(Vhat,old_V)


def test_global_pca_is_deterministic_and_truth_free():
    rng=np.random.default_rng(9);X=rng.normal(size=(30,5));V=rng.normal(size=(30,5))
    a=global_pca_state(X,V,2);b=global_pca_state(X,V,2)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert "truth" not in global_pca_state.__code__.co_varnames
