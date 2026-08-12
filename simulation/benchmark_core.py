"""Frozen configuration and evaluation utilities for formal simulations."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from scripts.benchmark_scenarios import HAIRPIN_DEFAULT_SEPARATION, SETS


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
PILOT_SEED = 41000
SCENARIO_LABELS = {
    "circle": "Circle", "s_curve": "S-curve", "curved_hairpin": "Curved hairpin",
    "flat_rotation_annulus": "Flat rotation annulus",
    "half_sphere_tangent": "Half-sphere tangent field", "y_branch": "Y-branch",
    "near_intersection": "Near-intersection",
    "swiss_roll": "Swiss roll", "saddle_surface": "Saddle surface",
}


def load_frozen_config() -> dict:
    """Load the already-selected non-GraphVelo settings without re-tuning."""
    selected = json.loads((ROOT / "results/manfitvelo_benchmark/selected_hyperparameters.json").read_text())
    if selected.get("selection_uses_final_seeds") is not False:
        raise AssertionError("frozen configuration does not certify disjoint final seeds")
    return selected


# --- Neighborhood scaling rule k(n,d) (Weekly Plan v1.1 section 4; C
# --- simplified to a single global constant under current_plan.md P0.1) ---------
#
# Applies to Cosine Kernel / Local PCA / Position-only MANFIT (M5) / ManfitVelo
# (M6) so that the neighborhood size is a data-adaptive function of (n, d)
# rather than a per-scenario tuned constant. The MISE-optimal local-linear
# bandwidth h_n ~ n^{-1/(d+4)} implies a neighborhood point count
#   k(n, d) = ceil(C * n^(4/(d+4))).
#
# C used to be two separate constants C_1/C_2, reverse-engineered by forcing
# k(n0, d) = 40 on one anchor scenario per intrinsic dimension (Circle for
# d=1, Flat Rotation Annulus for d=2). Per current_plan.md P0.1 (2026-08-12), this
# was replaced by a single dimension-independent C, selected once via
# simulation/run_c_selection.py:
#   - candidates C in {0.30, 0.45, 0.60, 0.75, 0.90} (5 values spanning and
#     straddling the two old anchors C_1~=0.361, C_2~=0.713);
#   - for each candidate, per-scenario k derived by the same two-stage
#     procedure as here (Stage-1 ceiling from this candidate C, then the
#     unchanged Stage-2 curvature-aware refinement below);
#   - scored by ManfitVelo (M6)'s pooled tuning_score on TUNING_SEEDS only,
#     mean over all 9 canonical scenarios (see
#     simulation/run_manfitvelo_benchmark.py:tuning_score); other VMF/
#     Position-only hyperparameters held at their then-current frozen values,
#     not re-tuned per candidate.
#   - winner: C = 0.60 (results/c_selection/c_selection_summary.csv).
# Known limitation discovered during this selection (accepted as-is by the
# user 2026-08-12, not fixed): Stage 2's argmin(slope) turn detection (below)
# assumes the log-log residual-vs-k slope curve is unimodal. At large enough
# Stage-1 ceilings (observed for Curved Hairpin at the C=0.75/0.90 candidates,
# ceiling 105/126), the slope curve is not unimodal -- it dips early (normal
# finite-sample-bias regime), rises (curvature bias), then dips again at very
# large k (mechanism not investigated), and argmin picks that spurious second
# dip, returning k close to the ceiling instead of the true early optimum.
# Confirmed this does NOT affect the winning C=0.60 candidate (nor 0.30/0.45)
# on any of the 9 scenarios -- see results/c_selection/c_selection_k_table.csv
# -- so it does not change the selection outcome, only inflates how much worse
# the two losing higher-C candidates look. Left as a known limitation of Stage
# 2 rather than fixed, since fixing it is a change shared by every scenario's
# k selection (not scoped to this C choice) and was out of scope for P0.1.
NEIGHBOR_SCALING_CONSTANT = 0.60
NEIGHBOR_COUNT_CLIP = (10, 200)


def neighbor_count(n: int, d: int) -> int:
    """Shared k(n, d) neighborhood rule; frozen, no scenario-specific tuning."""
    exponent = 4.0 / (d + 4)
    raw = NEIGHBOR_SCALING_CONSTANT * (float(n) ** exponent)
    k_min, k_max = NEIGHBOR_COUNT_CLIP
    return int(np.clip(np.ceil(raw), k_min, k_max))


# --- Curvature-aware refinement of k(n, d) (added 2026-08-11; see log.md) --
#
# k(n, d) above only sees sample size and intrinsic dimension, so it silently
# overshoots on curved geometry: a larger neighborhood always reduces
# noise-averaging variance, but on a curved manifold it also reaches far
# enough to pick up curvature bias in the local tangent/normal estimate,
# which grows with the neighborhood's geodesic radius. On a genuinely flat
# patch that bias term is ~0, so bigger k is strictly better; on curved
# patches it isn't. neighbor_count(n, d) has no way to tell the two cases
# apart. This refinement adds a second, purely data-driven signal (no ground
# truth, so it also applies to real data): sweep k from a small floor up to
# the neighbor_count(n, d) ceiling, and at each k measure the population-mean
# *normal-direction* residual from a local-PCA fit (local_pca_denoise's
# "mean_local_spectrum", summed over the ambient-d smallest eigenvalues --
# i.e. how much of each neighborhood's local spread is NOT explained by a
# rank-d tangent plane). Under pure noise this stays flat as k grows; once
# neighborhoods are wide enough to feel curvature, it starts climbing. On a
# log-log(residual) vs log(k) plot this shows up as the growth *rate*
# (slope) first decreasing (finite-sample covariance-estimation bias
# shrinking) then, only for curved geometry, turning around and increasing
# again (curvature bias taking over). The chosen k is the point right after
# that slope hits its minimum -- i.e. the last neighborhood size before the
# growth rate starts trending back up. For a flat manifold the slope never
# turns back up, so this reduces to the neighbor_count(n, d) ceiling
# unchanged (confirmed on flat_rotation_annulus / s_curve).
#
# Calibrated once (not per scenario) on the k-grid density / floor below;
# validated against clean_point_rmse_rel on all 7 formal scenarios' dev
# seeds -- 4 clear improvements over the plain formula, 2 ties, 2 losses of
# <10% relative (near-noise-level) -- see simulation/log.md for the full
# comparison table.
CURVATURE_GRID_POINTS = 14


def curvature_probe_k_grid(n: int, d: int, *, num: int = CURVATURE_GRID_POINTS) -> list[int]:
    """Candidate k values to probe, from a small floor up to neighbor_count(n, d)."""
    ceiling = neighbor_count(n, d)
    floor = min(max(2 * d + 2, 8), ceiling)
    if floor >= ceiling:
        return [ceiling]
    return sorted({int(round(v)) for v in np.geomspace(floor, ceiling, num=num)})


def local_pca_normal_residual(Y: np.ndarray, d: int, k_grid: list[int]) -> list[float]:
    """Population-mean normal-direction local-PCA residual at each k in k_grid."""
    from scripts.pca_denoisers import local_pca_denoise

    residual = []
    for k in k_grid:
        _, info = local_pca_denoise(Y, d, n_neighbors=k, return_info=True)
        residual.append(float(np.sum(info["mean_local_spectrum"][d:])))
    return residual


def curvature_aware_neighbor_count(k_grid: list[int], residual_curves: list[list[float]]) -> tuple[int, dict]:
    """Pick k from one or more residual curves (e.g. one per development seed).

    residual_curves are averaged first (reduces per-draw noise), then the
    log-log slope of residual vs k is swept for its minimum; the returned k
    is the grid point right after that minimum. See the module-level note
    above for the full rationale.
    """
    if len(k_grid) < 3:
        return int(k_grid[-1]), {"k_grid": list(k_grid), "residual_curve": [], "loglog_slope": [], "turn_index": None}
    residual = np.mean(np.asarray(residual_curves, dtype=float), axis=0)
    log_k, log_r = np.log(k_grid), np.log(np.maximum(residual, EPS))
    slope = np.diff(log_r) / np.diff(log_k)
    turn = int(np.argmin(slope))
    safe_k = int(k_grid[turn + 1])
    return safe_k, {
        "k_grid": list(k_grid),
        "residual_curve": residual.tolist(),
        "loglog_slope": slope.tolist(),
        "turn_index": turn,
    }


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def angle_mae(estimate: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    estimate, target = estimate[valid], target[valid]
    en, tn = np.linalg.norm(estimate, axis=1), np.linalg.norm(target, axis=1)
    keep = (en > 1e-8) & (tn > 1e-8)
    if not np.any(keep):
        return np.nan
    cosine = np.sum(estimate[keep] * target[keep], axis=1) / (en[keep] * tn[keep])
    return float(np.degrees(np.mean(np.arccos(np.clip(cosine, -1, 1)))))


def vector_rmse(estimate: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    difference = estimate[valid] - target[valid]
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def joint_error(Xhat, Vhat, Xclean, Vtrue, tau: float) -> float:
    difference = (Xhat + tau * Vhat) - (Xclean + tau * Vtrue)
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def observed_tau(Y: np.ndarray, W: np.ndarray, neighbors: np.ndarray) -> tuple[float, float, float]:
    edge_distance = np.linalg.norm(Y[neighbors] - Y[:, None, :], axis=2)
    median_knn = float(np.median(edge_distance))
    median_speed = float(np.median(np.linalg.norm(W, axis=1)))
    return 0.5 * median_knn / max(median_speed, EPS), median_knn, median_speed


@lru_cache(maxsize=None)
def dense_truth_support(scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario == "circle":
        t = np.linspace(0, 2*np.pi, 12000, endpoint=False)
        X = np.c_[np.cos(t), np.sin(t), np.zeros(len(t))]
        V = np.c_[-np.sin(t), np.cos(t), np.zeros(len(t))]
        labels = np.zeros(len(t), int)
    elif scenario == "s_curve":
        t = np.linspace(-1.4, 1.4, 16000)
        X = np.c_[np.sin(1.6*t), t, np.zeros(len(t))]
        V = np.c_[1.6*np.cos(1.6*t), np.ones(len(t)), np.zeros(len(t))]
        V /= np.linalg.norm(V, axis=1, keepdims=True); labels = np.zeros(len(t), int)
    elif scenario == "curved_hairpin":
        L, r, curvature = 1.2, HAIRPIN_DEFAULT_SEPARATION/2, .15
        x1 = np.linspace(-L, L, 7000); phase = np.pi*(x1+L)/(2*L)
        shape = curvature*(1-np.cos(phase)); derivative = curvature*np.sin(phase)*np.pi/(2*L)
        lower = np.c_[x1, -r+shape, np.zeros(len(x1))]
        v1 = np.c_[np.ones(len(x1)), derivative, np.zeros(len(x1))]; v1 /= np.linalg.norm(v1,axis=1,keepdims=True)
        angle = np.linspace(-np.pi/2, np.pi/2, 1500); center_y = 2*curvature
        turn = np.c_[L+r*np.cos(angle), center_y+r*np.sin(angle), np.zeros(len(angle))]
        v2 = np.c_[-np.sin(angle), np.cos(angle), np.zeros(len(angle))]
        x3 = np.linspace(L, -L, 7000); phase3 = np.pi*(x3+L)/(2*L)
        shape3 = curvature*(1-np.cos(phase3)); derivative3 = curvature*np.sin(phase3)*np.pi/(2*L)
        upper = np.c_[x3, r+shape3, np.zeros(len(x3))]
        v3 = -np.c_[np.ones(len(x3)), derivative3, np.zeros(len(x3))]; v3 /= np.linalg.norm(v3,axis=1,keepdims=True)
        X, V = np.vstack([lower,turn,upper]), np.vstack([v1,v2,v3])
        labels = np.r_[np.zeros(len(lower),int),np.ones(len(turn),int),np.full(len(upper),2,int)]
    elif scenario == "y_branch":
        t=np.linspace(0,1,6000); X=np.vstack([np.c_[0*t,-t,0*t],np.c_[-.8*t,.8*t,0*t],np.c_[.8*t,.8*t,0*t]])
        V=np.vstack([np.tile([0.,1.,0.],(len(t),1)),np.tile(np.array([-.8,.8,0])/np.sqrt(1.28),(len(t),1)),np.tile(np.array([.8,.8,0])/np.sqrt(1.28),(len(t),1))])
        labels=np.r_[np.zeros(len(t),int),np.ones(len(t),int),np.full(len(t),2,int)]
    elif scenario == "near_intersection":
        t=np.linspace(-1.1,1.1,10000); sep,curvature=.13,.3
        X=np.vstack([np.c_[t,sep/2+curvature*t*t,0*t],np.c_[t,-sep/2-curvature*t*t,0*t]])
        vu=np.c_[np.ones(len(t)),2*curvature*t,0*t]; vl=-np.c_[np.ones(len(t)),-2*curvature*t,0*t]
        vu/=np.linalg.norm(vu,axis=1,keepdims=True);vl/=np.linalg.norm(vl,axis=1,keepdims=True);V=np.vstack([vu,vl])
        labels=np.r_[np.zeros(len(t),int),np.ones(len(t),int)]
    elif scenario == "swiss_roll":
        tmax=3.5*np.pi  # one full winding; see scripts/benchmark_scenarios.py
        t=np.linspace(1.5*np.pi,tmax,170); y=np.linspace(-1,1,90)
        tt,yy=np.meshgrid(t,y,indexing="ij"); tt=tt.ravel(); yy=yy.ravel()
        X=np.c_[tt*np.cos(tt)/tmax,yy,tt*np.sin(tt)/tmax]
        dXdt=np.c_[(np.cos(tt)-tt*np.sin(tt))/tmax,np.zeros(len(tt)),(np.sin(tt)+tt*np.cos(tt))/tmax]
        V=dXdt/np.maximum(np.linalg.norm(dXdt,axis=1,keepdims=True),EPS)
        labels=np.zeros(len(tt),int)
    elif scenario == "saddle_surface":
        a=.45
        u=np.linspace(-1,1,170); v=np.linspace(-1,1,90)
        uu,vv=np.meshgrid(u,v,indexing="ij"); uu=uu.ravel(); vv=vv.ravel()
        X=np.c_[uu,vv,a*(uu*uu-vv*vv)]
        Ju=np.c_[np.ones(len(uu)),np.zeros(len(uu)),2*a*uu]
        V=Ju/np.maximum(np.linalg.norm(Ju,axis=1,keepdims=True),EPS)
        labels=np.zeros(len(uu),int)
    else:
        raise KeyError(scenario)
    return X,V,labels


@lru_cache(maxsize=None)
def support_indexes(scenario: str):
    X,_,labels=dense_truth_support(scenario); unrestricted=NearestNeighbors(n_neighbors=1).fit(X); branch={}
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label); branch[int(label)]=(NearestNeighbors(n_neighbors=1).fit(X[indices]),indices)
    return unrestricted,branch


def project_location_truth(scenario: str, Xhat: np.ndarray, data: dict, *, branch_aware: bool) -> dict:
    Xhat=np.asarray(Xhat,float); labels=np.asarray(data["labels"],int)
    if scenario == "half_sphere_tangent":
        projected=Xhat/np.maximum(np.linalg.norm(Xhat,axis=1,keepdims=True),EPS); below=projected[:,2]<0
        if np.any(below):
            rho=np.linalg.norm(projected[below,:2],axis=1);projected[below]=np.c_[projected[below,0]/np.maximum(rho,EPS),projected[below,1]/np.maximum(rho,EPS),np.zeros(below.sum())]
        direction=np.array([.7,-.4,.6]);raw=direction-projected*(projected@direction)[:,None]
        clean_raw=direction-data["P"]*(data["P"]@direction)[:,None];velocity=raw/np.mean(np.linalg.norm(clean_raw,axis=1));branch_switch=np.zeros(len(Xhat),bool)
    else:
        support,velocity_support,support_labels=dense_truth_support(scenario);unrestricted,branches=support_indexes(scenario)
        unrestricted_indices=unrestricted.kneighbors(Xhat,return_distance=False)[:,0];branch_switch=support_labels[unrestricted_indices]!=labels
        if branch_aware and scenario in {"curved_hairpin","y_branch","near_intersection"}:
            chosen=np.empty(len(Xhat),int)
            for label,(index,support_indices) in branches.items():
                mask=labels==label;chosen[mask]=support_indices[index.kneighbors(Xhat[mask],return_distance=False)[:,0]]
        else: chosen=unrestricted_indices
        projected=support[chosen];velocity=velocity_support[chosen]
    valid=np.ones(len(Xhat),bool)
    if scenario=="y_branch": valid=np.linalg.norm(projected[:,:2],axis=1)>=.05
    return {"position":projected,"velocity":velocity,"valid":valid,"branch_switch_fraction":float(branch_switch.mean()),"projection_rmse":float(np.sqrt(np.mean(np.sum((Xhat-projected)**2,axis=1))))}


def evaluation_targets(scenario: str, Xhat: np.ndarray, data: dict) -> dict:
    valid=np.ones(len(Xhat),bool)
    if np.array_equal(Xhat,np.asarray(data["P"])):
        position=np.asarray(data["P"]).copy();velocity=np.asarray(data["truth"]).copy();projection_rmse=0.;branch_switch=0.
    elif scenario=="flat_rotation_annulus":
        position=np.asarray(Xhat,float).copy();position[:,2]=0;radius=np.linalg.norm(position[:,:2],axis=1);position[:,:2]*=(np.clip(radius,.35,1)/np.maximum(radius,EPS))[:,None]
        velocity=np.c_[-position[:,1],position[:,0],np.zeros(len(Xhat))];projection_rmse=float(np.sqrt(np.mean(np.sum((Xhat-position)**2,axis=1))));branch_switch=0.
    else:
        result=project_location_truth(scenario,Xhat,data,branch_aware=True);position=result["position"];velocity=result["velocity"];valid=result["valid"];projection_rmse=result["projection_rmse"];branch_switch=result["branch_switch_fraction"]
    unrestricted=projection_rmse if scenario=="flat_rotation_annulus" else project_location_truth(scenario,Xhat,data,branch_aware=False)["projection_rmse"]
    return {"location_position":position,"location_velocity":velocity,"location_valid":valid,"distance_to_manifold":float(unrestricted),"location_projection_rmse":projection_rmse,"unrestricted_branch_switch_fraction":branch_switch}
