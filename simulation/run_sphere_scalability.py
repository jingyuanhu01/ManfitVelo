"""Ambient-dimension scalability benchmark on S^2 embedded in R^D."""

from __future__ import annotations

import argparse
import base64
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.graphvelo_official_adapter import (
    GRAPHVELO_CONFIG, GRAPHVELO_PROVENANCE, GRAPHVELO_STANDARDIZATION,
    graphvelo_velocity_standardized,
)
from scripts.pca_denoisers import local_pca_denoise
from scripts.benchmark_scenarios import fit_vmf_variant, position_only_trajectory
from scripts.simulation_baselines import (
    cosine_kernel_projection, downstream_velocity, joint_low_rank_state,
    local_pca_state, restore_noisy_speed, shared_knn_graph,
)
from simulation.benchmark_core import array_hash, joint_error, load_frozen_config, neighbor_count, observed_tau


DIMENSIONS = (3, 5, 10, 20, 50)
FORMAL_NOISE_MODEL = "fixed_coordinate"
SUPPORTED_NOISE_REGIMES = ("fixed_total", "fixed_coordinate")
FINAL_SEEDS = tuple(range(43000, 43015))
METHODS = ("ambient_noisy", "cosine_kernel", "graphvelo", "joint_low_rank", "local_pca", "position_only_manfit", "manfitvelo")
LABELS = {"ambient_noisy":"Ambient noisy input","cosine_kernel":"Cosine kernel","graphvelo":"GraphVelo","joint_low_rank":"Joint Low-Rank (M3)","local_pca":"Local PCA","position_only_manfit":"Position-only MANFIT","manfitvelo":"ManfitVelo"}
N = 480
INTRINSIC_DIMENSION = 2
TAU_X = 0.04 * np.sqrt(3.0)
TAU_V = 0.10 * np.sqrt(3.0)
ANGLE_SPEED_THRESHOLD = 0.20
# Shared, curvature-aware k(n,d) neighborhood rule (Weekly Plan v1.1 section
# 4, refined per simulation/benchmark_core.py::curvature_aware_neighbor_count
# -- see simulation/log.md). N/INTRINSIC_DIMENSION here match the
# half_sphere_tangent scenario in run_manfitvelo_benchmark.py exactly
# (n=480, d=2), so main() below reuses that scenario's already-computed
# frozen k straight from the loaded config instead of recomputing it --
# ambient dimension D is swept while n and d stay fixed, matching the plan's
# Scan D design (control intrinsic dimension, vary ambient D). The module-level
# value here is only a fallback default (e.g. for --report-only, which never
# calls fit_method); main() overwrites it with the frozen half_sphere_tangent
# k before any fitting happens.
K = neighbor_count(N, INTRINSIC_DIMENSION)
EPS = 1e-12


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/sphere_scalability")
    parser.add_argument("--report-only",action="store_true")
    return parser.parse_args()


def orthogonal_embedding(seed: int, ambient_dimension: int) -> np.ndarray:
    rng=np.random.default_rng(np.random.SeedSequence([seed,ambient_dimension,9137]))
    raw=rng.normal(size=(ambient_dimension,3)); q,r=np.linalg.qr(raw,mode="reduced")
    signs=np.sign(np.diag(r)); signs[signs==0]=1
    return q*signs


def sphere_data(seed: int, ambient_dimension: int, regime: str) -> dict:
    if regime not in SUPPORTED_NOISE_REGIMES: raise KeyError(regime)
    latent_rng=np.random.default_rng(np.random.SeedSequence([seed,771]))
    z=latent_rng.normal(size=(N,3)); z/=np.linalg.norm(z,axis=1,keepdims=True)
    omega=np.array([.35,-.55,.8]); raw_velocity=np.cross(np.broadcast_to(omega,z.shape),z)
    raw_velocity/=np.median(np.linalg.norm(raw_velocity,axis=1))
    Q=orthogonal_embedding(seed,ambient_dimension); X=z@Q.T; V=raw_velocity@Q.T
    noise_rng=np.random.default_rng(np.random.SeedSequence([seed,ambient_dimension,331]))
    zx=noise_rng.normal(size=(N,ambient_dimension)); zv=noise_rng.normal(size=(N,ambient_dimension))
    divisor=np.sqrt(ambient_dimension) if regime=="fixed_total" else np.sqrt(3.0)
    Y=X+(TAU_X/divisor)*zx; W=V+(TAU_V/divisor)*zv
    projector=np.einsum("da,nab,eb->nde",Q,np.eye(3)[None]-np.einsum("ni,nj->nij",z,z),Q)
    return {"Y":Y,"field":W,"P":X,"truth":V,"z":z,"Q":Q,"true_projector":projector,"d":2,"labels":np.zeros(N,int)}


def analytic_sphere_projection(X: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
    coordinates=X@Q; radius=np.linalg.norm(coordinates,axis=1); unit=coordinates/np.maximum(radius[:,None],EPS)
    projected=unit@Q.T; distance=np.linalg.norm(X-projected,axis=1)
    return projected,unit,float(np.sqrt(np.mean(distance**2)))


def diagnostic_projectors(X: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    _,info=local_pca_denoise(X,2,n_neighbors=K,return_info=True)
    return info["projectors"],np.asarray(info["mean_local_spectrum"])


def fit_method(method: str, data: dict, config: dict, seed: int):
    X,V=data["Y"],data["field"]
    if method=="ambient_noisy": Xhat,Vhat,info=X.copy(),V.copy(),{}
    elif method=="cosine_kernel":
        direction,info=cosine_kernel_projection(X,V,shared_knn_graph(X,K));Vhat,speed=restore_noisy_speed(direction,V);info.update(speed);Xhat=X.copy()
    elif method=="graphvelo": Xhat=X.copy();Vhat,info=graphvelo_velocity_standardized(X,V)
    elif method=="joint_low_rank": Xhat,Vhat,info=joint_low_rank_state(X,V)
    elif method=="local_pca": Xhat,Vhat,info=local_pca_state(X,V,2,K)
    elif method=="position_only_manfit":
        cfg=config["position_only_manfit"]["half_sphere_tangent"]
        Xhat=position_only_trajectory(X,V,2,cfg["k"],cfg["T"],cfg["eta_g"])[-1][1]
        Vhat,info=downstream_velocity(Xhat,V,2,K)
    elif method=="manfitvelo":
        result=fit_vmf_variant(X,V,2,config["velocity_manifold_fitter"]["half_sphere_tangent"],seed)
        Xhat,Vhat,info=result["X"],result["V"],{"projectors":result["P"]}
    else: raise KeyError(method)
    return np.asarray(Xhat),np.asarray(Vhat),info


def neighborhood_diagnostics(Xhat: np.ndarray, clean: np.ndarray) -> tuple[float,float,float]:
    clean_graph=shared_knn_graph(clean,K); estimate_graph=shared_knn_graph(Xhat,K)
    recall=np.mean([len(set(a).intersection(b))/K for a,b in zip(clean_graph,estimate_graph)])
    distances=np.linalg.norm(Xhat[estimate_graph]-Xhat[:,None,:],axis=2)
    _,info=local_pca_denoise(Xhat,2,n_neighbors=K,return_info=True); spectrum=np.asarray(info["mean_local_spectrum"])
    eigengap=float(spectrum[1]/max(spectrum[2],EPS)) if len(spectrum)>2 else np.inf
    return float(recall),eigengap,float(np.median(distances))


def evaluate(method: str, data: dict, Xhat: np.ndarray, Vhat: np.ndarray, info: dict, tau: float) -> dict:
    projected,zloc,distance=analytic_sphere_projection(Xhat,data["Q"])
    omega=np.array([.35,-.55,.8]); local=np.cross(np.broadcast_to(omega,zloc.shape),zloc)
    local/=np.median(np.linalg.norm(np.cross(np.broadcast_to(omega,data["z"].shape),data["z"]),axis=1))
    Vloc=local@data["Q"].T
    clean_speed=np.linalg.norm(data["truth"],axis=1); mask=clean_speed>ANGLE_SPEED_THRESHOLD
    estimate_speed=np.linalg.norm(Vhat,axis=1); angle_keep=mask&(estimate_speed>1e-8)
    cosine=np.sum(Vhat[angle_keep]*data["truth"][angle_keep],axis=1)/(estimate_speed[angle_keep]*clean_speed[angle_keep])
    recall,eigengap,radius=neighborhood_diagnostics(Xhat,data["P"])
    if "projectors" in info:
        estimated_projectors=info["projectors"]
    else:
        # joint_low_rank has no pointwise tangent projector (it's a global
        # rank-r subspace of the concatenated [X,V] space, not a per-point
        # position-space projector); diagnose the reconstructed positions'
        # local tangent structure post hoc, same as ambient_noisy/cosine_kernel/graphvelo.
        estimated_projectors,_=diagnostic_projectors(Xhat)
    projector_error=float(np.sqrt(np.mean(np.sum((estimated_projectors-data["true_projector"])**2,axis=(1,2)))))
    return {
        "clean_point_rmse":float(np.sqrt(np.mean(np.sum((Xhat-data["P"])**2,axis=1)))),
        "distance_to_sphere_rmse":distance,
        "velocity_rmse_id":float(np.sqrt(np.mean(np.sum((Vhat-data["truth"])**2,axis=1)))),
        "velocity_angle_mae_id":float(np.degrees(np.mean(np.arccos(np.clip(cosine,-1,1))))),
        "angle_valid_fraction":float(angle_keep.mean()),
        "velocity_rmse_loc":float(np.sqrt(np.mean(np.sum((Vhat-Vloc)**2,axis=1)))),
        "short_step_euler_rmse":joint_error(Xhat,Vhat,data["P"],data["truth"],tau),
        "knn_recall":recall,"tangent_projector_error":projector_error,"local_covariance_eigengap":eigengap,"median_knn_radius":radius,
    }


def warm_up(config: dict):
    data=sphere_data(41000,3,"fixed_coordinate")
    for method in METHODS: fit_method(method,data,config,41000)


def run(config: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    warm_up(config); rows=[]; runtime=[]
    for D in DIMENSIONS:
        for seed in FINAL_SEEDS:
            data=sphere_data(seed,D,FORMAL_NOISE_MODEL); graph=shared_knn_graph(data["Y"],K);tau,_,_=observed_tau(data["Y"],data["field"],graph)
            sample_hash=array_hash(data["Y"],data["field"],data["P"],data["truth"])
            for method in METHODS:
                start=time.perf_counter();Xhat,Vhat,info=fit_method(method,data,config,seed);elapsed=time.perf_counter()-start
                # Lightweight numerical-failure audit (near-zero cost); see simulation/log.md.
                nan_inf_count=int(np.sum(~np.isfinite(Xhat))+np.sum(~np.isfinite(Vhat)))
                if nan_inf_count:
                    Xhat=np.nan_to_num(Xhat,nan=0.0,posinf=0.0,neginf=0.0);Vhat=np.nan_to_num(Vhat,nan=0.0,posinf=0.0,neginf=0.0)
                rows.append({"ambient_dimension":D,"seed":seed,"method":method,"method_label":LABELS[method],"n":N,"intrinsic_dimension":2,"tau":tau,"sample_hash":sample_hash,"nan_inf_count":nan_inf_count,**evaluate(method,data,Xhat,Vhat,info,tau)})
                runtime.append({"ambient_dimension":D,"seed":seed,"method":method,"runtime_seconds":elapsed,"peak_rss_bytes":np.nan,"peak_memory_reliable":False})
    return pd.DataFrame(rows),pd.DataFrame(runtime)


METRICS=("clean_point_rmse","distance_to_sphere_rmse","velocity_rmse_id","velocity_angle_mae_id","velocity_rmse_loc","short_step_euler_rmse","knn_recall","tangent_projector_error","local_covariance_eigengap","median_knn_radius","angle_valid_fraction")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for keys,g in frame.groupby(["ambient_dimension","method","method_label"],sort=False):
        row=dict(zip(["ambient_dimension","method","method_label"],keys))
        for metric in METRICS: row|={f"{metric}_median":float(g[metric].median()),f"{metric}_q25":float(g[metric].quantile(.25)),f"{metric}_q75":float(g[metric].quantile(.75))}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_group(summary: pd.DataFrame, output: Path, name: str, metrics: tuple[str,...]) -> Path:
    fig,axes=plt.subplots(1,len(metrics),figsize=(5*len(metrics),4),constrained_layout=True);axes=np.atleast_1d(axes)
    for ax,metric in zip(axes,metrics):
        for method,g in summary.groupby("method",sort=False):
            g=g.sort_values("ambient_dimension");line="-"
            ax.plot(g.ambient_dimension,g[f"{metric}_median"],line,marker="o",label=LABELS[method])
        ax.set(xlabel="Ambient dimension D",ylabel=metric);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6,ncol=2);path=output/f"{name}.png";fig.savefig(path,dpi=160);plt.close(fig);return path


def image_uri(path: Path) -> str: return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()


class AuditParser(HTMLParser):
    def __init__(self): super().__init__();self.images=[];self.external=[]
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=="img":self.images.append(values.get("src",""))
        if tag in {"link","script"} and (values.get("href") or values.get("src")):self.external.append(values.get("href") or values.get("src"))


def build_report(output: Path, summary: pd.DataFrame, runtime: pd.DataFrame) -> dict:
    error=plot_group(summary,output,"errors_vs_dimension",("clean_point_rmse","velocity_rmse_id","short_step_euler_rmse"))
    diagnostics=plot_group(summary,output,"diagnostics_vs_dimension",("knn_recall","tangent_projector_error","local_covariance_eigengap"))
    rsummary=runtime.groupby(["ambient_dimension","method"],as_index=False).runtime_seconds.median();rsummary["method_label"]=rsummary.method.map(LABELS)
    runtime_path=plot_group(rsummary.rename(columns={"runtime_seconds":"runtime_seconds_median"}),output,"runtime_vs_dimension",("runtime_seconds",))
    table=summary[["ambient_dimension","method_label","clean_point_rmse_median","velocity_rmse_id_median","short_step_euler_rmse_median","knn_recall_median","tangent_projector_error_median"]].to_html(index=False,border=0,float_format=lambda x:f"{x:.4g}")
    style="body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f8;color:#17212b;margin:0}main{max-width:1350px;margin:auto;padding:28px}.card{background:white;border:1px solid #d8dee7;border-radius:9px;padding:18px;margin:16px 0;overflow:auto}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:11px}th,td{border:1px solid #d8dee7;padding:5px}th{background:#edf1f5}p{line-height:1.5}"
    html=f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Sphere scalability</title><style>{style}</style></head><body><main><h1>S² → Rᴰ scalability</h1><section class='card'><p>This module fixes d=2, n=480, final seeds 43000–43014, all method configurations, and D∈{{3,5,10,20,50}}. Each ambient coordinate has constant noise variance, so total noise increases with √D.</p><p>GraphVelo uses the same truth-free standardized official TSP pipeline as the main benchmark: median noisy 15-NN positional distance and median noisy velocity norm define the units. Angles use the prespecified clean-speed threshold {ANGLE_SPEED_THRESHOLD:.2f}; retained fractions are in the CSV. Runtime includes only method fitting after one untimed warm-up. Peak RSS is not reported because process-level measurements cannot reliably isolate native allocations per call.</p></section><section class='card'><img src='{image_uri(error)}'></section><section class='card'><img src='{image_uri(runtime_path)}'></section><section class='card'><img src='{image_uri(diagnostics)}'></section><section class='card'><h2>Median results</h2>{table}</section></main></body></html>"
    path=output/"scalability_report.html";path.write_text(html,encoding="utf-8");parser=AuditParser();parser.feed(html)
    return {"self_contained_html":len(parser.images)==3 and all(x.startswith("data:image/png;base64,") for x in parser.images) and not parser.external,"embedded_figure_count":len(parser.images),"obsolete_labels_absent":bool("noise_regime" not in html and "fixed_coordinate" not in html),"fixed_total_absent":bool("fixed_total" not in html and "Fixed-total" not in html),"trend_interpretation_absent":bool("Trend interpretation" not in html)}


def environment() -> dict:
    return {"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},"graphvelo":GRAPHVELO_PROVENANCE,"graphvelo_config":GRAPHVELO_CONFIG,"graphvelo_standardization":GRAPHVELO_STANDARDIZATION,"memory_note":"Per-method peak RSS unavailable: process RSS and Python tracers do not reliably isolate native BLAS/scipy allocations."}


def validate(frame: pd.DataFrame, audit: dict) -> dict:
    expected=len(DIMENSIONS)*len(FINAL_SEEDS)*len(METHODS)
    checks={"expected_seed_rows":bool(len(frame)==expected),"compact_output_schema":bool("noise_regime" not in frame.columns),"no_duplicate_rows":bool(not frame.duplicated(["ambient_dimension","seed","method"]).any()),"angle_fraction_recorded":bool(frame.angle_valid_fraction.between(0,1).all()),**audit}
    checks["all_checks_pass"]=bool(all(checks.values()))
    if not checks["all_checks_pass"]: raise AssertionError(checks)
    return checks


def main():
    global K
    args=parse_args();output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        frame=pd.read_csv(output/"seed_metrics.csv");summary=pd.read_csv(output/"summary_metrics.csv");runtime=pd.read_csv(output/"runtime_memory.csv")
    else:
        config=load_frozen_config();K=int(config["shared_graph_k"]["half_sphere_tangent"]);frame,runtime=run(config);summary=summarize(frame)
        frame.to_csv(output/"seed_metrics.csv",index=False);summary.to_csv(output/"summary_metrics.csv",index=False);runtime.to_csv(output/"runtime_memory.csv",index=False)
        (output/"config.json").write_text(json.dumps({"dimensions":DIMENSIONS,"noise_model":{"per_coordinate_position_sd":TAU_X/np.sqrt(3.0),"per_coordinate_velocity_sd":TAU_V/np.sqrt(3.0),"total_noise_growth":"sqrt(D)"},"seeds":FINAL_SEEDS,"n":N,"d":2,"tau_x":TAU_X,"tau_v":TAU_V,"angle_speed_threshold":ANGLE_SPEED_THRESHOLD,"algorithms":config,"graphvelo":GRAPHVELO_CONFIG,"graphvelo_standardization":GRAPHVELO_STANDARDIZATION},indent=2)+"\n")
        (output/"environment_provenance.json").write_text(json.dumps(environment(),indent=2)+"\n")
    audit=build_report(output,summary,runtime);checks=validate(frame,audit);(output/"sanity_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(output/"scalability_report.html")


if __name__=="__main__": main()
