import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys

from simulation.run_sphere_scalability import (
    analytic_sphere_projection, orthogonal_embedding, sphere_data,
)


def test_sphere_generator_geometry_and_tangency():
    data=sphere_data(43000,20,"fixed_total");Q=data["Q"]
    assert np.allclose(Q.T@Q,np.eye(3),atol=1e-12)
    assert np.allclose(np.linalg.norm(data["P"],axis=1),1,atol=1e-12)
    assert np.allclose(np.sum(data["P"]*data["truth"],axis=1),0,atol=1e-12)
    expected=np.einsum("da,nab,eb->nde",Q,np.eye(3)[None]-np.einsum("ni,nj->nij",data["z"],data["z"]),Q)
    assert np.allclose(data["true_projector"],expected)


def test_noise_regimes_are_exactly_equal_at_d3():
    a=sphere_data(43001,3,"fixed_total");b=sphere_data(43001,3,"fixed_coordinate")
    assert np.array_equal(a["Y"],b["Y"]) and np.array_equal(a["field"],b["field"])


def test_analytic_projection_is_exact_for_normal_and_off_span_noise():
    data=sphere_data(43002,10,"fixed_total");Q=data["Q"];X=data["P"]
    orth=np.linalg.svd(Q.T,full_matrices=True)[2][3]
    perturbed=1.2*X+.3*orth
    projected,_,distance=analytic_sphere_projection(perturbed,Q)
    assert np.allclose(projected,X,atol=1e-12)
    assert np.isclose(distance,np.sqrt(.2**2+.3**2),atol=1e-12)


def test_scalability_output_and_report_only_contract():
    root=Path(__file__).resolve().parents[1];output=root/"results/sphere_scalability"
    metrics=output/"seed_metrics.csv";before=metrics.stat().st_mtime_ns
    subprocess.run([sys.executable,str(root/"simulation/run_sphere_scalability.py"),"--report-only"],cwd=root,check=True,capture_output=True,text=True)
    assert metrics.stat().st_mtime_ns==before
    html=(output/"scalability_report.html").read_text()
    seed_metrics=pd.read_csv(output/"seed_metrics.csv")
    summary_metrics=pd.read_csv(output/"summary_metrics.csv")
    runtime_memory=pd.read_csv(output/"runtime_memory.csv")
    assert html.count("data:image/png;base64,")==3 and "http://" not in html and "https://" not in html
    assert "fixed_total" not in html and "Fixed-total" not in html
    assert "fixed_coordinate" not in html and "noise_regime" not in html
    assert "Trend interpretation" not in html
    assert "noise_regime" not in seed_metrics.columns
    assert "noise_regime" not in summary_metrics.columns
    assert "noise_regime" not in runtime_memory.columns
