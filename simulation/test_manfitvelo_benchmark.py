import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys

from simulation.run_manfitvelo_benchmark import (
    cross_branch_mask,
    hairpin_reach_diagnostics,
    tuning_score,
)


def test_reach_selection_is_geometry_only_and_meets_conditions():
    summary, points, selected = hairpin_reach_diagnostics()
    row = summary[summary.selected].iloc[0]
    assert selected["selection_uses_method_results"] is False
    assert selected["selection_uses_final_seeds"] is False
    assert row.meets_all_conditions
    assert row.reach_over_sigma >= 4
    assert row.q90_knn_radius_over_reach < 0.5
    assert row.cross_arm_edge_fraction < 0.05
    assert len(points) > 0


def test_hairpin_cross_arm_mask_excludes_turn_connections():
    labels = np.array([0, 1, 2])
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    clean = np.zeros((3, 3))
    mask = cross_branch_mask("curved_hairpin", neighbors, labels, clean)
    assert mask[0, 1] and mask[2, 0]
    assert not mask[0, 0] and not mask[1].any()


def test_tuning_score_is_mean_log_of_four_prespecified_errors():
    relative = {
        "clean_point_rmse_rel": 0.5,
        "velocity_rmse_id_rel": 0.5,
        "velocity_angle_mae_id_rel": 0.5,
        "joint_euler_state_rmse_rel": 0.5,
    }
    assert np.isclose(tuning_score(relative), np.log(0.5))


def test_formal_output_has_exact_design_and_no_graphvelo_tuning():
    output=Path(__file__).resolve().parents[1]/"results/manfitvelo_benchmark"
    frame=pd.read_csv(output/"final_seed_metrics.csv");tuning=pd.read_csv(output/"tuning_audit.csv")
    assert len(frame)==945
    assert not frame.duplicated(["scenario","seed","method"]).any()
    assert set(frame.method)=={"ambient_noisy","cosine_kernel","graphvelo","joint_low_rank","local_pca","position_only_manfit","manfitvelo"}
    assert np.allclose(frame.loc[frame.method.eq("ambient_noisy"),[c for c in frame if c.endswith("_rel")]],1)
    assert not tuning.tuning_stage.astype(str).str.contains("graphvelo",case=False).any()
    audit=pd.read_csv(output/"graphvelo_scale_audit.csv")
    assert len(audit)==9*15*2
    assert set(audit.variant)=={"standardized_primary","raw_official_sensitivity"}
    assert not audit.normalization_uses_clean_truth.any()
    assert not audit.selected_by_performance.any()
    assert not audit.oracle_enters_primary_ranking.any()


def test_report_only_preserves_algorithm_outputs_and_figures():
    root=Path(__file__).resolve().parents[1];output=root/"results/manfitvelo_benchmark"
    tracked=[output/"final_seed_metrics.csv",output/"figures/state_circle.png"]
    before=[p.stat().st_mtime_ns for p in tracked]
    subprocess.run([sys.executable,str(root/"simulation/run_manfitvelo_benchmark.py"),"--report-only"],cwd=root,check=True,capture_output=True,text=True)
    assert before==[p.stat().st_mtime_ns for p in tracked]
    html=(output/"final_report.html").read_text()
    assert html.count("data:image/png;base64,")==9 and "http://" not in html and "https://" not in html
    assert all(term in html for term in ("Experiment parameters","Mathematical definition","GraphVelo scale-equivariance audit"))
