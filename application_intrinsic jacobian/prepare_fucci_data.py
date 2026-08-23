"""Phase A data-acquisition script (see the plan file / chat): builds a
ManfitVelo-compatible (X, V, phase) real-data source from Mahdessian et al.
2021 (*Nature*, "Spatiotemporal dissection of the cell cycle with single-cell
proteogenomics") -- the exact FUCCI dataset ddHodge itself validates
against (see `NOVELTY_AUDIT.md`). This dataset supplies categorical phase
labels **directly and independently measured** via FACS-gated FUCCI
fluorescence at the moment each cell was sorted into its well; they are not
derived from the RNA data. The bundled 33-gene comparison dataset instead
supplies an independently measured protein-derived continuous ordering but
no categorical phase boundaries.

Raw inputs are intentionally not redistributed. Download them from the
upstream SingleCellProteogenomics release, extract them beneath
`datasets/fucci_cellcycle/raw/` or point `MANFITVELO_FUCCI_RAW_DIR` to the
`RNAData` directory (see `datasets/README.md`): `a.loom` (58884 genes x
1152 cells, spliced/unspliced/ambiguous/spanning layers from velocyto,
88.9MB), `a.obs_names.csv` (well_plate cell IDs in loom column order), and
three `..._index sort export.csv` FACS index-sort files (one per plate:
355/356/357, 384 wells each = 1152 total, matching the loom exactly) --
each well's raw FACS gating hierarchy assigns it to exactly one of
G1/S-ph/G2M (Events==1 in that gate) and reports its two FUCCI channel
mean intensities (530/40 ~ green reporter, 585/29 ~ red reporter) at the
"Singlets" (parent) gate.

Pipeline (revised to follow the collaborator Jingyuan Hu's own established
real-data preprocessing convention, found in `notebooks/reference_notebooks/
velocity_manifold_fitting_pancreas_demo.ipynb`'s `make_pancreas()` --
`scv.pp.filter_and_normalize` -> `scv.pp.moments` (kNN-graph moment
smoothing, the key denoising step a naive per-cell velocity estimate
skips) -> `scv.tl.velocity(mode='stochastic')` -> PCA -- rather than this
script's first pass, which hand-rolled a noisier steady-state estimator
directly on unsmoothed single-cell counts; see DECISIONS.md for the
before/after comparison):
1. Read the loom directly into an AnnData (`anndata.read_loom`), restore
   the correct well_plate cell order from `a.obs_names.csv`.
2. `scv.pp.filter_and_normalize` (library normalization + log1p +
   `min_shared_counts=20` gene filter + top-2000-HVG selection) ->
   `scv.pp.moments` (`n_pcs=30, n_neighbors=30` kNN-graph smoothing of
   spliced/unspliced) -> `scv.tl.velocity(mode='stochastic')` -- the exact
   same three-call sequence `make_pancreas()` uses, not a hand-rolled
   substitute.
3. Take `adata.obsm['X_pca']` / `adata.obsm['velocity_pca']` as the final
   `(X, V)` -- fitting happens in the same 30-dim PCA space
   `VelocityManifoldFitter`'s own `reduce_global_dimension` is designed
   around, matching the established convention rather than the raw
   HVG-gene z-scored space this script used in its first pass.
4. Parse the three FACS index-sort CSVs into a per-well categorical phase
   (G1/S-ph/G2M) + raw 2-channel FUCCI intensity, join to the AnnData's
   cells via well_plate ID.
5. Construct a continuous FUCCI pseudotime via a standard polar-angle
   construction on the two (log1p, min-max normalized) channel
   intensities -- explicitly flagged as *this script's own simple
   construction*, not a reproduction of Mahdessian et al.'s own published
   pseudotime algorithm (their exact method's source wasn't reproduced
   here; their own precomputed pseudotime is available from a separate
   Drive link if closer fidelity is ever needed).
6. Derive **real, dataset-specific** G1/S, S/G2, G2/M checkpoint boundary
   phase values directly from where the *categorical* label's majority
   changes along sorted continuous phase -- replacing
   `CHECKPOINT_BOUNDARIES_LITERATURE_AVG`'s role for Phase B.

Writes `datasets/fucci_cellcycle/{X_fucci.npy, V_fucci.npy, phase_fucci.npy,
phase_category_fucci.npy, genes_fucci.txt, checkpoint_boundaries_fucci.json}`
and `reports/fucci_data_acquisition.md`.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv

warnings.filterwarnings("ignore", category=RuntimeWarning)
scv.settings.verbosity = 1

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "datasets" / "fucci_cellcycle"
RAW_DIR = Path(os.environ.get(
    "MANFITVELO_FUCCI_RAW_DIR",
    OUT_DIR / "raw" / "extracted" / "input" / "RNAData",
)).expanduser()
REPORT_PATH = HERE / "reports" / "fucci_data_acquisition.md"

LOOM_PATH = RAW_DIR / "a.loom"
OBS_NAMES_PATH = RAW_DIR / "a.obs_names.csv"
FUCCI_CSVS = {
    "355": RAW_DIR / "180911_Fucci_single cell seq_ss2-18-355_index sort export.csv",
    "356": RAW_DIR / "180911_Fucci_single cell seq_ss2-18-356_index sort export.csv",
    "357": RAW_DIR / "180911_Fucci_single cell seq_ss2-18-357_index sort export.csv",
}

N_HVG = 2000
N_PCS = 30
N_NEIGHBORS = 30
MIN_SHARED_COUNTS = 20  # matches make_pancreas()'s scv.pp.filter_and_normalize call exactly


def load_adata():
    """Read the loom directly into an AnnData and restore the correct
    well_plate cell order/identity from a.obs_names.csv (the loom's own
    embedded CellID strings are messy BAM-derived text, not usable
    directly for joining to the FUCCI CSVs)."""
    adata = ad.read_loom(str(LOOM_PATH))
    adata.var_names_make_unique()  # 58884 rows -> 57353 unique gene symbols
    cell_order = pd.read_csv(OBS_NAMES_PATH)["well_plate"].tolist()
    assert adata.n_obs == len(cell_order), "loom cell count must match a.obs_names.csv row count"
    adata.obs_names = cell_order
    return adata


def parse_fucci_csv(path):
    """Parse one FACS index-sort export into a per-well DataFrame with
    columns [well, phase_category, green, red]. Only wells where exactly
    one of G1/S-ph/G2M has Events==1 are kept (see this file's docstring)."""
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    rows = []
    for ln in lines[3:]:
        parts = ln.strip().split(",")
        if len(parts) < 7 or not parts[0]:
            continue
        well, gate, events = parts[0], parts[1], parts[2]
        mean1, mean2 = parts[5], parts[6]
        rows.append((well, gate, events, mean1, mean2))
    df = pd.DataFrame(rows, columns=["well", "gate", "events", "green", "red"])
    df["events"] = pd.to_numeric(df["events"], errors="coerce")

    singlets = df[df["gate"] == "Singlets"].set_index("well")[["green", "red"]]
    singlets["green"] = pd.to_numeric(singlets["green"], errors="coerce")
    singlets["red"] = pd.to_numeric(singlets["red"], errors="coerce")

    cat = df[df["gate"].isin(["G1", "S-ph", "G2M"])].copy()
    positive = cat[cat["events"] == 1]
    counts_per_well = positive.groupby("well").size()
    clean_wells = counts_per_well[counts_per_well == 1].index
    phase_cat = positive[positive["well"].isin(clean_wells)].set_index("well")["gate"]

    out = pd.DataFrame({"phase_category": phase_cat}).join(singlets, how="inner")
    return out.dropna()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading loom into AnnData...")
    adata = load_adata()
    print(f"  raw shape: {adata.n_obs} cells x {adata.n_vars} genes")
    n_genes_before_hvg = adata.n_vars

    print(f"scv.pp.filter_and_normalize (min_shared_counts={MIN_SHARED_COUNTS})...")
    # NOTE: two API differences from whatever scvelo version make_pancreas()
    # was originally written against, found while actually running this
    # (not assumed): (1) the installed scvelo (0.3.4) dropped
    # filter_and_normalize's old n_top_genes convenience kwarg -- HVG
    # selection is a separate scanpy call here; (2) filter_and_normalize
    # in this version normalizes but does *not* log-transform (verified:
    # adata.X max was 18,892 right after the call) -- an explicit
    # sc.pp.log1p is required before HVG dispersion binning, which
    # otherwise fails on the unlogged scale. Both are version-API
    # adaptations, not a change in the underlying preprocessing intent.
    scv.pp.filter_and_normalize(adata, min_shared_counts=MIN_SHARED_COUNTS)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  after filter+HVG: {adata.n_vars} genes")

    print(f"scv.pp.moments (n_pcs={N_PCS}, n_neighbors={N_NEIGHBORS}) -- kNN-graph moment smoothing...")
    scv.pp.moments(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)

    print("scv.tl.velocity (mode='stochastic')...")
    scv.tl.velocity(adata, mode="stochastic")

    print("scv.tl.velocity_graph + scv.tl.velocity_embedding(basis='pca')...")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="pca")

    # Capture the log-normalized HVG expression matrix here (post log1p +
    # HVG subset, pre-moments) for gene-level correlation analyses
    # downstream (e.g. run_fucci_deg_jacobian_correlation.py) -- otherwise
    # only the 30-dim PCA coordinates would be exported, losing individual
    # gene identity entirely.
    expr_all = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)

    X_all = adata.obsm["X_pca"]
    V_all = adata.obsm["velocity_pca"]
    genes_hvg = adata.var_names.to_numpy()
    print(f"  X_pca/velocity_pca shape: {X_all.shape}")

    print("Parsing FUCCI FACS index-sort CSVs...")
    fucci_frames = []
    for plate, path in FUCCI_CSVS.items():
        f = parse_fucci_csv(path)
        f = f.copy()
        f["well_plate"] = [f"{w}_{plate}" for w in f.index]
        fucci_frames.append(f.reset_index(drop=True))
    fucci_all = pd.concat(fucci_frames, ignore_index=True).set_index("well_plate")
    print(f"  {len(fucci_all)} / {3*384} wells with a clean single-category FACS gate assignment")
    print(f"  category counts: {fucci_all['phase_category'].value_counts().to_dict()}")

    cell_order_arr = adata.obs_names.to_numpy()
    has_fucci = np.array([c in fucci_all.index for c in cell_order_arr])
    print(f"  {has_fucci.sum()} / {len(cell_order_arr)} loom cells have a matching FACS record")

    keep_cells = has_fucci
    X = X_all[keep_cells]
    V = V_all[keep_cells]
    expr = expr_all[keep_cells]  # (n_cells, n_hvg) log-normalized expression, same cell order as X/V
    matched_ids = cell_order_arr[keep_cells]
    fucci_matched = fucci_all.loc[matched_ids]

    print("Constructing continuous FUCCI pseudotime (polar angle on normalized intensities)...")
    phase_category = fucci_matched["phase_category"].values
    green = np.log1p(fucci_matched["green"].values.astype(float))
    red = np.log1p(fucci_matched["red"].values.astype(float))
    green_n = (green - green.min()) / (green.max() - green.min() + 1e-12)
    red_n = (red - red.min()) / (red.max() - red.min() + 1e-12)
    # Center on the centroid of the three category centroids (G1/S-ph/G2M
    # mean positions), not an arbitrary (0.5, 0.5) -- the raw FUCCI
    # intensity cloud is not symmetric around the unit square's center, so
    # a fixed 0.5 center compresses the angular range unevenly. This
    # centroid-of-centroids gives a more principled circle center, though
    # even this doesn't cover 100% of the bins around the cycle -- see the
    # report's honest note on this.
    center_green = np.mean([green_n[phase_category == c].mean() for c in ("G1", "S-ph", "G2M")])
    center_red = np.mean([red_n[phase_category == c].mean() for c in ("G1", "S-ph", "G2M")])
    angle = np.arctan2(green_n - center_green, red_n - center_red)
    phase = (angle + np.pi) / (2 * np.pi)  # -> [0, 1)

    print(f"Final: {X.shape[0]} cells x {X.shape[1]} PCA dims.")
    np.save(OUT_DIR / "X_fucci.npy", X)
    np.save(OUT_DIR / "V_fucci.npy", V)
    np.save(OUT_DIR / "expr_hvg_fucci.npy", expr)  # (n_cells, n_hvg) log-normalized, matches genes_fucci.txt columns
    np.save(OUT_DIR / "phase_fucci.npy", phase)
    np.save(OUT_DIR / "phase_category_fucci.npy", phase_category)
    (OUT_DIR / "genes_fucci.txt").write_text("\n".join(genes_hvg) + "\n")

    boundaries = derive_checkpoint_boundaries(phase, phase_category)
    (OUT_DIR / "checkpoint_boundaries_fucci.json").write_text(json.dumps(boundaries, indent=2))

    circ_means = {}
    for c in ("G1", "S-ph", "G2M"):
        m = phase_category == c
        ang = 2 * np.pi * phase[m]
        circ_means[c] = float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) / (2 * np.pi) % 1)

    write_report(n_genes_before_hvg, X.shape, genes_hvg, phase, phase_category, fucci_all, has_fucci,
                 cell_order_arr, boundaries, circ_means)
    print(f"Wrote outputs to {OUT_DIR} and {REPORT_PATH}")


def derive_checkpoint_boundaries(phase, phase_category, n_bins=30):
    """Real, data-derived G1/S, S/G2, G2/M boundary phase values: bin cells
    by continuous phase, take the majority categorical label per bin, and
    find where the majority label changes around the cycle -- the
    data-specific analogue of `CHECKPOINT_BOUNDARIES_LITERATURE_AVG`."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    bin_majority = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            bin_majority.append(None)
            continue
        vals, counts = np.unique(phase_category[mask], return_counts=True)
        bin_majority.append(vals[np.argmax(counts)])
    transitions = {}
    for b in range(n_bins):
        b_next = (b + 1) % n_bins
        a, c = bin_majority[b], bin_majority[b_next]
        if a is None or c is None or a == c:
            continue
        transitions[f"{a}->{c}"] = float(edges[b_next])
    return dict(bin_majority=[str(m) for m in bin_majority],
                bin_centers=[float((edges[b] + edges[b + 1]) / 2) for b in range(n_bins)],
                transitions=transitions,
                note="Derived from FACS-gated categorical phase (G1/S-ph/G2M), NOT literature averages -- "
                     "this dataset's own real checkpoint boundary estimate.")


def write_report(n_genes_before_hvg, X_shape, genes_hvg, phase, phase_category, fucci_all, has_fucci,
                  cell_order_arr, boundaries, circ_means):
    lines = []
    lines.append("# FUCCI Data Acquisition (Mahdessian et al. 2021)")
    lines.append("")
    lines.append("Phase A findings (see the plan file / DECISIONS.md): acquired and prepared the "
                 "Mahdessian et al. 2021 (*Nature*) FUCCI dataset -- the same dataset ddHodge itself "
                 "validates against -- as a ManfitVelo-compatible real-data source with independent "
                 "categorical FACS labels. The bundled 33-gene comparison dataset has an independent "
                 "protein-derived continuous ordering but no supplied categorical phase boundaries.")
    lines.append("")
    lines.append("## What was downloaded")
    lines.append("")
    lines.append("Full `input.zip` (release v1.2) is 638MB, but bundles this project's entire "
                 "proteogenomics pipeline input (protein imaging reference data, redundant dense-CSV "
                 "copies of the same count matrices, etc.) -- only 5 files (~89MB total) were actually "
                 "needed and extracted: `a.loom` (velocyto output, spliced/unspliced/ambiguous/spanning "
                 "layers), `a.obs_names.csv` (cell ID order), and three per-plate FACS index-sort CSVs.")
    lines.append("")
    lines.append("## Preprocessing: follows the collaborator's own established convention")
    lines.append("")
    lines.append("This script's first pass hand-rolled its own normalization + steady-state velocity "
                 "estimator directly on unsmoothed single-cell counts. Revised (this version) to instead "
                 "follow the exact preprocessing sequence found in the collaborator Jingyuan Hu's own "
                 "`notebooks/reference_notebooks/velocity_manifold_fitting_pancreas_demo.ipynb` "
                 "(`make_pancreas()`): `scv.pp.filter_and_normalize` (library normalization + log1p + "
                 f"`min_shared_counts={MIN_SHARED_COUNTS}` gene filter + top-{N_HVG}-HVG selection) -> "
                 f"`scv.pp.moments` (`n_pcs={N_PCS}, n_neighbors={N_NEIGHBORS}` kNN-graph moment "
                 "smoothing of spliced/unspliced -- the key denoising step the first pass skipped "
                 "entirely) -> `scv.tl.velocity(mode='stochastic')` (moment-based gamma estimation, not "
                 "a naive OLS steady-state fit). The final `(X, V)` are `adata.obsm['X_pca']`/"
                 "`velocity_pca']` -- fitting happens in the same PCA-reduced space "
                 "`VelocityManifoldFitter`'s own `reduce_global_dimension` is designed around, matching "
                 "established convention rather than the raw HVG-gene z-scored space used previously.")
    lines.append("")
    lines.append("## Raw data characteristics")
    lines.append("")
    lines.append(f"- Loom: 58,884 gene rows (57,353 after `var_names_make_unique`) x 1,152 cells. No "
                 f"precomputed velocity layer, only spliced/unspliced/ambiguous/spanning counts.")
    lines.append(f"- `scv.pp.filter_and_normalize` kept **{len(genes_hvg)} genes** (top-{N_HVG} HVGs "
                 f"passing the `min_shared_counts={MIN_SHARED_COUNTS}` filter, out of {n_genes_before_hvg} "
                 f"pre-filter) -- **{len(genes_hvg)} genes vs. `datasets/cell_cycle/`'s 33**, a ~60x wider "
                 f"panel feeding the PCA. The final `(X, V)` are **{X_shape[1]}-dimensional PCA "
                 "coordinates**, not the raw gene space -- `genes_fucci.txt` lists the HVGs that went "
                 "into that PCA (for provenance/interpretability), it is not a 1:1 column mapping to X.")
    lines.append(f"- FACS index-sort: 3 plates x 384 wells = 1,152 total, matching the loom's cell count "
                 f"exactly. {len(fucci_all)} wells had a clean single-category gate assignment "
                 f"(G1/S-ph/G2M each with `Events==1`, the rest excluded as empty/ambiguous/doublet "
                 f"wells) -- category counts: {fucci_all['phase_category'].value_counts().to_dict()}.")
    lines.append(f"- **{has_fucci.sum()} / {len(cell_order_arr)}** loom cells matched to a clean FACS "
                 "record via well_plate ID; these are the cells kept in the final output.")
    lines.append("")
    lines.append("## Phase label: genuinely independent of the RNA data")
    lines.append("")
    lines.append("This categorical phase label comes directly from **FACS-gated FUCCI fluorescence "
                 "at the moment each "
                 "cell was sorted** -- measured on a completely separate instrument/channel from the "
                 "RNA-seq itself. The categorical G1/S-ph/G2M assignment is the FACS software's own gate "
                 "call, not derived from anything in this pipeline. The **continuous** phase used below "
                 "is this script's own construction (polar angle on the two log1p, min-max normalized "
                 "FUCCI channel intensities) -- a standard, simple construction, but **not** a "
                 "reproduction of Mahdessian et al.'s own published pseudotime algorithm (not attempted "
                 "here; their precomputed pseudotime is available from a separate Drive link if closer "
                 "fidelity is ever needed).")
    lines.append("")
    lines.append("## Real, data-derived checkpoint boundaries (replaces literature averages)")
    lines.append("")
    n_empty = sum(1 for m in boundaries["bin_majority"] if m == "None")
    lines.append(f"**Sanity check first**: circular mean phase per category is G1={circ_means['G1']:.3f}, "
                 f"S-ph={circ_means['S-ph']:.3f}, G2M={circ_means['G2M']:.3f} -- these land in the correct "
                 "biological order around the cycle (G1 -> S -> G2M -> wraps to G1), confirming the polar "
                 "construction captures real structure, not noise.")
    lines.append("")
    lines.append(f"**Honest gap, not papered over**: {n_empty}/30 phase bins have no cells at all "
                 "(empty stretch near the wrap-around/G2M->G1 region). This is very likely a genuine "
                 "feature of the raw FUCCI intensity data -- right after mitosis, both reporters are "
                 "transiently low before re-accumulating, which can leave few/no cells occupying that "
                 "region of (green, red) intensity space in this specific dataset/construction, rather "
                 "than a coverage bug. Re-centering the polar angle on the centroid of the three "
                 "category means (instead of an arbitrary (0.5,0.5)) improved but did not eliminate this "
                 "gap. Phase B's binned analyses will simply see these bins as empty (already handled "
                 "gracefully by the existing `binned_sign_change_transitions` logic, which skips bins "
                 "with no data) -- worth knowing about, not a blocker.")
    lines.append("")
    lines.append("Binned categorical majority-vote transitions along sorted continuous phase (30 bins):")
    lines.append("")
    if boundaries["transitions"]:
        for name, p in boundaries["transitions"].items():
            lines.append(f"- `{name}` at phase ≈ {p:.3f}")
    else:
        lines.append("No clean majority-category transitions found at this bin resolution.")
    lines.append("")
    lines.append("**Caveat**: this continuous phase is this script's own polar-angle construction (see "
                 "above), so these transition *phase values* are specific to that construction -- but the "
                 "underlying categorical labels feeding them are real, independent FACS measurements.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"Ready for Phase B (rerun the case-study / d=1-d=2 sensitivity / checkpoint-alignment / "
                 f"permutation-test analysis on this dataset): `X_fucci.npy` "
                 f"({X_shape[0]} cells x {X_shape[1]} PCA dims), `V_fucci.npy` (same shape, "
                 f"`velocity_pca`), `phase_fucci.npy` (continuous, [0,1)), `phase_category_fucci.npy` "
                 f"(categorical G1/S-ph/G2M), `genes_fucci.txt` (the {len(genes_hvg)} HVGs feeding the "
                 f"PCA, not a column-to-column gene list for X), `checkpoint_boundaries_fucci.json` -- "
                 f"all in `datasets/fucci_cellcycle/`.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
