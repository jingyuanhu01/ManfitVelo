# FUCCI Data Acquisition (Mahdessian et al. 2021)

Phase A findings (see the plan file / DECISIONS.md): acquired and prepared the Mahdessian et al. 2021 (*Nature*) FUCCI dataset -- the same dataset ddHodge itself validates against -- as a ManfitVelo-compatible real-data source with independent categorical FACS labels. The bundled 33-gene comparison dataset has an independent protein-derived continuous ordering but no supplied categorical phase boundaries.

## What was downloaded

Full `input.zip` (release v1.2) is 638MB, but bundles this project's entire proteogenomics pipeline input (protein imaging reference data, redundant dense-CSV copies of the same count matrices, etc.) -- only 5 files (~89MB total) were actually needed and extracted: `a.loom` (velocyto output, spliced/unspliced/ambiguous/spanning layers), `a.obs_names.csv` (cell ID order), and three per-plate FACS index-sort CSVs.

## Preprocessing: follows the collaborator's own established convention

This script's first pass hand-rolled its own normalization + steady-state velocity estimator directly on unsmoothed single-cell counts. Revised (this version) to instead follow the exact preprocessing sequence found in the collaborator Jingyuan Hu's own `notebooks/reference_notebooks/velocity_manifold_fitting_pancreas_demo.ipynb` (`make_pancreas()`): `scv.pp.filter_and_normalize` (library normalization + log1p + `min_shared_counts=20` gene filter + top-2000-HVG selection) -> `scv.pp.moments` (`n_pcs=30, n_neighbors=30` kNN-graph moment smoothing of spliced/unspliced -- the key denoising step the first pass skipped entirely) -> `scv.tl.velocity(mode='stochastic')` (moment-based gamma estimation, not a naive OLS steady-state fit). The final `(X, V)` are `adata.obsm['X_pca']`/`velocity_pca']` -- fitting happens in the same PCA-reduced space `VelocityManifoldFitter`'s own `reduce_global_dimension` is designed around, matching established convention rather than the raw HVG-gene z-scored space used previously.

## Raw data characteristics

- Loom: 58,884 gene rows (57,353 after `var_names_make_unique`) x 1,152 cells. No precomputed velocity layer, only spliced/unspliced/ambiguous/spanning counts.
- `scv.pp.filter_and_normalize` kept **2000 genes** (top-2000 HVGs passing the `min_shared_counts=20` filter, out of 58884 pre-filter) -- **2000 genes vs. `datasets/cell_cycle/`'s 33**, a ~60x wider panel feeding the PCA. The final `(X, V)` are **30-dimensional PCA coordinates**, not the raw gene space -- `genes_fucci.txt` lists the HVGs that went into that PCA (for provenance/interpretability), it is not a 1:1 column mapping to X.
- FACS index-sort: 3 plates x 384 wells = 1,152 total, matching the loom's cell count exactly. 1067 wells had a clean single-category gate assignment (G1/S-ph/G2M each with `Events==1`, the rest excluded as empty/ambiguous/doublet wells) -- category counts: {'G2M': 387, 'G1': 346, 'S-ph': 334}.
- **1067 / 1152** loom cells matched to a clean FACS record via well_plate ID; these are the cells kept in the final output.

## Phase label: genuinely independent of the RNA data

This phase label comes directly from **FACS-gated FUCCI fluorescence at the moment each cell was sorted** -- measured on a completely separate instrument/channel from the RNA-seq itself. The categorical G1/S-ph/G2M assignment is the FACS software's own gate call, not derived from anything in this pipeline. The **continuous** phase used below is this script's own construction (polar angle on the two log1p, min-max normalized FUCCI channel intensities) -- a standard, simple construction, but **not** a reproduction of Mahdessian et al.'s own published pseudotime algorithm (not attempted here; their precomputed pseudotime is available from a separate Drive link if closer fidelity is ever needed).

## Real, data-derived checkpoint boundaries (replaces literature averages)

**Sanity check first**: circular mean phase per category is G1=0.298, S-ph=0.587, G2M=0.890 -- these land in the correct biological order around the cycle (G1 -> S -> G2M -> wraps to G1), confirming the polar construction captures real structure, not noise.

**Honest gap, not papered over**: 5/30 phase bins have no cells at all (empty stretch near the wrap-around/G2M->G1 region). This is very likely a genuine feature of the raw FUCCI intensity data -- right after mitosis, both reporters are transiently low before re-accumulating, which can leave few/no cells occupying that region of (green, red) intensity space in this specific dataset/construction, rather than a coverage bug. Re-centering the polar angle on the centroid of the three category means (instead of an arbitrary (0.5,0.5)) improved but did not eliminate this gap. Phase B's binned analyses will simply see these bins as empty (already handled gracefully by the existing `binned_sign_change_transitions` logic, which skips bins with no data) -- worth knowing about, not a blocker.

Binned categorical majority-vote transitions along sorted continuous phase (30 bins):

- `G1->S-ph` at phase ≈ 0.433
- `S-ph->G2M` at phase ≈ 0.800

**Caveat**: this continuous phase is this script's own polar-angle construction (see above), so these transition *phase values* are specific to that construction -- but the underlying categorical labels feeding them are real, independent FACS measurements.

## Status

Ready for the executed FUCCI application notebook: `X_fucci.npy` (1067 cells x 30 PCA dims), `V_fucci.npy` (same shape, `velocity_pca`), `phase_fucci.npy` (continuous, [0,1)), `phase_category_fucci.npy` (categorical G1/S-ph/G2M), `genes_fucci.txt` (the 2000 HVGs feeding the PCA, not a column-to-column gene list for X), `checkpoint_boundaries_fucci.json` -- all in `datasets/fucci_cellcycle/`.
