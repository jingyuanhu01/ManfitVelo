"""Prepare data from Ding et al. 2019 protein latent-space paper.

The raw supplementary files are expected under
``data/protein_latent_paper/raw``. Outputs are written next to that directory.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "protein_latent_paper"
RAW_DIR = DATA_DIR / "raw"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_shared_strings(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("a:si", NS):
        values.append("".join(t.text or "" for t in item.findall(".//a:t", NS)))
    return values


def read_named_sequences(path: Path) -> dict[str, str]:
    """Extract FASTA-like name/sequence pairs from Supplementary Data 1."""
    strings = _xlsx_shared_strings(path)
    pairs: dict[str, str] = {}
    i = 0
    while i + 1 < len(strings):
        name = strings[i]
        seq = strings[i + 1]
        if not name.startswith(">"):
            break
        pairs[name[1:]] = seq
        i += 2
    return pairs


def read_p450_fitness(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "paper_index": row[""],
                    "sequence_id": row["Sequence"],
                    "t50": row["T50"],
                    "split": row["class"],
                    "predicted_t50_z2": row["predicted_T50(dim_Z = 2)"],
                    "predicted_t50_z30": row["predicted_T50(dim_Z = 30)"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_p450() -> dict[str, object]:
    named_sequences = read_named_sequences(RAW_DIR / "supplementary_data_1.xlsx")
    rows = read_p450_fitness(RAW_DIR / "supplementary_data_2.csv")

    cleaned = []
    numeric_rows = []
    named_rows = []
    for row in rows:
        sequence_id = row["sequence_id"]
        full_sequence = named_sequences.get(sequence_id, "")
        item = {
            **row,
            "full_sequence": full_sequence,
        }
        cleaned.append(item)
        if re.fullmatch(r"[123]{8}", sequence_id):
            numeric_rows.append(item)
        else:
            named_rows.append(item)

    write_csv(
        DATA_DIR / "p450_fitness.csv",
        cleaned,
        [
            "paper_index",
            "sequence_id",
            "t50",
            "split",
            "predicted_t50_z2",
            "predicted_t50_z30",
            "full_sequence",
        ],
    )
    write_csv(
        DATA_DIR / "p450_named_fitness.csv",
        named_rows,
        [
            "paper_index",
            "sequence_id",
            "t50",
            "split",
            "predicted_t50_z2",
            "predicted_t50_z30",
            "full_sequence",
        ],
    )

    with (DATA_DIR / "p450_named_sequences.fasta").open("w") as handle:
        for name, sequence in named_sequences.items():
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    genotypes = np.array([[int(ch) for ch in row["sequence_id"]] for row in numeric_rows], dtype=np.int8)
    onehot = np.zeros((len(genotypes), 8, 3), dtype=np.float32)
    for i, genotype in enumerate(genotypes):
        onehot[i, np.arange(8), genotype - 1] = 1.0

    np.save(DATA_DIR / "p450_numeric_genotypes.npy", genotypes)
    np.save(DATA_DIR / "p450_numeric_onehot.npy", onehot.reshape(len(genotypes), 24))
    np.save(DATA_DIR / "p450_numeric_t50.npy", np.array([float(row["t50"]) for row in numeric_rows], dtype=np.float32))
    np.save(DATA_DIR / "p450_numeric_split.npy", np.array([row["split"] for row in numeric_rows]))
    np.save(DATA_DIR / "p450_numeric_sequence_ids.npy", np.array([row["sequence_id"] for row in numeric_rows]))

    return {
        "p450_total_records": len(rows),
        "p450_numeric_fragment_records": len(numeric_rows),
        "p450_named_records": len(named_rows),
        "p450_named_sequences": len(named_sequences),
    }


def prepare_stability() -> dict[str, object]:
    source = RAW_DIR / "supplementary_data_3.csv"
    target = DATA_DIR / "stability_mutations.csv"
    rows = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: row[key] for key in reader.fieldnames or []})
    write_csv(target, rows, list(rows[0]))

    proteins = sorted({row["Name"] for row in rows})
    mutation_re = re.compile(r"^([A-Z])(\d+)([A-Z])$")
    # Protein one-hot, numeric residue position, WT amino acid one-hot, mutant amino acid one-hot.
    mutation_features = np.zeros((len(rows), len(proteins) + 41), dtype=np.float32)
    y_exp = np.zeros(len(rows), dtype=np.float32)
    y_vae = np.zeros(len(rows), dtype=np.float32)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    for i, row in enumerate(rows):
        mutation_features[i, proteins.index(row["Name"])] = 1.0
        y_exp[i] = float(row["delta_delta_G_exp"])
        y_vae[i] = float(row["delta_delta_G_VAE"])
        match = mutation_re.match(row["Mutation"])
        if match:
            wt, pos, mutant = match.groups()
            mutation_features[i, len(proteins)] = float(pos)
            mutation_features[i, len(proteins) + 1 + aa_index[wt]] = 1.0
            mutation_features[i, len(proteins) + 21 + aa_index[mutant]] = 1.0

    np.save(DATA_DIR / "stability_mutation_features.npy", mutation_features)
    np.save(DATA_DIR / "stability_ddg_exp.npy", y_exp)
    np.save(DATA_DIR / "stability_ddg_vae.npy", y_vae)
    return {
        "stability_records": len(rows),
        "stability_proteins": proteins,
        "stability_feature_columns": mutation_features.shape[1],
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    summary.update(prepare_p450())
    summary.update(prepare_stability())
    summary["source_article"] = "https://doi.org/10.1038/s41467-019-13633-0"
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
