#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def natural_key(text: str):
    # Split text into list of strings and ints for natural sort
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split("(\\d+)", str(text))]


def build_id_to_name_map(neurons_df: pd.DataFrame) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    for _, row in neurons_df.iterrows():
        body_id = int(row["bodyId"])
        instance = str(row.get("instance", "") or "").strip()
        ntype = str(row.get("type", "") or "").strip()
        if instance:
            id_to_name[body_id] = instance
        elif ntype:
            id_to_name[body_id] = ntype
        else:
            id_to_name[body_id] = str(body_id)
    return id_to_name


def filter_epg_neurons(neurons_df: pd.DataFrame, epg_regex: str = r"^EPG") -> pd.DataFrame:
    # EPG identification: match at start of 'type' or 'instance'
    type_is_epg = neurons_df["type"].fillna("").str.contains(epg_regex, case=False, regex=True)
    inst_is_epg = neurons_df["instance"].fillna("").str.contains(epg_regex, case=False, regex=True)
    return neurons_df[type_is_epg | inst_is_epg].copy()


def compute_epg_epg_weight_matrix(
    neurons_csv: Path,
    total_connections_csv: Path,
    epg_regex: str = r"^EPG",
) -> pd.DataFrame:
    # Load neurons (minimal schema: bodyId,type,instance)
    neurons = pd.read_csv(neurons_csv, dtype={"bodyId": "int64", "type": "string", "instance": "string"})
    neurons = neurons[["bodyId", "type", "instance"]]
    epg_neurons = filter_epg_neurons(neurons, epg_regex=epg_regex)
    if epg_neurons.empty:
        raise SystemExit("No EPG neurons were found using the given pattern.")

    epg_ids: List[int] = epg_neurons["bodyId"].astype("int64").tolist()
    id_to_name = build_id_to_name_map(epg_neurons)

    # Load total connections (schema: bodyId_pre,bodyId_post,weight)
    conns = pd.read_csv(
        total_connections_csv,
        dtype={"bodyId_pre": "int64", "bodyId_post": "int64", "weight": "int64"},
    )
    required_cols = {"bodyId_pre", "bodyId_post", "weight"}
    if not required_cols.issubset(conns.columns):
        missing = required_cols.difference(conns.columns)
        raise SystemExit(f"Missing required columns in connections CSV: {missing}")

    # Filter to EPG -> EPG connections
    conns_epg = conns[conns["bodyId_pre"].isin(epg_ids) & conns["bodyId_post"].isin(epg_ids)].copy()
    if conns_epg.empty:
        # Create an empty matrix with all EPG names if there are no direct connections recorded
        names = sorted([id_to_name[i] for i in epg_ids], key=natural_key)
        return pd.DataFrame(0, index=names, columns=names, dtype="int64")

    # Group and aggregate weights
    grouped = (
        conns_epg.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
        .rename(columns={"bodyId_pre": "pre", "bodyId_post": "post"})
    )
    grouped["pre_name"] = grouped["pre"].map(id_to_name).fillna(grouped["pre"].astype(str))
    grouped["post_name"] = grouped["post"].map(id_to_name).fillna(grouped["post"].astype(str))

    # Pivot to a square matrix
    matrix = grouped.pivot_table(
        index="pre_name",
        columns="post_name",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure same ordering for rows/cols and include all EPGs (even if absent in data)
    all_names = sorted({*matrix.index.tolist(), *matrix.columns.tolist()}, key=natural_key)
    matrix = matrix.reindex(index=all_names, columns=all_names, fill_value=0)
    matrix = matrix.astype("int64")
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute EPG-EPG weight matrix from traced CSVs (neuprint export)."
    )
    parser.add_argument(
        "--neurons",
        type=Path,
        required=True,
        help="Path to traced-neurons.csv (expects columns: bodyId,type,instance).",
    )
    parser.add_argument(
        "--total-connections",
        type=Path,
        required=True,
        help="Path to traced-total-connections.csv (columns: bodyId_pre,bodyId_post,weight).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the EPG-EPG matrix.",
    )
    parser.add_argument(
        "--epg-pattern",
        type=str,
        default=r"^EPG",
        help="Regex pattern to select EPG neurons by type/instance (default: ^EPG).",
    )
    args = parser.parse_args()

    if not args.neurons.exists():
        raise SystemExit(f"Neurons CSV not found: {args.neurons}")
    if not args.total_connections.exists():
        raise SystemExit(f"Total connections CSV not found: {args.total_connections}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    matrix = compute_epg_epg_weight_matrix(
        neurons_csv=args.neurons,
        total_connections_csv=args.total_connections,
        epg_regex=args.epg_pattern,
    )
    matrix.to_csv(args.output)

    print(f"EPG-EPG weight matrix written to: {args.output}")
    print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")


if __name__ == "__main__":
    main()

