#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split("(\\d+)", str(text))]


def build_id_to_name_map(neurons_df: pd.DataFrame) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    for _, row in neurons_df.iterrows():
        bid = int(row["bodyId"])
        inst = str(row.get("instance", "") or "").strip()
        ntype = str(row.get("type", "") or "").strip()
        id_to_name[bid] = inst if inst else (ntype if ntype else str(bid))
    return id_to_name


def parse_pattern_list(pattern: str) -> Set[str]:
    """
    Accept comma/space-separated ROI names (e.g., 'EBw01, EBw02, EBw03').
    """
    if not pattern:
        return set()
    parts = re.split(r"[\s,]+", pattern.strip())
    return {p for p in (s.strip() for s in parts) if p}


def discover_eb_roi_names(df: pd.DataFrame, eb_regex: str) -> Set[str]:
    roi_cols = [c for c in df.columns if c.lower().endswith("_roi")]  # pre_roi / post_roi
    names: Set[str] = set()
    if "roi" in df.columns:
        names |= set(df["roi"].dropna().astype(str))
    for c in roi_cols:
        names |= set(df[c].dropna().astype(str))
    eb_pat = re.compile(eb_regex, re.IGNORECASE)
    return {r for r in names if eb_pat.search(r)}


def aggregate_neuron_eb_rois(
    roi_df: pd.DataFrame,
    side: str,
    eb_regex: str,
    weight_threshold: int,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ['bodyId', 'roi', 'weight'] for EB ROIs only.
    side: 'pre' | 'post' | 'both'
    """
    eb_pat = re.compile(eb_regex, re.IGNORECASE)
    has_pre = "pre_roi" in roi_df.columns or "bodyId_pre" in roi_df.columns
    has_post = "post_roi" in roi_df.columns or "bodyId_post" in roi_df.columns

    records: List[Tuple[int, str, int]] = []

    def add_records(bid_series: pd.Series, roi_series: pd.Series, weight_series: pd.Series):
        for bid, roi, w in zip(bid_series, roi_series, weight_series):
            if pd.isna(roi):
                continue
            roi = str(roi)
            if not eb_pat.search(roi):
                continue
            if int(w) >= weight_threshold:
                records.append((int(bid), roi, int(w)))

    # Try to use explicit pre/post ROI columns if available
    if "pre_roi" in roi_df.columns and "post_roi" in roi_df.columns:
        if side in ("pre", "both"):
            add_records(roi_df["bodyId_pre"], roi_df["pre_roi"], roi_df["weight"])
        if side in ("post", "both"):
            add_records(roi_df["bodyId_post"], roi_df["post_roi"], roi_df["weight"])
    elif "roi" in roi_df.columns:
        # Fallback: a single ROI column; associate with both sides depending on availability
        if side in ("pre", "both") and "bodyId_pre" in roi_df.columns:
            add_records(roi_df["bodyId_pre"], roi_df["roi"], roi_df["weight"])
        if side in ("post", "both") and "bodyId_post" in roi_df.columns:
            add_records(roi_df["bodyId_post"], roi_df["roi"], roi_df["weight"])
    else:
        raise SystemExit(
            "ROI connections CSV must contain either 'pre_roi'/'post_roi' or a single 'roi' column."
        )

    if not records:
        return pd.DataFrame(columns=["bodyId", "roi", "weight"])

    df = pd.DataFrame(records, columns=["bodyId", "roi", "weight"])
    # Aggregate by (bodyId, roi)
    df = (
        df.groupby(["bodyId", "roi"], as_index=False)["weight"]
        .sum()
        .sort_values(["bodyId", "roi"], key=lambda s: s.map(natural_key))
    )
    return df


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_matches(
    neuron_rois: pd.DataFrame,
    pattern_rois: Set[str],
    min_score: float,
) -> pd.DataFrame:
    """
    neuron_rois: DataFrame ['bodyId','roi','weight'] for EB ROIs
    pattern_rois: set of ROI names to match
    """
    if not len(neuron_rois):
        return pd.DataFrame(columns=["bodyId", "score", "rois"])
    roi_sets = neuron_rois.groupby("bodyId")["roi"].apply(lambda s: set(s.tolist()))
    results = []
    for body_id, rois in roi_sets.items():
        score = jaccard(rois, pattern_rois)
        if score >= min_score:
            results.append((int(body_id), score, sorted(rois, key=natural_key)))
    out = pd.DataFrame(results, columns=["bodyId", "score", "rois"]).sort_values(
        ["score", "bodyId"], ascending=[False, True]
    )
    return out


def main():
    p = argparse.ArgumentParser(
        description="Find neurons whose EB ROI pattern matches an ideal pattern."
    )
    p.add_argument("--neurons", type=Path, required=True, help="Path to traced-neurons.csv")
    p.add_argument(
        "--roi-connections",
        type=Path,
        required=True,
        help="Path to traced-roi-connections.csv",
    )
    p.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Comma/space separated EB ROI names (e.g., 'EBw01,EBw02,EBw03').",
    )
    p.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["pre", "post", "both"],
        help="Use pre, post, or both sides to build EB ROI sets (default: both).",
    )
    p.add_argument(
        "--eb-regex",
        type=str,
        default=r"\bEB",
        help="Regex to identify EB ROI names (default: '\\bEB').",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum Jaccard score to include matches (default: 0.5).",
    )
    p.add_argument(
        "--weight-threshold",
        type=int,
        default=1,
        help="Minimum connection weight per (body,roi) to consider presence (default: 1).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for matches.",
    )
    args = p.parse_args()

    if not args.neurons.exists():
        raise SystemExit(f"Neurons CSV not found: {args.neurons}")
    if not args.roi_connections.exists():
        raise SystemExit(f"ROI connections CSV not found: {args.roi_connections}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    neurons = pd.read_csv(args.neurons, dtype={"bodyId": "int64", "type": "string", "instance": "string"})
    id_to_name = build_id_to_name_map(neurons)
    roi_conns = pd.read_csv(
        args.roi_connections,
        dtype={
            "bodyId_pre": "int64",
            "bodyId_post": "int64",
            "weight": "int64",
            "pre_roi": "string",
            "post_roi": "string",
            "roi": "string",
        },
    )

    # Helpful printout of available EB ROI names
    eb_rois = sorted(discover_eb_roi_names(roi_conns, args.eb_regex), key=natural_key)
    if not eb_rois:
        raise SystemExit("No EB ROI names discovered in ROI connections.")
    print(f"Discovered EB ROI names ({len(eb_rois)}): {', '.join(eb_rois[:24])}{' ...' if len(eb_rois)>24 else ''}")

    pattern_rois = parse_pattern_list(args.pattern)
    missing = pattern_rois.difference(set(eb_rois))
    if missing:
        print(f"Note: pattern contains ROI names not seen in data: {', '.join(sorted(missing))}")

    neuron_eb = aggregate_neuron_eb_rois(
        roi_df=roi_conns, side=args.side, eb_regex=args.eb_regex, weight_threshold=args.weight_threshold
    )
    matches = compute_matches(neuron_rois=neuron_eb, pattern_rois=pattern_rois, min_score=args.min_score)
    if not len(matches):
        print("No neurons matched the pattern with given thresholds.")
        # Still write an empty CSV with headers
        matches.to_csv(args.output, index=False)
        return

    # Add type/instance columns for readability
    matches["name"] = matches["bodyId"].map(id_to_name)
    # Flatten rois list to semicolon-separated string
    matches["rois"] = matches["rois"].apply(lambda lst: ";".join(lst))
    matches = matches[["bodyId", "name", "score", "rois"]]
    matches.to_csv(args.output, index=False)
    print(f"Wrote {len(matches)} matches to {args.output}")


if __name__ == "__main__":
    main()

