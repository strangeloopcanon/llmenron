#!/usr/bin/env python3
"""Detect Enron regime shifts and build an agent-replay scenario spec."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks

TOKEN_RE = re.compile(r"[a-z]{3,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "you",
    "your",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "will",
    "can",
    "could",
    "would",
    "should",
    "about",
    "re",
    "fw",
    "fwd",
    "subject",
    "please",
    "thanks",
    "thank",
    "need",
    "attached",
    "regards",
    "enron",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-metadata",
        type=Path,
        default=Path("results/rosetta/enron_rosetta_events_metadata.parquet"),
    )
    parser.add_argument(
        "--events-content",
        type=Path,
        default=Path("results/rosetta/enron_rosetta_events_content.parquet"),
    )
    parser.add_argument(
        "--header-cache",
        type=Path,
        default=Path("data/enron_headers_1997_2003.parquet"),
    )
    parser.add_argument("--regimes-dir", type=Path, default=Path("results/regimes"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("results/tasks"))
    parser.add_argument("--scenarios-dir", type=Path, default=Path("results/scenarios"))
    parser.add_argument("--coverage-threshold", type=float, default=0.60)
    parser.add_argument("--window-weeks", type=int, default=12)
    parser.add_argument("--min-break-gap-weeks", type=int, default=12)
    parser.add_argument("--consensus-tolerance-weeks", type=int, default=2)
    parser.add_argument("--z-threshold", type=float, default=2.2)
    parser.add_argument("--min-breaks", type=int, default=6)
    parser.add_argument("--max-breaks", type=int, default=10)
    parser.add_argument("--task-clusters", type=int, default=8)
    parser.add_argument("--dependency-window-hours", type=int, default=72)
    parser.add_argument("--dependency-min-weight", type=int, default=3)
    parser.add_argument("--dependency-max-edges", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def monday_start(ts: pd.Series) -> pd.Series:
    day = ts.dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    return day - pd.to_timedelta(day.dt.weekday, unit="D")


def parse_artifacts(payload: str) -> dict[str, Any]:
    base = {
        "subject": "",
        "norm_subject": "",
        "to_count": 0,
        "cc_count": 0,
        "bcc_count": 0,
        "is_reply": False,
        "is_forward": False,
        "is_escalation": False,
        "consult_legal_specialist": False,
        "consult_trading_specialist": False,
        "has_attachment_reference": False,
    }
    if not isinstance(payload, str) or not payload:
        return base
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return base
    out = base.copy()
    for key in out:
        if key in obj:
            out[key] = obj[key]
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std <= 1e-9:
            return np.zeros_like(vals)
        return (vals - mean) / std
    return (vals - med) / scale


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr >= 0]
    if len(arr) == 0:
        return 0.0
    if np.allclose(arr, 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n)


def centralization(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n <= 2:
        return 0.0
    k_max = float(np.max(arr))
    numerator = float(np.sum(k_max - arr))
    denominator = float((n - 1) * (n - 2))
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing events metadata file: {path}")

    events = pd.read_parquet(path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"]).copy()
    events["actor_id"] = events["actor_id"].fillna("").astype(str)
    events["target_id"] = events["target_id"].fillna("").astype(str)
    events["event_type"] = events["event_type"].fillna("message").astype(str)
    events["thread_task_id"] = events["thread_task_id"].fillna("").astype(str)

    parsed = [parse_artifacts(x) for x in events["artifacts"].fillna("").astype(str)]
    parsed_df = pd.DataFrame(parsed)
    events = pd.concat([events.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    for col in [
        "to_count",
        "cc_count",
        "bcc_count",
    ]:
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "is_reply",
        "is_forward",
        "is_escalation",
        "consult_legal_specialist",
        "consult_trading_specialist",
        "has_attachment_reference",
    ]:
        events[col] = events[col].fillna(False).astype(bool)

    events["is_specialist_consult"] = (
        events["consult_legal_specialist"] | events["consult_trading_specialist"]
    )
    events["fanout"] = events["to_count"] + events["cc_count"] + events["bcc_count"]
    events["is_assignment"] = (events["event_type"] == "assignment").astype(int)
    events["is_approval"] = (events["event_type"] == "approval").astype(int)
    events["is_escalation_event"] = (events["event_type"] == "escalation").astype(int)
    events["hour_utc"] = events["timestamp"].dt.tz_convert("UTC").dt.hour
    events["is_after_hours"] = ((events["hour_utc"] < 8) | (events["hour_utc"] > 18)).astype(int)
    events["day"] = events["timestamp"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    events["week"] = monday_start(events["timestamp"])
    events = events.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    return events


def coverage_series(events: pd.DataFrame, header_cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_day = events.groupby("day").size().rename("observed_count").to_frame()
    if header_cache_path.exists():
        headers = pd.read_parquet(header_cache_path, columns=["sent_ts"])
        headers["sent_ts"] = pd.to_datetime(headers["sent_ts"], utc=True, errors="coerce")
        headers = headers.dropna(subset=["sent_ts"]).copy()
        headers["day"] = headers["sent_ts"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
        header_day = headers.groupby("day").size().rename("header_count").to_frame()
    else:
        header_day = pd.DataFrame(index=observed_day.index, data={"header_count": np.nan})

    daily = observed_day.join(header_day, how="outer").fillna(0)
    if "header_count" not in daily:
        daily["header_count"] = 0
    daily["coverage"] = np.where(
        daily["header_count"] > 0,
        daily["observed_count"] / daily["header_count"],
        np.nan,
    )
    daily["week"] = daily.index - pd.to_timedelta(daily.index.weekday, unit="D")
    weekly = (
        daily.groupby("week")
        .agg(
            observed_count=("observed_count", "sum"),
            header_count=("header_count", "sum"),
            coverage_mean=("coverage", "mean"),
            coverage_min=("coverage", "min"),
        )
        .reset_index()
        .sort_values("week")
    )
    weekly["coverage_ratio"] = np.where(
        weekly["header_count"] > 0,
        weekly["observed_count"] / weekly["header_count"],
        np.nan,
    )
    return daily.reset_index().rename(columns={"index": "day"}), weekly


def build_thread_features(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby("thread_task_id", sort=False)

    base = grouped.agg(
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
        n_events=("event_id", "count"),
        n_participants=("actor_id", "nunique"),
        n_targets=("target_id", "nunique"),
        n_escalations=("is_escalation_event", "sum"),
        n_assignments=("is_assignment", "sum"),
        n_approvals=("is_approval", "sum"),
        n_forwards=("is_forward", "sum"),
        cc_total=("cc_count", "sum"),
        bcc_total=("bcc_count", "sum"),
        fanout_mean=("fanout", "mean"),
        specialist_consults=("is_specialist_consult", "sum"),
        attachment_refs=("has_attachment_reference", "sum"),
        after_hours_share=("is_after_hours", "mean"),
        anchor_subject=("subject", "first"),
        anchor_norm_subject=("norm_subject", "first"),
    ).reset_index()

    base["duration_hours"] = (
        (base["end_ts"] - base["start_ts"]).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    base["start_week"] = monday_start(base["start_ts"])
    base["end_week"] = monday_start(base["end_ts"])
    base["escalations_per_event"] = np.where(
        base["n_events"] > 0,
        base["n_escalations"] / base["n_events"],
        0.0,
    )
    base["assignments_per_event"] = np.where(
        base["n_events"] > 0,
        base["n_assignments"] / base["n_events"],
        0.0,
    )
    base["approvals_per_event"] = np.where(
        base["n_events"] > 0,
        base["n_approvals"] / base["n_events"],
        0.0,
    )

    # Fast per-thread handoff and reopen counts from sorted event stream.
    sorted_events = events.sort_values(["thread_task_id", "timestamp", "event_id"])
    handoff: dict[str, int] = {}
    reopen: dict[str, int] = {}
    thread_last_actor: dict[str, str] = {}
    thread_last_ts: dict[str, int] = {}
    for row in sorted_events.itertuples(index=False):
        tid = row.thread_task_id
        actor = row.actor_id
        ts_ns = int(row.timestamp.value)
        if tid not in handoff:
            handoff[tid] = 0
            reopen[tid] = 0
            thread_last_actor[tid] = actor
            thread_last_ts[tid] = ts_ns
            continue
        if actor != thread_last_actor[tid]:
            handoff[tid] += 1
        gap_hours = (ts_ns - thread_last_ts[tid]) / 3_600_000_000_000.0
        if gap_hours > 24.0 * 7.0:
            reopen[tid] += 1
        thread_last_actor[tid] = actor
        thread_last_ts[tid] = ts_ns

    base["handoff_count"] = base["thread_task_id"].map(handoff).fillna(0).astype(int)
    base["reopen_count"] = base["thread_task_id"].map(reopen).fillna(0).astype(int)
    base["reopen_flag"] = (base["reopen_count"] > 0).astype(int)

    # Subject entropy per thread.
    subject_counts = (
        events.groupby(["thread_task_id", "norm_subject"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_per_thread = subject_counts.groupby("thread_task_id")["n"].sum().rename("n_total")
    subject_counts = subject_counts.join(total_per_thread, on="thread_task_id")
    subject_counts["p"] = subject_counts["n"] / subject_counts["n_total"]
    subject_counts["h_term"] = -subject_counts["p"] * np.log(subject_counts["p"])
    entropy = subject_counts.groupby("thread_task_id")["h_term"].sum().rename("subject_entropy")
    base = base.join(entropy, on="thread_task_id")
    base["subject_entropy"] = base["subject_entropy"].fillna(0.0)

    return base


def weekly_graph_proxies(events: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    week_groups = events.groupby("week", sort=True)
    for week, g in week_groups:
        actor_counts = g.groupby("actor_id").size().to_numpy()
        target_counts = g.groupby("target_id").size().to_numpy()

        directed = g[
            g["target_id"].str.contains("@", na=False)
            & ~g["target_id"].str.startswith("group:", na=False)
            & ~g["target_id"].str.startswith("thread:", na=False)
        ][["actor_id", "target_id"]]
        edge_set = set(zip(directed["actor_id"], directed["target_id"], strict=False))
        if edge_set:
            reciprocated_edges = sum(
                1 for (src, dst) in edge_set if (dst, src) in edge_set
            )
            reciprocity = reciprocated_edges / len(edge_set)
            edge_density = safe_div(
                len(edge_set),
                len(set(directed["actor_id"])) * max(1, len(set(directed["target_id"]))),
            )
        else:
            reciprocity = 0.0
            edge_density = 0.0

        out_rows.append(
            {
                "week": week,
                "actor_out_gini": gini(actor_counts),
                "target_in_gini": gini(target_counts),
                "actor_out_centralization": centralization(actor_counts),
                "target_in_centralization": centralization(target_counts),
                "pair_reciprocity": float(reciprocity),
                "pair_density_proxy": float(edge_density),
            }
        )
    return pd.DataFrame(out_rows).sort_values("week")


def build_weekly_panel(
    events: pd.DataFrame,
    thread_features: pd.DataFrame,
    coverage_weekly: pd.DataFrame,
) -> pd.DataFrame:
    weekly = (
        events.groupby("week", sort=True)
        .agg(
            n_events=("event_id", "count"),
            n_threads_active=("thread_task_id", "nunique"),
            n_actors=("actor_id", "nunique"),
            n_targets=("target_id", "nunique"),
            forward_rate=("is_forward", "mean"),
            cc_rate=("cc_count", lambda s: float((s > 0).mean())),
            bcc_rate=("bcc_count", lambda s: float((s > 0).mean())),
            mean_fanout=("fanout", "mean"),
            specialist_consult_rate=("is_specialist_consult", "mean"),
            escalation_rate=("is_escalation_event", "mean"),
            assignment_rate=("is_assignment", "mean"),
            approval_rate=("is_approval", "mean"),
            message_reply_rate=("is_reply", "mean"),
            after_hours_share=("is_after_hours", "mean"),
        )
        .reset_index()
        .sort_values("week")
    )
    weekly["avg_events_per_thread"] = np.where(
        weekly["n_threads_active"] > 0,
        weekly["n_events"] / weekly["n_threads_active"],
        0.0,
    )

    thread_week = (
        thread_features.groupby("start_week", sort=True)
        .agg(
            threads_opened=("thread_task_id", "count"),
            median_thread_duration_h=("duration_hours", "median"),
            mean_handoff_count=("handoff_count", "mean"),
            reopen_rate=("reopen_flag", "mean"),
            mean_participants_per_new_thread=("n_participants", "mean"),
        )
        .reset_index()
        .rename(columns={"start_week": "week"})
    )
    thread_close_week = (
        thread_features.groupby("end_week", sort=True)
        .size()
        .rename("threads_closed")
        .reset_index()
        .rename(columns={"end_week": "week"})
    )
    weekly = weekly.merge(thread_week, on="week", how="left")
    weekly = weekly.merge(thread_close_week, on="week", how="left")

    graph = weekly_graph_proxies(events)
    weekly = weekly.merge(graph, on="week", how="left")
    weekly = weekly.merge(coverage_weekly, on="week", how="left")

    weekly = weekly.sort_values("week").reset_index(drop=True)
    for col in weekly.columns:
        if col == "week":
            continue
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce")
    weekly = weekly.fillna(0.0)
    return weekly


def choose_detection_panel(panel: pd.DataFrame, coverage_threshold: float, window_weeks: int) -> pd.DataFrame:
    filt = panel[panel["coverage_mean"] >= coverage_threshold].copy()
    if len(filt) >= 2 * window_weeks + 8:
        return filt
    return panel.copy()


def detect_breakpoints(panel: pd.DataFrame, args: argparse.Namespace) -> tuple[list[pd.Timestamp], dict[str, Any]]:
    feature_cols = [
        "n_events",
        "n_threads_active",
        "n_actors",
        "avg_events_per_thread",
        "forward_rate",
        "cc_rate",
        "mean_fanout",
        "specialist_consult_rate",
        "escalation_rate",
        "assignment_rate",
        "approval_rate",
        "message_reply_rate",
        "median_thread_duration_h",
        "mean_handoff_count",
        "reopen_rate",
        "actor_out_gini",
        "target_in_gini",
        "actor_out_centralization",
        "target_in_centralization",
        "pair_reciprocity",
    ]
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Weekly panel missing required features: {missing}")

    detect_panel = choose_detection_panel(panel, args.coverage_threshold, args.window_weeks)
    X = detect_panel[feature_cols].to_numpy(dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std <= 1e-9, 1.0, std)
    X = (X - mean) / std

    weeks = detect_panel["week"].to_numpy()
    n = len(detect_panel)
    w = int(args.window_weeks)
    if n < 2 * w + 8:
        raise RuntimeError(
            f"Insufficient weekly points ({n}) for change detection with window {w}."
        )

    coverage_weight = detect_panel["coverage_mean"].to_numpy(dtype=float)
    coverage_weight = np.where(np.isfinite(coverage_weight), coverage_weight, 1.0)
    coverage_weight = np.clip(coverage_weight, 0.05, 1.0)

    score_mean = np.full(n, np.nan)
    score_cov = np.full(n, np.nan)
    for idx in range(w, n - w):
        pre = X[idx - w : idx]
        post = X[idx : idx + w]
        d_mean = np.linalg.norm(np.nanmean(post, axis=0) - np.nanmean(pre, axis=0))
        score_mean[idx] = d_mean * math.sqrt(w) * coverage_weight[idx]

        cov_pre = np.cov(pre, rowvar=False)
        cov_post = np.cov(post, rowvar=False)
        d_cov = np.linalg.norm(cov_post - cov_pre, ord="fro")
        score_cov[idx] = d_cov * coverage_weight[idx]

    z_mean = robust_z(np.nan_to_num(score_mean, nan=0.0))
    z_cov = robust_z(np.nan_to_num(score_cov, nan=0.0))
    combined = 0.6 * z_mean + 0.4 * z_cov

    min_gap = int(args.min_break_gap_weeks)
    peaks_mean, _ = find_peaks(
        np.nan_to_num(z_mean, nan=0.0),
        height=float(args.z_threshold),
        distance=min_gap,
    )
    peaks_cov, _ = find_peaks(
        np.nan_to_num(z_cov, nan=0.0),
        height=float(args.z_threshold),
        distance=min_gap,
    )
    peaks_combined, _ = find_peaks(
        np.nan_to_num(combined, nan=0.0),
        height=max(1.5, float(args.z_threshold) - 0.8),
        distance=min_gap,
    )

    tol = int(args.consensus_tolerance_weeks)
    consensus: list[dict[str, Any]] = []
    for i in peaks_mean:
        if len(peaks_cov) == 0:
            continue
        nearest = int(peaks_cov[np.argmin(np.abs(peaks_cov - i))])
        if abs(nearest - i) <= tol:
            idx = int(round((i + nearest) / 2))
            consensus.append(
                {
                    "idx": idx,
                    "week": str(pd.Timestamp(weeks[idx]).date()),
                    "mean_shift_z": float(z_mean[idx]),
                    "cov_shift_z": float(z_cov[idx]),
                    "combined_z": float(combined[idx]),
                }
            )

    candidate_strength: dict[int, float] = {}
    for d in consensus:
        candidate_strength[d["idx"]] = max(candidate_strength.get(d["idx"], -1e9), float(d["combined_z"]))
    for idx in peaks_combined:
        candidate_strength[idx] = max(candidate_strength.get(int(idx), -1e9), float(combined[int(idx)]))
    for idx in peaks_mean:
        candidate_strength[int(idx)] = max(candidate_strength.get(int(idx), -1e9), float(z_mean[int(idx)]))
    for idx in peaks_cov:
        candidate_strength[int(idx)] = max(candidate_strength.get(int(idx), -1e9), float(z_cov[int(idx)]))

    ordered = sorted(candidate_strength.items(), key=lambda kv: kv[1], reverse=True)
    chosen_idx: list[int] = []
    for idx, _ in ordered:
        if idx < w or idx >= n - w:
            continue
        if any(abs(idx - existing) < min_gap for existing in chosen_idx):
            continue
        chosen_idx.append(int(idx))
        if len(chosen_idx) >= int(args.max_breaks):
            break

    if len(chosen_idx) < int(args.min_breaks):
        ordered_fallback = sorted(
            [(int(i), float(combined[int(i)])) for i in range(w, n - w)],
            key=lambda kv: kv[1],
            reverse=True,
        )
        for idx, _ in ordered_fallback:
            if any(abs(idx - existing) < min_gap for existing in chosen_idx):
                continue
            chosen_idx.append(idx)
            if len(chosen_idx) >= int(args.min_breaks):
                break

    chosen_idx = sorted(set(chosen_idx))
    break_weeks = [pd.Timestamp(weeks[i]) for i in chosen_idx]

    details = {
        "params": {
            "window_weeks": int(args.window_weeks),
            "min_break_gap_weeks": int(args.min_break_gap_weeks),
            "consensus_tolerance_weeks": int(args.consensus_tolerance_weeks),
            "z_threshold": float(args.z_threshold),
            "coverage_threshold": float(args.coverage_threshold),
        },
        "panel_points_used": int(n),
        "coverage_points_filtered_out": int(len(panel) - len(detect_panel)),
        "methods": {
            "mean_shift": [
                {
                    "week": str(pd.Timestamp(weeks[i]).date()),
                    "z": float(z_mean[i]),
                }
                for i in peaks_mean
            ],
            "cov_shift": [
                {
                    "week": str(pd.Timestamp(weeks[i]).date()),
                    "z": float(z_cov[i]),
                }
                for i in peaks_cov
            ],
            "combined": [
                {
                    "week": str(pd.Timestamp(weeks[i]).date()),
                    "z": float(combined[i]),
                }
                for i in peaks_combined
            ],
        },
        "consensus_candidates": consensus,
        "selected_breakpoints": [
            {
                "week": str(pd.Timestamp(weeks[i]).date()),
                "combined_z": float(combined[i]),
                "mean_shift_z": float(z_mean[i]),
                "cov_shift_z": float(z_cov[i]),
            }
            for i in chosen_idx
        ],
    }
    return break_weeks, details


def assign_regime_id(weeks: pd.Series, break_weeks: list[pd.Timestamp]) -> pd.Series:
    breaks = np.array(sorted(set(pd.to_datetime(break_weeks))), dtype="datetime64[ns]")
    vals = weeks.to_numpy(dtype="datetime64[ns]")
    regime_ids = np.searchsorted(breaks, vals, side="right")
    return pd.Series(regime_ids, index=weeks.index, dtype=int)


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def cluster_base_name(row: pd.Series, quant: dict[str, float]) -> str:
    if row["approvals_per_event"] >= quant["approvals_per_event_q75"]:
        return "approval_governance"
    if (
        row["specialist_consults"] >= quant["specialist_consults_q75"]
        and row["escalations_per_event"] >= quant["escalations_per_event_q50"]
    ):
        return "specialist_escalation"
    if (
        row["n_events"] >= quant["n_events_q75"]
        and row["fanout_mean"] >= quant["fanout_mean_q75"]
    ):
        return "high_volume_broadcast"
    if (
        row["handoff_count"] >= quant["handoff_count_q75"]
        and row["duration_hours"] >= quant["duration_hours_q75"]
    ):
        return "cross_team_program"
    if (
        row["n_events"] <= quant["n_events_q25"]
        and row["duration_hours"] <= quant["duration_hours_q25"]
    ):
        return "quick_resolution"
    return "ongoing_operations"


def assign_task_clusters(thread_features: pd.DataFrame, k: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "n_events",
        "n_participants",
        "duration_hours",
        "n_escalations",
        "n_assignments",
        "n_approvals",
        "fanout_mean",
        "specialist_consults",
        "handoff_count",
        "after_hours_share",
        "subject_entropy",
    ]
    X = thread_features[feature_cols].fillna(0.0).to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma <= 1e-9, 1.0, sigma)
    Xz = (X - mu) / sigma

    n = len(thread_features)
    if n < 3:
        thread_features = thread_features.copy()
        thread_features["task_cluster_id"] = 0
        thread_features["task_type"] = "ongoing_operational"
        profiles = pd.DataFrame(
            [{"task_cluster_id": 0, "task_type": "ongoing_operational"}]
        )
        return thread_features, profiles

    k_final = max(3, min(int(k), max(3, n // 500)))
    np.random.seed(seed)
    centroids, labels = kmeans2(Xz, k_final, minit="points", iter=30)
    _ = centroids  # centroids only used implicitly through labels.

    out = thread_features.copy()
    out["task_cluster_id"] = labels.astype(int)

    profile = (
        out.groupby("task_cluster_id")
        .agg(
            threads=("thread_task_id", "count"),
            n_events=("n_events", "mean"),
            n_participants=("n_participants", "mean"),
            duration_hours=("duration_hours", "mean"),
            n_escalations=("n_escalations", "mean"),
            n_assignments=("n_assignments", "mean"),
            n_approvals=("n_approvals", "mean"),
            fanout_mean=("fanout_mean", "mean"),
            specialist_consults=("specialist_consults", "mean"),
            handoff_count=("handoff_count", "mean"),
            after_hours_share=("after_hours_share", "mean"),
            escalations_per_event=("escalations_per_event", "mean"),
            approvals_per_event=("approvals_per_event", "mean"),
        )
        .reset_index()
    )
    quant = {
        "approvals_per_event_q75": float(profile["approvals_per_event"].quantile(0.75)),
        "specialist_consults_q75": float(profile["specialist_consults"].quantile(0.75)),
        "escalations_per_event_q50": float(profile["escalations_per_event"].quantile(0.50)),
        "n_events_q75": float(profile["n_events"].quantile(0.75)),
        "fanout_mean_q75": float(profile["fanout_mean"].quantile(0.75)),
        "handoff_count_q75": float(profile["handoff_count"].quantile(0.75)),
        "duration_hours_q75": float(profile["duration_hours"].quantile(0.75)),
        "n_events_q25": float(profile["n_events"].quantile(0.25)),
        "duration_hours_q25": float(profile["duration_hours"].quantile(0.25)),
    }
    profile["task_type_base"] = profile.apply(cluster_base_name, axis=1, quant=quant)
    profile["task_type"] = [
        f"{base}_c{int(cid)}"
        for base, cid in zip(profile["task_type_base"], profile["task_cluster_id"], strict=False)
    ]
    label_map = dict(
        zip(profile["task_cluster_id"].astype(int), profile["task_type"], strict=False)
    )
    out["task_type"] = out["task_cluster_id"].map(label_map).fillna("ongoing_operational")
    return out, profile


def tokenize_subject(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return toks


def build_token_counters(events: pd.DataFrame) -> dict[int, Counter[str]]:
    counters: dict[int, Counter[str]] = defaultdict(Counter)
    for row in events[["regime_id", "norm_subject"]].itertuples(index=False):
        rid = int(row.regime_id)
        for tok in tokenize_subject(str(row.norm_subject)):
            counters[rid][tok] += 1
    return counters


def top_log_odds(post: Counter[str], pre: Counter[str], top_n: int = 12) -> list[tuple[str, float]]:
    vocab = set(post) | set(pre)
    v = max(1, len(vocab))
    n_post = max(1, sum(post.values()))
    n_pre = max(1, sum(pre.values()))
    scored: list[tuple[str, float]] = []
    for tok in vocab:
        p_post = (post.get(tok, 0) + 1.0) / (n_post + v)
        p_pre = (pre.get(tok, 0) + 1.0) / (n_pre + v)
        scored.append((tok, math.log(p_post) - math.log(p_pre)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def infer_shift_label(delta_series: pd.Series) -> str:
    up = set(delta_series.sort_values(ascending=False).head(4).index.tolist())
    if {"escalation_rate", "specialist_consult_rate"} & up:
        return "escalation-specialist shock"
    if {"approval_rate", "actor_out_centralization"} & up:
        return "centralized approval bottleneck"
    if {"n_events", "n_threads_active", "mean_fanout"} & up:
        return "volume surge and coordination expansion"
    if {"reopen_rate", "mean_handoff_count"} & up:
        return "rework and cross-team handoff spike"
    return "mixed operational shift"


def build_dependency_graph(
    events: pd.DataFrame,
    thread_features: pd.DataFrame,
    window_hours: int,
    min_weight: int,
    max_edges: int,
) -> pd.DataFrame:
    events_small = events[["actor_id", "thread_task_id", "timestamp"]].copy()
    events_small = events_small.sort_values(["actor_id", "timestamp"])
    window_ns = int(window_hours * 3_600_000_000_000)

    pair_weight: defaultdict[tuple[str, str], int] = defaultdict(int)
    for _actor, grp in events_small.groupby("actor_id", sort=False):
        dq: deque[tuple[int, str]] = deque()
        active_counts: defaultdict[str, int] = defaultdict(int)
        for row in grp.itertuples(index=False):
            ts_ns = int(row.timestamp.value)
            tid = str(row.thread_task_id)

            while dq and ts_ns - dq[0][0] > window_ns:
                old_ts, old_tid = dq.popleft()
                _ = old_ts
                active_counts[old_tid] -= 1
                if active_counts[old_tid] <= 0:
                    active_counts.pop(old_tid, None)

            if active_counts:
                for other_tid in active_counts.keys():
                    if other_tid == tid:
                        continue
                    if tid < other_tid:
                        key = (tid, other_tid)
                    else:
                        key = (other_tid, tid)
                    pair_weight[key] += 1

            dq.append((ts_ns, tid))
            active_counts[tid] += 1

    rows = [
        {"thread_a": a, "thread_b": b, "temporal_coupling_count": w}
        for (a, b), w in pair_weight.items()
        if w >= min_weight
    ]
    dep = pd.DataFrame(rows)
    if dep.empty:
        return dep

    dep = dep.sort_values("temporal_coupling_count", ascending=False).head(max_edges).copy()
    t_regime = thread_features.set_index("thread_task_id")["regime_id"].to_dict()
    t_actors = (
        events.groupby("thread_task_id")["actor_id"]
        .agg(lambda s: set(s))
        .to_dict()
    )
    dep["regime_a"] = dep["thread_a"].map(t_regime)
    dep["regime_b"] = dep["thread_b"].map(t_regime)
    dep["same_regime"] = (dep["regime_a"] == dep["regime_b"]).astype(int)
    dep["shared_actor_count"] = [
        len(t_actors.get(a, set()) & t_actors.get(b, set()))
        for a, b in zip(dep["thread_a"], dep["thread_b"], strict=False)
    ]
    q95 = float(dep["temporal_coupling_count"].quantile(0.95))

    def dep_type(row: pd.Series) -> str:
        if row["same_regime"] == 0:
            return "cross_regime_carryover"
        if row["shared_actor_count"] >= 2:
            return "shared_staff_dependency"
        if row["temporal_coupling_count"] >= q95:
            return "burst_coupling"
        return "weak_coupling"

    dep["dependency_type"] = dep.apply(dep_type, axis=1)
    return dep.reset_index(drop=True)


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "null"
        return f"{float(value):.6g}"
    return json.dumps(str(value), ensure_ascii=True)


def to_yaml_lines(obj: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(to_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
        return lines
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(to_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(obj)}"]


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    args.regimes_dir.mkdir(parents=True, exist_ok=True)
    args.tasks_dir.mkdir(parents=True, exist_ok=True)
    args.scenarios_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.events_metadata)
    daily_cov, weekly_cov = coverage_series(events, args.header_cache)

    thread_features = build_thread_features(events)
    panel = build_weekly_panel(events, thread_features, weekly_cov)

    break_weeks, break_details = detect_breakpoints(panel, args)

    panel["regime_id"] = assign_regime_id(panel["week"], break_weeks)
    events["regime_id"] = assign_regime_id(events["week"], break_weeks)
    thread_features["regime_id"] = assign_regime_id(thread_features["start_week"], break_weeks)

    thread_features, task_profiles = assign_task_clusters(
        thread_features=thread_features,
        k=int(args.task_clusters),
        seed=int(args.seed),
    )

    # Complexity score for exemplar selection.
    complexity_cols = [
        "n_events",
        "n_participants",
        "n_escalations",
        "handoff_count",
        "duration_hours",
        "fanout_mean",
    ]
    complexity = pd.DataFrame(index=thread_features.index)
    for col in complexity_cols:
        val = thread_features[col]
        if col == "duration_hours":
            val = np.log1p(val)
        complexity[col] = zscore(val)
    thread_features["complexity_score"] = complexity.mean(axis=1)

    # Regime signatures.
    panel_numeric_cols = [c for c in panel.columns if c not in {"week", "regime_id"}]
    signatures = (
        panel.groupby("regime_id")
        .agg(
            start_week=("week", "min"),
            end_week=("week", "max"),
            weeks=("week", "count"),
            **{col: (col, "mean") for col in panel_numeric_cols},
        )
        .reset_index()
        .sort_values("regime_id")
    )

    # Task distributions by regime.
    task_dist = (
        thread_features.groupby(["regime_id", "task_type"])
        .size()
        .rename("threads")
        .reset_index()
    )
    total_per_reg = task_dist.groupby("regime_id")["threads"].sum().rename("reg_total")
    task_dist = task_dist.join(total_per_reg, on="regime_id")
    task_dist["share"] = np.where(task_dist["reg_total"] > 0, task_dist["threads"] / task_dist["reg_total"], 0.0)
    task_dist = task_dist.sort_values(["regime_id", "share"], ascending=[True, False])

    # Dependency graph + stats.
    dep = build_dependency_graph(
        events=events,
        thread_features=thread_features,
        window_hours=int(args.dependency_window_hours),
        min_weight=int(args.dependency_min_weight),
        max_edges=int(args.dependency_max_edges),
    )

    dep_stats_rows: list[dict[str, Any]] = []
    if not dep.empty:
        for regime_id, reg_df in dep[dep["same_regime"] == 1].groupby("regime_a"):
            nodes = set(reg_df["thread_a"]).union(set(reg_df["thread_b"]))
            n_nodes = len(nodes)
            n_edges = len(reg_df)
            density = safe_div(2.0 * n_edges, n_nodes * max(1, n_nodes - 1))
            dep_stats_rows.append(
                {
                    "regime_id": int(regime_id),
                    "dependency_edges": int(n_edges),
                    "dependency_nodes": int(n_nodes),
                    "dependency_density": float(density),
                    "mean_temporal_coupling": float(reg_df["temporal_coupling_count"].mean()),
                    "p90_temporal_coupling": float(reg_df["temporal_coupling_count"].quantile(0.90)),
                    "mean_shared_actor_count": float(reg_df["shared_actor_count"].mean()),
                }
            )
    dep_stats = pd.DataFrame(dep_stats_rows)

    # Regime explanations.
    counters = build_token_counters(events)
    excluded_for_explanations = {
        "observed_count",
        "header_count",
        "coverage_mean",
        "coverage_min",
        "coverage_ratio",
    }
    explanation_cols = [c for c in panel_numeric_cols if c not in excluded_for_explanations]
    regime_means = panel.groupby("regime_id")[explanation_cols].mean()
    actor_share = (
        events.groupby(["regime_id", "actor_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    actor_totals = actor_share.groupby("regime_id")["n"].sum().rename("n_total")
    actor_share = actor_share.join(actor_totals, on="regime_id")
    actor_share["share"] = np.where(actor_share["n_total"] > 0, actor_share["n"] / actor_share["n_total"], 0.0)

    break_sorted = sorted(set(break_weeks))
    explanation_lines = [
        "# Enron Regime Explanations",
        "",
        "This report is inferred from internal data signatures (volume, coordination, graph proxies, task mix).",
        "",
    ]

    shock_rows: list[dict[str, Any]] = []
    for idx, wk in enumerate(break_sorted):
        pre_id = idx
        post_id = idx + 1
        if pre_id not in regime_means.index or post_id not in regime_means.index:
            continue
        delta = regime_means.loc[post_id] - regime_means.loc[pre_id]
        top_up = delta.sort_values(ascending=False).head(5)
        top_down = delta.sort_values().head(5)
        label = infer_shift_label(delta)

        pre_actor = actor_share[actor_share["regime_id"] == pre_id].set_index("actor_id")["share"]
        post_actor = actor_share[actor_share["regime_id"] == post_id].set_index("actor_id")["share"]
        actor_delta = post_actor.sub(pre_actor, fill_value=0.0).sort_values(ascending=False)
        top_actor_up = actor_delta.head(5)

        post_terms = top_log_odds(counters.get(post_id, Counter()), counters.get(pre_id, Counter()), top_n=10)

        exemplar = thread_features[thread_features["regime_id"] == post_id].sort_values(
            "complexity_score", ascending=False
        ).head(3)

        explanation_lines.append(f"## Breakpoint {idx + 1}: {pd.Timestamp(wk).date()}")
        explanation_lines.append(f"- Inferred shift: **{label}**")
        explanation_lines.append("- Top feature increases:")
        for name, val in top_up.items():
            explanation_lines.append(f"  - `{name}`: {val:+.4f}")
        explanation_lines.append("- Top feature decreases:")
        for name, val in top_down.items():
            explanation_lines.append(f"  - `{name}`: {val:+.4f}")
        explanation_lines.append("- Actors gaining share:")
        for actor, val in top_actor_up.items():
            explanation_lines.append(f"  - `{actor}`: {val:+.4f}")
        explanation_lines.append("- Subject terms rising in prevalence:")
        for tok, val in post_terms:
            explanation_lines.append(f"  - `{tok}`: {val:+.3f}")
        explanation_lines.append("- Exemplar complex threads from new regime:")
        for row in exemplar.itertuples(index=False):
            explanation_lines.append(
                "  - "
                f"{row.thread_task_id} | task={row.task_type} | events={int(row.n_events)} | "
                f"participants={int(row.n_participants)} | escalations={int(row.n_escalations)} | "
                f"duration_h={float(row.duration_hours):.1f} | subject={str(row.anchor_subject)[:90]}"
            )
        explanation_lines.append("")

        shock_rows.append(
            {
                "break_week": str(pd.Timestamp(wk).date()),
                "from_regime": int(pre_id),
                "to_regime": int(post_id),
                "label": label,
                "top_feature_up": [str(x) for x in top_up.index.tolist()[:3]],
                "top_feature_down": [str(x) for x in top_down.index.tolist()[:3]],
            }
        )

    # Optional external-date sanity check.
    macro_dates = [
        ("western_crisis_period_start_proxy", pd.Timestamp("2000-06-01")),
        ("skilling_resignation", pd.Timestamp("2001-08-14")),
        ("oct_2001_disclosure_shock", pd.Timestamp("2001-10-16")),
        ("bankruptcy_filing", pd.Timestamp("2001-12-02")),
        ("ferc_investigation_start", pd.Timestamp("2002-02-13")),
    ]
    if break_sorted:
        explanation_lines.append("## External-Date Sanity Check (Post Hoc)")
        for tag, dt in macro_dates:
            nearest = min(break_sorted, key=lambda b: abs((pd.Timestamp(b) - dt).days))
            delta_days = abs((pd.Timestamp(nearest) - dt).days)
            explanation_lines.append(
                f"- `{tag}` ({dt.date()}): nearest breakpoint `{pd.Timestamp(nearest).date()}` ({delta_days} days away)"
            )
        explanation_lines.append("")

    # Scenario YAML structure.
    regime_rows = []
    dep_stats_map = (
        dep_stats.set_index("regime_id").to_dict(orient="index")
        if not dep_stats.empty
        else {}
    )
    for row in signatures.itertuples(index=False):
        rid = int(row.regime_id)
        mix = task_dist[task_dist["regime_id"] == rid][["task_type", "share"]]
        mix_dict = {str(r.task_type): float(r.share) for r in mix.itertuples(index=False)}
        dep_row = dep_stats_map.get(rid, {})
        regime_rows.append(
            {
                "regime_id": rid,
                "start_week": str(pd.Timestamp(row.start_week).date()),
                "end_week": str(pd.Timestamp(row.end_week).date()),
                "weeks": int(row.weeks),
                "avg_events_per_week": float(row.n_events),
                "avg_threads_active_per_week": float(row.n_threads_active),
                "avg_actors_per_week": float(row.n_actors),
                "escalation_rate": float(row.escalation_rate),
                "specialist_consult_rate": float(row.specialist_consult_rate),
                "mean_fanout": float(row.mean_fanout),
                "approval_rate": float(row.approval_rate),
                "task_type_distribution": mix_dict,
                "dependency_stats": dep_row,
            }
        )

    global_dep_stats = {}
    if not dep.empty:
        nodes = set(dep["thread_a"]).union(set(dep["thread_b"]))
        n_nodes = len(nodes)
        n_edges = len(dep)
        global_dep_stats = {
            "edges": int(n_edges),
            "nodes": int(n_nodes),
            "density": safe_div(2.0 * n_edges, n_nodes * max(1, n_nodes - 1)),
            "mean_temporal_coupling": float(dep["temporal_coupling_count"].mean()),
            "p90_temporal_coupling": float(dep["temporal_coupling_count"].quantile(0.90)),
            "dependency_type_mix": (
                dep["dependency_type"].value_counts(normalize=True).round(6).to_dict()
            ),
        }

    scenario_obj = {
        "meta": {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "source": "enron_rosetta",
            "coverage_note": "Calibrate and replay with the same task/dependency/shock structure; vary only coordination technology.",
        },
        "inputs": {
            "events_rows": int(len(events)),
            "threads_rows": int(len(thread_features)),
            "weekly_points": int(len(panel)),
            "coverage_threshold": float(args.coverage_threshold),
        },
        "breakpoints": [str(pd.Timestamp(w).date()) for w in break_sorted],
        "shock_schedule": shock_rows,
        "regimes": regime_rows,
        "task_types_global": (
            thread_features["task_type"].value_counts(normalize=True).round(6).to_dict()
        ),
        "dependency_graph_global": global_dep_stats,
        "replay_constraints": {
            "freeze_exogenous": [
                "task_arrivals_by_regime",
                "task_type_distribution_by_regime",
                "dependency_graph_statistics",
                "shock_schedule",
            ],
            "vary_only": [
                "routing_policy",
                "memory_policy",
                "delegation_policy",
                "response_latency_model",
            ],
        },
    }

    # Persist artifacts.
    panel.to_parquet(args.regimes_dir / "enron_weekly_panel.parquet", index=False)
    with (args.regimes_dir / "enron_breakpoints.json").open("w", encoding="utf-8") as f:
        json.dump(break_details, f, indent=2)
        f.write("\n")
    signatures.to_parquet(args.regimes_dir / "enron_regime_signatures.parquet", index=False)
    (args.regimes_dir / "enron_regime_explanations.md").write_text(
        "\n".join(explanation_lines) + "\n",
        encoding="utf-8",
    )
    daily_cov.to_parquet(args.regimes_dir / "enron_observability_daily.parquet", index=False)

    thread_features.to_parquet(args.tasks_dir / "enron_thread_features.parquet", index=False)
    task_profiles.to_parquet(args.tasks_dir / "enron_task_clusters.parquet", index=False)
    task_dist.to_parquet(args.tasks_dir / "enron_task_distribution_by_regime.parquet", index=False)
    dep.to_parquet(args.tasks_dir / "enron_task_dependency_graph.parquet", index=False)

    yaml_lines = to_yaml_lines(scenario_obj)
    (args.scenarios_dir / "enron_like_scenario.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {(args.regimes_dir / 'enron_weekly_panel.parquet').resolve()}")
    print(f"Wrote: {(args.regimes_dir / 'enron_breakpoints.json').resolve()}")
    print(f"Wrote: {(args.regimes_dir / 'enron_regime_signatures.parquet').resolve()}")
    print(f"Wrote: {(args.regimes_dir / 'enron_regime_explanations.md').resolve()}")
    print(f"Wrote: {(args.tasks_dir / 'enron_thread_features.parquet').resolve()}")
    print(f"Wrote: {(args.tasks_dir / 'enron_task_clusters.parquet').resolve()}")
    print(f"Wrote: {(args.tasks_dir / 'enron_task_dependency_graph.parquet').resolve()}")
    print(f"Wrote: {(args.scenarios_dir / 'enron_like_scenario.yaml').resolve()}")
    print(f"Detected breakpoints: {len(break_sorted)}")
    if break_sorted:
        print("Breakpoint weeks:", ", ".join(str(pd.Timestamp(w).date()) for w in break_sorted))


if __name__ == "__main__":
    main()
