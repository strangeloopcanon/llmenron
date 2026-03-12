#!/usr/bin/env python3
"""Compare two scratchpad frontier runs, emphasizing memory-probe outcomes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

RE_PROJECT_CODE = re.compile(r"\b([A-Z][0-9]{3})\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-run-dir", type=Path, required=True)
    p.add_argument("--candidate-run-dir", type=Path, required=True)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/scratchpad_frontier/scratchpad_policy_compare/policy_compare"),
    )
    return p.parse_args()


def load_run(run_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    msg_path = run_dir / "message_log.csv"
    n_path = run_dir / "n_summary.csv"
    if not msg_path.exists() or not n_path.exists():
        raise FileNotFoundError(f"Missing message_log.csv or n_summary.csv in {run_dir}")
    msg = pd.read_csv(msg_path)
    nsum = pd.read_csv(n_path)
    msg["run_label"] = label
    nsum["run_label"] = label
    return msg, nsum


def extract_project_code(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = RE_PROJECT_CODE.search(text)
        if match:
            return match.group(1)
    return ""


def ensure_binding_columns(msg: pd.DataFrame) -> pd.DataFrame:
    out = msg.copy()
    if "pred_target_project_code" not in out.columns:
        out["pred_target_project_code"] = out.apply(
            lambda row: extract_project_code(
                row.get("pred_facts_used", ""),
                row.get("pred_action_summary", ""),
                row.get("pred_draft_reply", ""),
            ),
            axis=1,
        )
    out["pred_target_project_code"] = out["pred_target_project_code"].fillna("").astype(str).str.strip()

    if "pred_binding_decision" not in out.columns:
        out["pred_binding_decision"] = out["pred_target_project_code"].map(lambda x: "bound" if x else "")
    out["pred_binding_decision"] = out["pred_binding_decision"].fillna("").astype(str).str.strip().str.lower()

    if "pred_binding_source" not in out.columns:
        out["pred_binding_source"] = ""
    out["pred_binding_source"] = out["pred_binding_source"].fillna("").astype(str).str.strip().str.lower()

    if "gold_project_code" not in out.columns:
        gold_from_probe = pd.Series(
            [
                str(row.get("gold_required_value", "")).strip()
                if str(row.get("gold_required_key", "")).strip() == "project_code"
                else ""
                for _, row in out.iterrows()
            ],
            index=out.index,
            dtype=object,
        )
        out["gold_project_code"] = gold_from_probe
    out["gold_project_code"] = out["gold_project_code"].fillna("").astype(str).str.strip()

    if "binding_attempt" not in out.columns:
        out["binding_attempt"] = (out["pred_target_project_code"] != "").astype(float)
    else:
        out["binding_attempt"] = out["binding_attempt"].fillna(0).astype(float)

    if "target_match" not in out.columns:
        out["target_match"] = (
            (out["pred_target_project_code"] != "")
            & (out["gold_project_code"] != "")
            & (out["pred_target_project_code"] == out["gold_project_code"])
        ).astype(float)
    else:
        out["target_match"] = out["target_match"].fillna(0).astype(float)

    if "safe_clarification" not in out.columns:
        out["safe_clarification"] = (
            (out["pred_binding_decision"] == "clarify")
            & (out["pred_target_project_code"] == "")
        ).astype(float)
    else:
        out["safe_clarification"] = out["safe_clarification"].fillna(0).astype(float)

    if "unsafe_wrong_target" not in out.columns:
        out["unsafe_wrong_target"] = (
            (out["pred_target_project_code"] != "")
            & (out["gold_project_code"] != "")
            & (out["pred_target_project_code"] != out["gold_project_code"])
        ).astype(float)
    else:
        out["unsafe_wrong_target"] = out["unsafe_wrong_target"].fillna(0).astype(float)

    return out


def memory_probe_summary(msg: pd.DataFrame) -> pd.DataFrame:
    mem = ensure_binding_columns(msg[msg["needs_memory"] == 1].copy())
    if mem.empty:
        return pd.DataFrame(
            columns=[
                "run_label",
                "n_threads",
                "probe_count",
                "memory_recall",
                "probe_target_match",
                "probe_safe_clarification",
                "probe_unsafe_wrong_target",
                "probe_binding_attempt_rate",
                "probe_binding_precision",
                "mean_latency_min",
            ]
        )
    mem["probe_binding_precision"] = mem.apply(
        lambda row: float(row["target_match"]) if float(row["binding_attempt"]) > 0 else float("nan"),
        axis=1,
    )
    out = (
        mem.groupby(["run_label", "n_threads"], as_index=False)
        .agg(
            probe_count=("message_id", "count"),
            memory_recall=("fact_recall", "mean"),
            probe_target_match=("target_match", "mean"),
            probe_safe_clarification=("safe_clarification", "mean"),
            probe_unsafe_wrong_target=("unsafe_wrong_target", "mean"),
            probe_binding_attempt_rate=("binding_attempt", "mean"),
            probe_binding_precision=("probe_binding_precision", "mean"),
            mean_latency_min=("latency_min", "mean"),
        )
        .sort_values(["n_threads", "run_label"])
    )
    return out


def main() -> None:
    args = parse_args()
    base_msg, base_n = load_run(args.baseline_run_dir, "baseline")
    cand_msg, cand_n = load_run(args.candidate_run_dir, "candidate")

    base_msg = ensure_binding_columns(base_msg)
    cand_msg = ensure_binding_columns(cand_msg)
    msg = pd.concat([base_msg, cand_msg], ignore_index=True)
    nsum = pd.concat([base_n, cand_n], ignore_index=True)
    mem = memory_probe_summary(msg)

    rows: list[dict[str, object]] = []
    for n_threads in sorted(nsum["n_threads"].unique().tolist()):
        base_row = nsum[(nsum["run_label"] == "baseline") & (nsum["n_threads"] == n_threads)]
        cand_row = nsum[(nsum["run_label"] == "candidate") & (nsum["n_threads"] == n_threads)]
        if base_row.empty or cand_row.empty:
            continue
        b = base_row.iloc[0]
        c = cand_row.iloc[0]

        base_mem = mem[(mem["run_label"] == "baseline") & (mem["n_threads"] == n_threads)]
        cand_mem = mem[(mem["run_label"] == "candidate") & (mem["n_threads"] == n_threads)]
        base_mem_recall = float(base_mem["memory_recall"].iloc[0]) if not base_mem.empty else 1.0
        cand_mem_recall = float(cand_mem["memory_recall"].iloc[0]) if not cand_mem.empty else 1.0
        base_target_match = float(base_mem["probe_target_match"].iloc[0]) if not base_mem.empty else 0.0
        cand_target_match = float(cand_mem["probe_target_match"].iloc[0]) if not cand_mem.empty else 0.0
        base_safe = float(base_mem["probe_safe_clarification"].iloc[0]) if not base_mem.empty else 0.0
        cand_safe = float(cand_mem["probe_safe_clarification"].iloc[0]) if not cand_mem.empty else 0.0
        base_wrong = float(base_mem["probe_unsafe_wrong_target"].iloc[0]) if not base_mem.empty else 0.0
        cand_wrong = float(cand_mem["probe_unsafe_wrong_target"].iloc[0]) if not cand_mem.empty else 0.0
        base_precision = float(base_mem["probe_binding_precision"].iloc[0]) if not base_mem.empty else 1.0
        cand_precision = float(cand_mem["probe_binding_precision"].iloc[0]) if not cand_mem.empty else 1.0
        probe_count = int(base_mem["probe_count"].iloc[0]) if not base_mem.empty else 0

        rows.append(
            {
                "n_threads": int(n_threads),
                "probe_count": probe_count,
                "baseline_mean_quality": float(b["mean_quality"]),
                "candidate_mean_quality": float(c["mean_quality"]),
                "baseline_memory_recall": base_mem_recall,
                "candidate_memory_recall": cand_mem_recall,
                "memory_recall_delta": cand_mem_recall - base_mem_recall,
                "baseline_probe_target_match": base_target_match,
                "candidate_probe_target_match": cand_target_match,
                "probe_target_match_delta": cand_target_match - base_target_match,
                "baseline_probe_safe_clarification": base_safe,
                "candidate_probe_safe_clarification": cand_safe,
                "probe_safe_clarification_delta": cand_safe - base_safe,
                "baseline_probe_unsafe_wrong_target": base_wrong,
                "candidate_probe_unsafe_wrong_target": cand_wrong,
                "probe_unsafe_wrong_target_delta": cand_wrong - base_wrong,
                "baseline_probe_binding_precision": base_precision,
                "candidate_probe_binding_precision": cand_precision,
                "probe_binding_precision_delta": cand_precision - base_precision,
                "baseline_p0_sla": float(b["p0_sla"]),
                "candidate_p0_sla": float(c["p0_sla"]),
                "baseline_input_tokens": float(b.get("input_tokens", 0.0)),
                "candidate_input_tokens": float(c.get("input_tokens", 0.0)),
                "baseline_output_tokens": float(b.get("output_tokens", 0.0)),
                "candidate_output_tokens": float(c.get("output_tokens", 0.0)),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("n_threads")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_prefix.with_suffix(".csv"), index=False)

    lines = [
        "# Scratchpad Memory Policy Comparison",
        "",
        f"- Baseline: `{args.baseline_run_dir}`",
        f"- Candidate: `{args.candidate_run_dir}`",
        "",
    ]
    for row in out_df.itertuples(index=False):
        lines.append(
            f"- N={row.n_threads}: probe_recall {row.baseline_memory_recall:.3f} -> {row.candidate_memory_recall:.3f} "
            f"(delta {row.memory_recall_delta:+.3f}); target_match {row.baseline_probe_target_match:.3f} -> {row.candidate_probe_target_match:.3f}; "
            f"safe_clarify {row.baseline_probe_safe_clarification:.3f} -> {row.candidate_probe_safe_clarification:.3f}; "
            f"wrong_target {row.baseline_probe_unsafe_wrong_target:.3f} -> {row.candidate_probe_unsafe_wrong_target:.3f}; "
            f"probe_precision {row.baseline_probe_binding_precision:.3f} -> {row.candidate_probe_binding_precision:.3f}; "
            f"quality {row.baseline_mean_quality:.3f} -> {row.candidate_mean_quality:.3f}; "
            f"p0_sla {row.baseline_p0_sla:.3f} -> {row.candidate_p0_sla:.3f}; "
            f"input_tokens {int(row.baseline_input_tokens)} -> {int(row.candidate_input_tokens)}"
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
