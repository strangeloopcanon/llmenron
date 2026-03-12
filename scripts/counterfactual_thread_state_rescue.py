#!/usr/bin/env python3
"""Estimate how much a system-maintained thread state would rescue an existing run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PRIORITY_SLA_MIN = {"P0": 5.0, "P1": 20.0, "P2": 120.0}
BANNED_COMMITMENTS = ["i approved", "already approved", "done", "fixed", "completed", "shipped"]
RE_PROJECT_CODE = re.compile(r"\b([A-Z][0-9]{3})\b")
RE_OWNER = re.compile(r"\bOwner ([A-Z][a-z]+ [A-Z][a-z]+)\b")
RE_DUE_DATE = re.compile(r"\bDue (\d{4}-\d{2}-\d{2})\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-name", default="thread_state_rescue")
    return p.parse_args()


def extract_explicit_thread_facts(*, subject: str, body: str) -> dict[str, str]:
    text = f"{subject}\n{body}"
    facts: dict[str, str] = {}
    project_match = RE_PROJECT_CODE.search(text)
    owner_match = RE_OWNER.search(text)
    due_match = RE_DUE_DATE.search(text)
    if project_match:
        facts["project_code"] = project_match.group(1)
    if owner_match:
        facts["owner"] = owner_match.group(1)
    if due_match:
        facts["due_date"] = due_match.group(1)
    return facts


def merge_thread_facts(existing: dict[str, str], new_facts: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for key, value in new_facts.items():
        if value and key not in merged:
            merged[key] = value
    return merged


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            return [value]
    return []


def hallucination_penalty(draft_reply: str, source_text: str) -> float:
    d = str(draft_reply).lower()
    s = str(source_text).lower()
    for phrase in BANNED_COMMITMENTS:
        if phrase in d and phrase not in s:
            return 1.0
    return 0.0


def score_row(row: pd.Series, *, facts_used: list[str]) -> tuple[float, float]:
    priority_acc = float(str(row["pred_priority"]) == str(row["gold_priority"]))
    reply_acc = float(str(row["pred_reply_type"]) == str(row["gold_reply_type"]))
    blob = " ".join(
        [
            str(row.get("pred_action_summary", "")),
            str(row.get("pred_draft_reply", "")),
            " ".join(str(x) for x in facts_used),
        ]
    ).lower()
    if str(row.get("gold_required_key", "none")) == "none":
        fact_recall = 1.0
    else:
        fact_recall = float(str(row.get("gold_required_value", "")).lower() in blob)
    halluc = hallucination_penalty(
        str(row.get("pred_draft_reply", "")),
        f"{row.get('subject', '')}\n{row.get('body', '')}",
    )
    latency_min = float(row.get("latency_min", 0.0))
    on_time = float(latency_min <= PRIORITY_SLA_MIN[str(row["gold_priority"])])
    quality = 0.40 * priority_acc + 0.30 * reply_acc + 0.30 * fact_recall - 0.20 * halluc
    if on_time < 1.0 and str(row["gold_priority"]) in {"P0", "P1"}:
        quality *= 0.2
    quality = max(0.0, min(1.0, quality))
    return fact_recall, quality


def main() -> None:
    args = parse_args()
    msg_path = args.run_dir / "message_log.csv"
    if not msg_path.exists():
        raise FileNotFoundError(f"Missing message log: {msg_path}")

    scenario = pd.read_csv(args.scenario_dir / "messages.csv")
    run = pd.read_csv(msg_path)
    need_cols = ["message_id", "subject", "body"]
    joined = run.merge(
        scenario[need_cols],
        on=["message_id"],
        how="left",
        validate="one_to_one",
    )
    joined = joined.sort_values(["arrival_min", "message_id"]).reset_index(drop=True)

    thread_state_store: dict[str, dict[str, str]] = {}
    rescued_rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        thread_id = str(row["thread_id"])
        state_before = thread_state_store.get(thread_id, {})
        facts_used = parse_json_list(row.get("pred_facts_used", []))
        rescued_facts_used = list(facts_used)

        project_code = str(state_before.get("project_code", "")).strip()
        if project_code and project_code not in rescued_facts_used:
            rescued_facts_used.append(project_code)

        rescued_fact_recall, rescued_quality = score_row(row, facts_used=rescued_facts_used)
        baseline_fact_recall, baseline_quality = score_row(row, facts_used=facts_used)
        rescued_rows.append(
            {
                "message_id": row["message_id"],
                "thread_state_has_project_code": int(bool(project_code)),
                "baseline_fact_recall_check": baseline_fact_recall,
                "rescued_fact_recall": rescued_fact_recall,
                "baseline_quality_check": baseline_quality,
                "rescued_quality": rescued_quality,
                "rescued_pred_facts_used": json.dumps(rescued_facts_used, ensure_ascii=True),
            }
        )

        observed_facts = extract_explicit_thread_facts(subject=str(row["subject"]), body=str(row["body"]))
        if observed_facts:
            thread_state_store[thread_id] = merge_thread_facts(state_before, observed_facts)

    rescued = joined.merge(pd.DataFrame(rescued_rows), on="message_id", how="left", validate="one_to_one")
    out_message_path = args.run_dir / f"{args.output_name}_message_log.csv"
    out_summary_path = args.run_dir / f"{args.output_name}_n_summary.csv"
    out_report_path = args.run_dir / f"{args.output_name}_report.md"
    rescued.to_csv(out_message_path, index=False)

    summary = (
        rescued.groupby("n_threads", as_index=False)
        .agg(
            messages=("message_id", "count"),
            baseline_quality=("baseline_quality_check", "mean"),
            rescued_quality=("rescued_quality", "mean"),
            baseline_fact_recall=("baseline_fact_recall_check", "mean"),
            rescued_fact_recall=("rescued_fact_recall", "mean"),
            baseline_probe_recall=("fact_recall", lambda s: float(s[rescued.loc[s.index, "needs_memory"] == 1].mean())),
            rescued_probe_recall=("rescued_fact_recall", lambda s: float(s[rescued.loc[s.index, "needs_memory"] == 1].mean())),
            probe_count=("needs_memory", "sum"),
            thread_state_probe_coverage=("thread_state_has_project_code", lambda s: float(s[rescued.loc[s.index, "needs_memory"] == 1].mean())),
        )
        .sort_values("n_threads")
    )
    summary["quality_delta"] = summary["rescued_quality"] - summary["baseline_quality"]
    summary["probe_recall_delta"] = summary["rescued_probe_recall"] - summary["baseline_probe_recall"]
    summary.to_csv(out_summary_path, index=False)

    lines = [
        "# Counterfactual Thread-State Rescue",
        "",
        f"- Scenario dir: `{args.scenario_dir}`",
        f"- Run dir: `{args.run_dir}`",
        "",
        "This counterfactual leaves the model's priority/reply decisions unchanged and only adds a system-maintained",
        "per-thread project code to `facts_used` whenever that code had already been explicitly observed earlier in the same thread.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- N={row.n_threads}: probe_recall {row.baseline_probe_recall:.3f} -> {row.rescued_probe_recall:.3f} "
            f"(delta {row.probe_recall_delta:+.3f}); mean_quality {row.baseline_quality:.3f} -> {row.rescued_quality:.3f} "
            f"(delta {row.quality_delta:+.3f}); probe_coverage={row.thread_state_probe_coverage:.3f}"
        )
    out_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_message_path}")
    print(f"Wrote {out_summary_path}")
    print(f"Wrote {out_report_path}")


if __name__ == "__main__":
    main()
