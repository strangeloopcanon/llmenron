#!/usr/bin/env python3
"""Build compact regime + breakpoint cards for simulator configuration."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime-signatures",
        type=Path,
        default=Path("experiments/org_simulator/regimes/enron_regime_signatures.parquet"),
    )
    parser.add_argument(
        "--breakpoints-json",
        type=Path,
        default=Path("experiments/org_simulator/regimes/enron_breakpoints.json"),
    )
    parser.add_argument(
        "--task-distribution",
        type=Path,
        default=Path("experiments/org_simulator/tasks/enron_task_distribution_by_regime.parquet"),
    )
    parser.add_argument(
        "--dependency-graph",
        type=Path,
        default=Path("experiments/org_simulator/tasks/enron_task_dependency_graph.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/org_simulator/scenarios"),
    )
    return parser.parse_args()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def level_from_quantiles(
    value: float,
    q25: float,
    q50: float,
    q75: float,
    q90: float,
) -> str:
    if value >= q90:
        return "very_high"
    if value >= q75:
        return "high"
    if value >= q50:
        return "medium"
    if value >= q25:
        return "low"
    return "very_low"


def category_label(
    traffic_level: str,
    risk_level: str,
    approval_level: str,
    dep_level: str,
) -> str:
    if traffic_level in {"high", "very_high"} and risk_level in {"high", "very_high"}:
        return "crisis_operations"
    if approval_level in {"high", "very_high"} and dep_level in {"high", "very_high"}:
        return "governance_bottleneck"
    if traffic_level in {"very_low", "low"} and risk_level in {"very_low", "low"}:
        return "quiet_steady_state"
    if dep_level in {"high", "very_high"}:
        return "interdependent_program_mode"
    return "operational_normal"


def to_float_dict(series: pd.Series) -> dict[str, float]:
    out = {}
    for k, v in series.items():
        out[str(k)] = float(v)
    return out


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


def build_dependency_stats(dep: pd.DataFrame) -> pd.DataFrame:
    if dep.empty:
        return pd.DataFrame(
            columns=[
                "regime_id",
                "dependency_edges",
                "dependency_nodes",
                "dependency_density",
                "dep_mean_coupling",
                "dep_p90_coupling",
                "dep_shared_actor_mean",
                "dep_burst_share",
                "dep_shared_staff_share",
            ]
        )

    rows: list[dict[str, Any]] = []
    same = dep[dep["same_regime"] == 1].copy()
    for regime_id, g in same.groupby("regime_a", sort=True):
        nodes = set(g["thread_a"]).union(set(g["thread_b"]))
        n_nodes = len(nodes)
        n_edges = len(g)
        density = safe_div(2.0 * n_edges, n_nodes * max(1, n_nodes - 1))
        rows.append(
            {
                "regime_id": int(regime_id),
                "dependency_edges": int(n_edges),
                "dependency_nodes": int(n_nodes),
                "dependency_density": float(density),
                "dep_mean_coupling": float(g["temporal_coupling_count"].mean()),
                "dep_p90_coupling": float(g["temporal_coupling_count"].quantile(0.90)),
                "dep_shared_actor_mean": float(g["shared_actor_count"].mean()),
                "dep_burst_share": float((g["dependency_type"] == "burst_coupling").mean()),
                "dep_shared_staff_share": float((g["dependency_type"] == "shared_staff_dependency").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime_id")


def top_task_map(task_dist: pd.DataFrame) -> dict[int, list[tuple[str, float]]]:
    out: dict[int, list[tuple[str, float]]] = {}
    for rid, g in task_dist.groupby("regime_id", sort=True):
        top = g.sort_values("share", ascending=False)[["task_type", "share"]].head(4)
        out[int(rid)] = [(str(r.task_type), float(r.share)) for r in top.itertuples(index=False)]
    return out


def build_regime_cards(
    sig: pd.DataFrame,
    dep_stats: pd.DataFrame,
    task_dist: pd.DataFrame,
) -> pd.DataFrame:
    cards = sig.copy()
    cards = cards.merge(dep_stats, on="regime_id", how="left")
    for col in [
        "dependency_edges",
        "dependency_nodes",
        "dependency_density",
        "dep_mean_coupling",
        "dep_p90_coupling",
        "dep_shared_actor_mean",
        "dep_burst_share",
        "dep_shared_staff_share",
    ]:
        if col not in cards:
            cards[col] = 0.0
    cards = cards.fillna(0.0)

    # Quantiles for plain-English level encoding.
    q = {}
    for col in [
        "n_events",
        "n_threads_active",
        "mean_fanout",
        "escalation_rate",
        "specialist_consult_rate",
        "approval_rate",
        "mean_handoff_count",
        "reopen_rate",
        "dependency_density",
        "dep_p90_coupling",
    ]:
        s = cards[col].astype(float)
        q[col] = {
            "q25": float(s.quantile(0.25)),
            "q50": float(s.quantile(0.50)),
            "q75": float(s.quantile(0.75)),
            "q90": float(s.quantile(0.90)),
        }

    top_tasks = top_task_map(task_dist)
    med_events = float(cards["n_events"].median())
    med_threads = float(cards["n_threads_active"].median())
    med_dep = float(cards["dependency_density"].median()) if len(cards) else 1.0

    records: list[dict[str, Any]] = []
    for row in cards.itertuples(index=False):
        rid = int(row.regime_id)
        traffic_level = level_from_quantiles(
            float(row.n_events),
            q["n_events"]["q25"],
            q["n_events"]["q50"],
            q["n_events"]["q75"],
            q["n_events"]["q90"],
        )
        thread_level = level_from_quantiles(
            float(row.n_threads_active),
            q["n_threads_active"]["q25"],
            q["n_threads_active"]["q50"],
            q["n_threads_active"]["q75"],
            q["n_threads_active"]["q90"],
        )
        coordination_level = level_from_quantiles(
            float(row.mean_fanout),
            q["mean_fanout"]["q25"],
            q["mean_fanout"]["q50"],
            q["mean_fanout"]["q75"],
            q["mean_fanout"]["q90"],
        )
        risk_signal = float(row.escalation_rate) + 0.6 * float(row.specialist_consult_rate)
        risk_q25 = q["escalation_rate"]["q25"] + 0.6 * q["specialist_consult_rate"]["q25"]
        risk_q50 = q["escalation_rate"]["q50"] + 0.6 * q["specialist_consult_rate"]["q50"]
        risk_q75 = q["escalation_rate"]["q75"] + 0.6 * q["specialist_consult_rate"]["q75"]
        risk_q90 = q["escalation_rate"]["q90"] + 0.6 * q["specialist_consult_rate"]["q90"]
        risk_level = level_from_quantiles(risk_signal, risk_q25, risk_q50, risk_q75, risk_q90)
        approval_level = level_from_quantiles(
            float(row.approval_rate),
            q["approval_rate"]["q25"],
            q["approval_rate"]["q50"],
            q["approval_rate"]["q75"],
            q["approval_rate"]["q90"],
        )
        handoff_level = level_from_quantiles(
            float(row.mean_handoff_count),
            q["mean_handoff_count"]["q25"],
            q["mean_handoff_count"]["q50"],
            q["mean_handoff_count"]["q75"],
            q["mean_handoff_count"]["q90"],
        )
        dependency_level = level_from_quantiles(
            float(row.dependency_density),
            q["dependency_density"]["q25"],
            q["dependency_density"]["q50"],
            q["dependency_density"]["q75"],
            q["dependency_density"]["q90"],
        )
        category = category_label(traffic_level, risk_level, approval_level, dependency_level)
        top = top_tasks.get(rid, [])
        top1 = top[0] if len(top) > 0 else ("unknown", 0.0)
        top2 = top[1] if len(top) > 1 else ("unknown", 0.0)
        top3 = top[2] if len(top) > 2 else ("unknown", 0.0)

        records.append(
            {
                "regime_id": rid,
                "start_week": str(pd.Timestamp(row.start_week).date()),
                "end_week": str(pd.Timestamp(row.end_week).date()),
                "weeks": int(row.weeks),
                "regime_category": category,
                "workload_level": traffic_level,
                "thread_load_level": thread_level,
                "coordination_level": coordination_level,
                "risk_level": risk_level,
                "approval_load_level": approval_level,
                "handoff_level": handoff_level,
                "dependency_level": dependency_level,
                "top_task_1": top1[0],
                "top_task_1_share": float(top1[1]),
                "top_task_2": top2[0],
                "top_task_2_share": float(top2[1]),
                "top_task_3": top3[0],
                "top_task_3_share": float(top3[1]),
                "events_per_week": float(row.n_events),
                "active_threads_per_week": float(row.n_threads_active),
                "actors_per_week": float(row.n_actors),
                "mean_fanout": float(row.mean_fanout),
                "escalation_rate": float(row.escalation_rate),
                "specialist_consult_rate": float(row.specialist_consult_rate),
                "approval_rate": float(row.approval_rate),
                "after_hours_share": float(row.after_hours_share),
                "mean_handoff_count": float(row.mean_handoff_count),
                "reopen_rate": float(row.reopen_rate),
                "dependency_density": float(row.dependency_density),
                "dependency_p90_coupling": float(row.dep_p90_coupling),
                "dependency_burst_share": float(row.dep_burst_share),
                "dependency_shared_staff_share": float(row.dep_shared_staff_share),
                "traffic_multiplier_vs_median": safe_div(float(row.n_events), med_events),
                "thread_multiplier_vs_median": safe_div(float(row.n_threads_active), med_threads),
                "dependency_multiplier_vs_median": safe_div(float(row.dependency_density), med_dep if med_dep > 0 else 1.0),
                "sim_arrival_rate": float(row.n_events),
                "sim_active_threads": float(row.n_threads_active),
                "sim_escalation_prob": float(row.escalation_rate),
                "sim_specialist_prob": float(row.specialist_consult_rate),
                "sim_approval_prob": float(row.approval_rate),
                "sim_fanout_target": float(row.mean_fanout),
                "sim_dependency_density": float(row.dependency_density),
                "sim_dependency_burst_prob": float(row.dep_burst_share),
            }
        )
    return pd.DataFrame(records).sort_values("regime_id").reset_index(drop=True)


def assign_shock_type(
    arrival_mult: float,
    thread_mult: float,
    escalation_delta: float,
    approval_delta: float,
    centralization_delta: float,
    dependency_delta: float,
) -> str:
    if arrival_mult >= 1.6 and escalation_delta >= 0.03:
        return "surge_plus_risk"
    if arrival_mult >= 1.6 or thread_mult >= 1.6:
        return "volume_surge"
    if arrival_mult <= 0.7 and escalation_delta >= 0.02:
        return "contraction_with_risk_escalation"
    if arrival_mult <= 0.7 and centralization_delta >= 0.20:
        return "contraction_centralization"
    if approval_delta >= 0.01 and centralization_delta >= 0.10:
        return "approval_bottleneck"
    if dependency_delta >= 0.001:
        return "dependency_shock"
    return "mixed_shift"


def routing_policy(risk_level: str, workload_level: str, dep_level: str) -> str:
    if risk_level in {"high", "very_high"}:
        return "risk_first_with_specialist_autoroute"
    if workload_level in {"high", "very_high"} and dep_level in {"high", "very_high"}:
        return "load_balanced_with_dependency_guardrails"
    if workload_level in {"high", "very_high"}:
        return "load_balanced_autotriage"
    return "balanced_manual_override"


def memory_mode(thread_level: str, handoff_level: str) -> str:
    if thread_level in {"high", "very_high"} or handoff_level in {"high", "very_high"}:
        return "long_horizon_scratchpad"
    if thread_level in {"medium"}:
        return "thread_local_scratchpad"
    return "compact_scratchpad"


def build_breakpoint_cards(
    regime_cards: pd.DataFrame,
    breakpoints_json: dict[str, Any],
) -> pd.DataFrame:
    reg = regime_cards.set_index("regime_id")
    selected = breakpoints_json.get("selected_breakpoints", [])
    rows: list[dict[str, Any]] = []
    for item in selected:
        wk = str(item.get("week"))
        # mapping follows regime boundary construction in the main pipeline.
        # first breakpoint is transition 0->1, second is 1->2, ...
        idx = selected.index(item)
        from_id = int(idx)
        to_id = int(idx + 1)
        if from_id not in reg.index or to_id not in reg.index:
            continue
        r0 = reg.loc[from_id]
        r1 = reg.loc[to_id]

        arrival_mult = safe_div(float(r1["events_per_week"]), float(r0["events_per_week"]))
        thread_mult = safe_div(float(r1["active_threads_per_week"]), float(r0["active_threads_per_week"]))
        escalation_delta = float(r1["escalation_rate"] - r0["escalation_rate"])
        specialist_delta = float(r1["specialist_consult_rate"] - r0["specialist_consult_rate"])
        approval_delta = float(r1["approval_rate"] - r0["approval_rate"])
        fanout_delta = float(r1["mean_fanout"] - r0["mean_fanout"])
        dependency_delta = float(r1["dependency_density"] - r0["dependency_density"])
        centralization_delta = 0.0
        # centralization is not in regime_cards directly; infer via dependency + fanout + approval as proxy
        centralization_delta = 0.5 * dependency_delta + 0.3 * fanout_delta + 2.0 * approval_delta

        shock = assign_shock_type(
            arrival_mult=arrival_mult,
            thread_mult=thread_mult,
            escalation_delta=escalation_delta,
            approval_delta=approval_delta,
            centralization_delta=centralization_delta,
            dependency_delta=dependency_delta,
        )

        workload_shift = "up" if arrival_mult > 1.10 else ("down" if arrival_mult < 0.90 else "flat")
        thread_shift = "up" if thread_mult > 1.10 else ("down" if thread_mult < 0.90 else "flat")
        risk_shift_score = escalation_delta + 0.6 * specialist_delta + 0.4 * approval_delta
        risk_shift = "up" if risk_shift_score > 0.015 else ("down" if risk_shift_score < -0.015 else "flat")

        rows.append(
            {
                "break_week": wk,
                "from_regime": from_id,
                "to_regime": to_id,
                "shock_type": shock,
                "combined_z": float(item.get("combined_z", 0.0)),
                "mean_shift_z": float(item.get("mean_shift_z", 0.0)),
                "cov_shift_z": float(item.get("cov_shift_z", 0.0)),
                "workload_shift": workload_shift,
                "thread_shift": thread_shift,
                "risk_shift": risk_shift,
                "arrival_multiplier": arrival_mult,
                "active_thread_multiplier": thread_mult,
                "escalation_delta": escalation_delta,
                "specialist_delta": specialist_delta,
                "approval_delta": approval_delta,
                "fanout_delta": fanout_delta,
                "dependency_density_delta": dependency_delta,
                "sim_arrival_rate_post": float(r1["sim_arrival_rate"]),
                "sim_active_threads_post": float(r1["sim_active_threads"]),
                "sim_escalation_prob_post": float(r1["sim_escalation_prob"]),
                "sim_specialist_prob_post": float(r1["sim_specialist_prob"]),
                "sim_approval_prob_post": float(r1["sim_approval_prob"]),
                "sim_fanout_target_post": float(r1["sim_fanout_target"]),
                "sim_dependency_density_post": float(r1["sim_dependency_density"]),
                "sim_dependency_burst_prob_post": float(r1["sim_dependency_burst_prob"]),
                "sim_routing_policy": routing_policy(
                    str(r1["risk_level"]),
                    str(r1["workload_level"]),
                    str(r1["dependency_level"]),
                ),
                "sim_memory_mode": memory_mode(
                    str(r1["thread_load_level"]),
                    str(r1["handoff_level"]),
                ),
                "top_task_after_1": str(r1["top_task_1"]),
                "top_task_after_1_share": float(r1["top_task_1_share"]),
                "top_task_after_2": str(r1["top_task_2"]),
                "top_task_after_2_share": float(r1["top_task_2_share"]),
            }
        )
    return pd.DataFrame(rows)


def build_cards_markdown(regime_cards: pd.DataFrame, breakpoint_cards: pd.DataFrame) -> str:
    lines = [
        "# Enron Regime Cards",
        "",
        "Compact cards for simulation setup. One row in `breakpoint_cards` is one shock transition to replay.",
        "",
        "## Breakpoint Cards",
    ]
    for row in breakpoint_cards.itertuples(index=False):
        lines.append(
            f"- {row.break_week}: {row.shock_type} | "
            f"arrival x{row.arrival_multiplier:.2f}, threads x{row.active_thread_multiplier:.2f}, "
            f"escalation Δ{row.escalation_delta:+.3f}, approval Δ{row.approval_delta:+.3f}, "
            f"routing={row.sim_routing_policy}, memory={row.sim_memory_mode}"
        )
    lines.extend(["", "## Regime Cards"])
    for row in regime_cards.itertuples(index=False):
        lines.append(
            f"- Regime {int(row.regime_id)} ({row.start_week} to {row.end_week}): {row.regime_category} | "
            f"workload={row.workload_level}, coordination={row.coordination_level}, risk={row.risk_level}, "
            f"top_tasks={row.top_task_1} ({row.top_task_1_share:.2f}), {row.top_task_2} ({row.top_task_2_share:.2f})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    signatures = pd.read_parquet(args.regime_signatures).sort_values("regime_id")
    with args.breakpoints_json.open("r", encoding="utf-8") as f:
        breakpoints = json.load(f)
    task_dist = pd.read_parquet(args.task_distribution)
    dep = pd.read_parquet(args.dependency_graph)

    dep_stats = build_dependency_stats(dep)
    regime_cards = build_regime_cards(signatures, dep_stats, task_dist)
    breakpoint_cards = build_breakpoint_cards(regime_cards, breakpoints)

    regime_cards.to_csv(args.output_dir / "enron_regime_cards.csv", index=False)
    breakpoint_cards.to_csv(args.output_dir / "enron_breakpoint_cards.csv", index=False)

    yaml_obj = {
        "meta": {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "source": "enron_rosetta",
        },
        "regime_cards": json.loads(regime_cards.to_json(orient="records")),
        "breakpoint_cards": json.loads(breakpoint_cards.to_json(orient="records")),
    }
    (args.output_dir / "enron_regime_cards.yaml").write_text(
        "\n".join(to_yaml_lines(yaml_obj)) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "enron_regime_cards.md").write_text(
        build_cards_markdown(regime_cards, breakpoint_cards),
        encoding="utf-8",
    )

    print(f"Wrote: {(args.output_dir / 'enron_regime_cards.csv').resolve()}")
    print(f"Wrote: {(args.output_dir / 'enron_breakpoint_cards.csv').resolve()}")
    print(f"Wrote: {(args.output_dir / 'enron_regime_cards.yaml').resolve()}")
    print(f"Wrote: {(args.output_dir / 'enron_regime_cards.md').resolve()}")
    print(f"Regime rows: {len(regime_cards)}")
    print(f"Breakpoint rows: {len(breakpoint_cards)}")


if __name__ == "__main__":
    main()
