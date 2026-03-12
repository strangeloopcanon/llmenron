#!/usr/bin/env python3
"""Export one simulator-ready JSON config per breakpoint shock."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--breakpoint-cards",
        type=Path,
        default=Path("experiments/org_simulator/scenarios/enron_breakpoint_cards.csv"),
    )
    parser.add_argument(
        "--regime-cards",
        type=Path,
        default=Path("experiments/org_simulator/scenarios/enron_regime_cards.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/org_simulator/scenarios/config_pack"),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("experiments/org_simulator/scenarios/config_pack/index.json"),
    )
    return parser.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def recommended_episode_shape(arrival_rate_week: float, active_threads_week: float) -> dict[str, int]:
    # Normalize to an 8-hour episode from weekly rates.
    base_messages = arrival_rate_week / 21.0
    base_threads = active_threads_week / 21.0
    messages = int(round(clamp(base_messages, 20, 220)))
    threads = int(round(clamp(base_threads, 6, 120)))
    return {
        "episode_hours": 8,
        "messages_per_episode": messages,
        "active_threads_target": threads,
    }


def regime_record(row: pd.Series) -> dict[str, Any]:
    top_tasks = [
        {"task_type": str(row.get("top_task_1", "")), "share": safe_float(row.get("top_task_1_share", 0.0))},
        {"task_type": str(row.get("top_task_2", "")), "share": safe_float(row.get("top_task_2_share", 0.0))},
        {"task_type": str(row.get("top_task_3", "")), "share": safe_float(row.get("top_task_3_share", 0.0))},
    ]
    top_tasks = [x for x in top_tasks if x["task_type"] and x["task_type"] != "unknown"]
    return {
        "regime_id": int(row["regime_id"]),
        "start_week": str(row["start_week"]),
        "end_week": str(row["end_week"]),
        "weeks": int(row["weeks"]),
        "regime_category": str(row["regime_category"]),
        "levels": {
            "workload": str(row["workload_level"]),
            "thread_load": str(row["thread_load_level"]),
            "coordination": str(row["coordination_level"]),
            "risk": str(row["risk_level"]),
            "approval_load": str(row["approval_load_level"]),
            "handoff": str(row["handoff_level"]),
            "dependency": str(row["dependency_level"]),
        },
        "simulation_knobs": {
            "arrival_rate_week": safe_float(row["sim_arrival_rate"]),
            "active_threads_week": safe_float(row["sim_active_threads"]),
            "escalation_prob": safe_float(row["sim_escalation_prob"]),
            "specialist_prob": safe_float(row["sim_specialist_prob"]),
            "approval_prob": safe_float(row["sim_approval_prob"]),
            "fanout_target": safe_float(row["sim_fanout_target"]),
            "dependency_density": safe_float(row["sim_dependency_density"]),
            "dependency_burst_prob": safe_float(row["sim_dependency_burst_prob"]),
        },
        "top_tasks": top_tasks,
    }


def main() -> None:
    args = parse_args()
    if not args.breakpoint_cards.exists():
        raise FileNotFoundError(f"Missing breakpoint cards: {args.breakpoint_cards}")
    if not args.regime_cards.exists():
        raise FileNotFoundError(f"Missing regime cards: {args.regime_cards}")

    bp = pd.read_csv(args.breakpoint_cards)
    reg = pd.read_csv(args.regime_cards)
    reg_map = {int(r["regime_id"]): r for _, r in reg.iterrows()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []

    for _, row in bp.iterrows():
        from_id = int(row["from_regime"])
        to_id = int(row["to_regime"])
        if from_id not in reg_map or to_id not in reg_map:
            continue
        from_reg = regime_record(reg_map[from_id])
        to_reg = regime_record(reg_map[to_id])

        post_knobs = to_reg["simulation_knobs"]
        pre_knobs = from_reg["simulation_knobs"]

        rec_pre = recommended_episode_shape(
            arrival_rate_week=safe_float(pre_knobs["arrival_rate_week"]),
            active_threads_week=safe_float(pre_knobs["active_threads_week"]),
        )
        rec_post = recommended_episode_shape(
            arrival_rate_week=safe_float(post_knobs["arrival_rate_week"]),
            active_threads_week=safe_float(post_knobs["active_threads_week"]),
        )

        break_week = str(row["break_week"])
        shock_id = f"shock_{break_week}_r{from_id}_to_r{to_id}"
        payload = {
            "meta": {
                "created_at_utc": datetime.now(tz=UTC).isoformat(),
                "source": "enron_regime_cards",
                "schema_version": "1.0",
                "shock_id": shock_id,
            },
            "shock": {
                "break_week": break_week,
                "from_regime": from_id,
                "to_regime": to_id,
                "shock_type": str(row["shock_type"]),
                "strength": {
                    "combined_z": safe_float(row["combined_z"]),
                    "mean_shift_z": safe_float(row["mean_shift_z"]),
                    "cov_shift_z": safe_float(row["cov_shift_z"]),
                },
                "delta": {
                    "arrival_multiplier": safe_float(row["arrival_multiplier"]),
                    "active_thread_multiplier": safe_float(row["active_thread_multiplier"]),
                    "escalation_delta": safe_float(row["escalation_delta"]),
                    "specialist_delta": safe_float(row["specialist_delta"]),
                    "approval_delta": safe_float(row["approval_delta"]),
                    "fanout_delta": safe_float(row["fanout_delta"]),
                    "dependency_density_delta": safe_float(row["dependency_density_delta"]),
                },
            },
            "regimes": {
                "pre": from_reg,
                "post": to_reg,
            },
            "simulator_defaults": {
                "routing_policy_post": str(row["sim_routing_policy"]),
                "memory_mode_post": str(row["sim_memory_mode"]),
                "recommended_pre": rec_pre,
                "recommended_post": rec_post,
                "episodes_default": 2,
                "seed_default": 42,
                "max_total_calls_default": 6000,
            },
        }

        out_file = args.output_dir / f"{shock_id}.json"
        out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        index_rows.append(
            {
                "shock_id": shock_id,
                "file": str(out_file),
                "break_week": break_week,
                "from_regime": from_id,
                "to_regime": to_id,
                "shock_type": str(row["shock_type"]),
                "arrival_multiplier": safe_float(row["arrival_multiplier"]),
                "thread_multiplier": safe_float(row["active_thread_multiplier"]),
                "recommended_post_messages": rec_post["messages_per_episode"],
                "recommended_post_threads": rec_post["active_threads_target"],
            }
        )

    args.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_file.write_text(json.dumps(index_rows, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(index_rows).to_csv(args.output_dir / "index.csv", index=False)

    print(f"Wrote configs: {len(index_rows)}")
    print(f"Wrote: {args.metadata_file.resolve()}")
    print(f"Wrote: {(args.output_dir / 'index.csv').resolve()}")


if __name__ == "__main__":
    main()
