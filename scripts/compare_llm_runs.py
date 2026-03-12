#!/usr/bin/env python3
"""Compare LLM eval runs and generate cleaned summaries without zero/error rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/llm_eval/llm_eval_runs"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("experiments/llm_eval/llm_eval_runs/run_comparison_gpt52"),
    )
    return parser.parse_args()


def summarize_run(run_dir: Path) -> dict[str, object] | None:
    msg_path = run_dir / "llm_eval_message_log.csv"
    n_path = run_dir / "llm_eval_n_summary.csv"
    if not msg_path.exists() or not n_path.exists():
        return None

    msg = pd.read_csv(msg_path)
    nsum = pd.read_csv(n_path)
    if msg.empty or nsum.empty:
        return None

    if "api_error" not in msg.columns:
        msg["api_error"] = 0
    if "input_tokens" not in msg.columns:
        msg["input_tokens"] = 0
    if "output_tokens" not in msg.columns:
        msg["output_tokens"] = 0

    is_heuristic = run_dir.name.startswith("heuristic")
    if is_heuristic:
        zero_or_error_mask = msg["api_error"] > 0
    else:
        zero_or_error_mask = (msg["api_error"] > 0) | (
            (msg["input_tokens"] <= 0) & (msg["output_tokens"] <= 0)
        )
    msg_clean = msg[~zero_or_error_mask].copy()

    msg_clean.to_csv(run_dir / "llm_eval_message_log_nonzero.csv", index=False)
    if len(msg_clean):
        clean_n = (
            msg_clean.groupby("n_threads")
            .agg(
                mean_quality=("quality_score", "mean"),
                priority_acc=("priority_acc", "mean"),
                reply_acc=("reply_acc", "mean"),
                action_presence_acc=("action_presence_acc", "mean"),
                p0_sla=("on_time", lambda s: float(s.mean())),
                invalid_rate=("invalid_output", "mean"),
                api_error_rate=("api_error", "mean"),
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                rows=("email_id", "count"),
            )
            .reset_index()
            .sort_values("n_threads")
        )
    else:
        clean_n = pd.DataFrame(
            columns=[
                "n_threads",
                "mean_quality",
                "priority_acc",
                "reply_acc",
                "action_presence_acc",
                "p0_sla",
                "invalid_rate",
                "api_error_rate",
                "input_tokens",
                "output_tokens",
                "rows",
            ]
        )
    clean_n.to_csv(run_dir / "llm_eval_n_summary_nonzero.csv", index=False)

    summary = {
        "run": run_dir.name,
        "n_values": ",".join(str(int(x)) for x in sorted(nsum["n_threads"].unique())),
        "rows_total": int(len(msg)),
        "rows_nonzero": int(len(msg_clean)),
        "zero_or_error_share": float(zero_or_error_mask.mean()),
        "api_error_share": float((msg["api_error"] > 0).mean()),
        "mean_quality_raw": float(msg["quality_score"].mean()),
        "mean_quality_nonzero": float(msg_clean["quality_score"].mean()) if len(msg_clean) else 0.0,
        "priority_acc_raw": float(msg["priority_acc"].mean()),
        "priority_acc_nonzero": float(msg_clean["priority_acc"].mean()) if len(msg_clean) else 0.0,
        "reply_acc_raw": float(msg["reply_acc"].mean()),
        "reply_acc_nonzero": float(msg_clean["reply_acc"].mean()) if len(msg_clean) else 0.0,
        "input_tokens_total": float(msg["input_tokens"].sum()),
        "output_tokens_total": float(msg["output_tokens"].sum()),
        "n_star_raw": (
            int(nsum[nsum["passes_threshold"] == True]["n_threads"].max())
            if (nsum["passes_threshold"] == True).any()
            else "none"
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = summarize_run(run_dir)
        if summary is not None:
            rows.append(summary)

    df = pd.DataFrame(rows).sort_values("run")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_prefix.with_suffix(".csv"), index=False)

    lines = [
        "# LLM Run Comparison (Raw vs Nonzero)",
        "",
        "- `nonzero` excludes rows where `api_error=1` or both token counts are zero.",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"- {r['run']}: zero_or_error_share={r['zero_or_error_share']:.3f}, "
            f"quality_raw={r['mean_quality_raw']:.3f}, quality_nonzero={r['mean_quality_nonzero']:.3f}, "
            f"priority_raw={r['priority_acc_raw']:.3f}, priority_nonzero={r['priority_acc_nonzero']:.3f}, "
            f"n_star_raw={r['n_star_raw']}"
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Focused file for this round.
    focus_names = {
        "heuristic_gpt52_grid_v2",
        "heuristic_gpt52_grid_v1",
        "openai_gpt52_auto_grid_v2",
        "openai_gpt52_auto_grid_v1",
        "openai_gpt52_high_grid_v2",
        "openai_gpt52_high_grid_v1",
        "openai_live_v3_gpt5_mini",
        "openai_responses_v1mini",
        "openai_responses_tuned_smoke",
    }
    focus = df[df["run"].isin(focus_names)].copy()
    focus.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_focus.csv"), index=False)
    focus_lines = [
        "# Focus Comparison (GPT-5.2 vs Legacy)",
        "",
    ]
    for _, r in focus.sort_values("run").iterrows():
        focus_lines.append(
            f"- {r['run']}: zero_or_error_share={r['zero_or_error_share']:.3f}, "
            f"quality_nonzero={r['mean_quality_nonzero']:.3f}, priority_nonzero={r['priority_acc_nonzero']:.3f}, "
            f"reply_nonzero={r['reply_acc_nonzero']:.3f}"
        )
    args.output_prefix.with_name(args.output_prefix.name + "_focus.md").write_text(
        "\n".join(focus_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.md')}")
    print(f"Wrote {args.output_prefix.with_name(args.output_prefix.name + '_focus.csv')}")
    print(f"Wrote {args.output_prefix.with_name(args.output_prefix.name + '_focus.md')}")


if __name__ == "__main__":
    main()
