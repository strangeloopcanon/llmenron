#!/usr/bin/env python3
"""Generate simple figures used in the README.

This is intentionally small and dependency-light (matplotlib + pandas).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("results/figures"))

    p.add_argument("--key-results-csv", type=Path, default=Path("results/summaries/key_results.csv"))
    p.add_argument(
        "--scratchpad-scenario-dir",
        type=Path,
        default=Path("experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105"),
    )
    p.add_argument(
        "--gpt-run-dir",
        type=Path,
        default=Path(
            "experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105/runs/openai_gpt-5.2_20260210T023527Z"
        ),
    )
    p.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=Path(
            "experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105/runs/heuristic_gpt-5.2_20260210T015547Z"
        ),
    )
    p.add_argument("--score-threshold-q", type=float, default=0.75)
    return p.parse_args()


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_human_thread_load_quantiles(*, key_results_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(key_results_csv)
    if df.empty:
        raise RuntimeError(f"Empty key results: {key_results_csv}")
    row = df.iloc[0]

    qs = ["q25", "q50", "q75", "q90"]
    mean_vals = [float(row[f"recommended_N_mean_{q}"]) for q in qs]
    p90_vals = [float(row[f"recommended_N_p90_{q}"]) for q in qs]

    x = list(range(len(qs)))
    w = 0.38

    plt.figure(figsize=(8.2, 4.2))
    plt.bar([i - w / 2 for i in x], mean_vals, width=w, label="Typical load (mean-active threads)", color="#2A5CAA")
    plt.bar([i + w / 2 for i in x], p90_vals, width=w, label="Stress load (p90-active threads)", color="#D98324")
    plt.xticks(x, ["25th", "50th", "75th", "90th"])
    plt.ylabel("Threads (N)")
    plt.title("Human Thread Load From Enron (14-day rolling)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, ncol=1, loc="upper left")

    save_fig(out_path)


def plot_synthetic_judged_quality(
    *,
    key_results_csv: Path,
    gpt_run_dir: Path,
    baseline_run_dir: Path,
    out_path: Path,
    score_threshold_q: float,
) -> None:
    row = pd.read_csv(key_results_csv).iloc[0]
    human_typical = float(row["recommended_N_mean_q50"])
    human_stress = float(row["recommended_N_p90_q50"])

    gpt = pd.read_csv(gpt_run_dir / "judged_v2_n_summary.csv")
    base = pd.read_csv(baseline_run_dir / "judged_v2_n_summary.csv")

    plt.figure(figsize=(8.2, 4.2))
    plt.plot(gpt["n_threads"], gpt["mean_quality"], marker="o", label="GPT-5.2 (agent)", color="#2A5CAA")
    plt.plot(base["n_threads"], base["mean_quality"], marker="o", label="Baseline bot (heuristic)", color="#666666")
    plt.axhline(float(score_threshold_q), color="#B22222", linestyle="--", linewidth=1.2, label=f"Threshold q={score_threshold_q:.2f}")

    # Anchor interpretation: show where the human median loads sit on the N axis.
    ymin, ymax = 0.0, 1.02
    plt.axvline(human_typical, color="#333333", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.axvline(human_stress, color="#333333", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.text(human_typical, ymin + 0.02, "human median typical (~50)", rotation=90, va="bottom", ha="right", fontsize=9, color="#333333")
    plt.text(human_stress, ymin + 0.02, "human median stress (~105)", rotation=90, va="bottom", ha="right", fontsize=9, color="#333333")

    plt.ylim(0.0, 1.02)
    plt.xlabel("Concurrent active threads (N)")
    plt.ylabel("Judged quality (0..1)")
    plt.title("Synthetic Scratchpad Pilot: Judged Quality vs Load")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc="lower left")

    save_fig(out_path)


def plot_synthetic_memory_recall(*, key_results_csv: Path, gpt_run_dir: Path, baseline_run_dir: Path, out_path: Path) -> None:
    row = pd.read_csv(key_results_csv).iloc[0]
    human_typical = float(row["recommended_N_mean_q50"])
    human_stress = float(row["recommended_N_p90_q50"])

    gpt = pd.read_csv(gpt_run_dir / "message_log.csv")
    base = pd.read_csv(baseline_run_dir / "message_log.csv")

    def mem_recall(df: pd.DataFrame) -> pd.DataFrame:
        m = df[df["needs_memory"] == 1].copy()
        if m.empty:
            return pd.DataFrame({"n_threads": [], "memory_recall": []})
        out = m.groupby("n_threads", sort=True)["fact_recall"].mean().reset_index()
        out = out.rename(columns={"fact_recall": "memory_recall"})
        return out

    g = mem_recall(gpt)
    b = mem_recall(base)

    plt.figure(figsize=(8.2, 4.2))
    # Stacked bars: share of memory-probe messages where the agent includes the required project code (success)
    # vs omits it (failure / needs clarification).
    n_vals = sorted(set(g["n_threads"].tolist() + b["n_threads"].tolist()))
    x = [float(n) for n in n_vals]
    if len(n_vals) > 1:
        gaps = [float(b - a) for a, b in zip(n_vals[:-1], n_vals[1:])]
        min_gap = min(gaps) if gaps else 20.0
    else:
        min_gap = 20.0
    w = max(8.0, min(25.0, 0.35 * float(min_gap)))

    def recall_for(df: pd.DataFrame, n: int) -> float:
        hit = df[df["n_threads"] == n]["memory_recall"]
        return float(hit.iloc[0]) if len(hit) else 0.0

    g_recall = [recall_for(g, n) for n in n_vals]
    b_recall = [recall_for(b, n) for n in n_vals]

    # GPT bars
    plt.bar([i - w / 2 for i in x], g_recall, width=w, color="#2A5CAA", label="GPT-5.2: recall")
    plt.bar(
        [i - w / 2 for i in x],
        [1.0 - v for v in g_recall],
        width=w,
        bottom=g_recall,
        color="#A6B8D8",
        label="GPT-5.2: missing",
    )

    # Baseline bars
    plt.bar([i + w / 2 for i in x], b_recall, width=w, color="#666666", label="Baseline: recall")
    plt.bar(
        [i + w / 2 for i in x],
        [1.0 - v for v in b_recall],
        width=w,
        bottom=b_recall,
        color="#C9C9C9",
        label="Baseline: missing",
    )

    def annotate_stack(*, x_pos: float, recall: float) -> None:
        # Keep labels readable: put recall in the dark segment (white text) when possible,
        # and missing in the light segment (dark text).
        recall_pct = int(round(100.0 * recall))
        missing_pct = int(round(100.0 * (1.0 - recall)))

        if recall >= 0.08:
            plt.text(
                x_pos,
                recall / 2.0,
                f"{recall_pct}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            # If recall is tiny, label just above the bar to avoid illegible text.
            plt.text(
                x_pos,
                recall + 0.02,
                f"{recall_pct}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111111",
                fontweight="bold",
            )

        if (1.0 - recall) >= 0.12:
            plt.text(
                x_pos,
                recall + (1.0 - recall) / 2.0,
                f"{missing_pct}%",
                ha="center",
                va="center",
                fontsize=9,
                color="#111111",
                fontweight="bold",
            )

    for xi, gr, br in zip(x, g_recall, b_recall):
        annotate_stack(x_pos=xi - w / 2.0, recall=float(gr))
        annotate_stack(x_pos=xi + w / 2.0, recall=float(br))

    ymin, ymax = 0.0, 1.02
    plt.axvline(human_typical, color="#333333", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.axvline(human_stress, color="#333333", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.text(human_typical, ymin + 0.02, "human median typical (~50)", rotation=90, va="bottom", ha="right", fontsize=9, color="#333333")
    plt.text(human_stress, ymin + 0.02, "human median stress (~105)", rotation=90, va="bottom", ha="right", fontsize=9, color="#333333")

    plt.ylim(0.0, 1.02)
    plt.xticks(x, [str(n) for n in n_vals])
    plt.xlabel("Concurrent active threads (N)")
    plt.ylabel("Share of memory-probe messages")
    plt.title("Synthetic Scratchpad Pilot: Thread Identity Recall (Project Code)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)

    save_fig(out_path)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)

    plot_human_thread_load_quantiles(
        key_results_csv=args.key_results_csv,
        out_path=args.out_dir / "human_thread_load_quantiles.png",
    )
    plot_synthetic_judged_quality(
        key_results_csv=args.key_results_csv,
        gpt_run_dir=args.gpt_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        out_path=args.out_dir / "synthetic_pilot_judged_quality.png",
        score_threshold_q=float(args.score_threshold_q),
    )
    plot_synthetic_memory_recall(
        key_results_csv=args.key_results_csv,
        gpt_run_dir=args.gpt_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        out_path=args.out_dir / "synthetic_pilot_memory_recall.png",
    )
    print(f"Wrote figures to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
