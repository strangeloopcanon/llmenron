#!/usr/bin/env python3
"""Generate a blog-header image from the Enron-derived data in this repo."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@enron\.com", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosetta-path",
        type=Path,
        default=Path("data/enron_rosetta_source.parquet"),
    )
    parser.add_argument(
        "--custodian-metrics-path",
        type=Path,
        default=Path("experiments/reference_data/custodian_metrics.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/figures/blog_header_enron_complexity.png"),
    )
    parser.add_argument(
        "--variant",
        choices=["labeled", "clean"],
        default="labeled",
    )
    parser.add_argument("--top-nodes", type=int, default=140)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def extract_internal_emails(value: object) -> list[str]:
    if value is None:
        return []
    out: list[str] = []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        for item in value:
            out.extend(extract_internal_emails(item))
        return out
    return [m.lower() for m in EMAIL_RE.findall(str(value))]


def build_internal_graph(rosetta_path: Path, top_nodes: int) -> tuple[nx.Graph, Counter[str]]:
    df = pd.read_parquet(rosetta_path, columns=["from_email", "to_raw", "cc_raw", "bcc_raw"])

    participant_counts: Counter[str] = Counter()
    for row in df.itertuples(index=False):
        sender = str(row.from_email).strip().lower() if row.from_email is not None else ""
        if not sender.endswith("@enron.com"):
            continue
        recipients = set()
        recipients.update(extract_internal_emails(row.to_raw))
        recipients.update(extract_internal_emails(row.cc_raw))
        recipients.update(extract_internal_emails(row.bcc_raw))
        recipients.discard(sender)
        participant_counts[sender] += 1
        for recipient in recipients:
            participant_counts[recipient] += 1

    chosen_nodes = {email for email, _ in participant_counts.most_common(top_nodes)}

    edge_weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in df.itertuples(index=False):
        sender = str(row.from_email).strip().lower() if row.from_email is not None else ""
        if sender not in chosen_nodes:
            continue
        recipients = set()
        recipients.update(extract_internal_emails(row.to_raw))
        recipients.update(extract_internal_emails(row.cc_raw))
        recipients.update(extract_internal_emails(row.bcc_raw))
        for recipient in recipients:
            if recipient == sender or recipient not in chosen_nodes:
                continue
            a, b = sorted((sender, recipient))
            edge_weights[(a, b)] += 1

    graph = nx.Graph()
    graph.add_nodes_from(chosen_nodes)
    for (a, b), weight in edge_weights.items():
        graph.add_edge(a, b, weight=weight, layout_weight=math.log1p(weight))

    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph, participant_counts


def email_label(email: str) -> str:
    local = email.split("@", 1)[0].replace(".", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in local.split())


def build_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "enron_header",
        ["#78A6D1", "#A7D3F2", "#F2C572", "#F28F3B"],
        N=256,
    )


def draw_network(
    ax: plt.Axes,
    graph: nx.Graph,
    participant_counts: Counter[str],
    metrics_df: pd.DataFrame,
    seed: int,
    *,
    variant: str,
) -> None:
    ax.set_facecolor("#07111D")
    cmap = build_cmap()

    metric_map = {
        str(row.email).strip().lower(): float(row.mean_active_threads_rolling)
        for row in metrics_df.itertuples(index=False)
        if isinstance(row.email, str) and row.email
    }
    node_metric = []
    node_size = []
    for node in graph.nodes():
        weighted_degree = graph.degree(node, weight="weight")
        metric = metric_map.get(node, np.nan)
        if math.isnan(metric):
            metric = min(120.0, 8.0 + math.sqrt(participant_counts[node]) * 3.5)
        node_metric.append(metric)
        node_size.append(40.0 + math.sqrt(max(weighted_degree, 1.0)) * 16.0)

    pos = nx.spring_layout(
        graph,
        seed=seed,
        weight="layout_weight",
        k=1.65 / math.sqrt(max(graph.number_of_nodes(), 1)),
        iterations=350,
    )

    edges = sorted(graph.edges(data=True), key=lambda item: item[2]["weight"])
    widths = [0.25 + math.log1p(data["weight"]) * 0.45 for _, _, data in edges]
    edge_alphas = [min(0.28, 0.04 + math.log1p(data["weight"]) * 0.035) for _, _, data in edges]
    for (u, v, _), width, alpha in zip(edges, widths, edge_alphas):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#78A6D1", linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)

    nodes = list(graph.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    colors = cmap((np.array(node_metric) - np.nanmin(node_metric)) / (np.nanmax(node_metric) - np.nanmin(node_metric) + 1e-9))

    ax.scatter(xs, ys, s=[s * 2.9 for s in node_size], c="#78A6D1", alpha=0.05, linewidths=0, zorder=2)
    ax.scatter(xs, ys, s=[s * 1.7 for s in node_size], c="#A7D3F2", alpha=0.08, linewidths=0, zorder=3)
    ax.scatter(xs, ys, s=node_size, c=colors, alpha=0.96, edgecolors="#EAF4FF", linewidths=0.3, zorder=4)

    top_labels = sorted(nodes, key=lambda n: graph.degree(n, weight="weight"), reverse=True)[:12]
    placed: list[tuple[float, float]] = []
    min_dist = 0.12
    max_labels = 8 if variant == "labeled" else 5
    for node in top_labels:
        if len(placed) >= max_labels:
            break
        x, y = pos[node]
        if any((x - px) ** 2 + (y - py) ** 2 < min_dist**2 for px, py in placed):
            continue
        placed.append((x, y))
        label = email_label(node)
        txt = ax.text(
            x,
            y + 0.03,
            label,
            color="#F6FBFF",
            fontsize=10,
            ha="center",
            va="bottom",
            zorder=5,
            fontweight="medium",
        )
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="#07111D")])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_load_panel(ax: plt.Axes, metrics_df: pd.DataFrame, *, variant: str) -> None:
    ax.set_facecolor("#07111D")
    ordered = metrics_df.sort_values("mean_active_threads_rolling").reset_index(drop=True)
    x = np.arange(len(ordered))
    mean_vals = ordered["mean_active_threads_rolling"].to_numpy(dtype=float)
    p90_vals = ordered["p90_active_threads_rolling"].to_numpy(dtype=float)

    ax.fill_between(x, 0, mean_vals, color="#78A6D1", alpha=0.28)
    ax.plot(x, mean_vals, color="#A7D3F2", linewidth=2.2)
    ax.plot(x, p90_vals, color="#F2C572", linewidth=2.6)
    ax.fill_between(x, mean_vals, p90_vals, color="#F28F3B", alpha=0.16)

    median_mean = float(np.median(mean_vals))
    median_p90 = float(np.median(p90_vals))
    ax.axhline(median_mean, color="#A7D3F2", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(median_p90, color="#F2C572", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.text(len(x) * 0.03, median_mean + 3, "median typical ~50", color="#D6ECFF", fontsize=10)
    ax.text(len(x) * 0.03, median_p90 + 4, "median stress ~105", color="#F7D8A1", fontsize=10)
    if variant == "labeled":
        ax.text(len(x) * 0.03, p90_vals.max() + 13, "Human thread load", color="#F6FBFF", fontsize=16, fontweight="bold")
        ax.text(len(x) * 0.03, p90_vals.max() + 2, "148 Enron custodians, ranked by average active threads", color="#C1D5E8", fontsize=10)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, max(p90_vals) * 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_title(fig: plt.Figure, *, variant: str) -> None:
    if variant != "labeled":
        fig.text(
            0.045,
            0.055,
            "Internal Enron email graph and human thread load from the corpus",
            color="#8EA7BA",
            fontsize=9,
        )
        return
    fig.text(0.045, 0.92, "Enron As A Communication System", color="#F6FBFF", fontsize=28, fontweight="bold")
    fig.text(
        0.045,
        0.875,
        "A real internal email network on the left. Human thread load on the right.",
        color="#C7D9E8",
        fontsize=13,
    )
    fig.text(0.045, 0.06, "Data sources: enron_rosetta_source.parquet and custodian_metrics.csv", color="#8EA7BA", fontsize=9)


def optimize_png(path: Path) -> None:
    if path.suffix.lower() != ".png":
        return
    with Image.open(path) as img:
        quantized = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        quantized.save(path, optimize=True)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    graph, participant_counts = build_internal_graph(args.rosetta_path, top_nodes=args.top_nodes)
    metrics_df = pd.read_csv(args.custodian_metrics_path)

    plt.close("all")
    fig = plt.figure(figsize=(16, 8), facecolor="#07111D")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.95, 1.0], left=0.03, right=0.985, top=0.86, bottom=0.11, wspace=0.08)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_load = fig.add_subplot(gs[0, 1])

    draw_network(ax_net, graph, participant_counts, metrics_df, seed=args.seed, variant=args.variant)
    draw_load_panel(ax_load, metrics_df, variant=args.variant)
    add_title(fig, variant=args.variant)

    fig.savefig(args.output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    optimize_png(args.output_path)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
