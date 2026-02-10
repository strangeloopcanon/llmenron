#!/usr/bin/env python3
"""Run human inbox topic-juggling analysis on Enron metadata."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

HF_PARQUET_URLS = [
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00000-of-00003.parquet",
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00001-of-00003.parquet",
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00002-of-00003.parquet",
]

RE_LIST_TAG = re.compile(r"^\s*\[[^\]]+\]\s*")
RE_REPLY_PREFIX = re.compile(r"^\s*(re|fw|fwd)\s*:\s*", re.IGNORECASE)
RE_WHITESPACE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custodian-tsv",
        type=Path,
        default=Path("raw/enrondata_repo/data/misc/edo_enron-custodians-data.tsv"),
        help="Path to enron custodian id -> title TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for analysis artifacts",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=200,
        help="Minimum mailbox message count required for inferential analysis",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Rolling window for active thread concurrency",
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=Path("data/enron_headers_1997_2003.parquet"),
        help="Local cache file for remotely queried Enron metadata headers",
    )
    return parser.parse_args()


def load_enron_metadata(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    url_sql = ", ".join(f"'{u}'" for u in HF_PARQUET_URLS)

    query = f"""
        WITH src AS (
            SELECT
                lower(trim(message_id)) AS message_id,
                subject,
                lower(trim("from")) AS from_email,
                CAST(date AS TIMESTAMP) AS sent_ts,
                file_name
            FROM read_parquet([{url_sql}], union_by_name=true)
        )
        SELECT
            message_id,
            subject,
            from_email,
            sent_ts,
            file_name,
            lower(split_part(file_name, '/', 1)) AS custodian_id,
            lower(split_part(file_name, '/', 2)) AS folder
        FROM src
        WHERE sent_ts IS NOT NULL
          AND file_name IS NOT NULL
          AND split_part(file_name, '/', 1) <> ''
          AND year(sent_ts) BETWEEN 1997 AND 2003
    """
    df = con.execute(query).fetch_df()
    con.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def normalize_subject(subject: str) -> str:
    if not isinstance(subject, str):
        return ""
    s = subject.strip()
    if not s:
        return ""
    while True:
        updated = RE_LIST_TAG.sub("", s)
        updated = RE_REPLY_PREFIX.sub("", updated)
        updated = RE_WHITESPACE.sub(" ", updated).strip().lower()
        if updated == s:
            break
        s = updated
    return s


def assign_subject_threads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["norm_subject"] = out["subject"].map(normalize_subject)
    out["sent_day"] = pd.to_datetime(out["sent_ts"]).dt.floor("D")
    out = out.sort_values(["custodian_id", "norm_subject", "sent_ts", "message_id"])

    out["thread_segment"] = 0
    has_subject = out["norm_subject"] != ""
    subject_df = out.loc[has_subject, ["custodian_id", "norm_subject", "sent_day"]].copy()
    day_gap = (
        subject_df.groupby(["custodian_id", "norm_subject"])["sent_day"].diff().dt.days.fillna(0)
    )
    split = (day_gap > 90).astype(int)
    out.loc[has_subject, "thread_segment"] = split.groupby(
        [subject_df["custodian_id"], subject_df["norm_subject"]]
    ).cumsum()

    def make_thread_id(row: pd.Series) -> str:
        if row["norm_subject"]:
            base = f"{row['custodian_id']}|{row['norm_subject']}|{int(row['thread_segment'])}"
        else:
            base = f"{row['custodian_id']}|mid|{row['message_id']}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    out["thread_id"] = out.apply(make_thread_id, axis=1)
    return out


def rolling_active_threads_per_person(
    df: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for custodian_id, person_df in df.groupby("custodian_id", sort=False):
        by_day = (
            person_df.groupby("sent_day")["thread_id"]
            .agg(lambda values: set(values.tolist()))
            .sort_index()
        )
        q: deque[tuple[pd.Timestamp, set[str]]] = deque()
        counts: Counter[str] = Counter()

        for day, thread_set in by_day.items():
            q.append((day, thread_set))
            for tid in thread_set:
                counts[tid] += 1

            cutoff = day - pd.Timedelta(days=window_days - 1)
            while q and q[0][0] < cutoff:
                _, old_threads = q.popleft()
                for tid in old_threads:
                    counts[tid] -= 1
                    if counts[tid] <= 0:
                        counts.pop(tid, None)

            records.append(
                {
                    "custodian_id": custodian_id,
                    "sent_day": day,
                    "active_threads_rolling": len(counts),
                }
            )
    return pd.DataFrame.from_records(records)


def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def title_to_seniority(title: str) -> tuple[str, int]:
    if not isinstance(title, str) or not title.strip():
        return ("unknown", 0)
    t = title.lower()
    if "vice president" in t or re.search(r"\bvp\b", t):
        return ("vice_president", 4)
    if any(token in t for token in ("chief", "ceo", "president", "chairman", "coo", "cfo")):
        return ("executive", 5)
    if "director" in t:
        return ("director", 3)
    if "manager" in t:
        return ("manager", 2)
    return ("individual_contributor", 1)


def load_custodian_titles(path: Path) -> pd.DataFrame:
    titles = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["custodian_id", "email", "name", "title"],
        dtype=str,
        keep_default_na=False,
    )
    titles["custodian_id"] = titles["custodian_id"].str.lower().str.strip()
    mapped = titles["title"].map(title_to_seniority).apply(pd.Series)
    mapped.columns = ["seniority_tier", "seniority_score"]
    return pd.concat([titles, mapped], axis=1)


def is_sent_folder(folder: str) -> bool:
    if not isinstance(folder, str):
        return False
    f = folder.lower()
    return (
        "sent" in f
        or f in {"_sent_mail", "sent_mail", "sent_items", "sent", "all documents"}
    )


def run_analysis(
    metadata_df: pd.DataFrame,
    title_df: pd.DataFrame,
    min_messages: int,
    window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    threaded = assign_subject_threads(metadata_df)
    threaded["is_sent_folder"] = threaded["folder"].map(is_sent_folder)

    rolling = rolling_active_threads_per_person(threaded, window_days=window_days)

    per_custodian = threaded.groupby("custodian_id").agg(
        total_messages=("message_id", "count"),
        sent_messages=("is_sent_folder", "sum"),
        unique_threads=("thread_id", "nunique"),
        first_date=("sent_day", "min"),
        last_date=("sent_day", "max"),
    )
    per_custodian["received_proxy_messages"] = (
        per_custodian["total_messages"] - per_custodian["sent_messages"]
    )
    span_days = (
        (per_custodian["last_date"] - per_custodian["first_date"]).dt.days.clip(lower=1)
    )
    per_custodian["msgs_per_day"] = per_custodian["total_messages"] / span_days

    entropy_df = (
        threaded.groupby("custodian_id")["thread_id"]
        .apply(shannon_entropy)
        .rename("thread_entropy")
        .to_frame()
    )
    entropy_df["effective_topics"] = np.exp(entropy_df["thread_entropy"])

    active_summary = rolling.groupby("custodian_id").agg(
        active_days=("sent_day", "count"),
        mean_active_threads_rolling=("active_threads_rolling", "mean"),
        p90_active_threads_rolling=("active_threads_rolling", lambda x: np.percentile(x, 90)),
        max_active_threads_rolling=("active_threads_rolling", "max"),
    )

    metrics = (
        per_custodian.join(entropy_df)
        .join(active_summary)
        .reset_index()
        .merge(
            title_df[
                ["custodian_id", "email", "name", "title", "seniority_tier", "seniority_score"]
            ],
            on="custodian_id",
            how="left",
        )
    )
    metrics["seniority_tier"] = metrics["seniority_tier"].fillna("unknown")
    metrics["seniority_score"] = metrics["seniority_score"].fillna(0).astype(int)
    metrics["title"] = metrics["title"].fillna("")
    metrics["name"] = metrics["name"].fillna("")

    inferential = metrics[metrics["total_messages"] >= min_messages].copy()
    inferential = inferential[inferential["active_days"] >= 20].copy()

    group_summary = (
        inferential.groupby("seniority_tier")
        .agg(
            custodians=("custodian_id", "nunique"),
            median_mean_active_threads=("mean_active_threads_rolling", "median"),
            median_p90_active_threads=("p90_active_threads_rolling", "median"),
            median_total_messages=("total_messages", "median"),
            median_effective_topics=("effective_topics", "median"),
        )
        .reset_index()
        .sort_values("median_mean_active_threads", ascending=False)
    )

    if len(inferential) >= 5:
        spearman = stats.spearmanr(
            inferential["seniority_score"],
            inferential["mean_active_threads_rolling"],
        )
        spearman_volume_mean = stats.spearmanr(
            inferential["total_messages"],
            inferential["mean_active_threads_rolling"],
        )
        spearman_volume_p90 = stats.spearmanr(
            inferential["total_messages"],
            inferential["p90_active_threads_rolling"],
        )
        X = inferential[["seniority_score", "total_messages"]].copy()
        X["log_total_messages"] = np.log1p(X["total_messages"])
        X = sm.add_constant(X[["seniority_score", "log_total_messages"]])
        y = inferential["mean_active_threads_rolling"]
        model = sm.OLS(y, X).fit()
        senior_coef = float(model.params["seniority_score"])
        senior_pvalue = float(model.pvalues["seniority_score"])
        senior_ci_low, senior_ci_high = model.conf_int().loc["seniority_score"].tolist()
        categorical_model = smf.ols(
            "mean_active_threads_rolling ~ log_total_messages + C(seniority_tier)",
            data=inferential.assign(log_total_messages=np.log1p(inferential["total_messages"])),
        ).fit()
        categorical_anova = anova_lm(categorical_model, typ=2)
        seniority_block_pvalue = float(categorical_anova.loc["C(seniority_tier)", "PR(>F)"])
    else:
        spearman = stats.spearmanr([0, 1], [0, 1])
        spearman_volume_mean = stats.spearmanr([0, 1], [0, 1])
        spearman_volume_p90 = stats.spearmanr([0, 1], [0, 1])
        senior_coef = math.nan
        senior_pvalue = math.nan
        senior_ci_low, senior_ci_high = (math.nan, math.nan)
        seniority_block_pvalue = math.nan

    n_levels_mean = {
        "q25": float(inferential["mean_active_threads_rolling"].quantile(0.25)),
        "q50": float(inferential["mean_active_threads_rolling"].quantile(0.50)),
        "q75": float(inferential["mean_active_threads_rolling"].quantile(0.75)),
        "q90": float(inferential["mean_active_threads_rolling"].quantile(0.90)),
    }
    n_levels_p90 = {
        "q25": float(inferential["p90_active_threads_rolling"].quantile(0.25)),
        "q50": float(inferential["p90_active_threads_rolling"].quantile(0.50)),
        "q75": float(inferential["p90_active_threads_rolling"].quantile(0.75)),
        "q90": float(inferential["p90_active_threads_rolling"].quantile(0.90)),
    }

    top_concurrency = metrics.sort_values(
        ["mean_active_threads_rolling", "p90_active_threads_rolling"],
        ascending=False,
    ).head(10)
    top_display = top_concurrency[
        [
            "custodian_id",
            "name",
            "title",
            "mean_active_threads_rolling",
            "p90_active_threads_rolling",
            "total_messages",
        ]
    ].copy()

    key = {
        "rows_total": int(len(threaded)),
        "custodians_total": int(metrics["custodian_id"].nunique()),
        "custodians_inferential": int(inferential["custodian_id"].nunique()),
        "median_mean_active_threads_all": float(metrics["mean_active_threads_rolling"].median()),
        "median_p90_active_threads_all": float(metrics["p90_active_threads_rolling"].median()),
        "spearman_rho_seniority_vs_mean_active": float(spearman.correlation),
        "spearman_pvalue": float(spearman.pvalue),
        "spearman_rho_volume_vs_mean_active": float(spearman_volume_mean.correlation),
        "spearman_pvalue_volume_vs_mean_active": float(spearman_volume_mean.pvalue),
        "spearman_rho_volume_vs_p90_active": float(spearman_volume_p90.correlation),
        "spearman_pvalue_volume_vs_p90_active": float(spearman_volume_p90.pvalue),
        "ols_coef_seniority_score": senior_coef,
        "ols_pvalue_seniority_score": senior_pvalue,
        "ols_ci95_low_seniority_score": float(senior_ci_low),
        "ols_ci95_high_seniority_score": float(senior_ci_high),
        "anova_pvalue_seniority_block": seniority_block_pvalue,
        "recommended_N_mean_q25": n_levels_mean["q25"],
        "recommended_N_mean_q50": n_levels_mean["q50"],
        "recommended_N_mean_q75": n_levels_mean["q75"],
        "recommended_N_mean_q90": n_levels_mean["q90"],
        "recommended_N_p90_q25": n_levels_p90["q25"],
        "recommended_N_p90_q50": n_levels_p90["q50"],
        "recommended_N_p90_q75": n_levels_p90["q75"],
        "recommended_N_p90_q90": n_levels_p90["q90"],
    }

    summary_lines = [
        "# Human Topic-Juggling Results (Enron)",
        "",
        f"- Messages analyzed: **{key['rows_total']:,}**",
        f"- Custodians analyzed: **{key['custodians_total']}**",
        f"- Inferential sample (>= {min_messages} messages, >=20 active days): **{key['custodians_inferential']} custodians**",
        f"- Median mean active threads ({window_days}-day rolling): **{key['median_mean_active_threads_all']:.2f}**",
        f"- Median 90th percentile active threads: **{key['median_p90_active_threads_all']:.2f}**",
        "",
        "## Rank/Success Proxy Effect",
        f"- Spearman rho(seniority score, mean active threads): **{key['spearman_rho_seniority_vs_mean_active']:.3f}** (p={key['spearman_pvalue']:.4f})",
        (
            "- OLS coefficient for seniority score "
            f"(controlling for log total messages): **{key['ols_coef_seniority_score']:.3f}** "
            f"(95% CI {key['ols_ci95_low_seniority_score']:.3f} to {key['ols_ci95_high_seniority_score']:.3f}, "
            f"p={key['ols_pvalue_seniority_score']:.4f})"
        ),
        f"- ANOVA p-value for seniority tiers (controlling for log volume): **{key['anova_pvalue_seniority_block']:.4f}**",
        "",
        "## What Actually Predicts Juggling",
        f"- Spearman rho(total messages, mean active threads): **{key['spearman_rho_volume_vs_mean_active']:.3f}** (p={key['spearman_pvalue_volume_vs_mean_active']:.2e})",
        f"- Spearman rho(total messages, p90 active threads): **{key['spearman_rho_volume_vs_p90_active']:.3f}** (p={key['spearman_pvalue_volume_vs_p90_active']:.2e})",
        "",
        "## Human-Derived N Levels For Next Experiment",
        (
            "- Mean-active quantiles (q25/q50/q75/q90): "
            f"**{key['recommended_N_mean_q25']:.1f} / {key['recommended_N_mean_q50']:.1f} / "
            f"{key['recommended_N_mean_q75']:.1f} / {key['recommended_N_mean_q90']:.1f}**"
        ),
        (
            "- Stress (p90-active) quantiles (q25/q50/q75/q90): "
            f"**{key['recommended_N_p90_q25']:.1f} / {key['recommended_N_p90_q50']:.1f} / "
            f"{key['recommended_N_p90_q75']:.1f} / {key['recommended_N_p90_q90']:.1f}**"
        ),
        "",
        "## Top Custodians by Mean Active Threads",
    ]
    for _, row in top_display.iterrows():
        label = row["name"] if row["name"] else row["custodian_id"]
        title = row["title"] if row["title"] else "n/a"
        summary_lines.append(
            f"- {label} ({row['custodian_id']}, {title}): "
            f"mean_active={row['mean_active_threads_rolling']:.2f}, "
            f"p90_active={row['p90_active_threads_rolling']:.2f}, "
            f"messages={int(row['total_messages']):,}"
        )

    return metrics, group_summary, key, "\n".join(summary_lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_enron_metadata(cache_path=args.metadata_cache)
    title_df = load_custodian_titles(args.custodian_tsv)
    metrics, group_summary, key, summary_md = run_analysis(
        metadata_df=metadata_df,
        title_df=title_df,
        min_messages=args.min_messages,
        window_days=args.window_days,
    )

    metrics.to_csv(args.output_dir / "custodian_metrics.csv", index=False)
    group_summary.to_csv(args.output_dir / "seniority_group_summary.csv", index=False)
    pd.DataFrame([key]).to_csv(args.output_dir / "key_results.csv", index=False)
    (args.output_dir / "human_analysis_summary.md").write_text(summary_md, encoding="utf-8")

    print(summary_md)
    print(f"Wrote artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
