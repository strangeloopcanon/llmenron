#!/usr/bin/env python3
"""Analyze Enron message intent types for LLM simulation design."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HF_PARQUET_URLS = [
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00000-of-00003.parquet",
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00001-of-00003.parquet",
    "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-00002-of-00003.parquet",
]

RE_LIST_TAG = re.compile(r"^\s*\[[^\]]+\]\s*")
RE_REPLY_PREFIX = re.compile(r"^\s*(re|fw|fwd)\s*:\s*", re.IGNORECASE)
RE_WHITESPACE = re.compile(r"\s+")

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "trading_market",
        [
            r"\btrade\b",
            r"\btrading\b",
            r"\bmarket\b",
            r"\bbid\b",
            r"\boffer\b",
            r"\bprice\b",
            r"\bgas\b",
            r"\bpower\b",
            r"\beol\b",
        ],
    ),
    (
        "legal_compliance",
        [
            r"\blegal\b",
            r"\bcontract\b",
            r"\bagreement\b",
            r"\bcompliance\b",
            r"\bregulatory\b",
            r"\bferc\b",
            r"\blitigation\b",
        ],
    ),
    (
        "request_action",
        [
            r"\bplease\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bneed you\b",
            r"\brequest\b",
            r"\baction required\b",
            r"\burgent\b",
        ],
    ),
    (
        "decision_approval",
        [
            r"\bapprove\b",
            r"\bapproval\b",
            r"\bsign[- ]off\b",
            r"\bdecision\b",
            r"\bauthorize\b",
            r"\bokay to\b",
        ],
    ),
    (
        "status_update",
        [
            r"\bstatus\b",
            r"\bupdate\b",
            r"\bprogress\b",
            r"\breport\b",
            r"\bsummary\b",
            r"\brecap\b",
            r"\bweekly\b",
            r"\bdaily\b",
        ],
    ),
    (
        "hr_admin",
        [
            r"\bhr\b",
            r"\bbenefits\b",
            r"\bpayroll\b",
            r"\bvacation\b",
            r"\binterview\b",
            r"\brecruit",
            r"\bexpense\b",
            r"\breimbursement\b",
        ],
    ),
    (
        "ops_technical",
        [
            r"\bsystem\b",
            r"\bserver\b",
            r"\boutage\b",
            r"\bissue\b",
            r"\bincident\b",
            r"\bfailure\b",
            r"\bwarning\b",
            r"\bcrawler\b",
            r"\bbug\b",
            r"\bfix\b",
            r"\bsupport\b",
            r"\bdeploy\b",
        ],
    ),
    (
        "meeting_scheduling",
        [
            r"\bmeeting\b",
            r"\bcall\b",
            r"\bconference\b",
            r"\bagenda\b",
            r"\bschedule\b",
            r"\bcalendar\b",
        ],
    ),
    (
        "announcement_broadcast",
        [
            r"\bannouncement\b",
            r"\bnotice\b",
            r"\bnewsletter\b",
            r"\ball employees\b",
            r"\bpolicy\b",
            r"\breminder\b",
        ],
    ),
    (
        "social_personal",
        [
            r"\blunch\b",
            r"\bdinner\b",
            r"\bparty\b",
            r"\bholiday\b",
            r"\bthanks?\b",
            r"\bthank you\b",
            r"\bcongrat",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headers-cache",
        type=Path,
        default=Path("data/enron_headers_1997_2003.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--body-sample-size",
        type=int,
        default=50000,
        help="Random sample of bodies for action-language calibration.",
    )
    parser.add_argument(
        "--body-sample-cache",
        type=Path,
        default=Path("data/enron_body_sample_50k.parquet"),
        help="Local cache for sampled bodies; regenerated if missing.",
    )
    return parser.parse_args()


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


def is_sent_folder(folder: str) -> bool:
    if not isinstance(folder, str):
        return False
    f = folder.lower()
    return "sent" in f or f in {"_sent_mail", "sent_mail", "sent_items", "sent", "all documents"}


def detect_intent(text: str) -> str:
    if not isinstance(text, str):
        return "uncategorized"
    t = text.lower()
    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, t):
                return intent
    return "uncategorized"


def load_headers(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["subject"] = df["subject"].fillna("")
    df["norm_subject"] = df["subject"].map(normalize_subject)
    df["is_reply"] = df["subject"].str.match(r"^\s*(re|fw|fwd)\s*:", case=False, na=False)
    df["sent_day"] = pd.to_datetime(df["sent_ts"]).dt.floor("D")
    df["intent"] = df["norm_subject"].map(detect_intent)
    df["is_sent"] = df["folder"].map(is_sent_folder)
    return df


def load_body_sample(sample_size: int, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url_sql = ", ".join(f"'{u}'" for u in HF_PARQUET_URLS)
    query = f"""
        SELECT sent_ts, subject, body
        FROM (
            SELECT
                CAST(date AS TIMESTAMP) AS sent_ts,
                subject,
                body
            FROM read_parquet([{url_sql}], union_by_name=true)
            WHERE date IS NOT NULL
              AND year(date) BETWEEN 1997 AND 2003
        ) t
        USING SAMPLE {sample_size} ROWS (reservoir)
    """

    last_error: Exception | None = None
    for attempt in range(1, 4):
        con = duckdb.connect()
        con.execute("LOAD httpfs;")
        try:
            df = con.execute(query).fetch_df()
            con.close()
            break
        except Exception as exc:  # pragma: no cover - network-dependent path
            last_error = exc
            con.close()
            if attempt == 3:
                raise
            # jittered backoff to avoid repeated 429s
            time.sleep((2**attempt) + np.random.uniform(0, 1))
    else:
        raise RuntimeError(f"body sample fetch failed: {last_error}")

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def write_outputs(headers: pd.DataFrame, body_sample: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    intent_dist = (
        headers.groupby("intent")
        .agg(
            messages=("intent", "count"),
            share=("intent", lambda x: len(x) / len(headers)),
            reply_rate=("is_reply", "mean"),
            sent_share=("is_sent", "mean"),
            unique_subjects=("norm_subject", "nunique"),
            active_days=("sent_day", "nunique"),
        )
        .reset_index()
        .sort_values("messages", ascending=False)
    )

    intent_density = (
        headers.groupby(["sent_day", "intent"])
        .size()
        .rename("count")
        .reset_index()
        .groupby("intent")["count"]
        .agg(["mean", "median", lambda s: np.percentile(s, 90)])
        .reset_index()
    )
    intent_density.columns = ["intent", "mean_daily_count", "median_daily_count", "p90_daily_count"]

    top_subjects = (
        headers[headers["norm_subject"] != ""]
        .groupby(["intent", "norm_subject"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["intent", "count"], ascending=[True, False])
        .groupby("intent")
        .head(15)
    )

    body = body_sample.copy()
    body["intent"] = body["subject"].map(normalize_subject).map(detect_intent)
    body["has_question"] = body["body"].str.contains(r"\?", regex=True)
    body["has_action_language"] = body["body"].str.contains(
        r"\b(?:please|can you|could you|need to|action required|deadline|by eod|asap)\b",
        case=False,
        regex=True,
    )
    body["has_time_reference"] = body["body"].str.contains(
        r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|by \d{1,2}(?::\d{2})?\s?(?:am|pm))\b",
        case=False,
        regex=True,
    )
    body["has_attachment_language"] = body["body"].str.contains(
        r"\b(?:attached|attachment|see attached|enclosed)\b",
        case=False,
        regex=True,
    )

    def body_archetype(row: pd.Series) -> str:
        if row["has_action_language"] and row["has_time_reference"]:
            return "deadline_request"
        if row["has_action_language"] and row["has_question"]:
            return "request_with_question"
        if row["has_action_language"]:
            return "direct_request"
        if row["has_question"]:
            return "information_request"
        if row["has_attachment_language"]:
            return "document_delivery"
        return "informational_update"

    body["archetype"] = body.apply(body_archetype, axis=1)

    body_intent_features = (
        body.groupby("intent")
        .agg(
            sample_messages=("intent", "count"),
            question_rate=("has_question", "mean"),
            action_language_rate=("has_action_language", "mean"),
            time_reference_rate=("has_time_reference", "mean"),
            median_body_chars=("body", lambda s: int(np.median(s.str.len()))),
        )
        .reset_index()
        .sort_values("sample_messages", ascending=False)
    )
    body_archetypes = (
        body.groupby("archetype")
        .agg(
            sample_messages=("archetype", "count"),
            share=("archetype", lambda x: len(x) / len(body)),
            median_body_chars=("body", lambda s: int(np.median(s.str.len()))),
            question_rate=("has_question", "mean"),
            action_language_rate=("has_action_language", "mean"),
            time_reference_rate=("has_time_reference", "mean"),
        )
        .reset_index()
        .sort_values("sample_messages", ascending=False)
    )

    # Intent mix by reply vs non-reply gives a rough sent/responded work split.
    reply_mix = (
        headers.groupby(["is_reply", "intent"])
        .size()
        .rename("count")
        .reset_index()
    )
    reply_mix["share_within_reply_flag"] = reply_mix["count"] / reply_mix.groupby("is_reply")["count"].transform("sum")

    intent_dist.to_csv(output_dir / "message_intent_distribution.csv", index=False)
    intent_density.to_csv(output_dir / "message_intent_density.csv", index=False)
    top_subjects.to_csv(output_dir / "message_intent_top_subjects.csv", index=False)
    body_intent_features.to_csv(output_dir / "message_intent_body_features_sample.csv", index=False)
    body_archetypes.to_csv(output_dir / "message_body_archetypes_sample.csv", index=False)
    reply_mix.to_csv(output_dir / "message_intent_reply_mix.csv", index=False)

    head_dist = intent_dist.head(8)
    summary_lines = [
        "# Enron Message Intent Snapshot",
        "",
        f"- Total messages analyzed: **{len(headers):,}**",
        f"- Body sample size for language features: **{len(body):,}**",
        "",
        "## Top Intent Buckets",
    ]
    for _, row in head_dist.iterrows():
        summary_lines.append(
            f"- {row['intent']}: {int(row['messages']):,} messages ({row['share']*100:.1f}%), "
            f"reply_rate={row['reply_rate']*100:.1f}%, sent_share={row['sent_share']*100:.1f}%"
        )

    if "uncategorized" in set(intent_dist["intent"]):
        unc = intent_dist[intent_dist["intent"] == "uncategorized"].iloc[0]
        summary_lines.append(
            f"- uncategorized share is {unc['share']*100:.1f}% (expected with keyword taxonomy; "
            "use this as residual class in simulation)."
        )

    summary_lines.extend(
        [
            "",
            "## LLM Simulation Implications",
            "- Generate episodes by sampling intents with these empirical shares.",
            "- Preserve empirical reply-rate differences by intent (requests/coordination have higher reply pressure).",
            "- Match per-intent action-language and time-reference rates for realistic urgency.",
            "- Generate message bodies by archetype (deadline_request, direct_request, informational_update, etc.) to reproduce behavioral load.",
            "",
            "## Files",
            "- message_intent_distribution.csv",
            "- message_intent_density.csv",
            "- message_intent_top_subjects.csv",
            "- message_intent_body_features_sample.csv",
            "- message_body_archetypes_sample.csv",
            "- message_intent_reply_mix.csv",
        ]
    )
    (output_dir / "message_intent_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    headers = load_headers(args.headers_cache)
    body_sample = load_body_sample(
        sample_size=args.body_sample_size,
        cache_path=args.body_sample_cache,
    )
    write_outputs(headers=headers, body_sample=body_sample, output_dir=args.output_dir)
    print(f"Wrote intent analysis artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
