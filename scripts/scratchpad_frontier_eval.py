#!/usr/bin/env python3
"""Prepare and run scratchpad-only inbox capacity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

RE_LIST_TAG = re.compile(r"^\s*\[[^\]]+\]\s*")
RE_REPLY_PREFIX = re.compile(r"^\s*(re|fw|fwd)\s*:\s*", re.IGNORECASE)
RE_WHITESPACE = re.compile(r"\s+")
RE_PROJECT_CODE = re.compile(r"\b([A-Z][0-9]{3})\b")
RE_OWNER = re.compile(r"\bOwner ([A-Z][a-z]+ [A-Z][a-z]+)\b")
RE_DUE_DATE = re.compile(r"\bDue (\d{4}-\d{2}-\d{2})\b")

PRIORITY_SLA_MIN = {"P0": 5.0, "P1": 20.0, "P2": 120.0}
REQUIRED_PRIORITIES = {"P0", "P1", "P2"}
REQUIRED_REPLY_TYPES = {"NONE", "ACK", "ANSWER", "REQUEST_INFO", "REDIRECT"}
REQUIRED_BINDING_DECISIONS = {"bound", "clarify", "unbound_guess"}
REQUIRED_BINDING_SOURCES = {"current_email", "scratchpad", "thread_state", "none"}
BANNED_COMMITMENTS = ["i approved", "already approved", "done", "fixed", "completed", "shipped"]
RUBRIC_STRICT = (
    "Decision rubric (strict): "
    "P0 = urgent deadline work; message has deadline/urgency language (urgent, asap, immediately, by EOD, deadline) "
    "and actionable request/approval/compliance intent. "
    "P1 = actionable but not immediate; requests, questions needing follow-up, legal/compliance/approval without urgent timing. "
    "P2 = informational/status/social/document-delivery with no immediate action. "
    "Reply types: ANSWER for deadline/action/approval/compliance requests, REQUEST_INFO for information-gathering questions, "
    "ACK for informational updates/announcements/social/document delivery, NONE when no response needed, REDIRECT when out-of-domain."
)
RUBRIC_MEANING = (
    "Decision rubric (meaning): "
    "P0 = genuinely urgent and time-sensitive (e.g., same-day deadlines, blocking incidents, escalations). "
    "P1 = important and actionable but not an emergency; should be handled soon (typically today). "
    "P2 = routine/FYI/social or non-urgent updates; can wait. "
    "Reply types: ANSWER when you can directly comply/decide/respond, REQUEST_INFO when you need clarifying information, "
    "ACK when acknowledging an update, NONE when no reply is needed, REDIRECT when the request belongs to a different owner/team."
)
MEMORY_CONTRACT = (
    "Memory contract: You are evaluated on recalling thread facts using ONLY your scratchpad. "
    "When you see a project code like A742 (pattern [A-Z][0-9]{3}), record it in your scratchpad keyed by thread_id. "
    "Later follow-ups may omit the code; use your scratchpad to include the code in facts_used and/or the reply."
)

TASK_PRIORITY = {
    "deadline_request": "P0",
    "request_with_question": "P1",
    "direct_request": "P1",
    "information_request": "P1",
    "informational_update": "P2",
    "document_delivery": "P2",
}

TASK_REPLY = {
    "deadline_request": "ANSWER",
    "request_with_question": "ANSWER",
    "direct_request": "ANSWER",
    "information_request": "REQUEST_INFO",
    "informational_update": "ACK",
    "document_delivery": "ACK",
}

DEFAULT_MODELS = ["gpt-5.2", "gpt-5-mini"]


def model_supports_temperature(model: str) -> bool:
    name = str(model or "").strip().lower()
    return name not in {"gpt-5-mini"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["prepare", "run"],
        default="prepare",
        help="prepare: generate scenario + run matrix. run: execute a prepared scenario.",
    )
    parser.add_argument("--scenario-tag", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/scratchpad_frontier/scratchpad_setup"))
    parser.add_argument("--scenario-dir", type=Path, default=None, help="Required for --mode run.")

    # Data calibration
    parser.add_argument("--key-results-csv", type=Path, default=Path("results/summaries/key_results.csv"))
    parser.add_argument(
        "--intent-dist-csv",
        type=Path,
        default=Path("experiments/reference_data/message_intent_distribution.csv"),
    )
    parser.add_argument(
        "--archetype-csv",
        type=Path,
        default=Path("experiments/reference_data/message_body_archetypes_sample.csv"),
    )
    parser.add_argument(
        "--body-sample-cache",
        type=Path,
        default=Path("data/enron_body_sample_20k.parquet"),
    )
    parser.add_argument(
        "--filler-source",
        choices=["template", "enron"],
        default="template",
        help=(
            "Controls optional 'realistic' filler appended to synthetic templates. "
            "template = no Enron text (recommended for leakage-robust eval); "
            "enron = append sanitized Enron snippets for realism."
        ),
    )
    parser.add_argument(
        "--context-spotcheck-md",
        type=Path,
        default=Path("experiments/reference_data/context_spotcheck_summary.md"),
    )

    # Scenario size
    parser.add_argument("--employees", type=int, default=24)
    parser.add_argument("--n-values", default="35,50,70,105,140")
    parser.add_argument("--episodes-per-n", type=int, default=40)
    parser.add_argument("--messages-per-episode", type=int, default=220)
    parser.add_argument(
        "--context-memory-rate",
        type=float,
        default=-1.0,
        help="Share of messages requiring prior context. -1 reads from spotcheck markdown.",
    )

    # Run settings
    parser.add_argument("--agent", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default=os.getenv("RESEARCH_MODEL", "gpt-5.2"))
    parser.add_argument(
        "--memory-policy",
        choices=["scratchpad_only", "thread_state"],
        default="scratchpad_only",
        help=(
            "scratchpad_only = free-text scratchpad carried across all threads; "
            "thread_state = system-maintained per-thread facts (project code, owner, due date) "
            "passed only for the current thread."
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["meaning", "strict"],
        default="meaning",
        help="meaning = tests judgment; strict = tests rubric compliance.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--openai-timeout-sec", type=int, default=90)
    parser.add_argument("--openai-max-attempts", type=int, default=3)
    parser.add_argument("--openai-reasoning-mode", choices=["auto", "high"], default="auto")
    parser.add_argument("--openai-max-output-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-calls", type=int, default=100000)
    parser.add_argument("--max-consecutive-api-errors", type=int, default=5)
    parser.add_argument("--score-threshold-q", type=float, default=0.75)
    parser.add_argument("--p0-sla-threshold", type=float, default=0.90)
    parser.add_argument("--input-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--scratchpad-char-budget", type=int, default=5000)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated subset of prepared episode_ids to run from scenario_dir.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def render_thread_state(thread_id: str, facts: dict[str, str]) -> dict[str, Any]:
    if not facts:
        return {}
    out: dict[str, Any] = {"thread_id": thread_id}
    for key in ["project_code", "owner", "due_date"]:
        value = str(facts.get(key, "")).strip()
        if value:
            out[key] = value
    return out


def normalize_project_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = RE_PROJECT_CODE.search(text)
    return match.group(1) if match else ""


def resolve_gold_project_code(
    *,
    explicit_value: Any,
    gold_required_key: Any,
    gold_required_value: Any,
    subject: str,
    body: str,
) -> str:
    explicit = normalize_project_code(explicit_value)
    if explicit:
        return explicit
    if str(gold_required_key or "").strip() == "project_code":
        from_probe = normalize_project_code(gold_required_value)
        if from_probe:
            return from_probe
    extracted = extract_explicit_thread_facts(subject=subject, body=body).get("project_code", "")
    return normalize_project_code(extracted)


def find_project_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = RE_PROJECT_CODE.search(text)
    return match.group(1) if match else ""


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except Exception:
            obj = None
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
        if value.strip():
            return [value.strip()]
    return []


def append_csv_row(path: Path, row: dict[str, Any], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    active_fieldnames = list(fieldnames)
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            existing_header = next(reader, [])
        if existing_header and set(existing_header) != set(fieldnames):
            raise RuntimeError(
                f"CSV schema mismatch for {path}: existing header differs from row fields. "
                "Use a fresh run dir or resume a run created by the current harness."
            )
        if existing_header:
            active_fieldnames = existing_header
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=active_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in active_fieldnames})


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_context_memory_rate(path: Path) -> float:
    if not path.exists():
        return 0.30
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Context proxy \(>=2 cue[s]?\): \*\*([0-9.]+)%\*\*", text)
    if not match:
        return 0.30
    pct = float(match.group(1))
    return max(0.0, min(1.0, pct / 100.0))


def parse_n_values(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No N values provided")
    return vals


def load_key_stats(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing key results file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Empty key results file: {path}")
    row = df.iloc[0]
    required = [
        "recommended_N_mean_q25",
        "recommended_N_mean_q50",
        "recommended_N_mean_q75",
        "recommended_N_p90_q50",
        "median_mean_active_threads_all",
        "median_p90_active_threads_all",
    ]
    missing = [c for c in required if c not in row.index]
    if missing:
        raise RuntimeError(f"Missing columns in key stats: {missing}")
    out = {k: float(row[k]) for k in required}
    return out


def load_intent_distribution(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing intent distribution file: {path}")
    df = pd.read_csv(path)
    need = {"intent", "share"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"{path} must include columns: {sorted(need)}")
    dist = df[["intent", "share"]].copy()
    dist = dist[dist["share"] > 0].copy()
    dist["share"] = dist["share"] / dist["share"].sum()
    return dist


def load_archetype_distribution(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Conservative fallback; still deterministic and explicit.
        return pd.DataFrame(
            {
                "archetype": list(TASK_PRIORITY.keys()),
                "share": [0.26, 0.09, 0.18, 0.14, 0.29, 0.04],
            }
        )
    df = pd.read_csv(path)
    if not {"archetype", "share"}.issubset(set(df.columns)):
        raise RuntimeError(f"{path} must include columns: archetype, share")
    keep = df[df["archetype"].isin(TASK_PRIORITY.keys())][["archetype", "share"]].copy()
    if keep.empty:
        raise RuntimeError(f"{path} did not include recognized archetypes")
    keep["share"] = keep["share"] / keep["share"].sum()
    return keep


def load_body_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing body sample parquet: {path}")
    df = pd.read_parquet(path)
    for col in ["subject", "body"]:
        if col not in df.columns:
            raise RuntimeError(f"{path} missing required column: {col}")
    out = df[["subject", "body"]].copy()
    out["subject"] = out["subject"].fillna("").astype(str)
    out["body"] = out["body"].fillna("").astype(str)
    out["norm_subject"] = out["subject"].map(normalize_subject)
    return out


def pick_owner_name(rng: np.random.Generator) -> str:
    first = [
        "Alex",
        "Jordan",
        "Sam",
        "Taylor",
        "Morgan",
        "Casey",
        "Jamie",
        "Riley",
        "Cameron",
        "Avery",
    ]
    last = [
        "Lee",
        "Patel",
        "Kim",
        "Garcia",
        "Chen",
        "Brown",
        "Shah",
        "Miller",
        "Clark",
        "Khan",
    ]
    return f"{rng.choice(first)} {rng.choice(last)}"


def make_project_code(rng: np.random.Generator) -> str:
    letter = str(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    return f"{letter}{int(rng.integers(100, 999))}"


def make_due_date(rng: np.random.Generator) -> str:
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    return f"2001-{month:02d}-{day:02d}"


def choose_urgency(rng: np.random.Generator, archetype: str) -> tuple[str, str]:
    if archetype == "deadline_request":
        if rng.random() < 0.65:
            return ("urgent", "today")
        return ("soon", "this week")
    if archetype in {"request_with_question", "direct_request", "information_request"}:
        return ("normal", "this week")
    return ("low", "none")


TASK_CARDS_BY_ARCHETYPE: dict[str, list[dict[str, str]]] = {
    "deadline_request": [
        {
            "title": "Approval needed",
            "detail": "Need your approval on the revised plan so we can proceed without slipping the due date.",
        },
        {
            "title": "Blocking incident",
            "detail": "A blocking issue is preventing progress; need a go/no-go decision on the fallback option.",
        },
        {
            "title": "Board deck numbers",
            "detail": "Please confirm the final numbers for the deck (key metrics + narrative) before distribution.",
        },
        {
            "title": "Contract signature",
            "detail": "Need sign-off on the final contract terms so we can execute before the deadline.",
        },
    ],
    "request_with_question": [
        {
            "title": "Decision recommendation",
            "detail": "Which option should we choose and why? Please recommend a path with 1-2 key tradeoffs.",
        },
        {
            "title": "Risk assessment",
            "detail": "Can you assess the main risks and propose mitigations before we commit?",
        },
        {
            "title": "Scope tradeoff",
            "detail": "If we need to cut scope, what would you remove first? Please recommend a minimal viable plan.",
        },
    ],
    "direct_request": [
        {
            "title": "Quick action needed",
            "detail": "Please take the requested action and confirm once done (or tell me what is blocked).",
        },
        {
            "title": "Review and respond",
            "detail": "Please review the latest draft and reply with any edits or approval to proceed.",
        },
        {
            "title": "Send requested data",
            "detail": "Please send the requested numbers (or point me to where they live) so I can finish the update.",
        },
    ],
    "information_request": [
        {
            "title": "Status check",
            "detail": "Question: what is the current status, and what is the next step?",
        },
        {
            "title": "Owner clarification",
            "detail": "Question: who owns the next action here, and what is the expected timeline?",
        },
        {
            "title": "Blocking details",
            "detail": "Question: what is blocked right now, and what do you need from me to unblock it?",
        },
    ],
    "document_delivery": [
        {
            "title": "Draft attached",
            "detail": "Sharing the latest draft for reference. No action required yet unless you spot an issue.",
        },
        {
            "title": "Notes for reference",
            "detail": "Sharing notes and context for reference. No response needed unless questions.",
        },
    ],
    "informational_update": [
        {
            "title": "FYI update",
            "detail": "FYI: we made progress and will follow up with a fuller update later.",
        },
        {
            "title": "Schedule change",
            "detail": "FYI: timeline shifted; we will adjust and confirm the new plan.",
        },
        {
            "title": "Heads up",
            "detail": "FYI: there is a minor change; no action needed from you right now.",
        },
    ],
}

TEMPLATE_FILLERS = [
    "Thanks.",
    "Regards.",
    "Best.",
    "Noted for reference.",
    "Looping you in for visibility.",
    "Sharing in case useful.",
]


def pick_task_card(rng: np.random.Generator, archetype: str) -> tuple[str, str]:
    cards = TASK_CARDS_BY_ARCHETYPE.get(archetype) or TASK_CARDS_BY_ARCHETYPE["informational_update"]
    card = cards[int(rng.integers(0, len(cards)))]
    return str(card["title"]), str(card["detail"])


def pick_template_filler(rng: np.random.Generator) -> str:
    return str(TEMPLATE_FILLERS[int(rng.integers(0, len(TEMPLATE_FILLERS)))])


def pick_filler_body(*, filler_source: str, body_pool: pd.DataFrame, rng: np.random.Generator) -> str:
    if filler_source == "enron":
        _, base_body = build_realistic_snippet(body_pool, rng)
        return base_body
    return pick_template_filler(rng)


def build_employee_profiles(stats: dict[str, float], employees: int, rng: np.random.Generator) -> pd.DataFrame:
    if employees < 6:
        raise ValueError("--employees must be >= 6")
    q25 = stats["recommended_N_mean_q25"]
    q50 = stats["recommended_N_mean_q50"]
    q75 = stats["recommended_N_mean_q75"]

    rows: list[dict[str, Any]] = []
    for i in range(employees):
        p = (i + 1) / employees
        if p <= 0.35:
            tier = "low_volume"
            target = max(8.0, rng.normal(q25, 6.0))
            volume = int(max(180, rng.normal(500, 140)))
        elif p <= 0.80:
            tier = "mid_volume"
            target = max(12.0, rng.normal(q50, 9.0))
            volume = int(max(300, rng.normal(900, 220)))
        else:
            tier = "high_volume"
            target = max(20.0, rng.normal(q75, 14.0))
            volume = int(max(500, rng.normal(1600, 350)))
        rows.append(
            {
                "employee_id": f"emp-{i+1:03d}",
                "persona_name": pick_owner_name(rng),
                "workload_tier": tier,
                "target_active_threads": round(float(target), 2),
                "monthly_message_volume_proxy": int(volume),
                "seniority_proxy": int(rng.integers(1, 6)),
            }
        )
    return pd.DataFrame(rows)


def build_realistic_snippet(body_pool: pd.DataFrame, rng: np.random.Generator) -> tuple[str, str]:
    row = body_pool.iloc[int(rng.integers(0, len(body_pool)))]
    subject = str(row["subject"]).strip()
    body = str(row["body"]).strip()
    if not subject:
        subject = "follow up"
    if len(body) > 260:
        body = body[:260]
    return subject, body


def sanitize_filler(text: str) -> str:
    """Remove obvious urgency/action cues from filler so archetype templates aren't contradicted."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    t = re.sub(r"[?]+", ".", t)
    t = re.sub(r"\b(?:urgent|asap|immediately|deadline|by eod)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:please|can you|could you|need you|action required|respond)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:220]


def compose_message(
    *,
    base_subject: str,
    base_body: str,
    task_detail: str,
    project_code: str,
    owner: str,
    due_date: str,
    urgency_word: str,
    timing_word: str,
    archetype: str,
    needs_memory: bool,
    anchor_msg: bool,
) -> tuple[str, str, str, str]:
    priority = TASK_PRIORITY.get(archetype, "P2")
    reply_type = TASK_REPLY.get(archetype, "ACK")
    filler = sanitize_filler(base_body)

    def body_line(line: str) -> str:
        if filler:
            return f"{line} {filler}"
        return line

    if anchor_msg:
        prefix = f"Thread anchor. Project {project_code}. Owner {owner}. Due {due_date}."
        if archetype == "deadline_request":
            subject = f"[URGENT] {base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail} Need action by {due_date}. This is {urgency_word}.")
        elif archetype == "request_with_question":
            subject = f"{base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail} Please take a look and reply with your recommendation.")
        elif archetype == "direct_request":
            subject = f"{base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail} Please handle this {timing_word}.")
        elif archetype == "information_request":
            subject = f"{base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail}")
        elif archetype == "document_delivery":
            subject = f"{base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail}")
        else:  # informational_update
            subject = f"{base_subject} | {project_code}"
            body = body_line(f"{prefix} {task_detail}")
        required_key = "none"
    elif needs_memory:
        subject = f"Re: {base_subject}"
        if archetype == "deadline_request":
            body = body_line(
                f"Quick follow-up on the prior thread. {task_detail} Please handle this {timing_word}. "
                f"Deadline is tight; treat as {urgency_word}."
            )
        elif archetype == "request_with_question":
            body = body_line(f"Quick follow-up. {task_detail} Can you share your recommendation {timing_word}?")
        elif archetype == "direct_request":
            body = body_line(f"Quick follow-up. {task_detail} Please take action {timing_word} and confirm once done.")
        elif archetype == "information_request":
            body = body_line(f"Quick follow-up. {task_detail}")
        else:
            # Avoid generating incoherent "memory-required" P2 messages.
            body = body_line("Quick follow-up for awareness. No action needed.")
        required_key = "project_code"
    else:
        if archetype == "deadline_request":
            subject = f"[URGENT] Re: {base_subject}"
            body = body_line(
                f"Follow-up for project {project_code}. {task_detail} Need action by {due_date}; please handle {timing_word}."
            )
        elif archetype == "request_with_question":
            subject = f"Re: {base_subject}"
            body = body_line(f"Follow-up for project {project_code}. {task_detail} Please reply {timing_word}.")
        elif archetype == "direct_request":
            subject = f"Re: {base_subject}"
            body = body_line(f"Follow-up for project {project_code}. {task_detail} Please take action {timing_word}.")
        elif archetype == "information_request":
            subject = f"Re: {base_subject}"
            body = body_line(f"Follow-up on project {project_code}. {task_detail}")
        elif archetype == "document_delivery":
            subject = f"Re: {base_subject}"
            body = body_line(f"Sharing for project {project_code}. {task_detail}")
        else:  # informational_update
            subject = f"Re: {base_subject}"
            body = body_line(f"FYI update on project {project_code}. {task_detail}")
        required_key = "none"
    return subject, body, priority, reply_type


def generate_scenario(
    *,
    output_dir: Path,
    scenario_tag: str,
    seed: int,
    n_values: list[int],
    episodes_per_n: int,
    messages_per_episode: int,
    employees: pd.DataFrame,
    body_pool: pd.DataFrame,
    filler_source: str,
    archetype_dist: pd.DataFrame,
    context_memory_rate: float,
) -> Path:
    rng = np.random.default_rng(seed)
    scenario_dir = output_dir / scenario_tag
    scenario_dir.mkdir(parents=True, exist_ok=True)

    archetypes = archetype_dist["archetype"].tolist()
    archetype_w = archetype_dist["share"].to_numpy()

    if messages_per_episode <= 0:
        raise ValueError("--messages-per-episode must be positive")

    message_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    thread_rows: list[dict[str, Any]] = []

    episode_id = 0
    for n_threads in n_values:
        for _ in range(episodes_per_n):
            employee = employees.iloc[int(rng.integers(0, len(employees)))]
            ep_id = f"ep-{episode_id:05d}"
            episode_id += 1
            current_time = 0.0
            thread_ids = [f"{ep_id}-th-{j+1:03d}" for j in range(n_threads)]
            facts_by_thread: dict[str, dict[str, str]] = {}
            thread_msg_count = {t_id: 0 for t_id in thread_ids}

            for t_id in thread_ids:
                project_code = make_project_code(rng)
                owner = pick_owner_name(rng)
                due_date = make_due_date(rng)
                thread_archetype = str(rng.choice(archetypes, p=archetype_w))
                task_title, task_detail = pick_task_card(rng, thread_archetype)
                urgency_word, timing_word = choose_urgency(rng, thread_archetype)
                facts_by_thread[t_id] = {
                    "project_code": project_code,
                    "owner": owner,
                    "due_date": due_date,
                    "archetype": thread_archetype,
                    "task_title": task_title,
                    "task_detail": task_detail,
                    "urgency_word": urgency_word,
                    "timing_word": timing_word,
                }
                thread_rows.append(
                    {
                        "episode_id": ep_id,
                        "thread_id": t_id,
                        "project_code": project_code,
                        "owner": owner,
                        "due_date": due_date,
                        "thread_archetype": thread_archetype,
                        "task_title": task_title,
                    }
                )

            anchor_count = min(n_threads, messages_per_episode)
            anchor_share = float(anchor_count) / float(messages_per_episode)
            if (1.0 - anchor_share) > 1e-6:
                followup_memory_rate = min(1.0, context_memory_rate / (1.0 - anchor_share))
            else:
                followup_memory_rate = 0.0

            current_thread = thread_ids[0]
            for msg_idx in range(messages_per_episode):
                if msg_idx < anchor_count:
                    t_id = thread_ids[msg_idx]
                else:
                    if rng.random() < 0.45:
                        t_id = current_thread
                    else:
                        t_id = str(rng.choice(thread_ids))
                    current_thread = t_id

                facts = facts_by_thread[t_id]
                base_subject = facts["task_title"]
                base_body = pick_filler_body(filler_source=filler_source, body_pool=body_pool, rng=rng)
                archetype = facts["archetype"]
                urgency_word = facts["urgency_word"]
                timing_word = facts["timing_word"]
                anchor_msg = thread_msg_count[t_id] == 0
                memory_eligible = archetype in {
                    "deadline_request",
                    "request_with_question",
                    "direct_request",
                    "information_request",
                }
                needs_memory = int((not anchor_msg) and memory_eligible and (rng.random() < followup_memory_rate))

                subject, body, priority, reply_type = compose_message(
                    base_subject=base_subject,
                    base_body=base_body,
                    task_detail=facts["task_detail"],
                    project_code=facts["project_code"],
                    owner=facts["owner"],
                    due_date=facts["due_date"],
                    urgency_word=urgency_word,
                    timing_word=timing_word,
                    archetype=archetype,
                    needs_memory=bool(needs_memory),
                    anchor_msg=anchor_msg,
                )
                required_key = "project_code" if needs_memory else "none"
                required_val = facts["project_code"] if needs_memory else ""

                gap = float(rng.exponential(scale=2.4))
                if rng.random() < 0.20:
                    gap = float(rng.exponential(scale=0.5))
                current_time += gap
                message_rows.append(
                    {
                        "episode_id": ep_id,
                        "employee_id": employee["employee_id"],
                        "message_id": f"{ep_id}-m-{msg_idx:04d}",
                        "thread_id": t_id,
                        "arrival_min": round(current_time, 3),
                        "subject": subject[:220],
                        "body": body[:2400],
                        "archetype": archetype,
                        "gold_priority": priority,
                        "gold_reply_type": reply_type,
                        "gold_project_code": facts["project_code"],
                        "gold_required_key": required_key,
                        "gold_required_value": required_val,
                        "needs_memory": needs_memory,
                        "is_anchor_message": int(anchor_msg),
                        "n_threads": n_threads,
                    }
                )
                thread_msg_count[t_id] += 1

            episode_rows.append(
                {
                    "episode_id": ep_id,
                    "employee_id": employee["employee_id"],
                    "n_threads": n_threads,
                    "messages_in_episode": messages_per_episode,
                }
            )

    messages = pd.DataFrame(message_rows).sort_values(["episode_id", "arrival_min", "message_id"])
    episodes = pd.DataFrame(episode_rows).sort_values(["n_threads", "episode_id"])
    threads = pd.DataFrame(thread_rows).sort_values(["episode_id", "thread_id"])

    employees.to_csv(scenario_dir / "employees.csv", index=False)
    episodes.to_csv(scenario_dir / "episodes.csv", index=False)
    threads.to_csv(scenario_dir / "threads.csv", index=False)
    messages.to_csv(scenario_dir / "messages.csv", index=False)

    return scenario_dir


def write_run_matrix(
    *,
    scenario_dir: Path,
    n_values: list[int],
    models: list[str],
    episodes_per_n: int,
    score_threshold_q: float,
    p0_sla_threshold: float,
) -> None:
    rows = []
    for model in models:
        for reasoning in ["auto", "high"]:
            for memory_policy in ["scratchpad_only", "thread_state"]:
                rows.append(
                    {
                        "run_name": f"{model.replace('/', '_')}_{memory_policy}_{reasoning}",
                        "agent": "openai",
                        "model": model,
                        "reasoning_mode": reasoning,
                        "n_values": ",".join(str(v) for v in n_values),
                        "episodes_per_n": episodes_per_n,
                        "score_threshold_q": score_threshold_q,
                        "p0_sla_threshold": p0_sla_threshold,
                        "memory_policy": memory_policy,
                    }
                )
    for memory_policy in ["scratchpad_only", "thread_state"]:
        rows.append(
            {
                "run_name": f"heuristic_{memory_policy}",
                "agent": "heuristic",
                "model": "n/a",
                "reasoning_mode": "n/a",
                "n_values": ",".join(str(v) for v in n_values),
                "episodes_per_n": episodes_per_n,
                "score_threshold_q": score_threshold_q,
                "p0_sla_threshold": p0_sla_threshold,
                "memory_policy": memory_policy,
            }
        )
    pd.DataFrame(rows).to_csv(scenario_dir / "run_matrix.csv", index=False)


def write_setup_notes(
    *,
    scenario_dir: Path,
    stats: dict[str, float],
    n_values: list[int],
    context_memory_rate: float,
    episodes_per_n: int,
    messages_per_episode: int,
    filler_source: str,
) -> None:
    lines = [
        "# Scratchpad Frontier Setup",
        "",
        "This folder is a prepared experiment pack. No API runs are executed during setup.",
        "",
        "## Calibration Anchors",
        f"- Median active threads (14d): **{stats['median_mean_active_threads_all']:.2f}**",
        f"- Median p90 active threads: **{stats['median_p90_active_threads_all']:.2f}**",
        f"- Workload grid N: **{', '.join(str(v) for v in n_values)}**",
        f"- Context-memory rate (hard context cues): **{100.0 * context_memory_rate:.1f}%**",
        f"- Filler source: **{filler_source}**",
        "",
        "## Files",
        "- `employees.csv`: synthetic employee personas with volume tiers",
        "- `episodes.csv`: episode-level load and assignment",
        "- `threads.csv`: per-thread facts (project code, owner, due date, archetype, task title)",
        "- `messages.csv`: message stream with objective gold targets",
        "- `run_matrix.csv`: planned model/baseline runs",
        "",
        "## Core Estimands",
        "- Capacity frontier N*: largest N passing quality and P0 SLA thresholds",
        "- Production function: quality-latency-cost under randomized load N",
        "",
        "## Suggested Commands (later, when you approve execution)",
        "```bash",
        f"python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir {scenario_dir} --agent heuristic --prompt-profile meaning",
        f"python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir {scenario_dir} --agent openai --model gpt-5.2 --memory-policy scratchpad_only --openai-reasoning-mode auto --prompt-profile meaning",
        f"python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir {scenario_dir} --agent openai --model gpt-5.2 --memory-policy thread_state --openai-reasoning-mode auto --prompt-profile meaning",
        f"python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir {scenario_dir} --agent openai --model gpt-5-mini --memory-policy thread_state --openai-reasoning-mode auto --prompt-profile meaning",
        "```",
        "",
        f"Setup defaults: {episodes_per_n} episodes per N, {messages_per_episode} messages per episode.",
    ]
    (scenario_dir / "SETUP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def trim_scratchpad(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    return text[-budget:]


def build_fail_closed_decision(email: dict[str, Any]) -> dict[str, Any]:
    text = f"{email['subject']} {email['body']}".lower()
    priority = "P2"
    if any(tok in text for tok in ["urgent", "today", "asap", "immediately", "blocking", "deadline"]):
        priority = "P0"
    elif any(tok in text for tok in ["please", "need", "follow-up", "follow up", "can you", "could you"]):
        priority = "P1"
    return {
        "priority": priority,
        "reply_type": "REQUEST_INFO",
        "target_project_code": "",
        "binding_decision": "clarify",
        "binding_source": "none",
        "action_summary": "Clarify the project before taking action.",
        "facts_used": [],
        "draft_reply": "I need the project code or thread reference before I can safely act on this.",
        "scratchpad_update": "",
    }


def reconstruct_episode_state(
    *,
    existing_rows: pd.DataFrame,
    env: ScratchpadEnv,
    memory_policy: str,
    scratchpad_char_budget: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "current_time": 0.0,
        "calls": 0,
        "scratchpad": "",
        "thread_state_store": {},
        "rows": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "raw_invalid_total": 0,
        "api_error_total": 0,
    }
    if existing_rows.empty:
        return state

    rows_df = existing_rows.sort_values(["process_end_min", "message_id"]).copy()
    for _, row in rows_df.iterrows():
        message_id = str(row["message_id"])
        env.submit(message_id, {"resumed": True})
        row_dict = row.to_dict()
        state["rows"].append(row_dict)
        state["current_time"] = max(state["current_time"], float(row.get("process_end_min", 0.0)))
        state["calls"] += 1
        state["total_input_tokens"] += int(row.get("input_tokens", 0) or 0)
        state["total_output_tokens"] += int(row.get("output_tokens", 0) or 0)
        state["raw_invalid_total"] += int(row.get("invalid_output", 0) or 0)
        state["api_error_total"] += int(row.get("api_error", 0) or 0)

        if memory_policy == "scratchpad_only":
            update = str(row.get("pred_scratchpad_update", "") or "").strip()
            if update:
                state["scratchpad"] = trim_scratchpad(
                    (str(state["scratchpad"]) + "\n" + update).strip(),
                    budget=scratchpad_char_budget,
                )
        else:
            msg = env.by_id.get(message_id)
            if msg is None:
                continue
            observed_facts = extract_explicit_thread_facts(subject=msg.subject, body=msg.body)
            if observed_facts:
                thread_id = str(row.get("thread_id", msg.thread_id))
                state["thread_state_store"][thread_id] = merge_thread_facts(
                    state["thread_state_store"].get(thread_id, {}),
                    observed_facts,
                )
    return state


def build_run_manifest(*, args: argparse.Namespace, scenario_dir: Path, run_dir: Path) -> dict[str, Any]:
    messages_path = scenario_dir / "messages.csv"
    episodes_path = scenario_dir / "episodes.csv"
    metadata_path = scenario_dir / "metadata.json"
    judge_script = Path("scripts/judge_scratchpad_frontier_run.py")
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "scenario_dir": str(scenario_dir),
        "run_dir": str(run_dir),
        "scenario_messages_sha256": sha256_file(messages_path),
        "scenario_episodes_sha256": sha256_file(episodes_path),
        "scenario_metadata_sha256": sha256_file(metadata_path) if metadata_path.exists() else "",
        "agent": str(args.agent),
        "model": str(args.model if args.agent == "openai" else "n/a"),
        "memory_policy": str(args.memory_policy),
        "prompt_profile": str(args.prompt_profile),
        "seed": int(getattr(args, "seed", 0)),
        "openai_reasoning_mode": str(getattr(args, "openai_reasoning_mode", "n/a")),
        "openai_timeout_sec": int(getattr(args, "openai_timeout_sec", 0)),
        "max_calls": int(args.max_calls),
        "max_consecutive_api_errors": int(args.max_consecutive_api_errors),
        "episode_ids": [x for x in str(getattr(args, "episode_ids", "")).split(",") if x.strip()],
        "eval_script_sha256": sha256_file(Path(__file__)),
        "judge_script_sha256": sha256_file(judge_script) if judge_script.exists() else "",
    }
    return manifest


@dataclass
class ScenarioMessage:
    message_id: str
    episode_id: str
    employee_id: str
    thread_id: str
    arrival_min: float
    subject: str
    body: str
    gold_priority: str
    gold_reply_type: str
    gold_project_code: str
    gold_required_key: str
    gold_required_value: str
    needs_memory: int
    n_threads: int


class ScratchpadEnv:
    def __init__(self, messages: list[ScenarioMessage]) -> None:
        self.messages = sorted(messages, key=lambda x: (x.arrival_min, x.message_id))
        self.by_id = {m.message_id: m for m in self.messages}
        self.decisions: dict[str, dict[str, Any]] = {}

    def list_unread(self, current_time_min: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in self.messages:
            if m.arrival_min <= current_time_min and m.message_id not in self.decisions:
                out.append(
                    {
                        "message_id": m.message_id,
                        "thread_id": m.thread_id,
                        "arrival_min": m.arrival_min,
                        "subject": m.subject,
                    }
                )
        return out

    def open_message(self, message_id: str) -> dict[str, Any]:
        m = self.by_id[message_id]
        return {
            "message_id": m.message_id,
            "thread_id": m.thread_id,
            "arrival_min": m.arrival_min,
            "subject": m.subject[:220],
            "body": m.body[:2400],
        }

    def submit(self, message_id: str, decision: dict[str, Any]) -> None:
        self.decisions[message_id] = decision

    def next_arrival_after(self, current_time_min: float) -> float | None:
        future = [m.arrival_min for m in self.messages if m.message_id not in self.decisions and m.arrival_min > current_time_min]
        return min(future) if future else None

    def unresolved_count(self) -> int:
        return len(self.messages) - len(self.decisions)


class HeuristicScratchpadAgent:
    def __init__(self, *, memory_policy: str) -> None:
        self.memory_policy = memory_policy
        self.name = "heuristic"

    def decide(
        self,
        email: dict[str, Any],
        scratchpad: str,
        unread_count: int,
        *,
        thread_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = unread_count
        text = f"{email['subject']} {email['body']}".lower()
        if any(tok in text for tok in ["urgent", "today", "asap"]):
            priority = "P0"
        elif any(tok in text for tok in ["please", "need", "follow-up", "follow up", "can you", "could you"]):
            priority = "P1"
        else:
            priority = "P2"

        if "?" in email["body"]:
            reply_type = "REQUEST_INFO"
        elif priority in {"P0", "P1"}:
            reply_type = "ANSWER"
        else:
            reply_type = "ACK"

        project_code = ""
        binding_source = "none"
        project_code = find_project_code(email["subject"]) or find_project_code(email["body"])
        if project_code:
            binding_source = "current_email"
        elif self.memory_policy == "thread_state" and thread_state:
            project_code = normalize_project_code(thread_state.get("project_code", ""))
            if project_code:
                binding_source = "thread_state"
        elif scratchpad:
            project_code = find_project_code(scratchpad)
            if project_code:
                binding_source = "scratchpad"
        binding_decision = "bound" if project_code else "clarify"
        action_summary = f"Respond and track thread {email['thread_id']}"
        if project_code:
            action_summary = f"Respond for project {project_code} and track thread {email['thread_id']}"
        draft = "Acknowledged. I will take the requested action and follow up."
        if binding_decision == "clarify":
            draft = "I can help, but I need the project code or thread reference before I act."
        update = f"{email['thread_id']}: {project_code or 'project unknown'} | priority={priority}"

        return (
            {
                "priority": priority,
                "reply_type": reply_type,
                "target_project_code": project_code,
                "binding_decision": binding_decision,
                "binding_source": binding_source,
                "action_summary": action_summary,
                "facts_used": [project_code] if project_code else [],
                "draft_reply": draft,
                "scratchpad_update": update,
            },
            {"latency_sec": 0.03, "input_tokens": 0, "output_tokens": 0, "raw_invalid": 0},
        )


class OpenAIScratchpadAgent:
    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_sec: int,
        max_attempts: int,
        reasoning_mode: str,
        max_output_tokens: int,
        temperature: float | None,
        prompt_profile: str,
        memory_policy: str,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_attempts = max(1, int(max_attempts))
        self.reasoning_mode = reasoning_mode
        self.max_output_tokens = max(0, int(max_output_tokens))
        self.temperature = temperature
        self.memory_policy = memory_policy
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.session = requests.Session()
        rubric = RUBRIC_MEANING if prompt_profile == "meaning" else RUBRIC_STRICT
        if memory_policy == "thread_state":
            self.system_prompt = (
                "You are an inbox triage assistant with system-maintained per-thread state. "
                "You only know: current email + structured thread_state for the current thread. "
                "The thread_state contains only facts that were explicitly observed earlier in this same thread. "
                "Use thread_state when available, and do not assume any hidden memory beyond what is shown. "
                "Return strict JSON with fields: priority, reply_type, target_project_code, binding_decision, binding_source, "
                "action_summary, facts_used, draft_reply, scratchpad_update. "
                "priority must be one of P0,P1,P2. reply_type must be one of NONE,ACK,ANSWER,REQUEST_INFO,REDIRECT. "
                f"{rubric} "
                "If you can confidently identify the project, set binding_decision=bound and fill target_project_code. "
                "If you cannot, set binding_decision=clarify and leave target_project_code empty. "
                "Use binding_source=current_email, thread_state, or none. "
                "When thread_state includes a project_code, include it in facts_used and use it in the reply/action when relevant."
            )
        else:
            self.system_prompt = (
                "You are an inbox triage assistant with scratchpad-only memory. "
                "You only know: current email + current scratchpad text. "
                "Do not assume hidden memory. "
                "Return strict JSON with fields: priority, reply_type, target_project_code, binding_decision, binding_source, "
                "action_summary, facts_used, draft_reply, scratchpad_update. "
                "priority must be one of P0,P1,P2. reply_type must be one of NONE,ACK,ANSWER,REQUEST_INFO,REDIRECT. "
                f"{rubric} "
                "If you can confidently identify the project, set binding_decision=bound and fill target_project_code. "
                "If you cannot, set binding_decision=clarify and leave target_project_code empty. "
                "Use binding_source=current_email, scratchpad, or none. "
                f"{MEMORY_CONTRACT}"
            )
        self.response_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
                "reply_type": {
                    "type": "string",
                    "enum": ["NONE", "ACK", "ANSWER", "REQUEST_INFO", "REDIRECT"],
                },
                "target_project_code": {"type": "string"},
                "binding_decision": {"type": "string", "enum": ["bound", "clarify", "unbound_guess"]},
                "binding_source": {"type": "string", "enum": ["current_email", "scratchpad", "thread_state", "none"]},
                "action_summary": {"type": "string"},
                "facts_used": {"type": "array", "items": {"type": "string"}},
                "draft_reply": {"type": "string"},
                "scratchpad_update": {"type": "string"},
            },
            "required": [
                "priority",
                "reply_type",
                "target_project_code",
                "binding_decision",
                "binding_source",
                "action_summary",
                "facts_used",
                "draft_reply",
                "scratchpad_update",
            ],
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt == self.max_attempts:
                        resp.raise_for_status()
                    time.sleep((2**attempt) + float(np.random.uniform(0, 1)))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:1200]}")
                return resp.json()
            except Exception as exc:  # pragma: no cover - network-dependent path
                last_error = exc
                if attempt == self.max_attempts:
                    raise
                time.sleep((2**attempt) + float(np.random.uniform(0, 1)))
        raise RuntimeError(f"OpenAI request failed: {last_error}")

    def decide(
        self,
        email: dict[str, Any],
        scratchpad: str,
        unread_count: int,
        *,
        thread_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_obj = {
            "unread_count": unread_count,
            "email": email,
            "requirements": {
                "memory_policy": self.memory_policy,
                "must_not_assume_hidden_context": True,
            },
        }
        if self.memory_policy == "thread_state":
            prompt_obj["thread_state"] = thread_state or {}
        else:
            prompt_obj["scratchpad"] = scratchpad
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "scratchpad_triage",
                    "schema": self.response_schema,
                    "strict": True,
                }
            },
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_mode == "high":
            payload["reasoning"] = {"effort": "high"}
        if self.temperature is not None and model_supports_temperature(self.model):
            payload["temperature"] = self.temperature
        t0 = time.time()
        response = self._request(payload)
        latency = time.time() - t0

        content = ""
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            content = output_text
        else:
            for item in response.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            content = part.get("text", "")
                            break
                        if part.get("type") == "output_json":
                            content = json.dumps(part.get("json"), ensure_ascii=True)
                            break
                elif item.get("type") == "output_json":
                    content = json.dumps(item.get("json") or {}, ensure_ascii=True)
                elif item.get("type") == "output_text":
                    content = item.get("text", "")
                if content:
                    break

        raw_invalid = 0
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            raw_invalid = 1
            decision = {}

        usage = response.get("usage", {})
        meta = {
            "latency_sec": float(latency),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "raw_invalid": raw_invalid,
        }
        return decision, meta


def validate_decision(decision: dict[str, Any]) -> tuple[dict[str, Any], int]:
    invalid = 0
    out = {
        "priority": decision.get("priority", "P2"),
        "reply_type": decision.get("reply_type", "NONE"),
        "target_project_code": decision.get("target_project_code", ""),
        "binding_decision": decision.get("binding_decision", "clarify"),
        "binding_source": decision.get("binding_source", "none"),
        "action_summary": decision.get("action_summary", ""),
        "facts_used": decision.get("facts_used", []),
        "draft_reply": decision.get("draft_reply", ""),
        "scratchpad_update": decision.get("scratchpad_update", ""),
    }
    if out["priority"] not in REQUIRED_PRIORITIES:
        out["priority"] = "P2"
        invalid = 1
    if out["reply_type"] not in REQUIRED_REPLY_TYPES:
        out["reply_type"] = "NONE"
        invalid = 1
    out["target_project_code"] = normalize_project_code(out["target_project_code"])
    if out["binding_decision"] not in REQUIRED_BINDING_DECISIONS:
        out["binding_decision"] = "clarify"
        invalid = 1
    if out["binding_source"] not in REQUIRED_BINDING_SOURCES:
        out["binding_source"] = "none"
        invalid = 1
    if not isinstance(out["action_summary"], str):
        out["action_summary"] = str(out["action_summary"])
        invalid = 1
    if not isinstance(out["facts_used"], list):
        out["facts_used"] = []
        invalid = 1
    out["facts_used"] = [str(x) for x in out["facts_used"] if str(x).strip()]
    if out["target_project_code"] and out["target_project_code"] not in out["facts_used"]:
        out["facts_used"].append(out["target_project_code"])
    if not isinstance(out["draft_reply"], str):
        out["draft_reply"] = str(out["draft_reply"])
        invalid = 1
    if not isinstance(out["scratchpad_update"], str):
        out["scratchpad_update"] = str(out["scratchpad_update"])
        invalid = 1
    if out["binding_decision"] == "clarify":
        out["target_project_code"] = ""
        if out["binding_source"] != "none":
            out["binding_source"] = "none"
    elif not out["target_project_code"]:
        out["binding_decision"] = "clarify"
        out["binding_source"] = "none"
        invalid = 1
    return out, invalid


def hallucination_penalty(draft_reply: str, source_text: str) -> float:
    d = draft_reply.lower()
    s = source_text.lower()
    for phrase in BANNED_COMMITMENTS:
        if phrase in d and phrase not in s:
            return 1.0
    return 0.0


def score_message(
    *,
    message: ScenarioMessage,
    decision: dict[str, Any],
    process_end_min: float,
) -> dict[str, float]:
    priority_acc = float(decision["priority"] == message.gold_priority)
    reply_acc = float(decision["reply_type"] == message.gold_reply_type)
    text_blob = (
        f"{decision['action_summary']} {decision['draft_reply']} "
        f"{decision['target_project_code']} {' '.join(str(x) for x in decision['facts_used'])}"
    ).lower()

    if message.gold_required_key == "none":
        fact_recall = 1.0
    else:
        fact_recall = float(message.gold_required_value.lower() in text_blob)

    pred_target = normalize_project_code(decision.get("target_project_code", ""))
    gold_target = normalize_project_code(message.gold_project_code)
    email_target = normalize_project_code(extract_explicit_thread_facts(subject=message.subject, body=message.body).get("project_code", ""))
    target_match = float(bool(pred_target) and pred_target == gold_target)
    binding_attempt = float(bool(pred_target))
    safe_clarification = float(
        decision.get("binding_decision") == "clarify"
        and not pred_target
        and not email_target
    )
    unsafe_wrong_target = float(bool(pred_target) and pred_target != gold_target)

    halluc = hallucination_penalty(decision["draft_reply"], f"{message.subject}\n{message.body}")
    latency_min = process_end_min - message.arrival_min
    sla_min = PRIORITY_SLA_MIN[message.gold_priority]
    on_time = float(latency_min <= sla_min)

    quality = 0.40 * priority_acc + 0.30 * reply_acc + 0.30 * fact_recall - 0.20 * halluc
    if on_time < 1.0 and message.gold_priority in {"P0", "P1"}:
        quality *= 0.2
    quality = max(0.0, min(1.0, quality))
    return {
        "priority_acc": priority_acc,
        "reply_acc": reply_acc,
        "fact_recall": fact_recall,
        "target_match": target_match,
        "binding_attempt": binding_attempt,
        "safe_clarification": safe_clarification,
        "unsafe_wrong_target": unsafe_wrong_target,
        "hallucination": halluc,
        "latency_min": latency_min,
        "on_time": on_time,
        "quality_score": quality,
    }


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) > 0 else 0.0


def run_episode(
    *,
    env: ScratchpadEnv,
    agent: HeuristicScratchpadAgent | OpenAIScratchpadAgent,
    scratchpad_char_budget: int,
    max_calls: int,
    max_consecutive_api_errors: int,
    existing_rows: pd.DataFrame | None = None,
    message_log_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    memory_policy = getattr(agent, "memory_policy", "scratchpad_only")
    resume_rows = existing_rows.copy() if existing_rows is not None and not existing_rows.empty else pd.DataFrame()
    state = reconstruct_episode_state(
        existing_rows=resume_rows,
        env=env,
        memory_policy=memory_policy,
        scratchpad_char_budget=scratchpad_char_budget,
    )
    current_time = float(state["current_time"])
    calls = int(state["calls"])
    scratchpad = str(state["scratchpad"])
    thread_state_store: dict[str, dict[str, str]] = dict(state["thread_state_store"])
    rows: list[dict[str, Any]] = list(state["rows"])
    total_input_tokens = int(state["total_input_tokens"])
    total_output_tokens = int(state["total_output_tokens"])
    raw_invalid_total = int(state["raw_invalid_total"])
    api_error_total = int(state["api_error_total"])
    consecutive_api_errors = 0

    total_messages = len(env.messages)
    while env.unresolved_count() > 0:
        unread = env.list_unread(current_time)
        if not unread:
            next_arrival = env.next_arrival_after(current_time)
            if next_arrival is None:
                break
            current_time = next_arrival
            continue

        unread_sorted = sorted(unread, key=lambda x: (x["arrival_min"], x["message_id"]))
        target = unread_sorted[0]
        email = env.open_message(target["message_id"])
        current_thread_state = render_thread_state(
            email["thread_id"],
            thread_state_store.get(email["thread_id"], {}),
        )
        scratchpad_chars_before = len(scratchpad)

        if calls >= max_calls:
            raise RuntimeError(f"Max calls reached ({max_calls})")
        calls += 1
        api_error = 0
        try:
            raw_decision, meta = agent.decide(
                email=email,
                scratchpad=scratchpad,
                unread_count=len(unread),
                thread_state=current_thread_state,
            )
            consecutive_api_errors = 0
        except Exception as exc:  # pragma: no cover - live API dependent path
            api_error = 1
            consecutive_api_errors += 1
            raw_decision = build_fail_closed_decision(email)
            meta = {
                "latency_sec": 1.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "raw_invalid": 1,
                "error": str(exc)[:300],
            }
            if consecutive_api_errors > int(max_consecutive_api_errors):
                raise RuntimeError(
                    f"Exceeded max consecutive API errors ({max_consecutive_api_errors}) "
                    f"while processing episode {env.by_id[target['message_id']].episode_id}"
                ) from exc
        decision, invalid = validate_decision(raw_decision)
        invalid_total = int(invalid + meta.get("raw_invalid", 0))
        raw_invalid_total += invalid_total
        api_error_total += api_error

        latency_sec = float(meta.get("latency_sec", 0.05))
        process_time_min = max(0.35, latency_sec / 60.0)
        current_time = max(current_time, float(email["arrival_min"])) + process_time_min
        env.submit(target["message_id"], decision)

        if memory_policy == "scratchpad_only" and decision["scratchpad_update"]:
            scratchpad = trim_scratchpad(
                (scratchpad + "\n" + decision["scratchpad_update"]).strip(),
                budget=scratchpad_char_budget,
            )
        elif memory_policy == "thread_state":
            observed_facts = extract_explicit_thread_facts(subject=email["subject"], body=email["body"])
            if observed_facts:
                thread_state_store[email["thread_id"]] = merge_thread_facts(
                    thread_state_store.get(email["thread_id"], {}),
                    observed_facts,
                )

        msg = env.by_id[target["message_id"]]
        scores = score_message(message=msg, decision=decision, process_end_min=current_time)
        total_input_tokens += int(meta.get("input_tokens", 0))
        total_output_tokens += int(meta.get("output_tokens", 0))

        rows.append(
            {
                "message_id": msg.message_id,
                "episode_id": msg.episode_id,
                "n_threads": msg.n_threads,
                "thread_id": msg.thread_id,
                "arrival_min": msg.arrival_min,
                "process_end_min": current_time,
                "gold_priority": msg.gold_priority,
                "pred_priority": decision["priority"],
                "gold_reply_type": msg.gold_reply_type,
                "pred_reply_type": decision["reply_type"],
                "gold_project_code": msg.gold_project_code,
                "pred_target_project_code": decision["target_project_code"],
                "pred_binding_decision": decision["binding_decision"],
                "pred_binding_source": decision["binding_source"],
                "pred_action_summary": decision["action_summary"],
                "pred_facts_used": json.dumps(decision["facts_used"], ensure_ascii=True),
                "pred_draft_reply": decision["draft_reply"],
                "pred_scratchpad_update": decision["scratchpad_update"],
                "needs_memory": msg.needs_memory,
                "gold_required_key": msg.gold_required_key,
                "gold_required_value": msg.gold_required_value,
                "invalid_output": invalid_total,
                "api_error": api_error,
                "input_tokens": int(meta.get("input_tokens", 0)),
                "output_tokens": int(meta.get("output_tokens", 0)),
                "memory_policy": memory_policy,
                "scratchpad_chars_before": scratchpad_chars_before,
                "scratchpad_chars_after": len(scratchpad),
                "thread_state_fact_count": len(current_thread_state),
                "thread_state_has_project_code": int(bool(current_thread_state.get("project_code"))),
                **scores,
            }
        )
        if message_log_path is not None:
            append_csv_row(message_log_path, rows[-1], fieldnames=list(rows[-1].keys()))
        if calls % 200 == 0 or calls == total_messages:
            print(
                f"[episode] agent={agent.name} processed={calls}/{total_messages} "
                f"scratchpad_chars={len(scratchpad)} api_errors={api_error_total} invalid={raw_invalid_total}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Episode produced no rows")

    p0 = df[df["gold_priority"] == "P0"]
    p1 = df[df["gold_priority"] == "P1"]
    mem = df[df["needs_memory"] == 1]
    metrics = {
        "messages": float(len(df)),
        "mean_quality": safe_mean(df["quality_score"]),
        "priority_acc": safe_mean(df["priority_acc"]),
        "reply_acc": safe_mean(df["reply_acc"]),
        "fact_recall": safe_mean(df["fact_recall"]),
        "target_match": safe_mean(df["target_match"]),
        "safe_clarification": safe_mean(df["safe_clarification"]),
        "unsafe_wrong_target": safe_mean(df["unsafe_wrong_target"]),
        "memory_fact_recall": safe_mean(mem["fact_recall"]) if len(mem) > 0 else 1.0,
        "memory_target_match": safe_mean(mem["target_match"]) if len(mem) > 0 else 1.0,
        "binding_precision_on_memory_probes": (
            safe_mean(mem[mem["binding_attempt"] > 0]["target_match"]) if len(mem[mem["binding_attempt"] > 0]) > 0 else 1.0
        ),
        "p0_sla": safe_mean(p0["on_time"]) if len(p0) > 0 else 1.0,
        "p1_sla": safe_mean(p1["on_time"]) if len(p1) > 0 else 1.0,
        "mean_latency_min": safe_mean(df["latency_min"]),
        "invalid_rate": safe_mean(df["invalid_output"]),
        "api_error_rate": safe_mean(df["api_error"]),
        "input_tokens": float(total_input_tokens),
        "output_tokens": float(total_output_tokens),
        "calls": float(calls),
        "raw_invalid_total": float(raw_invalid_total),
        "thread_state_entries": float(len(thread_state_store)),
    }
    return df, metrics


def summarize_n_level(df_episode: pd.DataFrame) -> pd.DataFrame:
    return (
        df_episode.groupby("n_threads")
        .agg(
            episodes=("episode_id", "nunique"),
            mean_quality=("mean_quality", "mean"),
            priority_acc=("priority_acc", "mean"),
            reply_acc=("reply_acc", "mean"),
            fact_recall=("fact_recall", "mean"),
            target_match=("target_match", "mean"),
            safe_clarification=("safe_clarification", "mean"),
            unsafe_wrong_target=("unsafe_wrong_target", "mean"),
            memory_fact_recall=("memory_fact_recall", "mean"),
            memory_target_match=("memory_target_match", "mean"),
            binding_precision_on_memory_probes=("binding_precision_on_memory_probes", "mean"),
            p0_sla=("p0_sla", "mean"),
            p1_sla=("p1_sla", "mean"),
            invalid_rate=("invalid_rate", "mean"),
            api_error_rate=("api_error_rate", "mean"),
            mean_latency_min=("mean_latency_min", "mean"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            calls=("calls", "sum"),
        )
        .reset_index()
        .sort_values("n_threads")
    )


def write_run_report(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    n_summary: pd.DataFrame,
    n_star: int | None,
    total_cost: float,
) -> None:
    lines = [
        "# Scratchpad Frontier Run Report",
        "",
        f"- Agent: **{args.agent}**",
        f"- Model: **{args.model if args.agent == 'openai' else 'n/a'}**",
        f"- Memory policy: **{args.memory_policy}**",
        f"- Prompt profile: **{args.prompt_profile}**",
        f"- Scenario dir: **{args.scenario_dir}**",
        f"- Scratchpad chars: **{args.scratchpad_char_budget}**",
        f"- Quality threshold q: **{args.score_threshold_q:.2f}**",
        f"- P0 SLA threshold: **{args.p0_sla_threshold:.2f}**",
        f"- Estimated API cost: **${total_cost:.4f}**",
        "",
        "This report is **rubric-compliance scoring** against synthetic gold labels. "
        "For **judgment scoring** (LLM-as-judge), run `python scripts/judge_scratchpad_frontier_run.py` on this run.",
        "",
        "## N-Level Summary",
    ]
    for _, row in n_summary.iterrows():
        lines.append(
            f"- N={int(row['n_threads'])}: quality={row['mean_quality']:.3f}, "
            f"p0_sla={row['p0_sla']:.3f}, memory_recall={row['memory_fact_recall']:.3f}, "
            f"target_match={row['target_match']:.3f}, safe_clarify={row['safe_clarification']:.3f}, "
            f"wrong_target={row['unsafe_wrong_target']:.3f}, api_error_rate={row['api_error_rate']:.3f}"
        )
    lines.append("")
    if n_star is None:
        lines.append("- Estimated capacity N*: **none passed threshold**")
    else:
        lines.append(f"- Estimated capacity N*: **{n_star}**")
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_mode(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    stats = load_key_stats(args.key_results_csv)
    n_values = parse_n_values(args.n_values)
    context_memory_rate = (
        read_context_memory_rate(args.context_spotcheck_md)
        if args.context_memory_rate < 0
        else float(args.context_memory_rate)
    )
    context_memory_rate = max(0.0, min(1.0, context_memory_rate))

    _ = load_intent_distribution(args.intent_dist_csv)  # locked as calibration anchor for reproducibility metadata.
    archetype_dist = load_archetype_distribution(args.archetype_csv)
    body_pool = (
        load_body_sample(args.body_sample_cache) if str(args.filler_source) == "enron" else pd.DataFrame({"subject": [], "body": []})
    )

    if args.scenario_tag.strip():
        scenario_tag = args.scenario_tag.strip()
    else:
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        scenario_tag = f"scratchpad_frontier_{ts}"

    employees = build_employee_profiles(stats=stats, employees=args.employees, rng=rng)
    scenario_dir = generate_scenario(
        output_dir=args.output_dir,
        scenario_tag=scenario_tag,
        seed=seed,
        n_values=n_values,
        episodes_per_n=int(args.episodes_per_n),
        messages_per_episode=int(args.messages_per_episode),
        employees=employees,
        body_pool=body_pool,
        filler_source=str(args.filler_source),
        archetype_dist=archetype_dist,
        context_memory_rate=context_memory_rate,
    )
    write_run_matrix(
        scenario_dir=scenario_dir,
        n_values=n_values,
        models=DEFAULT_MODELS,
        episodes_per_n=int(args.episodes_per_n),
        score_threshold_q=float(args.score_threshold_q),
        p0_sla_threshold=float(args.p0_sla_threshold),
    )
    write_setup_notes(
        scenario_dir=scenario_dir,
        stats=stats,
        n_values=n_values,
        context_memory_rate=context_memory_rate,
        episodes_per_n=int(args.episodes_per_n),
        messages_per_episode=int(args.messages_per_episode),
        filler_source=str(args.filler_source),
    )

    metadata = {
        "scenario_tag": scenario_tag,
        "seed": seed,
        "n_values": n_values,
        "episodes_per_n": int(args.episodes_per_n),
        "messages_per_episode": int(args.messages_per_episode),
        "employees": int(args.employees),
        "context_memory_rate": context_memory_rate,
        "median_active_threads": stats["median_mean_active_threads_all"],
        "median_p90_active_threads": stats["median_p90_active_threads_all"],
        "filler_source": str(args.filler_source),
    }
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared scenario: {scenario_dir.resolve()}")
    print("No model runs were executed.")


def run_mode(args: argparse.Namespace) -> None:
    if args.scenario_dir is None:
        raise ValueError("--scenario-dir is required for --mode run")
    scenario_dir = args.scenario_dir
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario dir not found: {scenario_dir}")
    load_dotenv(Path(".env"))

    messages_df = pd.read_csv(scenario_dir / "messages.csv")
    episodes_df = pd.read_csv(scenario_dir / "episodes.csv")
    selected_episode_ids = [x.strip() for x in str(args.episode_ids).split(",") if x.strip()]
    if selected_episode_ids:
        episodes_df = episodes_df[episodes_df["episode_id"].astype(str).isin(selected_episode_ids)].copy()
        if episodes_df.empty:
            raise ValueError(f"--episode-ids did not match any prepared episodes: {selected_episode_ids}")
        messages_df = messages_df[messages_df["episode_id"].astype(str).isin(selected_episode_ids)].copy()
    n_values = sorted(episodes_df["n_threads"].unique().tolist())

    if args.agent == "heuristic":
        agent: HeuristicScratchpadAgent | OpenAIScratchpadAgent = HeuristicScratchpadAgent(
            memory_policy=str(args.memory_policy)
        )
    else:
        agent = OpenAIScratchpadAgent(
            model=args.model,
            base_url=args.openai_base_url,
            timeout_sec=args.openai_timeout_sec,
            max_attempts=args.openai_max_attempts,
            reasoning_mode=args.openai_reasoning_mode,
            max_output_tokens=args.openai_max_output_tokens,
            temperature=args.temperature,
            prompt_profile=str(args.prompt_profile),
            memory_policy=str(args.memory_policy),
        )

    model_tag = args.model.replace("/", "_").replace(":", "_")
    policy_tag = str(args.memory_policy).replace("-", "_")
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"--resume-run-dir not found: {run_dir}")
    else:
        run_tag = args.run_tag.strip() or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = scenario_dir / "runs" / f"{args.agent}_{policy_tag}_{model_tag}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    message_log_path = run_dir / "message_log.csv"
    episode_summary_path = run_dir / "episode_summary.csv"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_scenario_hash = sha256_file(scenario_dir / "messages.csv")
        if manifest.get("scenario_messages_sha256") != current_scenario_hash:
            raise RuntimeError("Resume run dir points at a different scenario/messages.csv hash.")
        if str(manifest.get("memory_policy")) != str(args.memory_policy):
            raise RuntimeError("Resume run dir memory_policy does not match current args.")
        if str(manifest.get("agent")) != str(args.agent):
            raise RuntimeError("Resume run dir agent does not match current args.")
        if args.agent == "openai" and str(manifest.get("model")) != str(args.model):
            raise RuntimeError("Resume run dir model does not match current args.")
        manifest_episode_ids = [str(x) for x in manifest.get("episode_ids", [])]
        if manifest_episode_ids != selected_episode_ids:
            raise RuntimeError("Resume run dir episode_ids do not match current args.")
    else:
        manifest = build_run_manifest(args=args, scenario_dir=scenario_dir, run_dir=run_dir)
        atomic_write_json(manifest_path, manifest)

    existing_message_log = pd.read_csv(message_log_path) if message_log_path.exists() else pd.DataFrame()
    if not existing_message_log.empty and "episode_id" not in existing_message_log.columns:
        existing_message_log = existing_message_log.merge(
            messages_df[["message_id", "episode_id", "n_threads"]],
            on="message_id",
            how="left",
            validate="many_to_one",
        )
    episode_summary_existing = pd.read_csv(episode_summary_path) if episode_summary_path.exists() else pd.DataFrame()
    completed_eps = set(episode_summary_existing.get("episode_id", pd.Series(dtype=str)).astype(str).tolist())
    calls_total = 0

    ep_ids = episodes_df["episode_id"].tolist()
    total_eps = len(ep_ids)
    for idx, ep_id in enumerate(ep_ids, start=1):
        if str(ep_id) in completed_eps:
            if not episode_summary_existing.empty:
                prev = episode_summary_existing[episode_summary_existing["episode_id"].astype(str) == str(ep_id)]
                if not prev.empty:
                    calls_total += int(prev.iloc[0].get("calls", 0))
            continue
        episode_messages = messages_df[messages_df["episode_id"] == ep_id].copy()
        records: list[ScenarioMessage] = []
        for _, row in episode_messages.iterrows():
            records.append(
                ScenarioMessage(
                    message_id=str(row["message_id"]),
                    episode_id=str(row["episode_id"]),
                    employee_id=str(row["employee_id"]),
                    thread_id=str(row["thread_id"]),
                    arrival_min=float(row["arrival_min"]),
                    subject=str(row["subject"]),
                    body=str(row["body"]),
                    gold_priority=str(row["gold_priority"]),
                    gold_reply_type=str(row["gold_reply_type"]),
                    gold_project_code=resolve_gold_project_code(
                        explicit_value=row.get("gold_project_code", ""),
                        gold_required_key=row.get("gold_required_key", ""),
                        gold_required_value=row.get("gold_required_value", ""),
                        subject=str(row["subject"]),
                        body=str(row["body"]),
                    ),
                    gold_required_key=str(row["gold_required_key"]),
                    gold_required_value=str(row["gold_required_value"]),
                    needs_memory=int(row["needs_memory"]),
                    n_threads=int(row["n_threads"]),
                )
            )
        env = ScratchpadEnv(records)
        existing_rows = pd.DataFrame()
        if not existing_message_log.empty:
            existing_rows = existing_message_log[existing_message_log["episode_id"].astype(str) == str(ep_id)].copy()
        msg_df, metrics = run_episode(
            env=env,
            agent=agent,
            scratchpad_char_budget=int(args.scratchpad_char_budget),
            max_calls=int(args.max_calls),
            max_consecutive_api_errors=int(args.max_consecutive_api_errors),
            existing_rows=existing_rows,
            message_log_path=message_log_path,
        )
        n_threads = int(episode_messages["n_threads"].iloc[0])
        episode_row = {"episode_id": ep_id, "n_threads": n_threads, **metrics}
        append_csv_row(episode_summary_path, episode_row, fieldnames=list(episode_row.keys()))
        completed_eps.add(str(ep_id))
        if existing_message_log.empty:
            existing_message_log = msg_df.copy()
        else:
            existing_message_log = pd.concat(
                [
                    existing_message_log[existing_message_log["episode_id"].astype(str) != str(ep_id)],
                    msg_df,
                ],
                ignore_index=True,
            )
        calls_total += int(metrics["calls"])
        if idx % 5 == 0 or idx == total_eps:
            print(
                f"[run] progress episodes={idx}/{total_eps} calls_total={calls_total} "
                f"last_n={n_threads} last_quality={metrics.get('mean_quality', 0.0):.3f} "
                f"last_api_error_rate={metrics.get('api_error_rate', 0.0):.3f}",
                flush=True,
            )
        if calls_total > int(args.max_calls):
            raise RuntimeError(f"Max calls reached ({args.max_calls})")

    if not message_log_path.exists():
        raise RuntimeError(f"No message log written to {message_log_path}")
    message_log = pd.read_csv(message_log_path)
    episode_summary = pd.read_csv(episode_summary_path)
    n_summary = summarize_n_level(episode_summary)

    total_input_tokens = float(episode_summary["input_tokens"].sum())
    total_output_tokens = float(episode_summary["output_tokens"].sum())
    total_cost = (
        total_input_tokens * float(args.input_cost_per_1m) + total_output_tokens * float(args.output_cost_per_1m)
    ) / 1_000_000.0

    n_summary["passes_threshold"] = (
        (n_summary["mean_quality"] >= float(args.score_threshold_q))
        & (n_summary["p0_sla"] >= float(args.p0_sla_threshold))
    )
    passed = n_summary[n_summary["passes_threshold"]]["n_threads"].tolist()
    n_star = int(max(passed)) if passed else None

    n_summary.to_csv(run_dir / "n_summary.csv", index=False)
    write_run_report(run_dir=run_dir, args=args, n_summary=n_summary, n_star=n_star, total_cost=total_cost)
    manifest["completed_at"] = datetime.now(tz=UTC).isoformat()
    manifest["completed_episodes"] = int(episode_summary["episode_id"].nunique())
    manifest["n_star"] = int(n_star) if n_star is not None else None
    atomic_write_json(manifest_path, manifest)

    print(f"Run finished: {run_dir.resolve()}")
    print(f"N*: {n_star if n_star is not None else 'none'}")
    print(f"N values: {n_values}")


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()
