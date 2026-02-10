#!/usr/bin/env python3
"""Evaluate inbox topic-capacity for heuristic and LLM agents."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

RE_LIST_TAG = re.compile(r"^\s*\[[^\]]+\]\s*")
RE_REPLY_PREFIX = re.compile(r"^\s*(re|fw|fwd)\s*:\s*", re.IGNORECASE)
RE_WHITESPACE = re.compile(r"\s+")

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "trading_market",
        [r"\btrade\b", r"\btrading\b", r"\bmarket\b", r"\bbid\b", r"\boffer\b", r"\bprice\b", r"\bgas\b", r"\bpower\b", r"\beol\b"],
    ),
    (
        "legal_compliance",
        [r"\blegal\b", r"\bcontract\b", r"\bagreement\b", r"\bcompliance\b", r"\bregulatory\b", r"\bferc\b", r"\blitigation\b"],
    ),
    (
        "ops_technical",
        [r"\bsystem\b", r"\bserver\b", r"\boutage\b", r"\bissue\b", r"\bincident\b", r"\bfailure\b", r"\bwarning\b", r"\bcrawler\b", r"\bbug\b", r"\bfix\b", r"\bsupport\b", r"\bdeploy\b"],
    ),
    (
        "meeting_scheduling",
        [r"\bmeeting\b", r"\bcall\b", r"\bconference\b", r"\bagenda\b", r"\bschedule\b", r"\bcalendar\b"],
    ),
    (
        "request_action",
        [r"\bplease\b", r"\bcan you\b", r"\bcould you\b", r"\bneed you\b", r"\brequest\b", r"\baction required\b", r"\burgent\b"],
    ),
    (
        "decision_approval",
        [r"\bapprove\b", r"\bapproval\b", r"\bsign[- ]off\b", r"\bdecision\b", r"\bauthorize\b", r"\bokay to\b"],
    ),
    (
        "status_update",
        [r"\bstatus\b", r"\bupdate\b", r"\bprogress\b", r"\breport\b", r"\bsummary\b", r"\brecap\b", r"\bweekly\b", r"\bdaily\b"],
    ),
    (
        "hr_admin",
        [r"\bhr\b", r"\bbenefits\b", r"\bpayroll\b", r"\bvacation\b", r"\binterview\b", r"\brecruit", r"\bexpense\b", r"\breimbursement\b"],
    ),
    (
        "announcement_broadcast",
        [r"\bannouncement\b", r"\bnotice\b", r"\bnewsletter\b", r"\ball employees\b", r"\bpolicy\b", r"\breminder\b"],
    ),
    (
        "social_personal",
        [r"\blunch\b", r"\bdinner\b", r"\bparty\b", r"\bholiday\b", r"\bthanks?\b", r"\bthank you\b", r"\bcongrat"],
    ),
]

PRIORITY_SLA_MIN = {"P0": 5.0, "P1": 20.0, "P2": 120.0}
REQUIRED_PRIORITIES = {"P0", "P1", "P2"}
REQUIRED_REPLY_TYPES = {"NONE", "ACK", "ANSWER", "REQUEST_INFO", "REDIRECT"}
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
    "P0 = genuinely urgent and time-sensitive (same-day deadlines, blocking incidents, escalations). "
    "P1 = important and actionable but not an emergency; should be handled soon (typically today). "
    "P2 = routine/FYI/social or non-urgent updates; can wait. "
    "Reply types: ANSWER when you can directly comply/decide/respond, REQUEST_INFO when you need clarifying information, "
    "ACK when acknowledging an update, NONE when no reply is needed, REDIRECT when the request belongs to a different owner/team."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default=os.getenv("RESEARCH_MODEL", "gpt-5"))
    parser.add_argument(
        "--prompt-profile",
        choices=["meaning", "strict"],
        default="meaning",
        help="meaning = tests judgment; strict = tests rubric compliance.",
    )
    parser.add_argument(
        "--pool-source",
        choices=["enron", "template"],
        default="enron",
        help=(
            "Message pool source for subject/body text. "
            "enron = sample real Enron bodies (may be in model training data); "
            "template = fully synthetic pool (no Enron text), but same harness."
        ),
    )
    parser.add_argument("--n-values", default="35,50,70,100", help="Comma-separated concurrent thread levels.")
    parser.add_argument("--episodes-per-n", type=int, default=3)
    parser.add_argument("--messages-per-episode", type=int, default=50)
    parser.add_argument("--thread-context-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold-q", type=float, default=0.75)
    parser.add_argument("--p0-sla-threshold", type=float, default=0.90)
    parser.add_argument("--headers-cache", type=Path, default=Path("data/enron_headers_1997_2003.parquet"))
    parser.add_argument("--body-sample-cache", type=Path, default=Path("data/enron_body_sample_20k.parquet"))
    parser.add_argument("--intent-dist-csv", type=Path, default=Path("results/message_intent_distribution.csv"))
    parser.add_argument(
        "--archetype-dist-csv",
        type=Path,
        default=Path("results/message_body_archetypes_sample.csv"),
        help="Archetype distribution for template pool generation.",
    )
    parser.add_argument("--template-pool-size", type=int, default=20000, help="Number of synthetic messages in template pool.")
    parser.add_argument("--template-topic-group-size", type=int, default=5, help="Messages per synthetic subject group.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-tag", default="", help="Optional run tag. If omitted, a timestamped tag is generated.")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-timeout-sec", type=int, default=60)
    parser.add_argument("--openai-max-attempts", type=int, default=3)
    parser.add_argument(
        "--openai-reasoning-mode",
        choices=["auto", "high"],
        default="auto",
        help="Reasoning mode for Responses API. 'auto' omits reasoning field.",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=0,
        help="Optional cap for Responses API. 0 means omit field.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature; omit for model default.")
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--input-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1m", type=float, default=0.0)
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


def detect_intent(text: str) -> str:
    if not isinstance(text, str):
        return "uncategorized"
    t = text.lower()
    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, t):
                return intent
    return "uncategorized"


def message_archetype(body: str) -> str:
    t = body.lower() if isinstance(body, str) else ""
    has_question = "?" in t
    has_action = bool(re.search(r"\b(?:please|can you|could you|need to|action required|deadline|by eod|asap)\b", t))
    has_time = bool(re.search(r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|by \d{1,2}(?::\d{2})?\s?(?:am|pm))\b", t))
    has_attachment = bool(re.search(r"\b(?:attached|attachment|see attached|enclosed)\b", t))
    if has_action and has_time:
        return "deadline_request"
    if has_action and has_question:
        return "request_with_question"
    if has_action:
        return "direct_request"
    if has_question:
        return "information_request"
    if has_attachment:
        return "document_delivery"
    return "informational_update"


def infer_gold_priority(intent: str, archetype: str, text: str) -> str:
    t = text.lower()
    urgency = bool(re.search(r"\b(?:urgent|asap|immediately|today|by eod|deadline)\b", t))
    if archetype == "deadline_request":
        return "P0" if urgency else "P1"
    if intent in {"legal_compliance", "decision_approval"} and urgency:
        return "P0"
    if intent in {"request_action", "legal_compliance", "decision_approval"}:
        return "P1"
    if archetype in {"direct_request", "request_with_question", "information_request"}:
        return "P1"
    return "P2"


def infer_gold_reply_type(intent: str, archetype: str) -> str:
    if archetype == "information_request":
        return "REQUEST_INFO"
    if archetype in {"deadline_request", "direct_request", "request_with_question"}:
        return "ANSWER"
    if intent in {"status_update", "announcement_broadcast", "social_personal"}:
        return "ACK"
    if intent in {"trading_market", "legal_compliance", "decision_approval"}:
        return "ANSWER"
    return "NONE"


def infer_gold_action_items(priority: str, subject: str) -> list[dict[str, Any]]:
    if priority == "P2":
        return []
    action = normalize_subject(subject)[:80] or "review and respond"
    return [{"action": action, "owner": "me", "due": "none", "blocking": priority == "P0"}]


def infer_heuristic_priority(intent: str, archetype: str, text: str) -> str:
    t = text.lower()
    urgent = bool(re.search(r"\b(?:urgent|asap|immediately|today|by eod|deadline|critical)\b", t))
    if urgent and (archetype in {"deadline_request", "request_with_question"} or intent in {"legal_compliance", "decision_approval"}):
        return "P0"
    if intent in {"request_action", "legal_compliance", "decision_approval"}:
        return "P1"
    if archetype in {"direct_request", "request_with_question", "information_request"}:
        return "P1"
    return "P2"


def infer_heuristic_reply_type(intent: str, archetype: str) -> str:
    if intent in {"legal_compliance", "decision_approval"}:
        return "ANSWER"
    if archetype in {"information_request"}:
        return "REQUEST_INFO"
    if archetype in {"deadline_request", "direct_request", "request_with_question"}:
        return "ANSWER"
    if intent in {"status_update", "announcement_broadcast", "social_personal"}:
        return "ACK"
    if archetype == "document_delivery":
        return "ACK"
    return "NONE"


def infer_heuristic_action_items(priority: str, subject: str) -> list[dict[str, Any]]:
    if priority == "P2":
        return []
    action = normalize_subject(subject)[:60] or "triage and respond"
    due = "none" if priority == "P1" else "today"
    return [{"action": action, "owner": "me", "due": due, "blocking": priority == "P0"}]


def load_message_pool(body_sample_cache: Path) -> pd.DataFrame:
    if not body_sample_cache.exists():
        raise FileNotFoundError(f"Body sample missing: {body_sample_cache}")
    df = pd.read_parquet(body_sample_cache)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["norm_subject"] = df["subject"].map(normalize_subject)
    text = (df["norm_subject"] + " " + df["body"].str.slice(0, 1000)).str.lower()
    df["intent"] = text.map(detect_intent)
    df["archetype"] = df["body"].map(message_archetype)
    return df


def load_intent_distribution(intent_dist_csv: Path, pool: pd.DataFrame) -> pd.DataFrame:
    if intent_dist_csv.exists():
        dist = pd.read_csv(intent_dist_csv)
        dist = dist[["intent", "share"]].copy()
    else:
        dist = pool["intent"].value_counts(normalize=True).rename_axis("intent").reset_index(name="share")
    dist = dist[dist["share"] > 0].copy()
    dist["share"] = dist["share"] / dist["share"].sum()
    return dist


def load_archetype_distribution(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Conservative fallback.
        return pd.DataFrame(
            {
                "archetype": [
                    "informational_update",
                    "deadline_request",
                    "direct_request",
                    "information_request",
                    "request_with_question",
                    "document_delivery",
                ],
                "share": [0.30, 0.25, 0.18, 0.14, 0.09, 0.04],
            }
        )
    df = pd.read_csv(path)
    if not {"archetype", "share"}.issubset(set(df.columns)):
        raise RuntimeError(f"{path} must include columns: archetype, share")
    keep = df[["archetype", "share"]].copy()
    keep = keep[keep["share"] > 0].copy()
    keep["share"] = keep["share"] / keep["share"].sum()
    return keep


SUBJECT_TEMPLATES: dict[str, list[str]] = {
    "trading_market": [
        "Gas trade bid/offer {topic}",
        "Power market price check {topic}",
        "Trading recap {topic}",
    ],
    "legal_compliance": [
        "Contract review {topic}",
        "Agreement terms {topic}",
        "FERC compliance item {topic}",
        "Litigation matter {topic}",
    ],
    "ops_technical": [
        "Server incident {topic}",
        "System outage {topic}",
        "Deploy failure {topic}",
        "Bug fix {topic}",
    ],
    "meeting_scheduling": [
        "Meeting schedule {topic}",
        "Agenda for call {topic}",
        "Conference timing {topic}",
    ],
    "request_action": [
        "Action required {topic}",
        "Request for help {topic}",
        "Urgent request {topic}",
    ],
    "decision_approval": [
        "Approval needed {topic}",
        "Sign-off request {topic}",
        "Decision needed {topic}",
    ],
    "status_update": [
        "Status report {topic}",
        "Weekly summary {topic}",
        "Progress recap {topic}",
    ],
    "hr_admin": [
        "Expense reimbursement {topic}",
        "Vacation request {topic}",
        "Payroll question {topic}",
    ],
    "announcement_broadcast": [
        "Policy notice {topic}",
        "Organization announcement {topic}",
        "Newsletter notice {topic}",
    ],
    "social_personal": [
        "Thank you {topic}",
        "Congratulations {topic}",
        "Holiday plans {topic}",
    ],
    "uncategorized": [
        "Project note {topic}",
        "Planning item {topic}",
        "Follow-up item {topic}",
    ],
}


def _pick_subject(intent: str, topic: str, rng: np.random.Generator) -> str:
    templates = SUBJECT_TEMPLATES.get(intent) or SUBJECT_TEMPLATES["uncategorized"]
    return str(rng.choice(templates)).format(topic=topic)


def _intent_phrase(intent: str) -> str:
    # Each phrase aims to trigger detect_intent() for the intended class,
    # while avoiding earlier classes' keywords.
    return {
        "trading_market": "gas trade",
        "legal_compliance": "contract",
        "ops_technical": "server incident",
        "meeting_scheduling": "meeting schedule",
        "request_action": "action required",
        "decision_approval": "approval",
        "status_update": "status report",
        "hr_admin": "expense reimbursement",
        "announcement_broadcast": "policy notice",
        "social_personal": "thanks",
        "uncategorized": "project item",
    }.get(intent, "project item")


def _action_token_for_intent(intent: str) -> str:
    # For intents that appear *after* request_action in INTENT_PATTERNS order,
    # avoid "please/can you/request/urgent" which would be classified as request_action.
    if intent in {"decision_approval", "status_update", "hr_admin", "announcement_broadcast", "social_personal", "uncategorized"}:
        return "need to"
    if intent == "request_action":
        return "please"
    return "please"


def _time_token(rng: np.random.Generator, urgent: bool) -> str:
    if urgent:
        # Use time markers that also register as time in message_archetype().
        # (Its regex does NOT treat "by EOD" as a time reference.)
        return str(rng.choice(["today", "by 3pm", "deadline today"]))
    return str(rng.choice(["tomorrow", "monday", "by 3pm"]))


def _compose_body(intent: str, archetype: str, topic: str, rng: np.random.Generator) -> str:
    phrase = _intent_phrase(intent)
    if archetype == "informational_update":
        # Avoid "update" unless status_update intent, so we don't accidentally reclassify.
        if intent == "status_update":
            return f"FYI {phrase} for {topic}. Summary: progress continues; no immediate action needed."
        return f"FYI {phrase} for {topic}. For awareness only; no immediate action needed."
    if archetype == "document_delivery":
        return f"Attached is the {phrase} for {topic}. See attached for details."
    if archetype == "information_request":
        return f"Question: for {topic} ({phrase}), what is the next step?"
    if archetype == "direct_request":
        action = _action_token_for_intent(intent)
        return f"We {action} review {topic} ({phrase}) and reply with next steps."
    if archetype == "request_with_question":
        action = _action_token_for_intent(intent)
        return f"We {action} review {topic} ({phrase}). Which option should we choose and why?"
    if archetype == "deadline_request":
        action = _action_token_for_intent(intent)
        urgent = bool(rng.random() < 0.65)
        when = _time_token(rng, urgent=urgent)
        extra = " This is urgent." if urgent else ""
        return f"We {action} finalize {topic} ({phrase}) {when}.{extra}"
    return f"FYI {phrase} for {topic}."


def generate_template_pool(
    *,
    pool_size: int,
    topic_group_size: int,
    seed: int,
    intent_dist: pd.DataFrame,
    archetype_dist: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if pool_size <= 0:
        raise ValueError("--template-pool-size must be positive")
    if topic_group_size < 2:
        raise ValueError("--template-topic-group-size must be >= 2")

    intents = intent_dist["intent"].astype(str).tolist()
    intent_w = intent_dist["share"].to_numpy(dtype=float)
    archetypes = archetype_dist["archetype"].astype(str).tolist()
    arche_w = archetype_dist["share"].to_numpy(dtype=float)

    n_topics = max(1, int(pool_size // topic_group_size))
    rows: list[dict[str, Any]] = []
    for topic_id in range(n_topics):
        intent_target = str(rng.choice(intents, p=intent_w))
        topic = f"T{topic_id:04d}"
        base_subject = _pick_subject(intent=intent_target, topic=topic, rng=rng)
        for j in range(topic_group_size):
            archetype_target = str(rng.choice(archetypes, p=arche_w))
            body = _compose_body(intent=intent_target, archetype=archetype_target, topic=topic, rng=rng)
            subject = base_subject if j == 0 else (f"Re: {base_subject}" if rng.random() < 0.6 else base_subject)
            norm = normalize_subject(subject)
            text = (norm + " " + body[:1000]).lower()
            intent = detect_intent(text)
            archetype = message_archetype(body)
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "norm_subject": norm,
                    "intent": intent,
                    "archetype": archetype,
                    "intent_target": intent_target,
                    "archetype_target": archetype_target,
                }
            )
            if len(rows) >= pool_size:
                break
        if len(rows) >= pool_size:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Generated empty template pool")
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["norm_subject"] = df["norm_subject"].fillna("").astype(str)
    df["intent"] = df["intent"].fillna("uncategorized").astype(str)
    df["archetype"] = df["archetype"].fillna("informational_update").astype(str)
    return df


GENERIC_THREAD_SUBJECTS = {
    "",
    "(no subject)",
    "no subject",
    "meeting",
    "lunch",
    "hey",
    "hi",
    "hello",
    "hello!",
    "fyi",
    "test",
    "update",
    "reminder",
}


@dataclass(frozen=True)
class ThreadTopicPool:
    subjects: list[str]
    indices_by_subject: dict[str, np.ndarray]
    top_intent_by_subject: dict[str, str]
    subjects_by_top_intent: dict[str, list[str]]


def build_thread_topic_pool(pool: pd.DataFrame) -> ThreadTopicPool:
    """Build a subject-grouped pool so per-thread context is coherent.

    Without this, threads are random mixes of unrelated Enron messages, and the provided
    thread_context is actively misleading.
    """
    if "norm_subject" not in pool.columns:
        raise RuntimeError("pool missing norm_subject; did load_message_pool() run?")
    if "intent" not in pool.columns:
        raise RuntimeError("pool missing intent; did load_message_pool() run?")

    sizes = pool.groupby("norm_subject").size().rename("n").reset_index()
    sizes["norm_subject"] = sizes["norm_subject"].fillna("").astype(str)
    sizes["len"] = sizes["norm_subject"].map(len)
    sizes["words"] = sizes["norm_subject"].map(lambda s: len([w for w in str(s).split() if w]))

    # Keep enough candidates to support high-N episodes, but filter out empty/generic subjects
    # that would merge unrelated content (e.g., "", "meeting", "fyi").
    cand = sizes[
        (sizes["n"] >= 2)
        & (sizes["len"] >= 12)
        & (sizes["words"] >= 2)
        & (~sizes["norm_subject"].str.strip().isin(GENERIC_THREAD_SUBJECTS))
    ].copy()
    if cand.empty:
        # Fallback: allow shorter subjects if the dataset is extremely sparse.
        cand = sizes[(sizes["n"] >= 2) & (sizes["norm_subject"].str.strip() != "")].copy()
    subjects = cand["norm_subject"].astype(str).tolist()

    indices_by_subject: dict[str, np.ndarray] = {}
    top_intent_by_subject: dict[str, str] = {}
    subjects_by_top_intent: dict[str, list[str]] = {}

    for subj, grp in pool[pool["norm_subject"].isin(subjects)].groupby("norm_subject"):
        idxs = grp.index.to_numpy()
        if len(idxs) < 2:
            continue
        indices_by_subject[str(subj)] = idxs
        # Dominant intent is a rough, metadata-only proxy for topic type.
        top_intent = str(grp["intent"].value_counts().index[0])
        top_intent_by_subject[str(subj)] = top_intent
        subjects_by_top_intent.setdefault(top_intent, []).append(str(subj))

    final_subjects = sorted(indices_by_subject.keys())
    if not final_subjects:
        raise RuntimeError("No eligible subject groups found for coherent threading")
    return ThreadTopicPool(
        subjects=final_subjects,
        indices_by_subject=indices_by_subject,
        top_intent_by_subject=top_intent_by_subject,
        subjects_by_top_intent=subjects_by_top_intent,
    )


@dataclass
class Message:
    email_id: str
    thread_id: str
    topic_norm_subject: str
    arrival_min: float
    subject: str
    body: str
    intent: str
    archetype: str
    gold_priority: str
    gold_reply_type: str
    gold_action_items: list[dict[str, Any]]


class InboxEnv:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = sorted(messages, key=lambda x: (x.arrival_min, x.email_id))
        self.by_id = {m.email_id: m for m in self.messages}
        self.decisions: dict[str, dict[str, Any]] = {}

    def list_unread(self, current_time_min: float) -> list[dict[str, Any]]:
        unread = []
        for m in self.messages:
            if m.arrival_min <= current_time_min and m.email_id not in self.decisions:
                unread.append(
                    {
                        "email_id": m.email_id,
                        "thread_id": m.thread_id,
                        "topic_norm_subject": m.topic_norm_subject,
                        "arrival_min": m.arrival_min,
                        "subject": m.subject[:120],
                        "intent": m.intent,
                    }
                )
        return unread

    def open_email(self, email_id: str) -> dict[str, Any]:
        m = self.by_id[email_id]
        return {
            "email_id": m.email_id,
            "thread_id": m.thread_id,
            "topic_norm_subject": m.topic_norm_subject,
            "subject": m.subject,
            "body": m.body[:3500],
            "arrival_min": m.arrival_min,
            "intent": m.intent,
            "archetype": m.archetype,
        }

    def get_thread(self, thread_id: str, k: int, current_time_min: float) -> list[dict[str, Any]]:
        rows = [
            m
            for m in self.messages
            if m.thread_id == thread_id and m.arrival_min <= current_time_min
        ]
        rows = rows[-k:]
        return [{"email_id": m.email_id, "arrival_min": m.arrival_min, "subject": m.subject[:120], "body_excerpt": m.body[:280]} for m in rows]

    def submit_decision(self, email_id: str, decision: dict[str, Any]) -> None:
        self.decisions[email_id] = decision

    def next_arrival_after(self, current_time_min: float) -> float | None:
        future = [m.arrival_min for m in self.messages if m.email_id not in self.decisions and m.arrival_min > current_time_min]
        return min(future) if future else None

    def unresolved_count(self) -> int:
        return len(self.messages) - len(self.decisions)


class HeuristicAgent:
    name = "heuristic"

    def decide(self, email: dict[str, Any], thread_context: list[dict[str, Any]], unread_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = thread_context, unread_count
        text = f"{email['subject']} {email['body']}".lower()
        intent = detect_intent(text)
        archetype = message_archetype(email["body"])
        priority = infer_heuristic_priority(intent=intent, archetype=archetype, text=text)
        reply_type = infer_heuristic_reply_type(intent=intent, archetype=archetype)
        action_items = infer_heuristic_action_items(priority, email["subject"])
        draft = "Thanks. I will review and follow up shortly."
        if priority == "P0":
            draft = "Received. Treating as urgent and responding now."
        decision = {
            "priority": priority,
            "action_items": action_items,
            "reply_type": reply_type,
            "reply_key_points": [normalize_subject(email["subject"])[:80] or "acknowledge request"],
            "draft_reply": draft,
        }
        meta = {"latency_sec": 0.05, "input_tokens": 0, "output_tokens": 0, "raw_invalid": 0}
        return decision, meta


class OpenAIAgent:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_sec: int,
        temperature: float | None,
        max_attempts: int,
        reasoning_mode: str,
        max_output_tokens: int,
        prompt_profile: str,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.max_attempts = max(1, int(max_attempts))
        self.reasoning_mode = reasoning_mode
        self.max_output_tokens = max(0, int(max_output_tokens))
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.session = requests.Session()
        rubric = RUBRIC_MEANING if str(prompt_profile) == "meaning" else RUBRIC_STRICT
        self.system_prompt = (
            "You are an inbox triage agent. "
            "Return strict JSON with fields: priority, action_items, reply_type, reply_key_points, draft_reply. "
            "Use priority in {P0,P1,P2}. reply_type in {NONE,ACK,ANSWER,REQUEST_INFO,REDIRECT}. "
            f"{rubric} "
            "Do not claim actions completed unless explicitly present in email/thread."
        )
        self.response_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string"},
                            "owner": {"type": "string"},
                            "due": {"type": "string"},
                            "blocking": {"type": "boolean"},
                        },
                        "required": ["action", "owner", "due", "blocking"],
                    },
                },
                "reply_type": {
                    "type": "string",
                    "enum": ["NONE", "ACK", "ANSWER", "REQUEST_INFO", "REDIRECT"],
                },
                "reply_key_points": {"type": "array", "items": {"type": "string"}},
                "draft_reply": {"type": "string"},
            },
            "required": [
                "priority",
                "action_items",
                "reply_type",
                "reply_key_points",
                "draft_reply",
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
                    sleep_sec = (2**attempt) + float(np.random.uniform(0, 1))
                    time.sleep(sleep_sec)
                    continue
                if resp.status_code >= 400:
                    # Include server payload for actionable diagnostics.
                    msg = resp.text[:1200]
                    raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {msg}")
                return resp.json()
            except Exception as exc:  # pragma: no cover - network-dependent path
                last_error = exc
                if attempt == self.max_attempts:
                    raise
                sleep_sec = (2**attempt) + float(np.random.uniform(0, 1))
                time.sleep(sleep_sec)
        raise RuntimeError(f"OpenAI request failed: {last_error}")

    def decide(self, email: dict[str, Any], thread_context: list[dict[str, Any]], unread_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_obj = {
            "unread_count": unread_count,
            "email": email,
            "thread_context": thread_context,
            "output_schema": {
                "priority": "P0|P1|P2",
                "action_items": [{"action": "...", "owner": "me|sender|third_party|unknown", "due": "YYYY-MM-DD|YYYY-MM-DDTHH:MM|none", "blocking": True}],
                "reply_type": "NONE|ACK|ANSWER|REQUEST_INFO|REDIRECT",
                "reply_key_points": ["...", "..."],
                "draft_reply": "...",
            },
        }
        payload = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "triage_output",
                    "schema": self.response_schema,
                    "strict": True,
                }
            },
        }
        if self.reasoning_mode == "high":
            payload["reasoning"] = {"effort": "high"}
        if self.max_output_tokens > 0:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
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
        "action_items": decision.get("action_items", []),
        "reply_type": decision.get("reply_type", "NONE"),
        "reply_key_points": decision.get("reply_key_points", []),
        "draft_reply": decision.get("draft_reply", ""),
    }
    if out["priority"] not in REQUIRED_PRIORITIES:
        out["priority"] = "P2"
        invalid = 1
    if out["reply_type"] not in REQUIRED_REPLY_TYPES:
        out["reply_type"] = "NONE"
        invalid = 1
    if not isinstance(out["action_items"], list):
        out["action_items"] = []
        invalid = 1
    if not isinstance(out["reply_key_points"], list):
        out["reply_key_points"] = []
        invalid = 1
    if not isinstance(out["draft_reply"], str):
        out["draft_reply"] = str(out["draft_reply"])
        invalid = 1
    return out, invalid


def action_presence_score(pred_items: list[Any], gold_items: list[Any]) -> float:
    pred_has = len(pred_items) > 0
    gold_has = len(gold_items) > 0
    return 1.0 if pred_has == gold_has else 0.0


def hallucination_penalty(draft_reply: str, source_text: str) -> float:
    d = draft_reply.lower()
    s = source_text.lower()
    for phrase in BANNED_COMMITMENTS:
        if phrase in d and phrase not in s:
            return 1.0
    return 0.0


def score_message(
    message: Message,
    decision: dict[str, Any],
    process_end_min: float,
) -> dict[str, float]:
    priority_acc = float(decision["priority"] == message.gold_priority)
    reply_acc = float(decision["reply_type"] == message.gold_reply_type)
    action_score = action_presence_score(decision["action_items"], message.gold_action_items)
    halluc = hallucination_penalty(decision["draft_reply"], f"{message.subject}\n{message.body}")
    latency_min = process_end_min - message.arrival_min
    sla_min = PRIORITY_SLA_MIN[message.gold_priority]
    on_time = float(latency_min <= sla_min)

    quality = 0.40 * priority_acc + 0.30 * reply_acc + 0.30 * action_score - 0.20 * halluc
    if on_time < 1.0 and message.gold_priority in {"P0", "P1"}:
        quality *= 0.2
    quality = max(0.0, min(1.0, quality))
    return {
        "priority_acc": priority_acc,
        "reply_acc": reply_acc,
        "action_presence_acc": action_score,
        "hallucination": halluc,
        "latency_min": latency_min,
        "on_time": on_time,
        "quality_score": quality,
    }


def generate_episode(
    pool: pd.DataFrame,
    intent_dist: pd.DataFrame,
    topic_pool: ThreadTopicPool,
    n_threads: int,
    messages_per_episode: int,
    seed: int,
    episode_idx: int,
) -> list[Message]:
    rng = np.random.default_rng(seed + episode_idx * 1009 + n_threads * 17)
    intents = intent_dist["intent"].tolist()
    weights = intent_dist["share"].to_numpy()

    messages: list[Message] = []
    current_time = 0.0
    thread_ids = [f"th-{i+1}" for i in range(n_threads)]
    current_thread = thread_ids[0]

    # Assign each thread a coherent subject group (topic). Prefer sampling subject groups with
    # a dominant intent matching the empirical intent distribution.
    used_subjects: set[str] = set()
    thread_subject: dict[str, str] = {}
    thread_seq: dict[str, list[int]] = {}
    thread_pos: dict[str, int] = {}

    for t_id in thread_ids:
        desired_intent = str(rng.choice(intents, p=weights))
        candidates = topic_pool.subjects_by_top_intent.get(desired_intent) or topic_pool.subjects
        # Prefer unused subjects within-episode for clearer separation of threads.
        fresh = [s for s in candidates if s not in used_subjects]
        pick_from = fresh if fresh else candidates
        subj = str(rng.choice(pick_from))
        used_subjects.add(subj)
        thread_subject[t_id] = subj
        idxs = topic_pool.indices_by_subject[subj]
        thread_seq[t_id] = rng.permutation(idxs).tolist()
        thread_pos[t_id] = 0

    for i in range(messages_per_episode):
        if i < n_threads:
            thread_id = thread_ids[i]
        else:
            if rng.random() < 0.45:
                thread_id = current_thread
            else:
                thread_id = str(rng.choice(thread_ids))
            current_thread = thread_id

        subj = thread_subject[thread_id]
        seq = thread_seq[thread_id]
        pos = thread_pos[thread_id]
        if pos >= len(seq):
            # Re-shuffle once exhausted; keep coherent topic but avoid repeating the same
            # single row endlessly for small groups.
            idxs = topic_pool.indices_by_subject[subj]
            seq = rng.permutation(idxs).tolist()
            thread_seq[thread_id] = seq
            pos = 0
        row = pool.loc[int(seq[pos])]
        thread_pos[thread_id] = pos + 1

        gap = float(rng.exponential(scale=2.5))
        if rng.random() < 0.20:
            gap = float(rng.exponential(scale=0.5))
        current_time += gap

        subject = str(row["subject"])[:220]
        body = str(row["body"])[:3500]
        intent = str(row["intent"])
        archetype = str(row["archetype"])
        text = f"{subject}\n{body}"
        gold_priority = infer_gold_priority(intent=intent, archetype=archetype, text=text)
        gold_reply_type = infer_gold_reply_type(intent=intent, archetype=archetype)
        gold_action_items = infer_gold_action_items(priority=gold_priority, subject=subject)
        messages.append(
            Message(
                email_id=f"ep{episode_idx}-m{i+1}",
                thread_id=thread_id,
                topic_norm_subject=subj,
                arrival_min=round(current_time, 3),
                subject=subject,
                body=body,
                intent=intent,
                archetype=archetype,
                gold_priority=gold_priority,
                gold_reply_type=gold_reply_type,
                gold_action_items=gold_action_items,
            )
        )
    return messages


def run_episode(
    env: InboxEnv,
    agent: HeuristicAgent | OpenAIAgent,
    thread_context_k: int,
    max_calls: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    current_time = 0.0
    calls = 0
    rows: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    raw_invalid_total = 0
    api_error_total = 0

    while env.unresolved_count() > 0:
        unread = env.list_unread(current_time)
        if not unread:
            next_arrival = env.next_arrival_after(current_time)
            if next_arrival is None:
                break
            current_time = next_arrival
            continue

        unread_sorted = sorted(unread, key=lambda x: (x["arrival_min"], x["email_id"]))
        target = unread_sorted[0]
        email = env.open_email(target["email_id"])
        thread_context = env.get_thread(email["thread_id"], k=thread_context_k, current_time_min=current_time)

        if calls >= max_calls:
            raise RuntimeError(f"Max calls reached ({max_calls})")
        calls += 1
        api_error = 0
        try:
            raw_decision, meta = agent.decide(
                email=email,
                thread_context=thread_context,
                unread_count=len(unread),
            )
        except Exception as exc:  # pragma: no cover - live API dependent path
            # Treat transient/deterministic API failures as degraded decisions, not crashed runs.
            api_error = 1
            raw_decision = {
                "priority": "P2",
                "action_items": [],
                "reply_type": "NONE",
                "reply_key_points": [],
                "draft_reply": "",
            }
            meta = {
                "latency_sec": 1.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "raw_invalid": 1,
                "error": str(exc)[:300],
            }
        decision, invalid = validate_decision(raw_decision)
        invalid_total = int(invalid + meta.get("raw_invalid", 0))
        raw_invalid_total += invalid_total
        api_error_total += api_error

        # Capture what the agent actually saw for audit/judging.
        thread_context_json = json.dumps(thread_context, ensure_ascii=True)

        latency_sec = float(meta.get("latency_sec", 0.05))
        process_time_min = max(0.40, latency_sec / 60.0)
        current_time = max(current_time, float(email["arrival_min"])) + process_time_min
        env.submit_decision(target["email_id"], decision)

        msg = env.by_id[target["email_id"]]
        scores = score_message(message=msg, decision=decision, process_end_min=current_time)
        total_input_tokens += int(meta.get("input_tokens", 0))
        total_output_tokens += int(meta.get("output_tokens", 0))

        rows.append(
            {
                "email_id": msg.email_id,
                "thread_id": msg.thread_id,
                "topic_norm_subject": msg.topic_norm_subject,
                "arrival_min": msg.arrival_min,
                "process_end_min": current_time,
                "subject": msg.subject[:220],
                "body": msg.body[:3500],
                "thread_context_json": thread_context_json,
                "intent": msg.intent,
                "archetype": msg.archetype,
                "gold_priority": msg.gold_priority,
                "pred_priority": decision["priority"],
                "gold_reply_type": msg.gold_reply_type,
                "pred_reply_type": decision["reply_type"],
                "pred_action_items": json.dumps(decision["action_items"], ensure_ascii=True),
                "pred_reply_key_points": json.dumps(decision["reply_key_points"], ensure_ascii=True),
                "pred_draft_reply": decision["draft_reply"],
                "invalid_output": invalid_total,
                "api_error": api_error,
                "input_tokens": int(meta.get("input_tokens", 0)),
                "output_tokens": int(meta.get("output_tokens", 0)),
                **scores,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Episode produced no rows")

    def safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) > 0 else 0.0

    p0 = df[df["gold_priority"] == "P0"]
    p1 = df[df["gold_priority"] == "P1"]
    metrics = {
        "messages": float(len(df)),
        "mean_quality": safe_mean(df["quality_score"]),
        "priority_acc": safe_mean(df["priority_acc"]),
        "reply_acc": safe_mean(df["reply_acc"]),
        "action_presence_acc": safe_mean(df["action_presence_acc"]),
        "p0_sla": safe_mean(p0["on_time"]) if len(p0) > 0 else 1.0,
        "p1_sla": safe_mean(p1["on_time"]) if len(p1) > 0 else 1.0,
        "mean_latency_min": safe_mean(df["latency_min"]),
        "invalid_rate": safe_mean(df["invalid_output"]),
        "input_tokens": float(total_input_tokens),
        "output_tokens": float(total_output_tokens),
        "calls": float(calls),
        "raw_invalid_total": float(raw_invalid_total),
        "api_error_rate": float(api_error_total / len(df)),
    }
    return df, metrics


def summarize_n_level(df_episode: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_episode.groupby("n_threads")
        .agg(
            episodes=("episode", "nunique"),
            mean_quality=("mean_quality", "mean"),
            priority_acc=("priority_acc", "mean"),
            reply_acc=("reply_acc", "mean"),
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
    return out


def write_report(
    n_summary: pd.DataFrame,
    args: argparse.Namespace,
    agent_name: str,
    n_star: int | None,
    total_cost: float,
    output_dir: Path,
) -> None:
    lines = [
        "# LLM Capacity Evaluation Report",
        "",
        f"- Agent: **{agent_name}**",
        f"- Model: **{args.model if agent_name == 'openai' else 'n/a'}**",
        f"- Prompt profile: **{args.prompt_profile}**",
        f"- Pool source: **{args.pool_source}**",
        f"- N values: **{args.n_values}**",
        f"- Episodes per N: **{args.episodes_per_n}**",
        f"- Messages per episode: **{args.messages_per_episode}**",
        f"- Capacity threshold q: **{args.score_threshold_q:.2f}**",
        f"- P0 SLA threshold: **{args.p0_sla_threshold:.2f}**",
        f"- Estimated API cost: **${total_cost:.4f}**",
        "",
        "This report is **rubric-compliance scoring** against heuristic gold labels. "
        "For **judgment scoring** (LLM-as-judge) and an `info_sufficient` fairness check, "
        "run `python scripts/judge_llm_capacity_run.py` on this run directory.",
        "",
        "## N-Level Summary",
    ]
    for _, row in n_summary.iterrows():
        lines.append(
            f"- N={int(row['n_threads'])}: quality={row['mean_quality']:.3f}, "
            f"p0_sla={row['p0_sla']:.3f}, p1_sla={row['p1_sla']:.3f}, "
            f"priority_acc={row['priority_acc']:.3f}, invalid_rate={row['invalid_rate']:.3f}, "
            f"api_error_rate={row['api_error_rate']:.3f}"
        )
    lines.append("")
    if n_star is None:
        lines.append("- Estimated capacity N*: **none passed threshold**")
    else:
        lines.append(f"- Estimated capacity N*: **{n_star}**")
    (output_dir / "llm_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_tag.strip():
        run_tag = args.run_tag.strip()
    else:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_tag = args.model.replace("/", "_").replace(":", "_")
        run_tag = f"{args.agent}_{model_tag}_{ts}"
    run_dir = args.output_dir / "llm_eval_runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]

    if str(args.pool_source) == "template":
        # For leakage-robust eval, generate a synthetic pool that still supports coherent threading.
        dummy_pool = pd.DataFrame({"intent": []})
        intent_dist = load_intent_distribution(args.intent_dist_csv, pool=dummy_pool)
        archetype_dist = load_archetype_distribution(args.archetype_dist_csv)
        pool = generate_template_pool(
            pool_size=int(args.template_pool_size),
            topic_group_size=int(args.template_topic_group_size),
            seed=int(args.seed),
            intent_dist=intent_dist,
            archetype_dist=archetype_dist,
        )
    else:
        pool = load_message_pool(args.body_sample_cache)
        intent_dist = load_intent_distribution(args.intent_dist_csv, pool=pool)

    topic_pool = build_thread_topic_pool(pool)

    # Persist minimal pool metadata for auditability (and to prove non-Enron when pool_source=template).
    pool_meta = {
        "pool_source": str(args.pool_source),
        "pool_rows": int(len(pool)),
        "unique_norm_subjects": int(pool["norm_subject"].nunique()) if "norm_subject" in pool.columns else 0,
    }
    (run_dir / "pool_meta.json").write_text(json.dumps(pool_meta, indent=2) + "\n", encoding="utf-8")

    if args.agent == "heuristic":
        agent: HeuristicAgent | OpenAIAgent = HeuristicAgent()
    else:
        agent = OpenAIAgent(
            model=args.model,
            base_url=args.openai_base_url,
            timeout_sec=args.openai_timeout_sec,
            temperature=args.temperature,
            max_attempts=args.openai_max_attempts,
            reasoning_mode=args.openai_reasoning_mode,
            max_output_tokens=args.openai_max_output_tokens,
            prompt_profile=str(args.prompt_profile),
        )

    all_message_rows: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    episode_counter = 0

    for n_threads in n_values:
        for ep in range(args.episodes_per_n):
            messages = generate_episode(
                pool=pool,
                intent_dist=intent_dist,
                topic_pool=topic_pool,
                n_threads=n_threads,
                messages_per_episode=args.messages_per_episode,
                seed=args.seed,
                episode_idx=episode_counter,
            )
            env = InboxEnv(messages)
            msg_df, metrics = run_episode(
                env=env,
                agent=agent,
                thread_context_k=args.thread_context_k,
                max_calls=args.max_calls,
            )
            msg_df["episode"] = episode_counter
            msg_df["n_threads"] = n_threads
            all_message_rows.append(msg_df)
            episode_rows.append(
                {
                    "episode": episode_counter,
                    "n_threads": n_threads,
                    **metrics,
                }
            )
            episode_counter += 1

    message_log = pd.concat(all_message_rows, ignore_index=True)
    episode_summary = pd.DataFrame(episode_rows)
    n_summary = summarize_n_level(episode_summary)

    total_input_tokens = float(episode_summary["input_tokens"].sum())
    total_output_tokens = float(episode_summary["output_tokens"].sum())
    total_cost = (total_input_tokens * args.input_cost_per_1m + total_output_tokens * args.output_cost_per_1m) / 1_000_000.0

    n_summary["passes_threshold"] = (
        (n_summary["mean_quality"] >= args.score_threshold_q)
        & (n_summary["p0_sla"] >= args.p0_sla_threshold)
    )
    passed_n = n_summary[n_summary["passes_threshold"]]["n_threads"].tolist()
    n_star = int(max(passed_n)) if passed_n else None

    message_log.to_csv(run_dir / "llm_eval_message_log.csv", index=False)
    episode_summary.to_csv(run_dir / "llm_eval_episode_summary.csv", index=False)
    n_summary.to_csv(run_dir / "llm_eval_n_summary.csv", index=False)
    write_report(
        n_summary=n_summary,
        args=args,
        agent_name=agent.name,
        n_star=n_star,
        total_cost=total_cost,
        output_dir=run_dir,
    )

    print(f"Agent: {agent.name}")
    print(f"N*: {n_star if n_star is not None else 'none'}")
    print(f"Run tag: {run_tag}")
    print(f"Wrote: {(run_dir / 'llm_eval_report.md').resolve()}")


if __name__ == "__main__":
    main()
