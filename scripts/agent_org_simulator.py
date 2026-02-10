#!/usr/bin/env python3
"""Run multi-agent organization simulation from breakpoint config pack."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

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
MEMORY_CONTRACT = (
    "Memory contract: You are evaluated on recalling thread facts using ONLY your scratchpad. "
    "When you see a project code like A742 (pattern [A-Z][0-9]{3}), record it in your scratchpad keyed by thread_id. "
    "Later follow-ups may omit the code; use your scratchpad to include the code in facts_used and/or the reply."
)

TASK_KIND_MAP = {
    "quick_resolution": "quick",
    "ongoing_operations": "ops",
    "approval_governance": "approval",
    "specialist_escalation": "specialist",
    "cross_team_program": "program",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("results/scenarios/config_pack"),
    )
    parser.add_argument(
        "--shock-id",
        default="",
        help="Specific shock id from config filename stem. Empty runs all configs in index.",
    )
    parser.add_argument("--agent", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default=os.getenv("RESEARCH_MODEL", "gpt-5.2"))
    parser.add_argument(
        "--prompt-profile",
        choices=["meaning", "strict"],
        default="meaning",
        help="meaning = tests judgment; strict = tests rubric compliance.",
    )
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-timeout-sec", type=int, default=90)
    parser.add_argument("--openai-max-attempts", type=int, default=3)
    parser.add_argument("--openai-reasoning-mode", choices=["auto", "high"], default="auto")
    parser.add_argument("--openai-max-output-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--episodes-per-phase", type=int, default=2)
    parser.add_argument("--episode-hours", type=int, default=0, help="0 uses config recommendation (8h).")
    parser.add_argument(
        "--messages-per-episode",
        type=int,
        default=0,
        help="0 uses config recommendation.",
    )
    parser.add_argument(
        "--threads-per-episode",
        type=int,
        default=0,
        help="0 uses config recommendation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total-calls", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulator_runs"))
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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_task_kind(task_type: str) -> str:
    t = str(task_type)
    for prefix, kind in TASK_KIND_MAP.items():
        if t.startswith(prefix):
            return kind
    return "ops"


def pick_task_type(task_probs: list[tuple[str, float]], rng: np.random.Generator) -> str:
    if not task_probs:
        return "quick_resolution_c4"
    labels = [x[0] for x in task_probs]
    weights = np.array([x[1] for x in task_probs], dtype=float)
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / np.sum(weights)
    return str(rng.choice(labels, p=weights))


def make_project_code(rng: np.random.Generator) -> str:
    return f"{str(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')))}{int(rng.integers(100, 999))}"


def build_message_text(
    *,
    task_kind: str,
    thread_id: str,
    project_code: str,
    needs_memory: bool,
    escalation_flag: bool,
    specialist_flag: bool,
    approval_flag: bool,
    fanout_hint: int,
    anchor: bool,
) -> tuple[str, str, str, str]:
    if task_kind == "approval":
        subject = f"Approval needed for {project_code}"
        reply_type = "ANSWER"
        priority = "P1" if not escalation_flag else "P0"
    elif task_kind == "specialist":
        subject = f"Specialist consult on {project_code}"
        reply_type = "REQUEST_INFO"
        priority = "P1" if not escalation_flag else "P0"
    elif task_kind == "program":
        subject = f"Cross-team update {project_code}"
        reply_type = "ACK"
        priority = "P1"
    elif task_kind == "quick":
        subject = f"Quick action {project_code}"
        reply_type = "ANSWER"
        priority = "P2"
    else:
        subject = f"Operations follow-up {project_code}"
        reply_type = "ANSWER"
        priority = "P1"

    if escalation_flag and priority != "P0":
        priority = "P0"
    if approval_flag and task_kind != "approval":
        subject = f"Re: Approval check {project_code}"
        reply_type = "ANSWER"
        priority = "P1" if priority != "P0" else "P0"
    if specialist_flag and task_kind not in {"specialist", "approval"}:
        subject = f"Re: Specialist input needed {project_code}"
        reply_type = "REQUEST_INFO"

    if anchor:
        body = (
            f"Thread anchor for {thread_id}. Project code {project_code}. "
            f"Stakeholders fanout {fanout_hint}. Please keep this as canonical reference."
        )
        required_key = "none"
    elif needs_memory:
        body = (
            f"Follow-up on prior thread details. As discussed earlier, act now. "
            f"Need concise response and next step by today. This message omits prior code intentionally."
        )
        required_key = "project_code"
    else:
        body = (
            f"Follow-up for project {project_code}. Please respond with status and owner action."
        )
        required_key = "none"

    if escalation_flag:
        body += " URGENT escalation path triggered."
    if specialist_flag:
        body += " Legal/trading specialist input requested."
    if approval_flag:
        body += " Approval checkpoint required."

    return subject[:220], body[:2500], priority, reply_type, required_key


@dataclass
class SimMessage:
    message_id: str
    phase: str
    episode_id: int
    arrival_min: float
    thread_id: str
    subject: str
    body: str
    gold_priority: str
    gold_reply_type: str
    gold_required_key: str
    gold_required_value: str
    escalation_flag: int
    specialist_flag: int
    approval_flag: int


def generate_episode_messages(
    *,
    phase: str,
    episode_id: int,
    rng: np.random.Generator,
    task_probs: list[tuple[str, float]],
    messages_per_episode: int,
    threads_target: int,
    episode_hours: int,
    escalation_prob: float,
    specialist_prob: float,
    approval_prob: float,
    fanout_target: float,
    dependency_burst_prob: float,
) -> list[SimMessage]:
    n_msgs = int(max(10, messages_per_episode))
    n_threads = int(max(3, min(threads_target, n_msgs)))
    episode_minutes = float(max(60, episode_hours * 60))
    avg_gap = episode_minutes / n_msgs
    threads = [f"{phase}_ep{episode_id:03d}_th{i+1:03d}" for i in range(n_threads)]
    thread_facts = {tid: {"project_code": make_project_code(rng)} for tid in threads}
    thread_seen = {tid: 0 for tid in threads}

    current_time = 0.0
    current_thread = threads[0]
    out: list[SimMessage] = []

    for i in range(n_msgs):
        if i < n_threads:
            tid = threads[i]
        else:
            if rng.random() < 0.45:
                tid = current_thread
            else:
                tid = str(rng.choice(threads))
            current_thread = tid

        project_code = thread_facts[tid]["project_code"]
        anchor = thread_seen[tid] == 0
        needs_memory = (not anchor) and (rng.random() < clamp(0.20 + 0.4 * specialist_prob, 0.05, 0.6))
        escalation_flag = int(rng.random() < escalation_prob)
        specialist_flag = int(rng.random() < specialist_prob)
        approval_flag = int(rng.random() < approval_prob)
        fanout_hint = int(max(1, round(fanout_target + rng.normal(0, 1))))

        task_type = pick_task_type(task_probs, rng)
        task_kind = parse_task_kind(task_type)

        subject, body, priority, reply_type, required_key = build_message_text(
            task_kind=task_kind,
            thread_id=tid,
            project_code=project_code,
            needs_memory=needs_memory,
            escalation_flag=bool(escalation_flag),
            specialist_flag=bool(specialist_flag),
            approval_flag=bool(approval_flag),
            fanout_hint=fanout_hint,
            anchor=anchor,
        )
        required_val = project_code if required_key == "project_code" else ""

        gap = float(rng.exponential(scale=max(0.2, avg_gap)))
        if rng.random() < dependency_burst_prob:
            gap = float(rng.exponential(scale=max(0.05, avg_gap / 4.0)))
        current_time = min(episode_minutes, current_time + gap)

        out.append(
            SimMessage(
                message_id=f"{phase}_ep{episode_id:03d}_m{i+1:04d}",
                phase=phase,
                episode_id=episode_id,
                arrival_min=round(current_time, 3),
                thread_id=tid,
                subject=subject,
                body=body,
                gold_priority=priority,
                gold_reply_type=reply_type,
                gold_required_key=required_key,
                gold_required_value=required_val,
                escalation_flag=escalation_flag,
                specialist_flag=specialist_flag,
                approval_flag=approval_flag,
            )
        )
        thread_seen[tid] += 1

    return sorted(out, key=lambda x: (x.arrival_min, x.message_id))


class HeuristicAgent:
    name = "heuristic"

    def decide(self, email: dict[str, Any], scratchpad: str, backlog: int) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = backlog
        text = f"{email['subject']} {email['body']}".lower()
        if "urgent" in text or "escalation" in text or "today" in text:
            priority = "P0"
        elif "approval" in text or "specialist" in text or "follow-up" in text or "follow up" in text:
            priority = "P1"
        else:
            priority = "P2"
        if "specialist" in text or "?" in email["body"]:
            reply_type = "REQUEST_INFO"
        elif "approval" in text:
            reply_type = "ANSWER"
        elif priority == "P2":
            reply_type = "ACK"
        else:
            reply_type = "ANSWER"

        code = ""
        for src in [email["subject"], email["body"], scratchpad]:
            m = re.search(r"\b[A-Z][0-9]{3}\b", str(src))
            if m:
                code = m.group(0)
                break
        facts = [code] if code else []
        action = f"Handle thread {email['thread_id']}"
        if code:
            action += f" for project {code}"
        draft = "Acknowledged. I will triage and follow up."
        update = f"{email['thread_id']}: {code or 'unknown'}|p={priority}|r={reply_type}"
        return (
            {
                "priority": priority,
                "reply_type": reply_type,
                "action_summary": action,
                "facts_used": facts,
                "draft_reply": draft,
                "scratchpad_update": update,
            },
            {"latency_sec": 0.03, "input_tokens": 0, "output_tokens": 0, "raw_invalid": 0},
        )


class OpenAIAgent:
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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_attempts = max(1, int(max_attempts))
        self.reasoning_mode = reasoning_mode
        self.max_output_tokens = max(0, int(max_output_tokens))
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.session = requests.Session()
        rubric = RUBRIC_MEANING if str(prompt_profile) == "meaning" else RUBRIC_STRICT
        self.system_prompt = (
            "You are an organizational inbox triage agent. "
            "Use only current email + provided scratchpad. "
            "Return strict JSON with fields: priority, reply_type, action_summary, facts_used, draft_reply, scratchpad_update. "
            "priority in {P0,P1,P2}; reply_type in {NONE,ACK,ANSWER,REQUEST_INFO,REDIRECT}. "
            f"{rubric} "
            f"{MEMORY_CONTRACT} "
            "Do not invent completed actions that are not in context."
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
                "action_summary": {"type": "string"},
                "facts_used": {"type": "array", "items": {"type": "string"}},
                "draft_reply": {"type": "string"},
                "scratchpad_update": {"type": "string"},
            },
            "required": [
                "priority",
                "reply_type",
                "action_summary",
                "facts_used",
                "draft_reply",
                "scratchpad_update",
            ],
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            except Exception:
                if attempt == self.max_attempts:
                    raise
                time.sleep((2**attempt) + float(np.random.uniform(0, 1)))
        raise RuntimeError("Unreachable")

    def decide(self, email: dict[str, Any], scratchpad: str, backlog: int) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_obj = {"backlog": backlog, "email": email, "scratchpad": scratchpad}
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent_org_triage",
                    "schema": self.response_schema,
                    "strict": True,
                }
            },
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_mode == "high":
            payload["reasoning"] = {"effort": "high"}
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
        "reply_type": decision.get("reply_type", "NONE"),
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
    if not isinstance(out["facts_used"], list):
        out["facts_used"] = []
        invalid = 1
    for key in ["action_summary", "draft_reply", "scratchpad_update"]:
        if not isinstance(out[key], str):
            out[key] = str(out[key])
            invalid = 1
    return out, invalid


def hallucination_penalty(draft_reply: str, source_text: str) -> float:
    d = draft_reply.lower()
    s = source_text.lower()
    for phrase in BANNED_COMMITMENTS:
        if phrase in d and phrase not in s:
            return 1.0
    return 0.0


def team_size_from_policy(policy: str) -> int:
    if "risk_first" in policy:
        return 6
    if "load_balanced" in policy:
        return 5
    return 3


def choose_agent_idx(
    *,
    policy: str,
    message: SimMessage,
    rng: np.random.Generator,
    available_times: list[float],
) -> int:
    n = len(available_times)
    if n == 1:
        return 0
    if "risk_first" in policy:
        if message.escalation_flag or message.specialist_flag or message.approval_flag:
            return 0  # specialist gate
        return int(np.argmin(available_times[1:])) + 1
    if "load_balanced" in policy:
        if rng.random() < 0.1:
            return int(rng.integers(0, n))
        return int(np.argmin(available_times))
    # balanced_manual_override
    return int(np.argmin(available_times))


def trim_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def score_message(msg: SimMessage, decision: dict[str, Any], process_end_min: float) -> dict[str, float]:
    priority_acc = float(decision["priority"] == msg.gold_priority)
    reply_acc = float(decision["reply_type"] == msg.gold_reply_type)
    text_blob = f"{decision['action_summary']} {decision['draft_reply']} {' '.join(str(x) for x in decision['facts_used'])}".lower()
    if msg.gold_required_key == "none":
        fact_recall = 1.0
    else:
        fact_recall = float(msg.gold_required_value.lower() in text_blob)
    halluc = hallucination_penalty(decision["draft_reply"], f"{msg.subject}\n{msg.body}")
    latency_min = process_end_min - msg.arrival_min
    on_time = float(latency_min <= PRIORITY_SLA_MIN[msg.gold_priority])
    quality = 0.40 * priority_acc + 0.30 * reply_acc + 0.30 * fact_recall - 0.20 * halluc
    if on_time < 1.0 and msg.gold_priority in {"P0", "P1"}:
        quality *= 0.2
    quality = max(0.0, min(1.0, quality))
    return {
        "priority_acc": priority_acc,
        "reply_acc": reply_acc,
        "fact_recall": fact_recall,
        "hallucination": halluc,
        "latency_min": latency_min,
        "on_time": on_time,
        "quality_score": quality,
    }


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else 0.0


def run_episode(
    *,
    agent: HeuristicAgent | OpenAIAgent,
    policy: str,
    messages: list[SimMessage],
    scratchpad_budget: int,
    rng: np.random.Generator,
    call_budget: int,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    team_n = team_size_from_policy(policy)
    agent_available = [0.0 for _ in range(team_n)]
    scratchpads = ["" for _ in range(team_n)]
    rows: list[dict[str, Any]] = []

    total_input_tokens = 0
    total_output_tokens = 0
    raw_invalid_total = 0
    api_error_total = 0
    calls = 0

    for msg in messages:
        if calls >= call_budget:
            raise RuntimeError(f"Max calls reached ({call_budget})")
        agent_idx = choose_agent_idx(
            policy=policy,
            message=msg,
            rng=rng,
            available_times=agent_available,
        )
        start_min = max(float(msg.arrival_min), float(agent_available[agent_idx]))
        backlog = int(sum(1 for t in agent_available if t > msg.arrival_min))
        email = {
            "message_id": msg.message_id,
            "thread_id": msg.thread_id,
            "subject": msg.subject,
            "body": msg.body,
            "arrival_min": msg.arrival_min,
            "escalation_flag": bool(msg.escalation_flag),
            "specialist_flag": bool(msg.specialist_flag),
            "approval_flag": bool(msg.approval_flag),
        }
        calls += 1
        api_error = 0
        try:
            raw_decision, meta = agent.decide(
                email=email,
                scratchpad=scratchpads[agent_idx],
                backlog=backlog,
            )
        except Exception as exc:  # pragma: no cover
            api_error = 1
            raw_decision = {
                "priority": "P2",
                "reply_type": "NONE",
                "action_summary": "",
                "facts_used": [],
                "draft_reply": "",
                "scratchpad_update": "",
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

        latency_sec = float(meta.get("latency_sec", 0.03))
        process_min = max(0.25, latency_sec / 60.0)
        end_min = start_min + process_min
        agent_available[agent_idx] = end_min

        if decision["scratchpad_update"]:
            scratchpads[agent_idx] = trim_text(
                (scratchpads[agent_idx] + "\n" + decision["scratchpad_update"]).strip(),
                max_chars=scratchpad_budget,
            )

        scores = score_message(msg, decision, process_end_min=end_min)
        total_input_tokens += int(meta.get("input_tokens", 0))
        total_output_tokens += int(meta.get("output_tokens", 0))
        rows.append(
            {
                "phase": msg.phase,
                "episode_id": msg.episode_id,
                "message_id": msg.message_id,
                "thread_id": msg.thread_id,
                "arrival_min": msg.arrival_min,
                "start_min": start_min,
                "end_min": end_min,
                "agent_idx": agent_idx,
                "routing_policy": policy,
                "subject": msg.subject,
                "body": msg.body,
                "gold_priority": msg.gold_priority,
                "pred_priority": decision["priority"],
                "gold_reply_type": msg.gold_reply_type,
                "pred_reply_type": decision["reply_type"],
                "pred_action_summary": decision["action_summary"],
                "pred_facts_used": json.dumps(decision["facts_used"], ensure_ascii=True),
                "pred_draft_reply": decision["draft_reply"],
                "pred_scratchpad_update": decision["scratchpad_update"],
                "gold_required_key": msg.gold_required_key,
                "gold_required_value": msg.gold_required_value,
                "invalid_output": invalid_total,
                "api_error": api_error,
                "input_tokens": int(meta.get("input_tokens", 0)),
                "output_tokens": int(meta.get("output_tokens", 0)),
                "scratchpad_len": len(scratchpads[agent_idx]),
                "escalation_flag": msg.escalation_flag,
                "specialist_flag": msg.specialist_flag,
                "approval_flag": msg.approval_flag,
                **scores,
            }
        )

    df = pd.DataFrame(rows)
    p0 = df[df["gold_priority"] == "P0"]
    p1 = df[df["gold_priority"] == "P1"]
    mem = df[df["gold_required_key"] != "none"]
    metrics = {
        "messages": float(len(df)),
        "mean_quality": safe_mean(df["quality_score"]),
        "priority_acc": safe_mean(df["priority_acc"]),
        "reply_acc": safe_mean(df["reply_acc"]),
        "fact_recall": safe_mean(df["fact_recall"]),
        "memory_fact_recall": safe_mean(mem["fact_recall"]) if len(mem) else 1.0,
        "p0_sla": safe_mean(p0["on_time"]) if len(p0) else 1.0,
        "p1_sla": safe_mean(p1["on_time"]) if len(p1) else 1.0,
        "mean_latency_min": safe_mean(df["latency_min"]),
        "invalid_rate": safe_mean(df["invalid_output"]),
        "api_error_rate": safe_mean(df["api_error"]),
        "input_tokens": float(total_input_tokens),
        "output_tokens": float(total_output_tokens),
        "calls": float(calls),
        "raw_invalid_total": float(raw_invalid_total),
        "team_size": float(team_n),
    }
    return df, metrics, calls


def load_configs(config_dir: Path, shock_id: str) -> list[Path]:
    index_json = config_dir / "index.json"
    if not index_json.exists():
        raise FileNotFoundError(f"Missing config pack index: {index_json}")
    items = json.loads(index_json.read_text(encoding="utf-8"))
    paths = [Path(x["file"]) for x in items]
    if shock_id:
        target = config_dir / f"{shock_id}.json"
        if target.exists():
            return [target]
        # fallback scan by stem
        matched = [p for p in paths if p.stem == shock_id]
        if not matched:
            raise FileNotFoundError(f"Shock id not found: {shock_id}")
        return matched
    return paths


def top_tasks(cfg: dict[str, Any], phase: str) -> list[tuple[str, float]]:
    rows = cfg["regimes"][phase].get("top_tasks", [])
    out = []
    for r in rows:
        t = str(r.get("task_type", ""))
        w = float(r.get("share", 0.0))
        if t:
            out.append((t, w))
    if not out:
        return [("quick_resolution_c4", 1.0)]
    total = sum(max(0.0, w) for _, w in out)
    if total <= 0:
        return [(t, 1.0 / len(out)) for t, _ in out]
    return [(t, max(0.0, w) / total) for t, w in out]


def resolve_episode_settings(cfg: dict[str, Any], phase: str, args: argparse.Namespace) -> tuple[int, int, int]:
    rec = cfg["simulator_defaults"][f"recommended_{phase}"]
    hours = int(args.episode_hours) if int(args.episode_hours) > 0 else int(rec["episode_hours"])
    msgs = int(args.messages_per_episode) if int(args.messages_per_episode) > 0 else int(rec["messages_per_episode"])
    threads = int(args.threads_per_episode) if int(args.threads_per_episode) > 0 else int(rec["active_threads_target"])
    return hours, msgs, threads


def write_report(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    transition_summary: pd.DataFrame,
    total_cost: float,
) -> None:
    lines = [
        "# Agent Organization Simulation Report",
        "",
        f"- Agent: **{args.agent}**",
        f"- Model: **{args.model if args.agent == 'openai' else 'n/a'}**",
        f"- Episodes per phase: **{int(args.episodes_per_phase)}**",
        f"- OpenAI reasoning mode: **{args.openai_reasoning_mode if args.agent == 'openai' else 'n/a'}**",
        f"- Estimated API cost: **${total_cost:.4f}**",
        "",
        "## Transition Summary",
    ]
    for row in transition_summary.itertuples(index=False):
        lines.append(
            f"- {row.shock_id}: quality pre={row.mean_quality_pre:.3f}, post={row.mean_quality_post:.3f}, "
            f"delta={row.delta_mean_quality:+.3f}; p0_sla pre={row.p0_sla_pre:.3f}, post={row.p0_sla_post:.3f}; "
            f"memory_recall pre={row.memory_fact_recall_pre:.3f}, post={row.memory_fact_recall_post:.3f}; "
            f"api_error post={row.api_error_rate_post:.3f}"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg_paths = load_configs(args.config_dir, args.shock_id)
    if args.agent == "openai":
        agent: HeuristicAgent | OpenAIAgent = OpenAIAgent(
            model=args.model,
            base_url=args.openai_base_url,
            timeout_sec=args.openai_timeout_sec,
            max_attempts=args.openai_max_attempts,
            reasoning_mode=args.openai_reasoning_mode,
            max_output_tokens=args.openai_max_output_tokens,
            temperature=args.temperature,
            prompt_profile=str(args.prompt_profile),
        )
    else:
        agent = HeuristicAgent()

    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    model_tag = args.model.replace("/", "_").replace(":", "_")
    run_tag = f"{args.agent}_{model_tag}_{ts}"
    run_dir = args.output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    message_logs: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    total_calls_used = 0

    for cfg_path in cfg_paths:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        shock_id = str(cfg["meta"]["shock_id"])
        post_policy = str(cfg["simulator_defaults"]["routing_policy_post"])
        pre_policy = "balanced_manual_override"

        phase_metrics: dict[str, list[dict[str, Any]]] = {"pre": [], "post": []}
        for phase in ["pre", "post"]:
            reg = cfg["regimes"][phase]
            knobs = reg["simulation_knobs"]
            task_probs = top_tasks(cfg, phase)
            policy = pre_policy if phase == "pre" else post_policy
            hours, msg_target, thread_target = resolve_episode_settings(cfg, phase, args)
            for ep in range(int(args.episodes_per_phase)):
                rng = np.random.default_rng(args.seed + ep * 1009 + hash(shock_id + phase) % 100_000)
                msgs = generate_episode_messages(
                    phase=phase,
                    episode_id=ep,
                    rng=rng,
                    task_probs=task_probs,
                    messages_per_episode=int(msg_target),
                    threads_target=int(thread_target),
                    episode_hours=int(hours),
                    escalation_prob=clamp(float(knobs["escalation_prob"]), 0.0, 1.0),
                    specialist_prob=clamp(float(knobs["specialist_prob"]), 0.0, 1.0),
                    approval_prob=clamp(float(knobs["approval_prob"]), 0.0, 1.0),
                    fanout_target=max(0.0, float(knobs["fanout_target"])),
                    dependency_burst_prob=clamp(float(knobs["dependency_burst_prob"]), 0.0, 0.9),
                )
                remaining = max(0, int(args.max_total_calls) - total_calls_used)
                if remaining <= 0:
                    raise RuntimeError(f"Max total calls reached ({args.max_total_calls})")
                msg_df, metrics, calls_used = run_episode(
                    agent=agent,
                    policy=policy,
                    messages=msgs,
                    scratchpad_budget=5000,
                    rng=rng,
                    call_budget=remaining,
                )
                total_calls_used += calls_used
                msg_df["shock_id"] = shock_id
                msg_df["phase"] = phase
                msg_df["episode_id"] = ep
                msg_df["policy"] = policy
                message_logs.append(msg_df)
                ep_row = {
                    "shock_id": shock_id,
                    "phase": phase,
                    "episode_id": ep,
                    "policy": policy,
                    "messages_target": int(msg_target),
                    "threads_target": int(thread_target),
                    **metrics,
                }
                episode_rows.append(ep_row)
                phase_metrics[phase].append(ep_row)

        df_pre = pd.DataFrame(phase_metrics["pre"])
        df_post = pd.DataFrame(phase_metrics["post"])
        summary = {
            "shock_id": shock_id,
            "break_week": str(cfg["shock"]["break_week"]),
            "shock_type": str(cfg["shock"]["shock_type"]),
            "routing_policy_post": post_policy,
            "mean_quality_pre": float(df_pre["mean_quality"].mean()),
            "mean_quality_post": float(df_post["mean_quality"].mean()),
            "delta_mean_quality": float(df_post["mean_quality"].mean() - df_pre["mean_quality"].mean()),
            "p0_sla_pre": float(df_pre["p0_sla"].mean()),
            "p0_sla_post": float(df_post["p0_sla"].mean()),
            "delta_p0_sla": float(df_post["p0_sla"].mean() - df_pre["p0_sla"].mean()),
            "memory_fact_recall_pre": float(df_pre["memory_fact_recall"].mean()),
            "memory_fact_recall_post": float(df_post["memory_fact_recall"].mean()),
            "delta_memory_fact_recall": float(df_post["memory_fact_recall"].mean() - df_pre["memory_fact_recall"].mean()),
            "mean_latency_pre": float(df_pre["mean_latency_min"].mean()),
            "mean_latency_post": float(df_post["mean_latency_min"].mean()),
            "delta_mean_latency": float(df_post["mean_latency_min"].mean() - df_pre["mean_latency_min"].mean()),
            "api_error_rate_pre": float(df_pre["api_error_rate"].mean()),
            "api_error_rate_post": float(df_post["api_error_rate"].mean()),
            "calls_pre": float(df_pre["calls"].sum()),
            "calls_post": float(df_post["calls"].sum()),
            "input_tokens_pre": float(df_pre["input_tokens"].sum()),
            "input_tokens_post": float(df_post["input_tokens"].sum()),
            "output_tokens_pre": float(df_pre["output_tokens"].sum()),
            "output_tokens_post": float(df_post["output_tokens"].sum()),
        }
        transition_rows.append(summary)

    message_log = pd.concat(message_logs, ignore_index=True) if message_logs else pd.DataFrame()
    episode_summary = pd.DataFrame(episode_rows)
    transition_summary = pd.DataFrame(transition_rows)

    total_input_tokens = float(episode_summary["input_tokens"].sum()) if not episode_summary.empty else 0.0
    total_output_tokens = float(episode_summary["output_tokens"].sum()) if not episode_summary.empty else 0.0
    total_cost = (
        total_input_tokens * float(args.input_cost_per_1m) + total_output_tokens * float(args.output_cost_per_1m)
    ) / 1_000_000.0

    message_log.to_csv(run_dir / "message_log.csv", index=False)
    episode_summary.to_csv(run_dir / "episode_summary.csv", index=False)
    transition_summary.to_csv(run_dir / "transition_summary.csv", index=False)
    write_report(run_dir=run_dir, args=args, transition_summary=transition_summary, total_cost=total_cost)

    print(f"Run dir: {run_dir.resolve()}")
    print(f"Shocks simulated: {len(transition_summary)}")
    print(f"Calls used: {total_calls_used}")
    print(f"Agent: {args.agent}")
    if not transition_summary.empty:
        print("Transition deltas (quality):")
        for row in transition_summary.itertuples(index=False):
            print(f"  {row.shock_id}: {row.delta_mean_quality:+.4f}")


if __name__ == "__main__":
    main()
