#!/usr/bin/env python3
"""Run multi-agent organization simulation from breakpoint config pack."""

from __future__ import annotations

import argparse
import hashlib
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

TASK_ROUTE_TO = {
    "approval": "approvals_queue",
    "specialist": "specialist_queue",
    "program": "program_pm",
    "quick": "ops_triage",
    "ops": "ops_triage",
}

TASK_REPLY_IDENTITIES = {
    "approval": "approvals_desk",
    "specialist": "specialist_desk",
    "program": "program_pm",
    "quick": "ops_desk",
    "ops": "ops_desk",
}

TASK_ASSIGNEES = {
    "approval": "agent_0",
    "specialist": "agent_0",
    "program": "agent_2",
    "quick": "agent_1",
    "ops": "agent_1",
}

ROUTE_TO_ASSIGNEE = {
    "approvals_queue": "agent_0",
    "specialist_queue": "agent_0",
    "program_pm": "agent_2",
    "ops_triage": "agent_1",
}

ROUTE_TO_CHOICES = ["ops_triage", "approvals_queue", "specialist_queue", "program_pm"]
RESPOND_AS_CHOICES = ["ops_desk", "approvals_desk", "specialist_desk", "program_pm"]

DEFAULT_BOARD_APPROVERS = {
    "approval": "agent_0",
    "specialist": "agent_0",
    "program": "agent_1",
    "quick": "",
    "ops": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/org_simulator/scenarios/config_pack"),
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
    parser.add_argument(
        "--board-mode",
        choices=["shared", "oracle", "off"],
        default="shared",
        help="shared = learned shared task board, oracle = preseed board with gold facts, off = no shared board.",
    )
    parser.add_argument(
        "--team-size-override",
        type=int,
        default=0,
        help="0 uses routing policy default; positive value forces the same team size in all phases.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/org_simulator/simulator_runs"))
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


def make_task_id(*, phase: str, episode_id: int, thread_idx: int) -> str:
    return f"TASK-{phase.upper()}-{episode_id:03d}-{thread_idx:03d}"


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
) -> tuple[str, str, str, str, str]:
    include_project_code = anchor or (not needs_memory)
    if task_kind == "approval":
        subject = f"Approval needed for {project_code}" if include_project_code else "Approval needed"
        reply_type = "ANSWER"
        priority = "P1" if not escalation_flag else "P0"
    elif task_kind == "specialist":
        subject = f"Specialist consult on {project_code}" if include_project_code else "Specialist consult"
        reply_type = "REQUEST_INFO"
        priority = "P1" if not escalation_flag else "P0"
    elif task_kind == "program":
        subject = f"Cross-team update {project_code}" if include_project_code else "Cross-team update"
        reply_type = "ACK"
        priority = "P1"
    elif task_kind == "quick":
        subject = f"Quick action {project_code}" if include_project_code else "Quick action"
        reply_type = "ANSWER"
        priority = "P2"
    else:
        subject = f"Operations follow-up {project_code}" if include_project_code else "Operations follow-up"
        reply_type = "ANSWER"
        priority = "P1"

    if escalation_flag and priority != "P0":
        priority = "P0"
    if approval_flag and task_kind != "approval":
        subject = f"Re: Approval check {project_code}" if include_project_code else "Re: Approval check"
        reply_type = "ANSWER"
        priority = "P1" if priority != "P0" else "P0"
    if specialist_flag and task_kind not in {"specialist", "approval"}:
        subject = f"Re: Specialist input needed {project_code}" if include_project_code else "Re: Specialist input needed"
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
    task_id: str
    subject: str
    body: str
    gold_project_code: str
    gold_owner: str
    gold_assignee: str
    gold_status: str
    gold_approver: str
    gold_allowed_responder: str
    gold_reply_identity: str
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
    thread_facts = {}
    for idx, tid in enumerate(threads, start=1):
        project_code = make_project_code(rng)
        task_kind = parse_task_kind(pick_task_type(task_probs, rng))
        thread_facts[tid] = {
            "task_id": make_task_id(phase=phase, episode_id=episode_id, thread_idx=idx),
            "project_code": project_code,
            "task_kind": task_kind,
            "owner": TASK_ROUTE_TO.get(task_kind, "ops_triage"),
            "assignee": TASK_ASSIGNEES.get(task_kind, "agent_1"),
            "status": "new",
            "approver": DEFAULT_BOARD_APPROVERS.get(task_kind, ""),
            "allowed_responder": TASK_REPLY_IDENTITIES.get(task_kind, "ops_desk"),
            "reply_identity": TASK_REPLY_IDENTITIES.get(task_kind, "ops_desk"),
        }
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

        task_facts = thread_facts[tid]
        project_code = str(task_facts["project_code"])
        anchor = thread_seen[tid] == 0
        needs_memory = (not anchor) and (rng.random() < clamp(0.20 + 0.4 * specialist_prob, 0.05, 0.6))
        escalation_flag = int(rng.random() < escalation_prob)
        specialist_flag = int(rng.random() < specialist_prob)
        approval_flag = int(rng.random() < approval_prob)
        fanout_hint = int(max(1, round(fanout_target + rng.normal(0, 1))))

        task_kind = str(task_facts["task_kind"])

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
                task_id=str(task_facts["task_id"]),
                subject=subject,
                body=body,
                gold_project_code=project_code,
                gold_owner=str(task_facts["owner"]),
                gold_assignee=str(task_facts["assignee"]),
                gold_status=str(task_facts["status"]),
                gold_approver=str(task_facts["approver"]),
                gold_allowed_responder=str(task_facts["allowed_responder"]),
                gold_reply_identity=str(task_facts["reply_identity"]),
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

    def decide(
        self,
        email: dict[str, Any],
        scratchpad: str,
        backlog: int,
        *,
        task_board_entry: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        board_project = str((task_board_entry or {}).get("project_code", "")).strip()
        board_route = str((task_board_entry or {}).get("route_to", "")).strip()
        board_reply_identity = str((task_board_entry or {}).get("reply_identity", "")).strip()
        if not board_route or not board_reply_identity:
            inferred_route, inferred_reply_identity = infer_actor_labels_from_text(text)
            board_route = board_route or inferred_route
            board_reply_identity = board_reply_identity or inferred_reply_identity
        for src in [board_project, email["subject"], email["body"], scratchpad]:
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
                "target_project_code": code,
                "draft_reply": draft,
                "scratchpad_update": update,
                "route_to": board_route,
                "respond_as": board_reply_identity,
                "needs_handoff": False,
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
            "Use only current email + provided scratchpad + optional task_board_entry for the current thread. "
            "Return strict JSON with fields: priority, reply_type, action_summary, facts_used, target_project_code, "
            "draft_reply, scratchpad_update, route_to, respond_as, needs_handoff. "
            "Use route_to for internal ownership/routing and respond_as for the outward reply identity or alias. "
            f"Allowed route_to values: {', '.join(ROUTE_TO_CHOICES)}. "
            f"Allowed respond_as values: {', '.join(RESPOND_AS_CHOICES)}. "
            "priority in {P0,P1,P2}; reply_type in {NONE,ACK,ANSWER,REQUEST_INFO,REDIRECT}. "
            f"{rubric} "
            f"{MEMORY_CONTRACT} "
            "When task_board_entry includes a project_code, route_to, or reply_identity, use those instead of guessing. "
            "Keep route_to and respond_as stable for a thread unless the email clearly indicates a handoff. "
            "If the target project is unclear, leave target_project_code empty. "
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
                "target_project_code": {"type": "string"},
                "draft_reply": {"type": "string"},
                "scratchpad_update": {"type": "string"},
                "route_to": {"type": "string", "enum": ROUTE_TO_CHOICES},
                "respond_as": {"type": "string", "enum": RESPOND_AS_CHOICES},
                "needs_handoff": {"type": "boolean"},
            },
            "required": [
                "priority",
                "reply_type",
                "action_summary",
                "facts_used",
                "target_project_code",
                "draft_reply",
                "scratchpad_update",
                "route_to",
                "respond_as",
                "needs_handoff",
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

    def decide(
        self,
        email: dict[str, Any],
        scratchpad: str,
        backlog: int,
        *,
        task_board_entry: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_obj = {"backlog": backlog, "email": email, "scratchpad": scratchpad}
        if task_board_entry:
            prompt_obj["task_board_entry"] = task_board_entry
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
        "target_project_code": decision.get("target_project_code", ""),
        "draft_reply": decision.get("draft_reply", ""),
        "scratchpad_update": decision.get("scratchpad_update", ""),
        "route_to": decision.get("route_to", ""),
        "respond_as": decision.get("respond_as", ""),
        "needs_handoff": bool(decision.get("needs_handoff", False)),
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
    for key in ["action_summary", "target_project_code", "draft_reply", "scratchpad_update", "route_to", "respond_as"]:
        if not isinstance(out[key], str):
            out[key] = str(out[key])
            invalid = 1
    if not isinstance(out["needs_handoff"], bool):
        out["needs_handoff"] = bool(out["needs_handoff"])
        invalid = 1
    if out["route_to"] and out["route_to"] not in ROUTE_TO_CHOICES:
        out["route_to"] = ""
        invalid = 1
    if out["respond_as"] and out["respond_as"] not in RESPOND_AS_CHOICES:
        out["respond_as"] = ""
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
    task_board: dict[str, dict[str, Any]] | None = None,
) -> int:
    n = len(available_times)
    if n == 1:
        return 0
    if task_board:
        entry = task_board.get(message.thread_id, {})
        assignee = str(entry.get("assignee", entry.get("owner", ""))).strip()
        if assignee.startswith("agent_"):
            try:
                idx = int(assignee.split("_", 1)[1])
            except ValueError:
                idx = -1
            if 0 <= idx < n:
                return idx
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


def model_supports_temperature(model: str) -> bool:
    model_name = str(model or "").strip().lower()
    return model_name not in {"gpt-5-mini"}


def assignee_for_route(route_to: str, team_n: int) -> str:
    raw = str(ROUTE_TO_ASSIGNEE.get(str(route_to).strip(), "agent_1"))
    if not raw.startswith("agent_"):
        return "agent_0"
    try:
        idx = int(raw.split("_", 1)[1])
    except ValueError:
        return "agent_0"
    if team_n <= 1:
        return "agent_0"
    idx = max(0, min(idx, team_n - 1))
    return f"agent_{idx}"


def infer_actor_labels_from_text(text: str) -> tuple[str, str]:
    s = str(text).lower()
    if "approval" in s:
        return "approvals_queue", "approvals_desk"
    if "specialist" in s or "legal" in s or "trading specialist" in s:
        return "specialist_queue", "specialist_desk"
    if "cross-team" in s or "program" in s:
        return "program_pm", "program_pm"
    return "ops_triage", "ops_desk"


def stable_seed_offset(*parts: str, mod: int = 100_000) -> int:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod


def score_message(msg: SimMessage, decision: dict[str, Any], process_end_min: float) -> dict[str, float]:
    priority_acc = float(decision["priority"] == msg.gold_priority)
    reply_acc = float(decision["reply_type"] == msg.gold_reply_type)
    pred_target = str(decision.get("target_project_code", "")).strip()
    pred_route = str(decision.get("route_to", "")).strip()
    pred_respond_as = str(decision.get("respond_as", "")).strip()
    text_blob = (
        f"{decision['action_summary']} {decision['draft_reply']} {pred_target} "
        f"{' '.join(str(x) for x in decision['facts_used'])}"
    ).lower()
    if msg.gold_required_key == "none":
        fact_recall = 1.0
    else:
        fact_recall = float(msg.gold_required_value.lower() in text_blob)
    target_match = float(bool(pred_target) and pred_target == msg.gold_project_code)
    owner_match = float(bool(pred_route) and pred_route == msg.gold_owner) if msg.gold_owner else 1.0
    reply_identity_match = (
        float(bool(pred_respond_as) and pred_respond_as == msg.gold_reply_identity)
        if msg.gold_reply_identity
        else 1.0
    )
    unauthorized_response = (
        float(bool(pred_respond_as) and pred_respond_as != msg.gold_reply_identity)
        if msg.gold_reply_identity
        else 0.0
    )
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
        "target_match": target_match,
        "owner_match": owner_match,
        "reply_identity_match": reply_identity_match,
        "unauthorized_response": unauthorized_response,
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
    board_mode: str,
    team_size_override: int,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    team_n = int(team_size_override) if int(team_size_override) > 0 else team_size_from_policy(policy)
    agent_available = [0.0 for _ in range(team_n)]
    scratchpads = ["" for _ in range(team_n)]
    task_board: dict[str, dict[str, Any]] | None = {} if board_mode != "off" else None
    rows: list[dict[str, Any]] = []

    total_input_tokens = 0
    total_output_tokens = 0
    raw_invalid_total = 0
    api_error_total = 0
    calls = 0

    for msg in messages:
        if calls >= call_budget:
            raise RuntimeError(f"Max calls reached ({call_budget})")
        if task_board is not None and msg.thread_id not in task_board:
            task_board[msg.thread_id] = {
                "task_id": msg.task_id,
                "project_code": msg.gold_project_code if board_mode == "oracle" else "",
                "route_to": msg.gold_owner if board_mode == "oracle" else "",
                "assignee": msg.gold_assignee if board_mode == "oracle" else "",
                "status": msg.gold_status if board_mode == "oracle" else "new",
                "approver": msg.gold_approver if board_mode == "oracle" else "",
                "reply_identity": msg.gold_reply_identity if board_mode == "oracle" else "",
                "last_actor": "",
                "last_update": float(msg.arrival_min),
            }
        agent_idx = choose_agent_idx(
            policy=policy,
            message=msg,
            rng=rng,
            available_times=agent_available,
            task_board=task_board,
        )
        start_min = max(float(msg.arrival_min), float(agent_available[agent_idx]))
        backlog = int(sum(1 for t in agent_available if t > msg.arrival_min))
        task_board_entry = dict(task_board[msg.thread_id]) if task_board is not None else {}
        email = {
            "message_id": msg.message_id,
            "thread_id": msg.thread_id,
            "task_id": msg.task_id,
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
                task_board_entry=task_board_entry,
            )
        except Exception as exc:  # pragma: no cover
            api_error = 1
            raw_decision = {
                "priority": "P2",
                "reply_type": "NONE",
                "action_summary": "",
                "facts_used": [],
                "target_project_code": "",
                "draft_reply": "",
                "scratchpad_update": "",
                "route_to": "",
                "respond_as": "",
                "needs_handoff": False,
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

        board_entry = {}
        if task_board is not None:
            board_entry = task_board[msg.thread_id]
            if not str(board_entry.get("project_code", "")).strip() and decision["target_project_code"]:
                board_entry["project_code"] = decision["target_project_code"]
            if str(decision.get("route_to", "")).strip():
                route_to = str(decision["route_to"]).strip()
                board_entry["route_to"] = route_to
                board_entry["assignee"] = assignee_for_route(route_to, team_n)
            elif not str(board_entry.get("assignee", "")).strip():
                board_entry["assignee"] = f"agent_{agent_idx}"
            if msg.approval_flag:
                board_entry["status"] = "awaiting_approval"
            elif decision["reply_type"] == "REQUEST_INFO":
                board_entry["status"] = "needs_info"
            else:
                board_entry["status"] = "triaged"
            if not str(board_entry.get("approver", "")).strip() and msg.approval_flag:
                board_entry["approver"] = "agent_0"
            if str(decision.get("respond_as", "")).strip():
                board_entry["reply_identity"] = str(decision["respond_as"]).strip()
            board_entry["last_actor"] = f"agent_{agent_idx}"
            board_entry["last_update"] = round(end_min, 3)

        scores = score_message(msg, decision, process_end_min=end_min)
        total_input_tokens += int(meta.get("input_tokens", 0))
        total_output_tokens += int(meta.get("output_tokens", 0))
        rows.append(
            {
                "phase": msg.phase,
                "episode_id": msg.episode_id,
                "message_id": msg.message_id,
                "thread_id": msg.thread_id,
                "task_id": msg.task_id,
                "arrival_min": msg.arrival_min,
                "start_min": start_min,
                "end_min": end_min,
                "agent_idx": agent_idx,
                "routing_policy": policy,
                "board_mode": board_mode,
                "subject": msg.subject,
                "body": msg.body,
                "gold_priority": msg.gold_priority,
                "gold_project_code": msg.gold_project_code,
                "gold_owner": msg.gold_owner,
                "gold_assignee": msg.gold_assignee,
                "gold_status": msg.gold_status,
                "gold_approver": msg.gold_approver,
                "gold_allowed_responder": msg.gold_allowed_responder,
                "gold_reply_identity": msg.gold_reply_identity,
                "pred_priority": decision["priority"],
                "gold_reply_type": msg.gold_reply_type,
                "pred_reply_type": decision["reply_type"],
                "pred_action_summary": decision["action_summary"],
                "pred_facts_used": json.dumps(decision["facts_used"], ensure_ascii=True),
                "pred_target_project_code": decision["target_project_code"],
                "pred_draft_reply": decision["draft_reply"],
                "pred_scratchpad_update": decision["scratchpad_update"],
                "pred_route_to": decision["route_to"],
                "pred_respond_as": decision["respond_as"],
                "pred_needs_handoff": int(bool(decision["needs_handoff"])),
                "task_board_entry": json.dumps(task_board_entry, ensure_ascii=True),
                "task_board_route_to_after": str(board_entry.get("route_to", "")),
                "task_board_assignee_after": str(board_entry.get("assignee", "")),
                "task_board_status_after": str(board_entry.get("status", "")),
                "task_board_reply_identity_after": str(board_entry.get("reply_identity", "")),
                "gold_required_key": msg.gold_required_key,
                "gold_required_value": msg.gold_required_value,
                "invalid_output": invalid_total,
                "api_error": api_error,
                "api_error_text": str(meta.get("error", "")),
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
        "target_match": safe_mean(df["target_match"]),
        "owner_match": safe_mean(df["owner_match"]),
        "reply_identity_match": safe_mean(df["reply_identity_match"]),
        "unauthorized_response_rate": safe_mean(df["unauthorized_response"]),
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
        "board_enabled": float(board_mode != "off"),
    }
    return df, metrics, calls


def load_configs(config_dir: Path, shock_id: str) -> list[Path]:
    index_json = config_dir / "index.json"
    if not index_json.exists():
        raise FileNotFoundError(f"Missing config pack index: {index_json}")
    items = json.loads(index_json.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for item in items:
        raw = Path(str(item["file"]))
        candidates = [
            raw,
            config_dir / raw.name,
            config_dir / raw,
            Path(str(raw).replace("results/scenarios/config_pack", str(config_dir))),
        ]
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"Config listed in index was not found: {raw}")
        paths.append(resolved)
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
        f"- Board mode: **{args.board_mode}**",
        f"- Team size override: **{int(args.team_size_override) if int(args.team_size_override) > 0 else 'policy default'}**",
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
            f"owner_match post={row.owner_match_post:.3f}; reply_identity post={row.reply_identity_match_post:.3f}; "
            f"unauthorized post={row.unauthorized_response_rate_post:.3f}; "
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
            temperature=args.temperature if model_supports_temperature(args.model) else None,
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
                rng = np.random.default_rng(
                    args.seed + ep * 1009 + stable_seed_offset(shock_id, phase)
                )
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
                    board_mode=str(args.board_mode),
                    team_size_override=int(args.team_size_override),
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
            "owner_match_pre": float(df_pre["owner_match"].mean()),
            "owner_match_post": float(df_post["owner_match"].mean()),
            "delta_owner_match": float(df_post["owner_match"].mean() - df_pre["owner_match"].mean()),
            "reply_identity_match_pre": float(df_pre["reply_identity_match"].mean()),
            "reply_identity_match_post": float(df_post["reply_identity_match"].mean()),
            "delta_reply_identity_match": float(
                df_post["reply_identity_match"].mean() - df_pre["reply_identity_match"].mean()
            ),
            "unauthorized_response_rate_pre": float(df_pre["unauthorized_response_rate"].mean()),
            "unauthorized_response_rate_post": float(df_post["unauthorized_response_rate"].mean()),
            "delta_unauthorized_response_rate": float(
                df_post["unauthorized_response_rate"].mean() - df_pre["unauthorized_response_rate"].mean()
            ),
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
