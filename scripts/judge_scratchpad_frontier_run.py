#!/usr/bin/env python3
"""LLM-as-judge scoring for scratchpad frontier runs.

Goal: score *judgment* (reasonableness) rather than compliance with heuristic labels.
We keep objective metrics (memory recall, hallucination flags, latency) and use an
independent judge to grade priority/reply_type decisions + whether messages are answerable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PRIORITY_SLA_MIN = {"P0": 5.0, "P1": 20.0, "P2": 120.0}
RE_PROJECT_CODE = re.compile(r"\b([A-Z][0-9]{3})\b")

RUBRIC_MEANING = (
    "Priority meanings: "
    "P0 = genuinely urgent and time-sensitive (same-day deadlines, blocking incidents, escalations). "
    "P1 = important and actionable but not an emergency; should be handled soon (typically today). "
    "P2 = routine/FYI/social or non-urgent updates; can wait. "
    "Reply types: ANSWER when you can directly comply/decide/respond, REQUEST_INFO when you need clarifying information, "
    "ACK when acknowledging an update, NONE when no reply is needed, REDIRECT when the request belongs to a different owner/team."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", type=Path, required=True, help="Scenario directory containing messages.csv.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory containing message_log.csv.")
    p.add_argument("--output-name", default="judged", help="Output file stem under run-dir.")

    p.add_argument("--judge-model", default=os.getenv("RESEARCH_JUDGE_MODEL", "gpt-5.2"))
    p.add_argument("--judge-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--judge-timeout-sec", type=int, default=90)
    p.add_argument("--judge-max-attempts", type=int, default=3)
    p.add_argument("--judge-reasoning-mode", choices=["auto", "high"], default="auto")
    p.add_argument("--judge-max-output-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=None)

    p.add_argument("--max-messages", type=int, default=0, help="0 = judge all; else limit for debugging.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Judge this many messages per API call (reduces cost/latency). Use 1 for per-message judging.",
    )
    p.add_argument("--resume", action="store_true", help="Skip messages already in prior judged outputs.")
    p.add_argument("--sleep-sec", type=float, default=0.0, help="Optional pacing between judge calls.")

    # Reporting / aggregation
    p.add_argument("--score-threshold-q", type=float, default=0.75)
    p.add_argument("--p0-sla-threshold", type=float, default=0.90)
    p.add_argument("--bootstrap-iters", type=int, default=400)
    return p.parse_args()


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


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else 0.0


def grade_to_score(grade: str) -> float:
    g = str(grade).strip().lower()
    if g == "good":
        return 1.0
    if g in {"borderline", "debatable"}:
        return 0.5
    return 0.0


@dataclass
class JudgeOutput:
    recommended_priority: str
    priority_grade: str
    recommended_reply_type: str
    reply_grade: str
    info_sufficient: bool
    rationale: str


class OpenAIJudge:
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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = int(timeout_sec)
        self.max_attempts = max(1, int(max_attempts))
        self.reasoning_mode = reasoning_mode
        self.max_output_tokens = max(0, int(max_output_tokens))
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.session = requests.Session()

        self.system_prompt = (
            "You are a strict evaluator of inbox triage decisions. "
            "Given the message text and an agent decision, grade whether the decision is reasonable. "
            "Do NOT grade against keyword rules; grade against the meaning of priorities and reply types. "
            "Grade only triage (priority + reply-type reasonableness), not whether the agent could fully execute the task. "
            f"{RUBRIC_MEANING} "
            "You may be given a single item or a list of items. "
            "When given a list, output results in the SAME ORDER as the input items. "
            "Keep rationale short (<= 25 words). "
            "Output strict JSON only."
        )
        self.item_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recommended_priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
                "priority_grade": {"type": "string", "enum": ["good", "borderline", "bad"]},
                "recommended_reply_type": {
                    "type": "string",
                    "enum": ["NONE", "ACK", "ANSWER", "REQUEST_INFO", "REDIRECT"],
                },
                "reply_grade": {"type": "string", "enum": ["good", "borderline", "bad"]},
                "info_sufficient": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "recommended_priority",
                "priority_grade",
                "recommended_reply_type",
                "reply_grade",
                "info_sufficient",
                "rationale",
            ],
        }
        self.batch_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": self.item_schema,
                    "minItems": 1,
                    "maxItems": 50,
                }
            },
            "required": ["results"],
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
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt == self.max_attempts:
                    raise
                time.sleep((2**attempt) + float(np.random.uniform(0, 1)))
        raise RuntimeError(f"OpenAI request failed: {last_error}")

    def judge_one(self, *, message: dict[str, Any], decision: dict[str, Any]) -> tuple[JudgeOutput, dict[str, Any]]:
        prompt_obj = {"message": message, "agent_decision": decision}
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "triage_judgment",
                    "schema": self.item_schema,
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
        resp = self._request(payload)
        latency = time.time() - t0

        content = ""
        output_text = resp.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            content = output_text
        else:
            for item in resp.get("output", []):
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
            obj = json.loads(content)
        except json.JSONDecodeError:
            raw_invalid = 1
            obj = {}
        usage = resp.get("usage", {})
        meta = {
            "latency_sec": float(latency),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "raw_invalid": raw_invalid,
        }
        out = JudgeOutput(
            recommended_priority=str(obj.get("recommended_priority", "P2")),
            priority_grade=str(obj.get("priority_grade", "bad")),
            recommended_reply_type=str(obj.get("recommended_reply_type", "NONE")),
            reply_grade=str(obj.get("reply_grade", "bad")),
            info_sufficient=bool(obj.get("info_sufficient", False)),
            rationale=str(obj.get("rationale", ""))[:600],
        )
        return out, meta

    def judge_batch(self, *, items: list[dict[str, Any]]) -> tuple[list[JudgeOutput], dict[str, Any]]:
        if not items:
            return [], {"latency_sec": 0.0, "input_tokens": 0, "output_tokens": 0, "raw_invalid": 0}

        prompt_obj = {"items": items}
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "triage_judgment_batch",
                    "schema": self.batch_schema,
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
        resp = self._request(payload)
        latency = time.time() - t0

        content = ""
        output_text = resp.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            content = output_text
        else:
            for item in resp.get("output", []):
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
            obj = json.loads(content)
        except json.JSONDecodeError:
            raw_invalid = 1
            obj = {}
        usage = resp.get("usage", {})
        meta = {
            "latency_sec": float(latency),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "raw_invalid": raw_invalid,
        }
        results: list[Any] = []
        if isinstance(obj, dict):
            results = obj.get("results", []) or []
        outputs: list[JudgeOutput] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            outputs.append(
                JudgeOutput(
                    recommended_priority=str(r.get("recommended_priority", "P2")),
                    priority_grade=str(r.get("priority_grade", "bad")),
                    recommended_reply_type=str(r.get("recommended_reply_type", "NONE")),
                    reply_grade=str(r.get("reply_grade", "bad")),
                    info_sufficient=bool(r.get("info_sufficient", False)),
                    rationale=str(r.get("rationale", ""))[:600],
                )
            )
        return outputs, meta


def compute_fact_recall(row: pd.Series) -> float:
    key = str(row.get("gold_required_key", "none"))
    if key == "none":
        return 1.0
    required = str(row.get("gold_required_value", "")).strip().lower()
    blob = " ".join(
        [
            str(row.get("pred_action_summary", "")),
            str(row.get("pred_draft_reply", "")),
            str(row.get("pred_facts_used", "")),
            str(row.get("pred_target_project_code", "")),
        ]
    ).lower()
    return float(required != "" and required in blob)


def extract_project_code(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = RE_PROJECT_CODE.search(text)
        if match:
            return match.group(1)
    return ""


def ensure_binding_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "pred_target_project_code" not in out.columns:
        out["pred_target_project_code"] = out.apply(
            lambda row: extract_project_code(
                row.get("pred_facts_used", ""),
                row.get("pred_action_summary", ""),
                row.get("pred_draft_reply", ""),
            ),
            axis=1,
        )
    out["pred_target_project_code"] = out["pred_target_project_code"].fillna("").astype(str).str.strip()

    if "pred_binding_decision" not in out.columns:
        out["pred_binding_decision"] = np.where(
            out["pred_target_project_code"] != "",
            "bound",
            "",
        )
    out["pred_binding_decision"] = out["pred_binding_decision"].fillna("").astype(str).str.strip().str.lower()

    if "pred_binding_source" not in out.columns:
        out["pred_binding_source"] = ""
    out["pred_binding_source"] = out["pred_binding_source"].fillna("").astype(str).str.strip().str.lower()

    if "gold_project_code" not in out.columns:
        gold_from_probe = np.where(
            out.get("gold_required_key", pd.Series(index=out.index, dtype=object)).fillna("").astype(str) == "project_code",
            out.get("gold_required_value", pd.Series(index=out.index, dtype=object)).fillna("").astype(str),
            "",
        )
        out["gold_project_code"] = gold_from_probe
    out["gold_project_code"] = out["gold_project_code"].fillna("").astype(str).str.strip()
    missing_gold = out["gold_project_code"] == ""
    if missing_gold.any():
        out.loc[missing_gold, "gold_project_code"] = out.loc[missing_gold].apply(
            lambda row: extract_project_code(row.get("subject", ""), row.get("body", "")),
            axis=1,
        )

    email_project_code = out.apply(lambda row: extract_project_code(row.get("subject", ""), row.get("body", "")), axis=1)
    pred_nonempty = out["pred_target_project_code"] != ""
    gold_nonempty = out["gold_project_code"] != ""

    if "binding_attempt" not in out.columns:
        out["binding_attempt"] = pred_nonempty.astype(float)
    else:
        out["binding_attempt"] = out["binding_attempt"].fillna(0).astype(float)

    if "target_match" not in out.columns:
        out["target_match"] = (pred_nonempty & gold_nonempty & (out["pred_target_project_code"] == out["gold_project_code"])).astype(float)
    else:
        out["target_match"] = out["target_match"].fillna(0).astype(float)

    if "safe_clarification" not in out.columns:
        out["safe_clarification"] = (
            (out["pred_binding_decision"] == "clarify")
            & (~pred_nonempty)
            & (email_project_code == "")
        ).astype(float)
    else:
        out["safe_clarification"] = out["safe_clarification"].fillna(0).astype(float)

    if "unsafe_wrong_target" not in out.columns:
        out["unsafe_wrong_target"] = (pred_nonempty & gold_nonempty & (out["pred_target_project_code"] != out["gold_project_code"])).astype(float)
    else:
        out["unsafe_wrong_target"] = out["unsafe_wrong_target"].fillna(0).astype(float)

    needs_memory = out.get("needs_memory", pd.Series(0, index=out.index)).fillna(0).astype(int)
    out["memory_probe_target_match"] = np.where(needs_memory == 1, out["target_match"], np.nan)
    out["memory_probe_safe_clarification"] = np.where(needs_memory == 1, out["safe_clarification"], np.nan)
    out["memory_probe_unsafe_wrong_target"] = np.where(needs_memory == 1, out["unsafe_wrong_target"], np.nan)
    out["binding_precision_on_memory_probes"] = np.where(
        (needs_memory == 1) & (out["binding_attempt"] > 0),
        out["target_match"],
        np.nan,
    )
    return out


def compute_hallucination(row: pd.Series) -> float:
    # Keep this conservative and objective: only flag banned commitments if absent from message.
    banned = ["i approved", "already approved", "done", "fixed", "completed", "shipped"]
    draft = str(row.get("pred_draft_reply", "")).lower()
    source = (str(row.get("subject", "")) + "\n" + str(row.get("body", ""))).lower()
    for phrase in banned:
        if phrase in draft and phrase not in source:
            return 1.0
    return 0.0


def add_episode_level(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["episode_id", "n_threads"], sort=False)
    out = g.agg(
        messages=("message_id", "count"),
        mean_quality=("quality_judged", "mean"),
        mean_priority_score=("priority_judge_score", "mean"),
        mean_reply_score=("reply_judge_score", "mean"),
        mean_fact_recall=("fact_recall", "mean"),
        mean_halluc=("hallucination", "mean"),
        p0_sla_judge=("on_time_judge_p0", "mean"),
        p1_sla_judge=("on_time_judge_p1", "mean"),
        info_sufficient_rate=("info_sufficient", "mean"),
        target_match=("target_match", "mean"),
        safe_clarification=("safe_clarification", "mean"),
        unsafe_wrong_target=("unsafe_wrong_target", "mean"),
        binding_attempt_rate=("binding_attempt", "mean"),
        memory_probe_target_match=("memory_probe_target_match", "mean"),
        memory_probe_safe_clarification=("memory_probe_safe_clarification", "mean"),
        memory_probe_unsafe_wrong_target=("memory_probe_unsafe_wrong_target", "mean"),
        binding_precision_on_memory_probes=("binding_precision_on_memory_probes", "mean"),
        api_error_rate=("api_error", "mean"),
        invalid_rate=("invalid_output", "mean"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
    ).reset_index()
    return out


def bootstrap_ci(values: np.ndarray, iters: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    means = []
    for _ in range(int(iters)):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means, dtype=float)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    batch_size = max(1, int(args.batch_size))
    min_tokens_for_batch = 250 * batch_size
    if batch_size > 1 and int(args.judge_max_output_tokens) < min_tokens_for_batch:
        raise ValueError(
            f"--judge-max-output-tokens too low for --batch-size={batch_size}. "
            f"Set --judge-max-output-tokens >= {min_tokens_for_batch} or reduce --batch-size."
        )

    scenario_dir = args.scenario_dir
    run_dir = args.run_dir
    if not (scenario_dir / "messages.csv").exists():
        raise FileNotFoundError(f"Missing messages.csv: {scenario_dir}")
    if not (run_dir / "message_log.csv").exists():
        raise FileNotFoundError(f"Missing message_log.csv: {run_dir}")

    # message_log already includes episode_id/n_threads; we only need text fields from the scenario.
    messages = pd.read_csv(scenario_dir / "messages.csv")[["message_id", "subject", "body"]]
    log = pd.read_csv(run_dir / "message_log.csv")
    df = log.merge(messages, on=["message_id"], how="left", validate="many_to_one")
    missing = int(df["subject"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} rows missing subject/body after join; wrong scenario-dir?")

    out_path = run_dir / f"{args.output_name}_message_log.csv"
    meta_path = run_dir / f"{args.output_name}_judge_meta.csv"
    summary_path = run_dir / f"{args.output_name}_n_summary.csv"
    report_path = run_dir / f"{args.output_name}_report.md"

    existing: set[str] = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path, usecols=["message_id"])
        existing = set(prev["message_id"].astype(str).tolist())

    judge = OpenAIJudge(
        model=args.judge_model,
        base_url=args.judge_base_url,
        timeout_sec=args.judge_timeout_sec,
        max_attempts=args.judge_max_attempts,
        reasoning_mode=args.judge_reasoning_mode,
        max_output_tokens=args.judge_max_output_tokens,
        temperature=args.temperature,
    )

    judged_rows: list[dict[str, Any]] = []
    judge_meta_rows: list[dict[str, Any]] = []

    n_total = len(df)
    limit = int(args.max_messages) if int(args.max_messages) > 0 else n_total
    df_work = df.head(limit).copy()

    batch_items: list[dict[str, Any]] = []
    batch_ids: list[str] = []

    for idx, row in df_work.iterrows():
        mid = str(row["message_id"])
        if mid in existing:
            continue
        message = {"subject": str(row["subject"]), "body": str(row["body"])}
        facts_used: Any = row.get("pred_facts_used", [])
        if isinstance(facts_used, str):
            try:
                facts_used = json.loads(facts_used)
            except Exception:
                pass
        decision = {
            "priority": str(row.get("pred_priority", "P2")),
            "reply_type": str(row.get("pred_reply_type", "NONE")),
            "action_summary": str(row.get("pred_action_summary", "")),
            "facts_used": facts_used,
            "draft_reply": str(row.get("pred_draft_reply", "")),
        }

        batch_items.append({"message": message, "agent_decision": decision})
        batch_ids.append(mid)
        if len(batch_items) < max(1, int(args.batch_size)):
            continue

        outs, meta = judge.judge_batch(items=batch_items)
        n = max(1, len(batch_items))
        per_item_meta = {
            "latency_sec": float(meta.get("latency_sec", 0.0)) / n,
            "input_tokens": int(meta.get("input_tokens", 0) or 0) // n,
            "output_tokens": int(meta.get("output_tokens", 0) or 0) // n,
            "raw_invalid": int(meta.get("raw_invalid", 0) or 0),
        }
        for i, msg_id in enumerate(batch_ids):
            j = outs[i] if i < len(outs) else JudgeOutput("P2", "bad", "NONE", "bad", False, "batch_missing")
            judged_rows.append(
                {
                    "message_id": msg_id,
                    "recommended_priority": j.recommended_priority,
                    "priority_grade": j.priority_grade,
                    "recommended_reply_type": j.recommended_reply_type,
                    "reply_grade": j.reply_grade,
                    "info_sufficient": int(bool(j.info_sufficient)),
                    "rationale": j.rationale,
                }
            )
            judge_meta_rows.append({"message_id": msg_id, **per_item_meta})
        batch_items = []
        batch_ids = []
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))
        if (len(judged_rows) % 100) == 0:
            print(f"[judge] processed={len(judged_rows)}", flush=True)

    if batch_items:
        outs, meta = judge.judge_batch(items=batch_items)
        n = max(1, len(batch_items))
        per_item_meta = {
            "latency_sec": float(meta.get("latency_sec", 0.0)) / n,
            "input_tokens": int(meta.get("input_tokens", 0) or 0) // n,
            "output_tokens": int(meta.get("output_tokens", 0) or 0) // n,
            "raw_invalid": int(meta.get("raw_invalid", 0) or 0),
        }
        for i, msg_id in enumerate(batch_ids):
            j = outs[i] if i < len(outs) else JudgeOutput("P2", "bad", "NONE", "bad", False, "batch_missing")
            judged_rows.append(
                {
                    "message_id": msg_id,
                    "recommended_priority": j.recommended_priority,
                    "priority_grade": j.priority_grade,
                    "recommended_reply_type": j.recommended_reply_type,
                    "reply_grade": j.reply_grade,
                    "info_sufficient": int(bool(j.info_sufficient)),
                    "rationale": j.rationale,
                }
            )
            judge_meta_rows.append({"message_id": msg_id, **per_item_meta})

    judged = pd.DataFrame(judged_rows)
    judge_meta = pd.DataFrame(judge_meta_rows)
    if out_path.exists() and args.resume:
        prev = pd.read_csv(out_path)
        judged = pd.concat([prev, judged], ignore_index=True).drop_duplicates(subset=["message_id"], keep="last")
    if meta_path.exists() and args.resume:
        prevm = pd.read_csv(meta_path)
        judge_meta = pd.concat([prevm, judge_meta], ignore_index=True).drop_duplicates(subset=["message_id"], keep="last")

    full = df.merge(judged, on="message_id", how="left")
    full = ensure_binding_columns(full)
    # Compute judged scores.
    full["priority_judge_score"] = full["priority_grade"].map(lambda x: grade_to_score(str(x)))
    full["reply_judge_score"] = full["reply_grade"].map(lambda x: grade_to_score(str(x)))
    full["fact_recall"] = full.apply(compute_fact_recall, axis=1)
    full["hallucination"] = full.apply(compute_hallucination, axis=1)

    # SLA vs judge-recommended priority.
    latency_min = full["process_end_min"] - full["arrival_min"]
    full["latency_min"] = latency_min
    full["on_time_judge_p0"] = ((full["recommended_priority"] == "P0") & (latency_min <= PRIORITY_SLA_MIN["P0"])) | (
        full["recommended_priority"] != "P0"
    )
    full["on_time_judge_p1"] = ((full["recommended_priority"] == "P1") & (latency_min <= PRIORITY_SLA_MIN["P1"])) | (
        full["recommended_priority"] != "P1"
    )

    full["quality_judged"] = (
        0.40 * full["priority_judge_score"]
        + 0.30 * full["reply_judge_score"]
        + 0.30 * full["fact_recall"]
        - 0.20 * full["hallucination"]
    ).clip(lower=0.0, upper=1.0)

    judged.to_csv(out_path, index=False)
    judge_meta.to_csv(meta_path, index=False)

    # Episode + N-level summaries.
    ep = add_episode_level(full)
    g = ep.groupby("n_threads", sort=True)
    n_summary = g.agg(
        episodes=("episode_id", "nunique"),
        mean_quality=("mean_quality", "mean"),
        mean_priority_score=("mean_priority_score", "mean"),
        mean_reply_score=("mean_reply_score", "mean"),
        mean_fact_recall=("mean_fact_recall", "mean"),
        mean_halluc=("mean_halluc", "mean"),
        p0_sla=("p0_sla_judge", "mean"),
        p1_sla=("p1_sla_judge", "mean"),
        info_sufficient_rate=("info_sufficient_rate", "mean"),
        target_match=("target_match", "mean"),
        safe_clarification=("safe_clarification", "mean"),
        unsafe_wrong_target=("unsafe_wrong_target", "mean"),
        binding_attempt_rate=("binding_attempt_rate", "mean"),
        memory_probe_target_match=("memory_probe_target_match", "mean"),
        memory_probe_safe_clarification=("memory_probe_safe_clarification", "mean"),
        memory_probe_unsafe_wrong_target=("memory_probe_unsafe_wrong_target", "mean"),
        binding_precision_on_memory_probes=("binding_precision_on_memory_probes", "mean"),
        api_error_rate=("api_error_rate", "mean"),
        invalid_rate=("invalid_rate", "mean"),
        judge_input_tokens=("input_tokens", "sum"),
        judge_output_tokens=("output_tokens", "sum"),
    ).reset_index()

    # Bootstrap CIs over episodes (per N).
    rng = np.random.default_rng(123)
    ci_lo = []
    ci_hi = []
    for _, row in n_summary.iterrows():
        n = int(row["n_threads"])
        vals = ep[ep["n_threads"] == n]["mean_quality"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals, iters=int(args.bootstrap_iters), rng=rng)
        ci_lo.append(lo)
        ci_hi.append(hi)
    n_summary["quality_ci95_lo"] = ci_lo
    n_summary["quality_ci95_hi"] = ci_hi

    n_summary["passes_threshold"] = (n_summary["mean_quality"] >= float(args.score_threshold_q)) & (
        n_summary["p0_sla"] >= float(args.p0_sla_threshold)
    )
    n_summary["passes_quality"] = n_summary["mean_quality"] >= float(args.score_threshold_q)
    n_summary["passes_p0_sla"] = n_summary["p0_sla"] >= float(args.p0_sla_threshold)

    passed = n_summary[n_summary["passes_threshold"]]["n_threads"].tolist()
    passed_q = n_summary[n_summary["passes_quality"]]["n_threads"].tolist()
    passed_sla = n_summary[n_summary["passes_p0_sla"]]["n_threads"].tolist()

    n_star = int(max(passed)) if passed else None
    n_star_quality = int(max(passed_q)) if passed_q else None
    n_star_sla = int(max(passed_sla)) if passed_sla else None

    n_summary.to_csv(summary_path, index=False)

    lines = [
        "# Scratchpad Frontier Judged Report",
        "",
        f"- Run dir: **{run_dir}**",
        f"- Scenario dir: **{scenario_dir}**",
        f"- Judge model: **{args.judge_model}**",
        f"- Judge reasoning mode: **{args.judge_reasoning_mode}**",
        f"- Quality threshold q: **{float(args.score_threshold_q):.2f}**",
        f"- P0 SLA threshold: **{float(args.p0_sla_threshold):.2f}**",
        "",
        "## N-Level Summary (Judged)",
    ]
    for _, r in n_summary.iterrows():
        lines.append(
            f"- N={int(r['n_threads'])}: quality={r['mean_quality']:.3f} "
            f"(ci95 {r['quality_ci95_lo']:.3f}-{r['quality_ci95_hi']:.3f}), "
            f"p0_sla={r['p0_sla']:.3f}, info_sufficient={r['info_sufficient_rate']:.3f}, "
            f"target_match={r['target_match']:.3f}, safe_clarify={r['safe_clarification']:.3f}, "
            f"wrong_target={r['unsafe_wrong_target']:.3f}, probe_precision={r['binding_precision_on_memory_probes']:.3f}"
        )
    lines.append("")
    lines.append(f"- Estimated N* (quality+SLA): **{n_star if n_star is not None else 'none'}**")
    lines.append(f"- Estimated N* (quality only): **{n_star_quality if n_star_quality is not None else 'none'}**")
    lines.append(f"- Estimated N* (SLA only): **{n_star_sla if n_star_sla is not None else 'none'}**")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    print(f"Wrote: {report_path.resolve()}")


if __name__ == "__main__":
    main()
