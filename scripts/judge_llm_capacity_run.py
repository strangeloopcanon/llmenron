#!/usr/bin/env python3
"""LLM-as-judge scoring for llm_capacity_eval runs (Enron bodies).

Goal: score *judgment* (reasonableness) rather than compliance with heuristic labels.

Inputs:
- run-dir: results/llm_eval_runs/<tag>/llm_eval_message_log.csv (produced by llm_capacity_eval.py)

Outputs (written under run-dir):
- judged_message_log.csv: per-message judge outputs + derived scores
- judged_judge_meta.csv: per-message judge latency/token usage
- judged_n_summary.csv: aggregated results by N, with bootstrap CIs
- judged_report.md: human-readable summary, including info_sufficient rates
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PRIORITY_SLA_MIN = {"P0": 5.0, "P1": 20.0, "P2": 120.0}

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
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory containing llm_eval_message_log.csv.")
    p.add_argument("--output-name", default="judged", help="Output file stem under run-dir.")

    p.add_argument("--judge-model", default=os.getenv("RESEARCH_JUDGE_MODEL", "gpt-5.2"))
    p.add_argument("--judge-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--judge-timeout-sec", type=int, default=90)
    p.add_argument("--judge-max-attempts", type=int, default=3)
    p.add_argument("--judge-reasoning-mode", choices=["auto", "high"], default="auto")
    p.add_argument("--judge-max-output-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=None)

    p.add_argument("--max-messages", type=int, default=0, help="0 = judge all; else limit for debugging.")
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


def grade_to_score(grade: str) -> float:
    g = str(grade).strip().lower()
    if g == "good":
        return 1.0
    if g in {"borderline", "debatable"}:
        return 0.5
    return 0.0


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else 0.0


def bootstrap_ci(values: np.ndarray, iters: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    means = []
    for _ in range(int(iters)):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.quantile(np.array(means, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


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
            "Output strict JSON only."
        )
        self.response_schema: dict[str, Any] = {
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

    def judge_one(
        self,
        *,
        message: dict[str, Any],
        thread_context: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> tuple[JudgeOutput, dict[str, Any]]:
        prompt_obj = {
            "message": message,
            "thread_context": thread_context,
            "agent_decision": decision,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": json.dumps(prompt_obj, ensure_ascii=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "triage_judgment",
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


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []


def compute_hallucination(row: pd.Series) -> float:
    banned = ["i approved", "already approved", "done", "fixed", "completed", "shipped"]
    draft = str(row.get("pred_draft_reply", "")).lower()
    source = (str(row.get("subject", "")) + "\n" + str(row.get("body", "")) + "\n" + str(row.get("thread_context_json", ""))).lower()
    for phrase in banned:
        if phrase in draft and phrase not in source:
            return 1.0
    return 0.0


def compute_action_presence_vs_judge(row: pd.Series) -> float:
    rec = str(row.get("recommended_priority", "P2"))
    items = parse_json_list(row.get("pred_action_items", "[]"))
    has = len(items) > 0
    if rec in {"P0", "P1"}:
        return 1.0 if has else 0.0
    return 1.0 if (not has) else 0.0


def add_episode_level(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["episode", "n_threads"], sort=False)
    return g.agg(
        messages=("email_id", "count"),
        mean_quality=("quality_judged", "mean"),
        mean_priority_score=("priority_judge_score", "mean"),
        mean_reply_score=("reply_judge_score", "mean"),
        mean_action_presence=("action_presence_judged", "mean"),
        mean_halluc=("hallucination_judged", "mean"),
        p0_sla_judge=("on_time_judge_p0", "mean"),
        p1_sla_judge=("on_time_judge_p1", "mean"),
        info_sufficient_rate=("info_sufficient", "mean"),
        api_error_rate=("api_error", "mean"),
        invalid_rate=("invalid_output", "mean"),
        judge_input_tokens=("judge_input_tokens", "sum"),
        judge_output_tokens=("judge_output_tokens", "sum"),
    ).reset_index()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))

    run_dir = args.run_dir
    log_path = run_dir / "llm_eval_message_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing llm_eval_message_log.csv: {log_path}")

    df = pd.read_csv(log_path)
    required = {
        "email_id",
        "episode",
        "n_threads",
        "arrival_min",
        "process_end_min",
        "subject",
        "body",
        "thread_context_json",
        "pred_priority",
        "pred_reply_type",
        "pred_action_items",
        "pred_reply_key_points",
        "pred_draft_reply",
        "api_error",
        "invalid_output",
    }
    missing = sorted(required.difference(set(df.columns)))
    if missing:
        raise RuntimeError(
            "Run log missing required columns. "
            "Re-run llm_capacity_eval.py after updating it to log decision fields.\n"
            f"Missing: {missing}"
        )

    out_path = run_dir / f"{args.output_name}_message_log.csv"
    meta_path = run_dir / f"{args.output_name}_judge_meta.csv"
    summary_path = run_dir / f"{args.output_name}_n_summary.csv"
    report_path = run_dir / f"{args.output_name}_report.md"

    existing: set[str] = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path, usecols=["email_id"])
        existing = set(prev["email_id"].astype(str).tolist())

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

    for _, row in df_work.iterrows():
        eid = str(row["email_id"])
        if eid in existing:
            continue
        message = {"subject": str(row["subject"]), "body": str(row["body"])}
        try:
            thread_context = json.loads(str(row.get("thread_context_json", "[]")))
            if not isinstance(thread_context, list):
                thread_context = []
        except Exception:
            thread_context = []
        decision = {
            "priority": str(row.get("pred_priority", "P2")),
            "reply_type": str(row.get("pred_reply_type", "NONE")),
            "action_items": parse_json_list(row.get("pred_action_items", "[]")),
            "reply_key_points": parse_json_list(row.get("pred_reply_key_points", "[]")),
            "draft_reply": str(row.get("pred_draft_reply", "")),
        }
        j, meta = judge.judge_one(message=message, thread_context=thread_context, decision=decision)
        judged_rows.append(
            {
                "email_id": eid,
                "recommended_priority": j.recommended_priority,
                "priority_grade": j.priority_grade,
                "recommended_reply_type": j.recommended_reply_type,
                "reply_grade": j.reply_grade,
                "info_sufficient": int(bool(j.info_sufficient)),
                "rationale": j.rationale,
            }
        )
        judge_meta_rows.append({"email_id": eid, **meta})
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))
        if (len(judged_rows) % 100) == 0:
            print(f"[judge] processed={len(judged_rows)}", flush=True)

    judged = pd.DataFrame(judged_rows)
    judge_meta = pd.DataFrame(judge_meta_rows).rename(
        columns={"input_tokens": "judge_input_tokens", "output_tokens": "judge_output_tokens"}
    )

    if out_path.exists() and args.resume:
        prev = pd.read_csv(out_path)
        judged = pd.concat([prev, judged], ignore_index=True).drop_duplicates(subset=["email_id"], keep="last")
    if meta_path.exists() and args.resume:
        prevm = pd.read_csv(meta_path)
        judge_meta = pd.concat([prevm, judge_meta], ignore_index=True).drop_duplicates(subset=["email_id"], keep="last")

    full = df.merge(judged, on="email_id", how="left").merge(judge_meta, on="email_id", how="left")

    full["priority_judge_score"] = full["priority_grade"].map(lambda x: grade_to_score(str(x)))
    full["reply_judge_score"] = full["reply_grade"].map(lambda x: grade_to_score(str(x)))
    full["hallucination_judged"] = full.apply(compute_hallucination, axis=1)
    full["action_presence_judged"] = full.apply(compute_action_presence_vs_judge, axis=1)

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
        + 0.30 * full["action_presence_judged"]
        - 0.20 * full["hallucination_judged"]
    ).clip(lower=0.0, upper=1.0)

    full.to_csv(out_path, index=False)
    judge_meta.to_csv(meta_path, index=False)

    ep = add_episode_level(full)
    g = ep.groupby("n_threads", sort=True)
    n_summary = g.agg(
        episodes=("episode", "nunique"),
        mean_quality=("mean_quality", "mean"),
        mean_priority_score=("mean_priority_score", "mean"),
        mean_reply_score=("mean_reply_score", "mean"),
        mean_action_presence=("mean_action_presence", "mean"),
        mean_halluc=("mean_halluc", "mean"),
        p0_sla=("p0_sla_judge", "mean"),
        p1_sla=("p1_sla_judge", "mean"),
        info_sufficient_rate=("info_sufficient_rate", "mean"),
        api_error_rate=("api_error_rate", "mean"),
        invalid_rate=("invalid_rate", "mean"),
        judge_input_tokens=("judge_input_tokens", "sum"),
        judge_output_tokens=("judge_output_tokens", "sum"),
    ).reset_index()

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
    passed = n_summary[n_summary["passes_threshold"]]["n_threads"].tolist()
    n_star = int(max(passed)) if passed else None

    n_summary.to_csv(summary_path, index=False)

    lines = [
        "# LLM Capacity Judged Report (Enron Bodies)",
        "",
        f"- Run dir: **{run_dir}**",
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
            f"p0_sla={r['p0_sla']:.3f}, info_sufficient={r['info_sufficient_rate']:.3f}"
        )
    lines.append("")
    lines.append(f"- Estimated N*: **{n_star if n_star is not None else 'none'}**")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    print(f"Wrote: {report_path.resolve()}")


if __name__ == "__main__":
    main()
