# Inbox Juggling: Human vs LLM Capacity

At human-realistic inbox loads (≈50 concurrent threads typical, ≈105 stress), GPT‑5.2 makes reasonable triage calls but often loses thread identity on context-dependent follow-ups. That limits autonomy and can create “right action, wrong object” risk in execution settings.

This repo measures how much “communication load” people face in real organizations (Enron), then stress-tests an LLM agent on a comparable synthetic workload where messages from many threads arrive interleaved.

## So What (So Far)

- Real organizations run at **high N**: ~50 active threads is typical, ~105 is a common stress level.
- In a pilot at those N values, **triage judgment holds up** for GPT‑5.2, but **thread identity does not** on context-dependent follow-ups.
- That matters because autonomy isn’t just “write a good reply.” It’s “take the right action on the right object.” Losing the identifier forces extra back-and-forth, or (in a more autonomous setup) risks misroutes or wrong approvals.

## How To Read The Charts (What You Should Learn)

There are two separate questions here:

1. **How much load do humans face?** (That’s the Enron chart.)
2. **Given that same load, how does an LLM behave?** (That’s the synthetic charts.)

We are **not** claiming “LLMs are better/worse than humans at email” yet, because we did not run humans through the same synthetic episodes. The “human vs LLM” comparison in this pilot is: **LLM performance at human-realistic N**.

Concrete interpretation:

- The Enron chart tells you what values of **N** (concurrent threads) are realistic: ~50 is typical; ~105 is a common stress level.
- The synthetic **quality** chart asks: at those N values, does the agent’s triage (priority + reply-type) look reasonable?
- The synthetic **memory** chart asks: at those N values, can the agent **bind** a follow-up to the right thread by recovering the project code from its scratchpad when the follow-up omits it?

What “triage holds up but memory degrades” means in practice:

- The agent can still tell which messages are urgent and what kind of response is appropriate.
- But as the number of active threads increases, it more often **can’t name the thing it’s acting on** (the project/contract/ticket identifier), so it has to ask clarifying questions or it risks acting on the wrong object.

## Objective

Estimate a **capacity frontier** for inbox work:

- **Load (N):** how many threads are active at once.
- **Quality:** are triage decisions reasonable (not “did you match our keyword rules”)?
- **Latency/SLA:** do urgent items get handled on time?
- **Memory / identity:** can the agent reliably recover the thread identifier (project code) under load using only a scratchpad?

## What We Ran

### 1) Human baseline (Enron)

We use Enron email metadata to estimate how many topics/threads a person is juggling at any given time (14‑day rolling “active threads”).

![Human thread load quantiles](results/figures/human_thread_load_quantiles.png)

Key results (from `results/key_results.csv` / `results/human_analysis_summary.md`):

- Median mean active threads (14‑day rolling): **49.97**
- Median p90 active threads: **104.50**
- **Volume predicts juggling strongly; seniority does not** (once you control for volume)

### 2) Synthetic “scratchpad frontier” pilot (non‑Enron text setup)

We generate an email stream with **N interleaved threads**. The agent processes messages sequentially. It has **scratchpad-only memory**: if a follow-up omits a project code, the agent only succeeds if it wrote the code down earlier.

We score “judgment” using an **LLM-as-judge** (reasonable vs borderline vs bad), plus objective metrics (SLA, memory recall, hallucination flags).

![Synthetic judged quality](results/figures/synthetic_pilot_judged_quality.png)

![Synthetic thread identity recall](results/figures/synthetic_pilot_memory_recall.png)

Pilot configuration:

- N values tested: **50** (typical) and **105** (stress)
- Episodes per N: **1** (this is a pilot)
- Messages per episode: **180**
- Agent model: **gpt‑5.2** (Responses API, reasoning=auto)
- Judge model: **gpt‑5.2** (batch judging)

Pilot results (judged):

| Metric | N=50 | N=105 |
| --- | --- | --- |
| **GPT‑5.2 judged quality** | 0.886 | 0.874 |
| **Baseline bot judged quality** | 0.584 | 0.598 |
| **GPT‑5.2 P0 SLA hit rate** | 1.00 | 1.00 |
| **Thread-ID recovered on probes (GPT‑5.2)** | 0.46 | 0.21 |
| **Thread-ID recovered on probes (baseline bot)** | 0.11 | 0.02 |
| **Probes needing clarification (GPT‑5.2)** | 0.54 | 0.79 |
| **Probes needing clarification (baseline bot)** | 0.89 | 0.98 |

### The Specific Risk: “Right Action, Wrong Object”

The critical failure mode isn’t “bad writing.” It’s **identity binding**.

At the stress load (**N≈105**), GPT‑5.2 recovered the thread’s project code on only **21%** of context-dependent follow-ups (meaning **79%** of the time it could not name which project it was acting on).

In this pilot, the model mostly handled missing IDs by **asking a clarifying question** (safe, but adds back-and-forth). If you move from “draft replies” to “do the thing” permissions, this becomes a hard gating problem: you need the system to prevent execution unless an explicit task/thread ID is present.

Where the pilot lives on disk:

- Scenario: `results/scratchpad_canonical_pilot/canonical_pilot_50_105`
- GPT run: `results/scratchpad_canonical_pilot/canonical_pilot_50_105/runs/openai_gpt-5.2_20260210T023527Z`
- Baseline run: `results/scratchpad_canonical_pilot/canonical_pilot_50_105/runs/heuristic_gpt-5.2_20260210T015547Z`

## Implications For Agent-Driven Organizations

This pilot suggests a specific bottleneck:

- **Triage scales first.** At human-realistic load levels (N≈50/105), GPT‑5.2’s priority/reply decisions look reasonable.
- **Execution hinges on identity.** Under load, the agent often can’t recover the thread identifier when the follow-up assumes shared context. In practice that means it has to ask for context again, or it risks acting on the wrong object.
- **Safety vs speed is a design choice.** In this pilot, GPT‑5.2 mostly responds to missing identifiers by asking clarifying questions (safe, but slower). If you give an agent “do things” permissions, you need a way to prevent confident action without a bound ID.
- **The org primitive that matters is a task object.** The clean path to autonomy is to move from “messages” to “tasks” with stable IDs, owners, and auditable state transitions, and require every action to attach to one.

## Caveats / Things To Note

This is deliberately a **pilot**, so don’t over-read the exact numbers yet.

- **Small sample (1 episode per N).** You should expect variance once we run 10+ episodes per N and report confidence intervals.
- **Judge choice matters.** Right now the judge is GPT‑5.2; that can bias results (self‑evaluation effects). A stronger next step is a different judge model (or multi‑judge agreement).
- **Low “info sufficient” rates on action messages.** Many generated requests are intentionally underspecified. The judged task is “triage + next step,” not “perfectly execute with hidden context.”
- **Memory probe metric is strict.** A “miss” includes cases where the agent behaves sensibly by asking clarifying questions instead of guessing the identifier.
- **FIFO processing order.** The environment processes the oldest unread message first. That makes SLA partly a throughput question, not just prioritization skill.
- **Synthetic ≠ real work.** The synthetic setup is meant to isolate load, timing, and memory mechanics. It’s not claiming to reproduce the full richness of any one organization.

## Suggested Next Enhancements

If we continue this track, the most valuable upgrades are:

- Scale to more N levels (e.g. 35/50/70/105/140/220) with 10–40 episodes each, then estimate a real frontier.
- Switch to an independent judge model and/or run a small judge calibration pass.
- Let the agent “scan then act” (choose which unread email to process) to separate prioritization skill from FIFO artifacts.
- Replace the scratchpad with structured state (per-thread notes) and measure how much that shifts memory degradation.

<details>
<summary>Reproduce The Figures</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_readme_figures.py
```

Outputs:

- `results/figures/human_thread_load_quantiles.png`
- `results/figures/synthetic_pilot_judged_quality.png`
- `results/figures/synthetic_pilot_memory_recall.png`
</details>

<details>
<summary>How The Synthetic Run + Judge Were Executed</summary>

Run the agent on an existing scenario:

```bash
cp .env.example .env  # add OPENAI_API_KEY

python scripts/scratchpad_frontier_eval.py \
  --mode run \
  --scenario-dir results/scratchpad_canonical_pilot/canonical_pilot_50_105 \
  --agent openai \
  --model gpt-5.2 \
  --openai-reasoning-mode auto \
  --prompt-profile meaning \
  --temperature 0
```

Judge the run (batching reduces call count):

```bash
python scripts/judge_scratchpad_frontier_run.py \
  --scenario-dir results/scratchpad_canonical_pilot/canonical_pilot_50_105 \
  --run-dir results/scratchpad_canonical_pilot/canonical_pilot_50_105/runs/openai_gpt-5.2_20260210T023527Z \
  --output-name judged_v2 \
  --judge-model gpt-5.2 \
  --judge-reasoning-mode auto \
  --temperature 0 \
  --batch-size 5 \
  --judge-max-output-tokens 2000
```
</details>
