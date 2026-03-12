# Scratchpad Frontier Setup

This folder is a prepared experiment pack. No API runs are executed during setup.

## Calibration Anchors
- Median active threads (14d): **49.97**
- Median p90 active threads: **104.50**
- Workload grid N: **50, 105**
- Context-memory rate (hard context cues): **30.7%**
- Filler source: **template**

## Files
- `employees.csv`: synthetic employee personas with volume tiers
- `episodes.csv`: episode-level load and assignment
- `threads.csv`: per-thread facts (project code, owner, due date, archetype, task title)
- `messages.csv`: message stream with objective gold targets
- `run_matrix.csv`: planned model/baseline runs

## Core Estimands
- Capacity frontier N*: largest N passing quality and P0 SLA thresholds
- Production function: quality-latency-cost under randomized load N

## Suggested Commands (later, when you approve execution)
```bash
python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105 --agent heuristic --prompt-profile meaning
python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105 --agent openai --model gpt-5.2 --openai-reasoning-mode auto --prompt-profile meaning
python scripts/scratchpad_frontier_eval.py --mode run --scenario-dir experiments/scratchpad_frontier/scratchpad_canonical_pilot/canonical_pilot_50_105 --agent openai --model gpt-5.2 --openai-reasoning-mode high --prompt-profile meaning
```

Setup defaults: 1 episodes per N, 180 messages per episode.
