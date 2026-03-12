# Canonical Scratchpad Pilot Summary

This is the small pilot that established the original repo claim: GPT-5.2 keeps triage quality fairly high at human-realistic inbox load, but thread identity recall drops sharply when follow-ups omit the project code.

Judged quality:

- `N=50`: GPT-5.2 `0.886`, heuristic baseline `0.584`
- `N=105`: GPT-5.2 `0.874`, heuristic baseline `0.598`

Objective memory recall on context-dependent probes:

- `N=50`: GPT-5.2 `0.459`, heuristic baseline `0.108`
- `N=105`: GPT-5.2 `0.208`, heuristic baseline `0.021`

So what:

- The model often still knows what kind of response is needed.
- Under load, it increasingly loses track of which project the response belongs to.
- The autonomy bottleneck looks more like task identity than broad inbox judgment.
