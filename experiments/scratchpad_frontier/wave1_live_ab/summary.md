# Wave 1 Live A/B Summary

This was the clean architecture test: same model, same `N=105` stress scenario, same judge, but compare `scratchpad_only` against `thread_state`.

Result:

- `thread_state` fixed the main failure mode.
- Memory-dependent target recovery improved from `0.213` to `1.000`.
- Overall target attachment improved from `0.835` to `0.976`.
- Wrong-target actions fell from `0.019` to `0.000`.
- Judged quality improved slightly from `0.875` to `0.886`.
- Input tokens fell from `971,663` to `332,557`.

So what:

- The strongest result in the repo is that explicit thread/task state beats a giant scratchpad.
- The first unlock for agent systems looks like better state structure, not just a longer prompt.
