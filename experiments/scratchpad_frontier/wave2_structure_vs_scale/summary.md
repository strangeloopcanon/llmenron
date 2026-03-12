# Wave 2 Structure-Vs-Scale Summary

This was the harder claim: can a smaller model with better state compete with a stronger model using worse state?

In this setup, the answer was no.

- `gpt-5-mini + thread_state` produced invalid outputs about `85.6%` of the time in the completed episode.
- Mean quality was `0.370`.
- Memory-dependent target recovery was only `0.054`.

So what:

- Better state helped the strong model in Wave 1.
- Better state did not rescue a smaller model that was unreliable at the output/control contract.
- Architecture matters, but it does not create competence out of nothing.
