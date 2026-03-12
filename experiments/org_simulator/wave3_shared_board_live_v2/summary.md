# Wave 3 Shared-Board Summary

This was the coordination test. “More agents” here means multiple worker instances of the same model processing the same org simulation in parallel. The question was whether extra agents help on their own, or whether shared state matters more.

Post-shock quality:

- Single agent, no board: `0.500`
- Single agent, shared board: `0.627`
- Multi-agent, no board: `0.460`
- Multi-agent, shared board: `0.633`

So what:

- Shared coordination state helped both the single-agent and multi-agent setups.
- More agents without a shared board were slightly worse than one agent without a board.
- The board looks like the real scaling primitive, not the swarm.
