# Wave 4 Actor-Identity Summary

This was a follow-up to the shared-board result. Once shared state looked important, the next question was what that state should contain. This pilot focused on actor identity: who should own the task internally, and who should respond externally.

Post-shock actor-identity metrics:

- No board: owner match `0.667`, reply-identity match `0.667`, unauthorized response rate `0.333`
- Shared board: owner match `1.000`, reply-identity match `1.000`, unauthorized response rate `0.000`
- Oracle board: owner match `1.000`, reply-identity match `1.000`, unauthorized response rate `0.000`

So what:

- Task identity and actor identity are separate problems.
- Once you give the system a shared board, that board should contain role fields, not just task fields.
- The system should know both what the work is and who is supposed to act or speak for it.
