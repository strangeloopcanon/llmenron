# Enron Message Intent Snapshot

- Total messages analyzed: **516,796**
- Body sample size for language features: **20,000**

## Top Intent Buckets
- uncategorized: 368,196 messages (71.2%), reply_rate=38.2%, sent_share=26.0%
- trading_market: 37,402 messages (7.2%), reply_rate=35.2%, sent_share=21.3%
- status_update: 25,795 messages (5.0%), reply_rate=22.2%, sent_share=14.0%
- meeting_scheduling: 24,117 messages (4.7%), reply_rate=34.4%, sent_share=22.5%
- legal_compliance: 21,313 messages (4.1%), reply_rate=41.0%, sent_share=25.9%
- request_action: 10,165 messages (2.0%), reply_rate=34.4%, sent_share=19.8%
- announcement_broadcast: 8,316 messages (1.6%), reply_rate=19.3%, sent_share=12.9%
- social_personal: 7,751 messages (1.5%), reply_rate=43.9%, sent_share=30.3%
- uncategorized share is 71.2% (expected with keyword taxonomy; use this as residual class in simulation).

## LLM Simulation Implications
- Generate episodes by sampling intents with these empirical shares.
- Preserve empirical reply-rate differences by intent (requests/coordination have higher reply pressure).
- Match per-intent action-language and time-reference rates for realistic urgency.
- Generate message bodies by archetype (deadline_request, direct_request, informational_update, etc.) to reproduce behavioral load.

## Files
- message_intent_distribution.csv
- message_intent_density.csv
- message_intent_top_subjects.csv
- message_intent_body_features_sample.csv
- message_body_archetypes_sample.csv
- message_intent_reply_mix.csv
