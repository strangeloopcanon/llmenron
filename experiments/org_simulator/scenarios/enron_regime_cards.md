# Enron Regime Cards

Compact cards for simulation setup. One row in `breakpoint_cards` is one shock transition to replay.

## Breakpoint Cards
- 1997-06-09: volume_surge | arrival x2.04, threads x2.10, escalation Δ-0.064, approval Δ+0.000, routing=balanced_manual_override, memory=compact_scratchpad
- 1997-10-20: contraction_with_risk_escalation | arrival x0.47, threads x0.62, escalation Δ+0.216, approval Δ+0.000, routing=balanced_manual_override, memory=compact_scratchpad
- 1998-11-30: volume_surge | arrival x2.93, threads x3.94, escalation Δ-0.005, approval Δ+0.020, routing=balanced_manual_override, memory=thread_local_scratchpad
- 1999-04-12: surge_plus_risk | arrival x102.85, threads x70.40, escalation Δ+0.118, approval Δ+0.002, routing=load_balanced_autotriage, memory=long_horizon_scratchpad
- 2001-09-17: volume_surge | arrival x1.64, threads x2.67, escalation Δ-0.006, approval Δ+0.001, routing=load_balanced_autotriage, memory=long_horizon_scratchpad
- 2002-02-04: contraction_with_risk_escalation | arrival x0.09, threads x0.09, escalation Δ+0.046, approval Δ-0.010, routing=risk_first_with_specialist_autoroute, memory=thread_local_scratchpad

## Regime Cards
- Regime 0 (1996-12-30 to 1997-06-02): quiet_steady_state | workload=very_low, coordination=very_low, risk=very_low, top_tasks=quick_resolution_c4 (0.79), ongoing_operations_c6 (0.14)
- Regime 1 (1997-06-09 to 1997-10-13): quiet_steady_state | workload=low, coordination=very_low, risk=very_low, top_tasks=quick_resolution_c4 (0.82), ongoing_operations_c6 (0.13)
- Regime 2 (1997-10-20 to 1998-11-23): interdependent_program_mode | workload=very_low, coordination=very_high, risk=medium, top_tasks=quick_resolution_c4 (0.88), ongoing_operations_c0 (0.06)
- Regime 3 (1998-11-30 to 1999-04-05): interdependent_program_mode | workload=medium, coordination=low, risk=medium, top_tasks=quick_resolution_c4 (0.80), ongoing_operations_c6 (0.12)
- Regime 4 (1999-04-12 to 2001-09-10): operational_normal | workload=high, coordination=medium, risk=medium, top_tasks=quick_resolution_c4 (0.69), ongoing_operations_c6 (0.19)
- Regime 5 (2001-09-17 to 2002-01-28): operational_normal | workload=very_high, coordination=high, risk=medium, top_tasks=ongoing_operations_c6 (0.49), quick_resolution_c4 (0.46)
- Regime 6 (2002-02-04 to 2002-09-23): operational_normal | workload=medium, coordination=medium, risk=very_high, top_tasks=quick_resolution_c4 (0.49), ongoing_operations_c6 (0.48)
