CMP Dummy Dataset (2 months, with drift/shift & auto-retrain)
Days=60, Tools=2, Runs/Day/Tool=60
Total runs ≈ 7200

Files:
- run_header.csv       : meta + monitoring({target_mrr_nmps, mrr_pred_nmps, mrr_err_*}, drift flags, auto_retrain)
- timeseries_process.csv: 1 Hz process signals (torque, motor_current with bias drift, AE, optical)
- timeseries_conditioning.csv: pad conditioning
- pcmp_clean.csv       : post-CMP clean recipe
- layout_features.csv  : 5x5 pattern density tiles
- metrology_labels.csv : KPIs (wiwnu, dishing, erosion, particles, yield)
- events.csv           : slurry lot change (weekly), pad/disk change (counters), maintenance (biweekly), auto_retrain
- composite_score.csv  : per-layer z-normalized composite quality score
- pass_fail.csv        : overall_pass (spec-based)

Design:
- Pad wear = slow drift in pad_factor → RR/quality drift.
- Slurry lot = weekly abrupt shift in pH center → distribution shift.
- Motor current bias = random walk + occasional step → sensor drift.
- Diurnal temperature = mild sinusoidal offset.
- MLOps:
  * mrr_pred is a "stale model" prediction using pad/lot refs at last retrain.
  * mrr_err_pct EWMA > threshold → auto_retrain event, model refs updated.
  * split_hint: train/val/test by time bands for temporal generalization tests.

Targets:
- Regression: target_mrr_nmps, wiwnu_pct, dishing_nm, particle_count_ge_50nm
- Classification: yield_flag, overall_pass
- Composite: composite_score (for multi-objective ranking)
