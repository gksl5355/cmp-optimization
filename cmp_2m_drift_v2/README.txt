CMP Dummy Dataset (2 months, with drift/shift & auto-retrain) — v2
Days=60, Tools=2, Runs/Day/Tool=60
Total runs ≈ 7200

Files:
- run_header.csv        : meta + endpoint({endpoint_time_s, actual_polish_time_s, overpolish_s, endpoint_class}) +
                          monitoring({target_mrr_nmps, mrr_pred_nmps, mrr_err_*}, drift_flag_mrr, auto_retrain) +
                          normalization meta({scaler_group_key, channel_stats_version})
- timeseries_process.csv: 1 Hz process signals (torque_norm, motor_current with bias drift, AE, optical)
- timeseries_conditioning.csv: pad conditioning
- pcmp_clean.csv        : post-CMP clean recipe
- layout_features.csv   : 5x5 pattern density tiles
- metrology_labels.csv  : KPIs (thickness_mean_nm, wiwnu, thickness_range, dishing, erosion, particles, yield)
- events.csv            : slurry lot change(weekly), pad/disk change(counters), maintenance(biweekly), auto_retrain
- group_keys.csv        : run_id별 그룹 라벨(tool/layer/recipe/pad_age_bin/pattern_density_bin/slurry_lot/pH_bin)
- composite_score.csv   : per-layer z-normalized composite quality score
- pass_fail.csv         : overall_pass (spec-based)
- channel_stats.json    : scaler_group_key(tool|layer)별 채널 스케일 통계(mean/std/median/iqr, version=v1)

Design:
- Pad wear = slow drift in pad_factor → RR/quality drift.
- Slurry lot = weekly abrupt shift in pH center → distribution shift.
- Motor current bias = random walk + occasional step → sensor drift.
- Diurnal temperature = mild sinusoidal offset.

MLOps:
- mrr_pred is a "stale model" prediction using pad/lot refs at last retrain.
- mrr_err_pct EWMA > threshold → auto_retrain event, model refs updated.
- split_hint: train/val/test by time bands (temporal generalization).
- 학습 입력 시 주의: run_header의 {target_mrr_nmps, mrr_pred_nmps, mrr_err_*}는 입력 제외(모니터링 전용).

Targets:
- Regression: wiwnu_pct, dishing_nm, particle_count_ge_50nm (선택: target_mrr_nmps 보조)
- Classification: yield_flag, overall_pass
- Composite: composite_score (multi-objective ranking)

Endpoint labels:
- endpoint_time_s = 모델 정렬 기준(권장: t' = t − endpoint_time_s).
- endpoint_class  ∈ {under, normal, over} (overpolish_s 기준).
