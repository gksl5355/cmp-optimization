# gen_cmp_2m_drift_mlops_v2.py
# 2개월 규모 CMP 더미 데이터 생성기 (드리프트/쉬프트 + 자동 재학습 + 엔드포인트/정규화메타/그룹키 추가)
# usage:
#   python gen_cmp_2m_drift_mlops_v2.py --days 60 --tools 2 --runs-per-day-per-tool 60 --out ./cmp_2m_drift_v2 --seed 42
import os, argparse, math, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ----------------------------
# Helper: EWMA tracker
# ----------------------------
class EWMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v
        return self.v

def zscore(col):
    mu, sd = np.nanmean(col), np.nanstd(col) + 1e-9
    return (col - mu) / sd

def iqr(a):
    q75 = np.nanpercentile(a, 75)
    q25 = np.nanpercentile(a, 25)
    return float(q75 - q25)

def main(days=60, tools=2, runs_per_day_per_tool=60, outdir="./cmp_2m_drift_v2", seed=42):
    rng = np.random.default_rng(seed)

    # ---------- Catalogs ----------
    process_types = ["Cu", "STI_Oxide"]
    recipes = ["RCP-A", "RCP-B", "RCP-C"]
    pad_types = ["IC1000", "NexPlanar-XP"]
    disk_types = ["Kinik-3.5", "3M-Std"]
    products = [f"P{n}" for n in range(100, 150)]
    tools_list = [f"CMP-{i:02d}" for i in range(1, tools+1)]
    start_time = datetime(2025, 7, 1, 8, 0, 0)

    # ---------- Events & drift params ----------
    pad_change_interval_runs = 800
    disk_change_interval_runs = 1000
    slurry_lot_change_days = 7
    maintenance_every_days = 14
    maintenance_hours = 4

    motor_bias_walk_sigma = 0.01
    motor_bias_step_prob = 0.002
    motor_bias_step_mag = 0.2

    def diurnal_offset(ts):
        h = ts.hour + ts.minute/60
        return 0.4 * math.sin(2*math.pi*h/24)

    # ---------- Containers ----------
    run_header_rows, ts_rows, cond_rows, pcmp_rows, layout_rows, met_rows, event_rows = ([] for _ in range(7))

    # ---------- Tool state (consumables & model) ----------
    tool_state = {}
    for t in tools_list:
        tool_state[t] = {
            "pad_type": rng.choice(pad_types),
            "disk_type": rng.choice(disk_types),
            "pad_age_wafers": int(rng.integers(30, 80)),
            "disk_age_wafers": int(rng.integers(30, 120)),
            "slurry_lot": f"LOT-S{rng.integers(1,9999):04d}",
            "runs_since_pad_change": 0,
            "runs_since_disk_change": 0,

            "motor_bias_A": 0.0,

            # stale model refs (updated on auto_retrain)
            "model_pad_factor_ref": 1.0,
            "model_lot_mu_pH": 7.5,
            "last_retrain_ts": start_time,
            "cooldown_runs": 0,
            "ewma_err": EWMA(alpha=0.2)
        }

    # MLOps thresholds
    err_pct_ewma_thresh = 10.0
    retrain_cooldown = 400

    # ---------- split hints ----------
    def split_hint(ts):
        day_idx = (ts - start_time).days
        if day_idx <= 34: return "train"
        if day_idx <= 46: return "val"
        return "test"

    # ---------- One run simulation ----------
    def simulate_run(run_id, tool_id, ts0):
        st = tool_state[tool_id]

        # Regimes
        product_id = rng.choice(products)
        process_type = rng.choice(process_types, p=[0.6, 0.4])
        layer_id = "M1" if process_type == "Cu" else "STI"
        recipe_id = rng.choice(recipes)

        # Pattern density (5x5)
        base_density = 30 if process_type == "Cu" else 40
        pdens = np.clip(base_density + 10*rng.standard_normal((5,5)), 5, 85)
        mean_density = float(pdens.mean())

        # Setpoints
        platen_rpm_sp = int(rng.choice([60, 80, 100]))
        head_rpm_sp = platen_rpm_sp - int(rng.choice([10, 20]))
        downforce_sp = int(rng.choice([15, 20, 25]))
        ring_kpa_sp = int(rng.choice([3, 5, 7]))
        zone_center_kpa = downforce_sp + int(rng.choice([-1,0,1]))
        zone_edge_kpa   = downforce_sp + int(rng.choice([-1,0,1]))
        slurry_flow_sp  = int(rng.choice([250, 300, 350, 400]))
        slurry_temp_c   = float(rng.normal(22, 0.5) + diurnal_offset(ts0))

        # Chemistry with lot shift
        lot_hash = abs(hash(st["slurry_lot"])) % 1000
        lot_shift_pH = (lot_hash - 500)/500 * 0.3  # ±0.3
        if process_type == "Cu":
            slurry_pH = float(rng.normal(3.5 + lot_shift_pH, 0.2))
            particle_size_nm = float(rng.normal(120, 10))
            oxidizer_pct = float(rng.normal(2.0, 0.2))
        else:
            slurry_pH = float(rng.normal(10.5 + lot_shift_pH, 0.2))
            particle_size_nm = float(rng.normal(70, 8))
            oxidizer_pct = float(rng.normal(0.2, 0.05))

        pre_thk = (950 if process_type == "Cu" else 1050) + rng.normal(0, 25)
        target_post_thk = 300 if process_type == "Cu" else 800

        # True MRR with pad wear drift
        chem_factor = (11 - slurry_pH) if process_type == "Cu" else (slurry_pH - 7)
        chem_factor = max(0.5, float(chem_factor))
        pad_factor_true = 1.0 - 0.00030*(st["pad_age_wafers"] - 200)
        pad_factor_true = float(np.clip(pad_factor_true, 0.60, 1.10))
        rr_true = 0.02 * platen_rpm_sp * (downforce_sp/20) * chem_factor * pad_factor_true
        rr_true = float(np.clip(rr_true, 3.0, 20.0))

        removal = max(0, pre_thk - target_post_thk)
        t_endpoint = int(np.clip(removal/rr_true, 60, 160))

        # ---- endpoint offset (under/over) ----
        if rng.random() < 0.15:
            # 15% 확률로 under-polish
            offset = -int(rng.integers(1, 8))   # 1~7 s 부족
        else:
            offset = int(rng.integers(5, 20))   # 5~19 s over
        planned_end = max(30, t_endpoint + offset)  # 최소 길이 30 s
        secs = np.arange(0, planned_end)
        n = len(secs)

        overpolish_s = planned_end - t_endpoint  # 음수면 under
        if overpolish_s < -3:
            endpoint_class = "under"
        elif overpolish_s <= 5:
            endpoint_class = "normal"
        else:
            endpoint_class = "over"

        # 1 Hz series with sensor bias drift
        platen_rpm = platen_rpm_sp + rng.normal(0, 0.8, n)
        head_rpm   = head_rpm_sp + rng.normal(0, 0.8, n)
        downforce  = downforce_sp + rng.normal(0, 0.5, n)
        ring_kpa   = ring_kpa_sp + rng.normal(0, 0.3, n)
        slurry_flow = slurry_flow_sp + rng.normal(0, 5, n)
        platen_temp = 24 + 0.02*secs + rng.normal(0, 0.2, n) + diurnal_offset(ts0)
        head_temp   = 24 + 0.015*secs + rng.normal(0, 0.2, n) + diurnal_offset(ts0)

        st["motor_bias_A"] += rng.normal(0, motor_bias_walk_sigma)
        if rng.random() < motor_bias_step_prob:
            st["motor_bias_A"] += rng.choice([-1, 1]) * motor_bias_step_mag

        mu_base = 0.35 if process_type == "Cu" else 0.25
        mu = mu_base + 0.002*(downforce - 20) + 0.005*np.tanh((secs - t_endpoint)/10)
        torque_norm = mu * downforce * (platen_rpm/80) + rng.normal(0, 0.2, n)
        motor_current = 8 + 0.1*torque_norm + rng.normal(0, 0.2, n) + st["motor_bias_A"]

        ae_rms = 0.5 + 0.4*np.exp(-((secs - t_endpoint)/8)**2) + 0.05*rng.standard_normal(n)
        optical_reflect = 1.0 - 0.005*secs + 0.1*(secs > t_endpoint) + 0.01*rng.standard_normal(n)

        # Conditioning
        cond_force_n = int(rng.choice([30, 40, 50]))
        cond_rpm     = int(rng.choice([60, 80]))
        cond_sweep   = int(rng.choice([10, 15, 20]))
        cond_time_s  = int(rng.choice([30, 45, 60]))

        # PCMP clean
        brush_speed = int(rng.choice([200, 220, 240]))
        contact_force_n = int(rng.choice([5, 7, 9]))
        megasonic_w = int(rng.choice([0, 150, 250]))
        megasonic_khz = 0 if megasonic_w == 0 else int(rng.choice([800, 1000]))
        chem_type = "acidic" if process_type == "Cu" else "alkaline"
        chem_conc_pct = float(rng.normal(0.5 if chem_type=="acidic" else 1.0, 0.1))
        di_temp_c = float(22.0 + rng.normal(0, 0.3) + diurnal_offset(ts0))
        rinse_s = int(rng.choice([20, 30, 40]))
        dry_type = str(rng.choice(["SR", "Marangoni"]))

        # KPI labels (use only positive overpolish effect for dishing/particles)
        over_pos = max(0, overpolish_s)
        post_mean = target_post_thk + rng.normal(0, 8) + 0.8*over_pos
        wiwnu_pct = 3.0 + 0.02*(zone_edge_kpa - zone_center_kpa)**2 + 0.02*np.std(pdens)
        thickness_range = rng.normal(20 + 0.15*wiwnu_pct, 3)
        dishing_nm = max(0, 2 + 0.15*mean_density + 0.6*over_pos + rng.normal(0, 3))
        erosion_pct = max(0, 0.05*mean_density + 0.1*over_pos + rng.normal(0, 0.5))
        particle_base = 60 + 0.1*mean_density + 0.3*over_pos
        particle_reduction = (megasonic_w>0)*15 + (chem_type=="alkaline")*5
        particle_count = int(max(0, particle_base - particle_reduction + rng.normal(0, 10)))
        scratch_count = int(max(0, rng.poisson(1 + 0.02*mean_density + 0.05*(contact_force_n-6))))
        electrical_res = 1.5e-2 + 1e-4*(post_mean - target_post_thk) + rng.normal(0, 1e-4)
        leakage_na = max(0, rng.normal(5 + 0.1*erosion_pct, 1.5))
        yield_flag = int(particle_count < 80 and wiwnu_pct < 6 and dishing_nm < 25)

        # ---- Monitoring model prediction (stale) ----
        model_pad_ref  = st["model_pad_factor_ref"]
        model_pH_mu    = st["model_lot_mu_pH"]
        chem_model     = (11 - model_pH_mu) if process_type == "Cu" else (model_pH_mu - 7)
        chem_model     = max(0.5, float(chem_model))
        rr_pred = 0.02 * platen_rpm_sp * (downforce_sp/20) * chem_model * model_pad_ref
        rr_pred = float(np.clip(rr_pred, 3.0, 20.0))

        err_abs = abs(rr_true - rr_pred)
        err_pct = 100.0 * err_abs / max(1e-6, rr_true)
        err_ewma = st["ewma_err"].update(err_pct)
        drift_flag_mrr = int(err_ewma > err_pct_ewma_thresh)

        auto_retrain = 0
        if drift_flag_mrr and st["cooldown_runs"] <= 0:
            st["model_pad_factor_ref"] = pad_factor_true
            st["model_lot_mu_pH"] = slurry_pH
            st["last_retrain_ts"] = ts0
            st["cooldown_runs"] = retrain_cooldown
            auto_retrain = 1
        else:
            st["cooldown_runs"] = max(0, st["cooldown_runs"] - 1)

        # Header
        run_start_ts = ts0
        run_end_ts = run_start_ts + timedelta(seconds=n)
        scaler_group_key = f"{tool_id}|{layer_id}"

        run_header_rows.append({
            "run_id": run_id,
            "split_hint": split_hint(run_start_ts),
            "fab_id": "H-FAB",
            "tool_id": tool_id,
            "recipe_id": recipe_id,
            "product_id": product_id,
            "layer_id": layer_id,
            "process_type": process_type,
            "wafer_id": f"W-{run_id}",
            "lot_id": f"LOT-{run_start_ts.strftime('%m%d')}",
            "slurry_lot": st["slurry_lot"],
            "run_start_ts": run_start_ts.isoformat(),
            "run_end_ts": run_end_ts.isoformat(),
            "pad_type": st["pad_type"],
            "conditioner_type": st["disk_type"],
            "slurry_type": "Cu-acidic" if process_type=="Cu" else "Oxide-alkaline",
            "slurry_pH": round(slurry_pH,3),
            "particle_size_nm": round(particle_size_nm,1),
            "oxidizer_pct": round(oxidizer_pct,3),
            "pad_age_wafers": int(st["pad_age_wafers"]),
            "disk_age_wafers": int(st["disk_age_wafers"]),
            "pre_thickness_nm": round(pre_thk,1),
            "target_post_thk_nm": target_post_thk,

            # endpoint annotations
            "endpoint_time_s": int(t_endpoint),
            "actual_polish_time_s": int(planned_end),
            "overpolish_s": int(overpolish_s),
            "endpoint_class": endpoint_class,

            # Monitoring & drift
            "target_mrr_nmps": round(rr_true,4),
            "mrr_pred_nmps": round(rr_pred,4),
            "mrr_err_abs": round(err_abs,4),
            "mrr_err_pct": round(err_pct,3),
            "mrr_err_ewma": round(err_ewma,3),
            "drift_flag_mrr": drift_flag_mrr,
            "auto_retrain": auto_retrain,
            "model_ref_pad_factor": round(st["model_pad_factor_ref"],4),
            "model_ref_lot_pH": round(st["model_lot_mu_pH"],3),
            "last_retrain_ts": st["last_retrain_ts"].isoformat(),

            # normalization meta
            "scaler_group_key": scaler_group_key,
            "channel_stats_version": "v1"
        })

        # Time series
        for i, s in enumerate(secs):
            ts_rows.append({
                "run_id": run_id,
                "ts": (run_start_ts + timedelta(seconds=int(s))).isoformat(),
                "sec": int(s),
                "platen_rpm": round(platen_rpm[i],3),
                "head_rpm": round(head_rpm[i],3),
                "downforce_kpa": round(downforce[i],3),
                "ring_kpa": round(ring_kpa[i],3),
                "zone_center_kpa": zone_center_kpa,
                "zone_edge_kpa": zone_edge_kpa,
                "slurry_flow_mlpm": round(slurry_flow[i],3),
                "slurry_temp_c": round(slurry_temp_c,3),
                "platen_temp_c": round(platen_temp[i],3),
                "head_temp_c": round(head_temp[i],3),
                "torque_norm": round(torque_norm[i],4),
                "motor_current_a": round(motor_current[i],4),
                "ae_rms": round(ae_rms[i],4),
                "optical_reflect": round(optical_reflect[i],4)
            })

        # Conditioning & PCMP
        cond_rows.append({
            "run_id": run_id,
            "cond_force_n": cond_force_n,
            "cond_rpm": cond_rpm,
            "cond_sweep_mms": cond_sweep,
            "cond_time_s": cond_time_s
        })
        pcmp_rows.append({
            "run_id": run_id,
            "brush_speed_rpm": brush_speed,
            "contact_force_n": contact_force_n,
            "chem_type": chem_type,
            "chem_conc_pct": round(chem_conc_pct,3),
            "megasonic_w": megasonic_w,
            "megasonic_khz": megasonic_khz,
            "di_temp_c": round(di_temp_c,3),
            "rinse_s": rinse_s,
            "dry_type": dry_type
        })

        # Layout tiles
        for ix in range(5):
            for iy in range(5):
                layout_rows.append({
                    "run_id": run_id,
                    "grid_x": ix,
                    "grid_y": iy,
                    "pattern_density_pct": round(float(pdens[iy,ix]),3)
                })

        # KPI labels (incl. thickness_mean_nm)
        met_rows.append({
            "run_id": run_id,
            "wafer_id": f"W-{run_id}",
            "thickness_mean_nm": round(float(post_mean),2),
            "wiwnu_pct": round(float(wiwnu_pct),3),
            "thickness_range_nm": round(float(thickness_range),2),
            "dishing_nm": round(float(dishing_nm),2),
            "erosion_pct": round(float(erosion_pct),3),
            "particle_count_ge_50nm": int(particle_count),
            "scratch_count": int(scratch_count),
            "electrical_res_ohm": round(float(electrical_res),6),
            "leakage_na": round(float(leakage_na),3),
            "yield_flag": int(yield_flag)
        })

        # Update consumables
        st["pad_age_wafers"]  += 1
        st["disk_age_wafers"] += 1
        st["runs_since_pad_change"]  += 1
        st["runs_since_disk_change"] += 1

        return run_end_ts, auto_retrain

    # ---------- Schedule ----------
    cur_time = {t: start_time for t in tools_list}
    run_idx = 0

    for day in range(days):
        day_date = start_time + timedelta(days=day)

        # Slurry lot swap (weekly)
        if day % slurry_lot_change_days == 0 and day > 0:
            for t in tools_list:
                tool_state[t]["slurry_lot"] = f"LOT-S{rng.integers(1,9999):04d}"
                event_rows.append({"ts": (day_date + timedelta(hours=7)).isoformat(),
                                   "tool_id": t, "event": "slurry_lot_change",
                                   "detail": tool_state[t]["slurry_lot"]})

        # Maintenance (biweekly)
        for ti, t in enumerate(tools_list):
            if (day - ti) % maintenance_every_days == 0 and day > 0:
                mt_start = day_date + timedelta(hours=18)
                mt_end = mt_start + timedelta(hours=maintenance_hours)
                event_rows.append({"ts": mt_start.isoformat(), "tool_id": t,
                                   "event": "maintenance_start", "detail": f"{maintenance_hours}h"})
                cur_time[t] = max(cur_time[t], mt_end)
                event_rows.append({"ts": mt_end.isoformat(), "tool_id": t,
                                   "event": "maintenance_end", "detail": ""})

        # Daily production
        for t in tools_list:
            for _ in range(runs_per_day_per_tool):
                # Pad/disk replacement
                if tool_state[t]["runs_since_pad_change"] >= pad_change_interval_runs:
                    tool_state[t]["pad_type"] = np.random.choice(pad_types)
                    tool_state[t]["pad_age_wafers"] = 0
                    tool_state[t]["runs_since_pad_change"] = 0
                    event_rows.append({"ts": cur_time[t].isoformat(), "tool_id": t,
                                       "event": "pad_change", "detail": tool_state[t]["pad_type"]})
                if tool_state[t]["runs_since_disk_change"] >= disk_change_interval_runs:
                    tool_state[t]["disk_type"] = np.random.choice(disk_types)
                    tool_state[t]["disk_age_wafers"] = 0
                    tool_state[t]["runs_since_disk_change"] = 0
                    event_rows.append({"ts": cur_time[t].isoformat(), "tool_id": t,
                                       "event": "disk_change", "detail": tool_state[t]["disk_type"]})

                run_id = f"RUN-{run_idx:06d}"
                run_end_ts, retrain_flag = simulate_run(run_id, t, cur_time[t])

                if retrain_flag:
                    event_rows.append({"ts": cur_time[t].isoformat(), "tool_id": t,
                                       "event": "auto_retrain", "detail": ""})

                gap_min = int(np.clip(np.round(np.random.normal(10, 2)), 6, 20))
                cur_time[t] = run_end_ts + timedelta(minutes=gap_min)
                run_idx += 1

    # ---------- Save CSVs ----------
    os.makedirs(outdir, exist_ok=True)
    def save_csv(name, rows):
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, name), index=False)
        return df

    df_header = save_csv("run_header.csv", run_header_rows)
    df_ts     = save_csv("timeseries_process.csv", ts_rows)
    df_cond   = save_csv("timeseries_conditioning.csv", cond_rows)
    df_pcmp   = save_csv("pcmp_clean.csv", pcmp_rows)
    df_layout = save_csv("layout_features.csv", layout_rows)
    df_met    = save_csv("metrology_labels.csv", met_rows)
    df_evt    = save_csv("events.csv", event_rows)

    # ---------- Composite score & pass/fail ----------
    m = df_header[["run_id","layer_id"]].merge(df_met, on="run_id", how="left")
    m["_z_wiwnu"]    = m.groupby("layer_id")["wiwnu_pct"].transform(zscore)
    m["_z_particle"] = m.groupby("layer_id")["particle_count_ge_50nm"].transform(zscore)
    m["_z_dishing"]  = m.groupby("layer_id")["dishing_nm"].transform(zscore)
    w1,w2,w3,w4 = 0.35,0.35,0.20,0.10
    comp = w1*(-m["_z_wiwnu"]) + w2*(-m["_z_particle"]) + w3*(-m["_z_dishing"]) + w4*(m["yield_flag"])
    comp = (comp - comp.mean())/(comp.std()+1e-9)
    pd.DataFrame({"run_id": m["run_id"], "composite_score": comp}).to_csv(os.path.join(outdir,"composite_score.csv"), index=False)

    thr_wiwnu, thr_particle, thr_dishing = 6.0, 80, 25.0
    overall_pass = ((m["wiwnu_pct"]<thr_wiwnu)&(m["particle_count_ge_50nm"]<thr_particle)&(m["dishing_nm"]<thr_dishing)).astype(int)
    pd.DataFrame({"run_id": m["run_id"], "overall_pass": overall_pass}).to_csv(os.path.join(outdir,"pass_fail.csv"), index=False)

    # ---------- Group keys (for GroupKFold / stress reports) ----------
    # pattern_density mean & bin
    pd_mean = df_layout.groupby("run_id")["pattern_density_pct"].mean().rename("pattern_density_mean")
    g = df_header[["run_id","tool_id","layer_id","recipe_id","pad_age_wafers","slurry_lot","slurry_pH"]].merge(pd_mean, on="run_id", how="left")
    def pad_bin(x):  # bins: new/mid/old
        if x < 150: return "pad_new"
        if x < 400: return "pad_mid"
        return "pad_old"
    def dens_bin(x):
        if x < 25: return "low"
        if x < 55: return "mid"
        return "high"
    def pH_bin(x):
        if x < 5: return "acidic"
        if x > 9: return "alkaline"
        return "neutral"
    g["pad_age_bin"] = g["pad_age_wafers"].apply(pad_bin)
    g["pattern_density_bin"] = g["pattern_density_mean"].apply(dens_bin)
    g["slurry_pH_bin"] = g["slurry_pH"].apply(pH_bin)
    g[["run_id","tool_id","layer_id","recipe_id","pad_age_bin","pattern_density_bin","slurry_lot","slurry_pH_bin"]].to_csv(os.path.join(outdir,"group_keys.csv"), index=False)

    # ---------- Channel stats (normalization meta) ----------
    # merge scaler_group_key to timeseries
    ts_with_group = df_ts.merge(df_header[["run_id","scaler_group_key"]], on="run_id", how="left")
    channels = ["platen_rpm","head_rpm","downforce_kpa","ring_kpa",
                "slurry_flow_mlpm","slurry_temp_c","platen_temp_c","head_temp_c",
                "torque_norm","motor_current_a","ae_rms","optical_reflect"]
    stats = {}
    for key, grp in ts_with_group.groupby("scaler_group_key"):
        stats[key] = {}
        for ch in channels:
            arr = grp[ch].to_numpy(dtype=float)
            stats[key][ch] = {
                "mean": float(np.nanmean(arr)),
                "std":  float(np.nanstd(arr) + 1e-9),
                "median": float(np.nanmedian(arr)),
                "iqr": iqr(arr)
            }
    with open(os.path.join(outdir,"channel_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"version":"v1", "by_group": stats}, f, ensure_ascii=False, indent=2)

    # ---------- README ----------
    with open(os.path.join(outdir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"""CMP Dummy Dataset (2 months, with drift/shift & auto-retrain) — v2
Days={days}, Tools={tools}, Runs/Day/Tool={runs_per_day_per_tool}
Total runs ≈ {len(df_header)}

Files:
- run_header.csv        : meta + endpoint({{endpoint_time_s, actual_polish_time_s, overpolish_s, endpoint_class}}) +
                          monitoring({{target_mrr_nmps, mrr_pred_nmps, mrr_err_*}}, drift_flag_mrr, auto_retrain) +
                          normalization meta({{scaler_group_key, channel_stats_version}})
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
- 학습 입력 시 주의: run_header의 {{target_mrr_nmps, mrr_pred_nmps, mrr_err_*}}는 입력 제외(모니터링 전용).

Targets:
- Regression: wiwnu_pct, dishing_nm, particle_count_ge_50nm (선택: target_mrr_nmps 보조)
- Classification: yield_flag, overall_pass
- Composite: composite_score (multi-objective ranking)

Endpoint labels:
- endpoint_time_s = 모델 정렬 기준(권장: t' = t − endpoint_time_s).
- endpoint_class  ∈ {{under, normal, over}} (overpolish_s 기준).
""")

    print("[DONE] saved to", os.path.abspath(outdir))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--tools", type=int, default=2)
    ap.add_argument("--runs-per-day-per-tool", type=int, default=60)
    ap.add_argument("--out", type=str, default="./cmp_2m_drift_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.days, args.tools, args.runs_per_day_per_tool, args.out, args.seed)
