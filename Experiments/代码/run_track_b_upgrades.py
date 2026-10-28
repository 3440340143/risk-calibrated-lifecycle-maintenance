#!/usr/bin/env python3
"""Reproduce full-pool lifecycle selection and retirement-score calibration on C-MAPSS.

The frozen forecast map, split seeds, matched policy budgets, and exact
record-crossing search follow the protocol in the accompanying paper.
"""
from __future__ import annotations
import sys
import os
import math
import json
import argparse
from pathlib import Path
from time import perf_counter
import platform

for thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_var] = '1'

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import run_public_benchmark as rpb

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / '公开基准' / 'results_full_pool_rerun'
CORE_PATH = Path(__file__).resolve().parent / 'decision_calibrated_pdm_suite.py'
RAW_DIR = ROOT / '公开基准' / 'raw'

def censored_retirement_calibration(sequences, lam_hist):
    """
    Compute observable retirement scores \tilde{B}_u under historical PM policy lam_hist:
    - If unit u is preventively replaced at tau_u(lam_hist) <= T_u - l (i.e. B_u <= lam_hist),
      its trajectory is right-censored at retirement U_u = tau_u(lam_hist) + l < T_u, and only
      decision cycles t <= tau_u(lam_hist) are observed before replacement commitment.
      The observable prefix critical score is \tilde{B}_u = min_{t <= tau_u(lam_hist)} a_{u,t} <= lam_hist.
    - If unit u suffers an unscheduled breakdown (B_u > lam_hist), no preventive replacement occurs
      before T_u - l, so natural failure T_u is observed uncensored and \tilde{B}_u = B_u > lam_hist.
    Also returns naive failures-only scores (which discard preventively maintained units).
    """
    ell = rpb.CONFIG["lead_time"]
    exact_scores = []
    tilde_scores = []
    failures_only_scores = []
    n_censored = 0
    for u in sequences:
        crossings = (u["point"] - ell) / u["scale"]
        pre_mask = (u["cycles"] + ell < u["T"])
        b_exact = float(np.min(crossings[pre_mask]))
        exact_scores.append(b_exact)
        
        eligible = np.flatnonzero(crossings <= float(lam_hist))
        tau = float(u["cycles"][eligible[0]]) if len(eligible) else np.inf
        if tau + ell < u["T"]:
            # Preventively replaced! Trajectory right-censored after tau
            n_censored += 1
            obs_mask = (u["cycles"] <= tau)
            b_tilde = float(np.min(crossings[obs_mask]))
            tilde_scores.append(b_tilde)
        else:
            # Unscheduled breakdown! Observed until T_u
            tilde_scores.append(b_exact)
            failures_only_scores.append(b_exact)
            
    rank = int(math.ceil((len(tilde_scores) + 1) * (1.0 - rpb.CONFIG["epsilon"])))
    q_rc = max(0.0, sorted(tilde_scores)[rank - 1]) if rank <= len(tilde_scores) else None
    q_exact = max(0.0, sorted(exact_scores)[rank - 1]) if rank <= len(exact_scores) else None
    
    if len(failures_only_scores) >= 1:
        rank_fo = int(math.ceil((len(failures_only_scores) + 1) * (1.0 - rpb.CONFIG["epsilon"])))
        q_fo = max(0.0, sorted(failures_only_scores)[min(rank_fo - 1, len(failures_only_scores) - 1)])
    else:
        q_fo = float(lam_hist)
        
    return {
        "exact_scores": np.array(exact_scores, dtype=float),
        "tilde_scores": np.array(tilde_scores, dtype=float),
        "n_cal": len(sequences),
        "n_censored": n_censored,
        "censoring_rate": n_censored / float(len(sequences)),
        "q_exact": q_exact,
        "q_rc": q_rc,
        "q_failures_only": q_fo,
    }

def main():
    global OUT_DIR, RAW_DIR, CORE_PATH
    parser = argparse.ArgumentParser(description='Reproduce the full-pool and retirement-score experiments.')
    parser.add_argument('--raw', type=Path, default=RAW_DIR)
    parser.add_argument('--output', type=Path, default=OUT_DIR)
    parser.add_argument('--core', type=Path, default=CORE_PATH)
    args = parser.parse_args()
    OUT_DIR, RAW_DIR, CORE_PATH = args.output.resolve(), args.raw.resolve(), args.core.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    core = rpb.load_core(CORE_PATH)
    if not (RAW_DIR / 'train_FD001.txt').exists():
        rpb.extract_raw(RAW_DIR, False)
    raw_extract = RAW_DIR

    seeds = [rpb.CONFIG["primary_seed"]] + rpb.CONFIG["sensitivity_seeds"]
    seed_rows = []
    all_units = []
    rc_rows = []
    
    for dataset in rpb.CONFIG["datasets"]:
        frame = rpb.load_dataset(raw_extract / f"train_{dataset}.txt")
        for seed in seeds:
            print(f"Replaying {dataset}, split {seed}", flush=True)
            parts, _ = rpb.split_frame(frame, seed)
            enc = core.CausalMonotoneEncoder(True, True).fit(parts["fit"])
            features = {key: enc.transform(df)[0] for key, df in parts.items()}
            
            for head_name, kind, focus in [("Ridge", "ridge", False), ("Linex", "linex", True)]:
                head = core.PrognosticHead().fit(
                    features["fit"], parts["fit"].true_rul.to_numpy(),
                    kind=kind, reg=rpb.CONFIG["regularization"], focus_window=focus
                )
                predictions = {
                    key: head.predict(x, parts[key], causal_smooth=(head_name == "Linex"))
                    for key, x in features.items()
                }
                if head_name == "Linex":
                    bias = float(np.mean(predictions["tune"] - parts["tune"].true_rul.to_numpy()))
                    predictions = {key: np.clip(p - 0.85 * bias, 0, rpb.CONFIG["rul_cap"]) for key, p in predictions.items()}
                sd = max(1e-6, float(np.std(predictions["tune"] - parts["tune"].true_rul.to_numpy())))
                scales = {
                    key: (sd * (0.30 + 0.70 * expit((37 - p) / 9)) if head_name == "Linex" else np.full(len(p), sd))
                    for key, p in predictions.items()
                }
                sequences = {key: rpb.unit_sequences(parts[key], predictions[key], scales[key]) for key in parts}
                
                q, scores, rank = rpb.calibration_threshold(sequences["cal"])
                tune_margin, tune_cost, _ = rpb.exact_margin_frontier(sequences["tune"])
                equal_margin, equal_cost, _ = rpb.exact_margin_frontier(sequences["tune"] + sequences["cal"])
                floor = None if q is None else max(q, tune_margin)
                constrained, constrained_cost, _ = rpb.exact_margin_frontier(sequences["tune"], q)
                
                # NEW: Theorem 1 — Full-Pool Conformal-Constrained Policy Optimization (FP-CCPO)
                fp_ccpo, fp_ccpo_cost, _ = rpb.exact_margin_frontier(sequences["tune"] + sequences["cal"], q)
                # Also Equal-Budget Floor: max(q, equal_margin)
                fp_floor = None if q is None else max(q, equal_margin)
                
                policies = [
                    ("ERM_Tune", tune_margin),
                    ("UnitCP", q),
                    ("Floor", floor),
                    ("Constrained", constrained),
                    ("ERM_EqualBudget", equal_margin),
                    ("FP_Floor", fp_floor),
                    ("FP_CCPO", fp_ccpo),
                ]
                if head_name == "Ridge":
                    residuals = predictions["cal"] - parts["cal"].true_rul.to_numpy()
                    k = math.ceil((len(residuals) + 1) * (1 - rpb.CONFIG["epsilon"]))
                    residual_q = max(0.0, np.sort(residuals)[min(k - 1, len(residuals) - 1)])
                    policies.append(("Cycle_CP", residual_q / sd))
                    
                for policy, margin in policies:
                    name = "Cycle_CP" if policy == "Cycle_CP" else f"{head_name}_{policy}"
                    units = rpb.at_margin(sequences["test"], margin)
                    for column, value in [("dataset", dataset), ("seed", seed), ("method", name)]:
                        units.insert(0, column, value)
                    all_units.append(units)
                    seed_rows.append(dict(
                        dataset=dataset, seed=seed, method=name, margin=margin,
                        n_fit=parts["fit"].unit_id.nunique(),
                        n_tune=parts["tune"].unit_id.nunique(),
                        n_cal=parts["cal"].unit_id.nunique(),
                        **rpb.summary(units)
                    ))
                    
                # --- Theorem 2: Preventive Maintenance Right-Censoring Verification ---
                tune_b = [float(np.min(((u["point"] - rpb.CONFIG["lead_time"]) / u["scale"])[u["cycles"] + rpb.CONFIG["lead_time"] < u["T"]])) for u in sequences["tune"]]
                for scen_name, lam_hist in [
                    ("Historical_ERM_Tune", float(tune_margin)),
                    ("Historical_Q70_Tune", float(np.quantile(tune_b, 0.70))),
                    ("Historical_Q85_Tune", float(np.quantile(tune_b, 0.85))),
                ]:
                    rc_res = censored_retirement_calibration(sequences["cal"], lam_hist)
                    b_exact = rc_res["exact_scores"]
                    b_tilde = rc_res["tilde_scores"]
                    
                    # Verify Theorem 2(a): Pathwise dominance B_u <= \tilde{B}_u for EVERY calibration unit
                    max_dom_violation = float(np.max(b_exact - b_tilde))
                    assert max_dom_violation <= 1e-12, f"Dominance violated: {max_dom_violation}"
                    
                    # Verify Theorem 2(b): Exact upper-tail identifiability above lam_hist
                    # Specifically, max(lam_hist, b_tilde) == max(lam_hist, b_exact) identically!
                    upper_tail_err = float(np.max(np.abs(np.maximum(b_tilde, lam_hist) - np.maximum(b_exact, lam_hist))))
                    assert upper_tail_err <= 1e-12, f"Upper-tail identifiability violated: {upper_tail_err}"
                    
                    q_exact = rc_res["q_exact"]
                    q_rc = rc_res["q_rc"]
                    q_fo = rc_res["q_failures_only"]
                    
                    env_err = abs(max(lam_hist, q_rc) - max(lam_hist, q_exact))
                    assert env_err <= 1e-12, f"Envelope threshold error: {env_err}"
                    
                    # Evaluate test units under:
                    # 1. Uncensored Oracle UnitCP (q_exact)
                    # 2. Right-Censored RC-CRC (q_rc)
                    # 3. Right-Censored Envelope max(lam_hist, q_rc)
                    # 4. Naive Failures-Only (q_fo)
                    u_or = rpb.at_margin(sequences["test"], q_exact)
                    u_rc = rpb.at_margin(sequences["test"], q_rc)
                    u_env = rpb.at_margin(sequences["test"], max(lam_hist, q_rc))
                    u_fo = rpb.at_margin(sequences["test"], q_fo)
                    
                    rc_rows.append({
                        "seed": seed,
                        "dataset": dataset,
                        "head": head_name,
                        "scenario": scen_name,
                        "n_cal": rc_res["n_cal"],
                        "n_censored": rc_res["n_censored"],
                        "censoring_pct": 100.0 * rc_res["censoring_rate"],
                        "lambda_hist": lam_hist,
                        "q_oracle": q_exact,
                        "q_rc_crc": q_rc,
                        "q_failures_only": q_fo,
                        "max_dominance_violation": max_dom_violation,
                        "upper_tail_identifiability_error": upper_tail_err,
                        "envelope_threshold_error": env_err,
                        "exact_q_diff": abs(q_rc - q_exact),
                        "oracle_j_life": rpb.summary(u_or)["j_life"],
                        "rc_crc_j_life": rpb.summary(u_rc)["j_life"],
                        "rc_env_j_life": rpb.summary(u_env)["j_life"],
                        "failures_only_j_life": rpb.summary(u_fo)["j_life"],
                        "oracle_fail_pct": rpb.summary(u_or)["fail_pct"],
                        "rc_crc_fail_pct": rpb.summary(u_rc)["fail_pct"],
                        "rc_env_fail_pct": rpb.summary(u_env)["fail_pct"],
                        "failures_only_fail_pct": rpb.summary(u_fo)["fail_pct"],
                        "oracle_total_cost": float(u_or.total_cost.sum()),
                        "oracle_service_cycles": float(u_or.service_cycles.sum()),
                        "rc_crc_total_cost": float(u_rc.total_cost.sum()),
                        "rc_crc_service_cycles": float(u_rc.service_cycles.sum()),
                        "failures_only_total_cost": float(u_fo.total_cost.sum()),
                        "failures_only_service_cycles": float(u_fo.service_cycles.sum()),
                        "oracle_fails": int(u_or.fail.sum()),
                        "rc_crc_fails": int(u_rc.fail.sum()),
                        "failures_only_fails": int(u_fo.fail.sum()),
                        "n_test": len(u_or),
                    })

    per_split_df = pd.DataFrame(seed_rows)
    units_df = pd.concat(all_units, ignore_index=True)
    rc_df = pd.DataFrame(rc_rows)
    
    per_split_df.to_csv(OUT_DIR / "cmapss_upgraded_per_split.csv", index=False, lineterminator='\n')
    units_df.to_csv(OUT_DIR / "cmapss_upgraded_per_unit.csv", index=False, lineterminator='\n')
    rc_df.to_csv(OUT_DIR / "cmapss_right_censored_verification.csv", index=False, lineterminator='\n')
    
    # Pooled primary holdout summary across all 286 test engines
    prim_units = units_df[units_df.seed == rpb.CONFIG["primary_seed"]]
    prim_pooled = []
    for method, grp in prim_units.groupby("method", sort=False):
        s = rpb.summary(grp)
        prim_pooled.append({"method": method, **s})
    prim_pooled_df = pd.DataFrame(prim_pooled).sort_values("j_life")
    prim_pooled_df.to_csv(OUT_DIR / "cmapss_upgraded_primary_pooled.csv", index=False, lineterminator='\n')
    
    # 6-Split pooled summary (for each seed, pool across the 4 datasets, then average across 6 seeds)
    seed_pooled = []
    for (seed, method), grp in units_df.groupby(["seed", "method"], sort=False):
        s = rpb.summary(grp)
        seed_pooled.append({"seed": seed, "method": method, **s})
    seed_pooled_df = pd.DataFrame(seed_pooled)
    seed_pooled_df.to_csv(OUT_DIR / "cmapss_upgraded_seed_pooled.csv", index=False, lineterminator='\n')
    
    multi_agg = seed_pooled_df.groupby("method", as_index=False).agg(
        mean_j_life=("j_life", "mean"),
        sd_j_life=("j_life", "std"),
        mean_fail_pct=("fail_pct", "mean"),
        sd_fail_pct=("fail_pct", "std"),
        max_fail_pct=("fail_pct", "max"),
        mean_waste_cycles=("waste_cycles", "mean"),
        mean_util_pct=("utilization_pct", "mean"),
    ).sort_values("mean_j_life")
    multi_agg.to_csv(OUT_DIR / "cmapss_upgraded_6split_summary.csv", index=False, lineterminator='\n')
    
    # Stratified Paired Bootstrap on Primary Holdout (4,000 draws, exact same RNG seed 471902 as original!)
    methods = prim_units.method.unique().tolist()
    by_dataset = {}
    for dataset, group in prim_units.groupby("dataset", sort=True):
        order = sorted(group.unit_id.unique())
        by_dataset[dataset] = {m: group[group.method == m].set_index("unit_id").loc[order] for m in methods}
    rng = np.random.default_rng(rpb.CONFIG["bootstrap_seed"])
    numerator = {m: np.zeros(rpb.CONFIG["bootstrap_replicates"]) for m in methods}
    denominator = {m: np.zeros(rpb.CONFIG["bootstrap_replicates"]) for m in methods}
    fail_num = {m: np.zeros(rpb.CONFIG["bootstrap_replicates"]) for m in methods}
    for dataset, records in by_dataset.items():
        count = len(next(iter(records.values())))
        draws = rng.integers(0, count, (rpb.CONFIG["bootstrap_replicates"], count))
        for method in methods:
            numerator[method] += records[method].total_cost.to_numpy()[draws].sum(axis=1)
            denominator[method] += records[method].service_cycles.to_numpy()[draws].sum(axis=1)
            fail_num[method] += records[method].fail.to_numpy()[draws].sum(axis=1)
    ratios = {m: 100.0 * numerator[m] / denominator[m] for m in methods}
    fail_rates = {m: 100.0 * fail_num[m] / 286.0 for m in methods}
    
    comparisons = [
        ("Linex_FP_CCPO", "Linex_ERM_EqualBudget"),
        ("Linex_FP_CCPO", "Linex_Constrained"),
        ("Linex_FP_CCPO", "Linex_Floor"),
        ("Linex_FP_CCPO", "Linex_ERM_Tune"),
        ("Linex_FP_CCPO", "Cycle_CP"),
        ("Ridge_FP_CCPO", "Ridge_Floor"),
        ("Ridge_FP_CCPO", "Ridge_Constrained"),
        ("Ridge_FP_CCPO", "Ridge_ERM_EqualBudget"),
        ("Ridge_FP_CCPO", "Ridge_ERM_Tune"),
        ("Ridge_FP_CCPO", "Cycle_CP"),
        ("Linex_FP_CCPO", "Ridge_FP_CCPO"),
    ]
    boot_rows = []
    for treat, comp in comparisons:
        diff_j = ratios[treat] - ratios[comp]
        diff_f = fail_rates[treat] - fail_rates[comp]
        pt_t = 100.0 * prim_units[prim_units.method == treat].total_cost.sum() / prim_units[prim_units.method == treat].service_cycles.sum()
        pt_c = 100.0 * prim_units[prim_units.method == comp].total_cost.sum() / prim_units[prim_units.method == comp].service_cycles.sum()
        pf_t = 100.0 * prim_units[prim_units.method == treat].fail.mean()
        pf_c = 100.0 * prim_units[prim_units.method == comp].fail.mean()
        lo_j, hi_j = np.quantile(diff_j, [0.025, 0.975])
        lo_f, hi_f = np.quantile(diff_f, [0.025, 0.975])
        boot_rows.append({
            "treatment": treat,
            "comparator": comp,
            "delta_j_life": pt_t - pt_c,
            "ci95_j_low": float(lo_j),
            "ci95_j_high": float(hi_j),
            "delta_fail_pct": pf_t - pf_c,
            "ci95_fail_low": float(lo_f),
            "ci95_fail_high": float(hi_f),
        })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "cmapss_upgraded_paired_bootstrap.csv", index=False, lineterminator='\n')
    
    print("=== PRIMARY HOLDOUT POOLED (286 ENGINES) ===")
    print(prim_pooled_df[["method", "failures", "fail_pct", "descriptive_cp_ucb95_pct", "j_life", "waste_cycles", "utilization_pct"]].to_string(index=False))
    
    print("\n=== 6-SPLIT POOLED AGGREGATE SUMMARY ===")
    print(multi_agg.to_string(index=False))
    
    print("\n=== PAIRED BOOTSTRAP CIs (4,000 REPLICATES, SEED 471902) ===")
    print(boot_df.to_string(index=False))
    
    print("\n=== THEOREM 2: RIGHT-CENSORED CALIBRATION POOLED ACROSS FD001-FD004 (PRIMARY SEED) ===")
    rc_prim = rc_df[rc_df.seed == rpb.CONFIG["primary_seed"]]
    for (scen, head), grp in rc_prim.groupby(["scenario", "head"]):
        tot_cal = grp.n_cal.sum()
        tot_cen = grp.n_censored.sum()
        or_j = 100.0 * grp.oracle_total_cost.sum() / grp.oracle_service_cycles.sum()
        rc_j = 100.0 * grp.rc_crc_total_cost.sum() / grp.rc_crc_service_cycles.sum()
        fo_j = 100.0 * grp.failures_only_total_cost.sum() / grp.failures_only_service_cycles.sum()
        or_f = 100.0 * grp.oracle_fails.sum() / grp.n_test.sum()
        rc_f = 100.0 * grp.rc_crc_fails.sum() / grp.n_test.sum()
        fo_f = 100.0 * grp.failures_only_fails.sum() / grp.n_test.sum()
        max_dom = grp.max_dominance_violation.max()
        max_ut = grp.upper_tail_identifiability_error.max()
        max_env = grp.envelope_threshold_error.max()
        max_qdiff = grp.exact_q_diff.max()
        print(f"{scen:24s} | {head:5s} | Censored: {tot_cen}/{tot_cal} ({100*tot_cen/tot_cal:.1f}%) | "
              f"DomViol: {max_dom:.1e} | UpperTailErr: {max_ut:.1e} | EnvErr: {max_env:.1e} | ExactQDiff: {max_qdiff:.4f} | "
              f"Oracle J={or_j:.4f} (Fail={or_f:.2f}%) | RC-CRC J={rc_j:.4f} (Fail={rc_f:.2f}%) | FailOnly J={fo_j:.4f} (Fail={fo_f:.2f}%)")

    runtime = dict(platform=platform.platform(), python=platform.python_version(),
                   numpy=np.__version__, pandas=pd.__version__,
                   elapsed_seconds=perf_counter()-started, thread_limit=1,
                   primary_seed=rpb.CONFIG['primary_seed'],
                   sensitivity_seeds=rpb.CONFIG['sensitivity_seeds'],
                   bootstrap_replicates=rpb.CONFIG['bootstrap_replicates'],
                   raw_directory=str(RAW_DIR), core_sha256=rpb.sha(CORE_PATH))
    (OUT_DIR / 'local_runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')

if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
