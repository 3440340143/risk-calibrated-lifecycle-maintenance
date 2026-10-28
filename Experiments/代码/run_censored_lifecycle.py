"""Censored-record lifecycle selection and scalar-structure audit.

The policy-selection functions accept observed retirement records and specified
scalar scenarios. Full lifetimes are supplied separately to numerical audits and
retrospective policy evaluation. Source predictors and data splits are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'Experiments' / '代码'))
import run_public_benchmark as rpb
import run_retirement_event_optimization as ro


def observable_states(records, lower):
    """One scan, one sort, then constant work per strict prefix record."""
    cfg = rpb.CONFIG
    crossings = []
    for i, record in enumerate(records):
        prefix_min = math.inf
        for score, cycle in zip(record['a'], record['cycles']):
            if score < prefix_min:
                if score > lower:
                    crossings.append((float(score), i, float(cycle)))
                prefix_min = float(score)
    costs, services, _ = ro.observed_replay(records, lower)
    csum, usum = float(costs.sum()), float(services.sum())
    result = [(float(lower), csum, usum)]
    crossings.sort(key=lambda item: item[0])
    for value, group in itertools.groupby(crossings, key=lambda item: item[0]):
        for _, i, cycle in group:
            record = records[i]
            failed = int(record['failed'] and cycle + cfg['lead_time'] >= record['age'])
            cost = 1.0 + (cfg['rho'] - 1.0) * failed
            service = min(cycle + cfg['lead_time'], record['age'])
            csum += cost - costs[i]
            usum += service - services[i]
            costs[i], services[i] = cost, service
        result.append((value, csum, usum))
    return np.asarray(result, dtype=float)


def cost_vector(states, scalar, cw):
    return 100.0 * (states[:, 1] + cw * scalar) / states[:, 2] - 100.0 * cw


def first_minimum(values):
    # Preserve the existing sweep's smallest-margin 1e-14 comparison convention.
    best, index = math.inf, 0
    for i, value in enumerate(values):
        if value < best - 1e-14:
            best, index = float(value), i
    return index


def optimize_scalar(states, scalar, cw):
    values = cost_vector(states, scalar, cw)
    index = first_minimum(values)
    return float(states[index, 0]), float(values[index]), index


def optimize_regret(states, low, upper, cw):
    if not np.isfinite(upper) or upper < low:
        raise ValueError('Endpoint regret requires a specified finite interval.')
    a, b = cost_vector(states, low, cw), cost_vector(states, upper, cw)
    regret = np.maximum(a - a.min(), b - b.min())
    index = first_minimum(regret)
    return float(states[index, 0]), float(regret[index]), index


def stability_upper(states, index, low, cw):
    """Largest upper endpoint over which the low-end minimizing state stays optimal.

    Larger-lambda competitors have at least as large slopes and cannot displace
    the selected state as total lifetime increases. A smaller-lambda competitor
    displaces it at its affine intersection, unless all slopes coincide.
    """
    intercept = 100.0 * states[:, 1] / states[:, 2] - 100.0 * cw
    slope = 100.0 * cw / states[:, 2]
    difference = slope[index] - slope
    competitors = difference > 1e-15
    if not competitors.any():
        return math.inf
    upper = float(np.min((intercept[competitors] - intercept[index]) / difference[competitors]))
    assert upper >= low - 1e-7
    return max(float(low), upper)


def paired_bootstrap(units, cfg):
    primary = units[units.seed == cfg['primary_seed']]
    rows = []
    for (head, scenario), group in primary.groupby(['head', 'scenario']):
        methods = list(group.method.unique())
        rng = np.random.default_rng(cfg['bootstrap_seed'])
        numerators = {m: np.zeros(cfg['bootstrap_replicates']) for m in methods}
        denominators = {m: np.zeros(cfg['bootstrap_replicates']) for m in methods}
        failures = {m: np.zeros(cfg['bootstrap_replicates']) for m in methods}
        for _, subset in group.groupby('dataset', sort=True):
            order = sorted(subset.unit_id.unique())
            draws = rng.integers(0, len(order), (cfg['bootstrap_replicates'], len(order)))
            for m in methods:
                data = subset[subset.method == m].set_index('unit_id').loc[order]
                numerators[m] += data.total_cost.to_numpy()[draws].sum(axis=1)
                denominators[m] += data.service_cycles.to_numpy()[draws].sum(axis=1)
                failures[m] += data.fail.to_numpy()[draws].sum(axis=1)
        totals = int(group[group.method == methods[0]].shape[0])
        for treatment in ['Censored_lifecycle_low', 'Censored_lifecycle_minimax']:
            for comparator in ['Event_sweep_existing', 'Retirement_floor', 'Historical', 'Complete_lifecycle_same_ray']:
                tv, cv = group[group.method == treatment], group[group.method == comparator]
                delta = 100 * tv.total_cost.sum() / tv.service_cycles.sum() - 100 * cv.total_cost.sum() / cv.service_cycles.sum()
                d = 100 * numerators[treatment] / denominators[treatment] - 100 * numerators[comparator] / denominators[comparator]
                fd = 100 * (failures[treatment] - failures[comparator]) / totals
                rows.append(dict(head=head, scenario=scenario, treatment=treatment, comparator=comparator,
                                 delta_j_life=delta, ci95_jl_low=np.quantile(d, .025), ci95_jl_high=np.quantile(d, .975),
                                 delta_fail_pct=100 * (tv.fail.mean() - cv.fail.mean()),
                                 ci95_fail_low=np.quantile(fd, .025), ci95_fail_high=np.quantile(fd, .975)))
    return pd.DataFrame(rows)


def pooled_summary(units, keys):
    rows = []
    for values, group in units.groupby(keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        rows.append(dict(zip(keys, values), **rpb.summary(group)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE / 'Experiments' / '公开基准' / 'results_censored_lifecycle')
    parser.add_argument('--primary-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cfg = rpb.CONFIG
    core = rpb.load_core(BASE / 'Experiments' / '代码' / 'decision_calibrated_pdm_suite.py')
    rows, unit_rows, states_rows, path_rows = [], [], [], []
    max_scalar_error = 0.0
    max_service_error = 0.0
    max_event_error = 0
    monotonic_violations = 0
    endpoint_violations = 0
    for dataset in cfg['datasets']:
        frame = rpb.load_dataset(BASE / 'Experiments' / '公开基准' / 'raw' / f'train_{dataset}.txt')
        for seed in [cfg['primary_seed']] + ([] if args.primary_only else cfg['sensitivity_seeds']):
            parts, _ = rpb.split_frame(frame, seed)
            encoder = core.CausalMonotoneEncoder(True, True).fit(parts['fit'])
            features = {key: encoder.transform(part)[0] for key, part in parts.items()}
            development = pd.concat([parts['fit'], parts['tune']])
            dev_lives = development.groupby('unit_id').cycle.max()
            dev_max, dev_mean = float(dev_lives.max()), float(dev_lives.mean())
            for head, kind, focus in [('Ridge', 'ridge', False), ('Linex', 'linex', True)]:
                model = core.PrognosticHead().fit(features['fit'], parts['fit'].true_rul.to_numpy(), kind=kind,
                                                  reg=cfg['regularization'], focus_window=focus)
                pred = {key: model.predict(value, parts[key], causal_smooth=head == 'Linex') for key, value in features.items()}
                if head == 'Linex':
                    bias = float(np.mean(pred['tune'] - parts['tune'].true_rul.to_numpy()))
                    pred = {key: np.clip(value - .85 * bias, 0, cfg['rul_cap']) for key, value in pred.items()}
                sd = max(1e-6, float(np.std(pred['tune'] - parts['tune'].true_rul.to_numpy())))
                scale = {key: sd * (.3 + .7 * expit((37 - value) / 9)) if head == 'Linex'
                         else np.full(len(value), sd) for key, value in pred.items()}
                seq = {key: rpb.unit_sequences(parts[key], pred[key], scale[key]) for key in parts}
                scores = np.array([np.min(((u['point'] - cfg['lead_time']) / u['scale'])[
                    u['cycles'] + cfg['lead_time'] < u['T']]) for u in seq['tune']])
                historical_erm, _, _ = rpb.exact_margin_frontier(seq['tune'])
                q_oracle, _, rank = rpb.calibration_threshold(seq['cal'])
                full_lam, full_jl, _ = rpb.exact_margin_frontier(seq['cal'], q_oracle)
                true_scalar = float(sum(u['T'] for u in seq['cal']))
                histories = [('Tune_Q70', float(np.quantile(scores, .70))),
                             ('Tune_Q85', float(np.quantile(scores, .85))), ('Tune_ERM', float(historical_erm))]
                for scenario, h in histories:
                    records = [ro.observe(u, h) for u in seq['cal']]
                    q_ret = max(0.0, sorted(r['score'] for r in records)[rank - 1])
                    lower = max(0.0, h, q_ret)
                    states = observable_states(records, lower)
                    cw = cfg['waste_cost']
                    low = float(sum(r['age'] + 1 - r['failed'] for r in records))
                    upper = float(sum(r['age'] if r['failed'] else max(r['age'] + 1, dev_max) for r in records))
                    mean = float(sum(r['age'] if r['failed'] else max(r['age'] + 1, dev_mean) for r in records))
                    lam_true, val_true, _ = optimize_scalar(states, true_scalar, cw)
                    lam_event, _, _ = optimize_scalar(states, 0.0, 0.0)
                    lam_low, _, low_index = optimize_scalar(states, low, cw)
                    lam_up, _, _ = optimize_scalar(states, upper, cw)
                    lam_mm, regret_bound, mm_index = optimize_regret(states, low, upper, cw)
                    lam_mean, _, _ = optimize_scalar(states, mean, cw)
                    ray_lam, ray_jl, _ = rpb.exact_margin_frontier(seq['cal'], lower)
                    assert abs(lam_true - ray_lam) < 1e-10 and abs(val_true - ray_jl) < 1e-10
                    stability = stability_upper(states, low_index, low, cw)
                    row = dict(dataset=dataset, seed=seed, head=head, scenario=scenario, n_cal=len(records),
                               n_censored=sum(1 - r['failed'] for r in records), h=h, q_oracle=q_oracle,
                               q_ret=q_ret, L_R=lower, S_T_low=low, S_T_true=true_scalar,
                               S_T_mean=mean, S_T_up=upper, oracle_full_lam=full_lam, oracle_full_jl=full_jl,
                               oracle_ray_lam=ray_lam, oracle_ray_jl=ray_jl, lam_st_true=lam_true,
                               lam_je=lam_event, lam_jl_low=lam_low, lam_jl_up=lam_up,
                               lam_jl_mm=lam_mm, lam_jl_mean=lam_mean, max_regret_bound=regret_bound,
                               invariant_interval=abs(lam_low - lam_up) < 1e-10,
                               stability_upper=stability, unbounded_invariance=math.isinf(stability),
                               stress_interval_contains_true=low <= true_scalar <= upper, candidate_states=len(states))
                    for suffix, lam in [('je', lam_event), ('low', lam_low), ('up', lam_up), ('mm', lam_mm), ('mean', lam_mean)]:
                        value = float(rpb.summary(rpb.at_margin(seq['cal'], lam))['j_life'])
                        row['cal_jl_' + suffix] = value
                        row[suffix + '_matches_ray'] = abs(value - ray_jl) < 1e-10
                    row['low_matches_full'] = abs(row['cal_jl_low'] - full_jl) < 1e-10
                    rows.append(row)
                    # Every candidate is independently replayed by complete histories.
                    for lam, csum, usum in states:
                        complete = rpb.at_margin(seq['cal'], lam)
                        direct = float(rpb.summary(complete)['j_life'])
                        error = abs(100 * (csum + cw * true_scalar) / usum - 100 * cw - direct)
                        max_scalar_error = max(max_scalar_error, error)
                        oc, ou, of = ro.observed_replay(records, lam)
                        max_service_error = max(max_service_error, float(np.max(np.abs(ou - complete.service_cycles))))
                        max_event_error = max(max_event_error, int(np.sum(of != complete.fail)))
                        states_rows.append(dict(dataset=dataset, seed=seed, head=head, scenario=scenario,
                                                margin=lam, event_cost_sum=csum, service_sum=usum,
                                                full_lifecycle_cost=direct, scalar_error=error))
                    previous = math.inf
                    for scalar in np.linspace(low, upper, 33):
                        lam, optimum, _ = optimize_scalar(states, scalar, cw)
                        monotonic_violations += int(lam > previous + 1e-10)
                        previous = lam
                        regret = float(cost_vector(states, scalar, cw)[mm_index] - optimum)
                        endpoint_violations += int(regret > regret_bound + 1e-10)
                        path_rows.append(dict(dataset=dataset, seed=seed, head=head, scenario=scenario,
                                              scalar=scalar, selected_margin=lam, optimum=optimum, mm_regret=regret))
                    policies = [('Historical', h), ('Retirement_floor', lower), ('Event_sweep_existing', lam_event),
                                ('Censored_lifecycle_low', lam_low), ('Censored_lifecycle_minimax', lam_mm),
                                ('Censored_lifecycle_mean', lam_mean), ('Complete_lifecycle_same_ray', ray_lam),
                                ('Complete_lifecycle_full_pool_cal', full_lam)]
                    for method, lam in policies:
                        g = rpb.at_margin(seq['test'], lam)
                        for _, unit in g.iterrows():
                            unit_rows.append(dict(dataset=dataset, seed=seed, head=head, scenario=scenario,
                                                  method=method, margin=lam, unit_id=int(unit.unit_id), T_u=float(unit.T_u),
                                                  fail=int(unit.fail), service_cycles=float(unit.service_cycles),
                                                  waste_cycles=float(unit.waste_cycles), event_cost=float(1 + 9 * unit.fail),
                                                  total_cost=float(unit.total_cost)))
            print(f'{dataset} seed={seed} completed', flush=True)
    configurations, units = pd.DataFrame(rows), pd.DataFrame(unit_rows)
    configurations.to_csv(args.out / 'lifecycle_identification.csv', index=False)
    units.to_csv(args.out / 'lifecycle_per_unit.csv', index=False)
    pd.DataFrame(states_rows).to_csv(args.out / 'lifecycle_candidate_states.csv', index=False)
    pd.DataFrame(path_rows).to_csv(args.out / 'lifecycle_scalar_paths.csv', index=False)
    pooled = pooled_summary(units, ['seed', 'head', 'scenario', 'method'])
    pooled.to_csv(args.out / 'lifecycle_pooled_by_seed.csv', index=False)
    pooled[pooled.seed == cfg['primary_seed']].to_csv(args.out / 'lifecycle_primary.csv', index=False)
    paired_bootstrap(units, cfg).to_csv(args.out / 'lifecycle_paired_bootstrap.csv', index=False)
    assert max_scalar_error < 1e-10 and max_service_error == 0 and max_event_error == 0
    assert monotonic_violations == 0 and endpoint_violations == 0
    audit = dict(passed=True, elapsed_seconds=time.perf_counter() - start,
                 configurations=len(rows), candidate_states=len(states_rows), scalar_path_evaluations=len(path_rows),
                 max_scalar_identity_error=max_scalar_error, max_service_error=max_service_error,
                 max_event_error=max_event_error, monotonic_violations=monotonic_violations,
                 endpoint_regret_violations=endpoint_violations,
                 event_matches_ray=int(configurations.je_matches_ray.sum()),
                 low_matches_ray=int(configurations.low_matches_ray.sum()),
                 minimax_matches_ray=int(configurations.mm_matches_ray.sum()),
                 mean_matches_ray=int(configurations['mean_matches_ray'].sum()),
                 low_matches_full=int(configurations.low_matches_full.sum()),
                 finite_interval_invariance=int(configurations.invariant_interval.sum()),
                 unbounded_invariance=int(configurations.unbounded_invariance.sum()),
                 stress_interval_coverage=int(configurations.stress_interval_contains_true.sum()),
                 environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__),
                 protocol=dict(development_stress_endpoint='maximum Fit+Tune lifetime, empirical scenario endpoint',
                               lifecycle_extension='retrospective evaluation after the original primary replay',
                               bootstraps=cfg['bootstrap_replicates'], bootstrap_seed=cfg['bootstrap_seed']))
    audit['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.glob('*.csv'))}
    (args.out / 'lifecycle_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
