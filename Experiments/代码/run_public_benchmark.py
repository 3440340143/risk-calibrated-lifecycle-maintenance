#!/usr/bin/env python3
"""Independent C-MAPSS replay with a fixed primary holdout and exact margin sweeps.

C-MAPSS is a NASA simulator benchmark, not a physical fleet measurement dataset.
Only complete official train trajectories are used; truncated official test
records cannot supply an unobserved future stopping sequence. Splits are by engine.
The old synthetic experimental core is imported without modifying its bytes.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import time
import urllib.request
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from threadpoolctl import threadpool_limits


CONFIG = {
    'datasets': ['FD001', 'FD002', 'FD003', 'FD004'],
    'primary_seed': 20261023,
    'sensitivity_seeds': [11, 22, 33, 44, 55],
    'split_fractions_fit_tune_cal_test': [0.30, 0.10, 0.20, 0.40],
    'selected_original_sensor_numbers': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
    'lead_time': 15.0, 'epsilon': 0.05, 'rho': 10.0, 'waste_cost': 0.045,
    'rul_cap': 125.0, 'regularization': 14.0,
    'bootstrap_replicates': 4000, 'bootstrap_seed': 471902,
    'margin_search': 'All strict prefix-record crossing values, no fixed upper limit; exact empirical lifecycle ratio',
    'primary_estimand': 'Pooled lifecycle ratio and paired difference on one untouched primary holdout; conditional stratified unit bootstrap',
    'repeat_interpretation': 'Five additional split seeds are descriptive robustness checks; do not pool overlapping units as independent trials',
    'tuning_labels': 'Tune-only ERM vs pure Cal-only CP vs Tune+Cal floor; equal-budget ERM gets Tune+Cal labels',
    'archive_url': 'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip',
    'official_source': 'https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/',
    'official_catalog': 'https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data',
    'core_sha256': '1392a8f83cc277d96d2fddfb436883b987385454acea60b357f6d87d61790300',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def load_core(path):
    assert sha(path) == CONFIG['core_sha256'], 'The reference numerical core changed.'
    spec = importlib.util.spec_from_file_location('dc_pdm_reference_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core._C_PAVA_LIB = False
    return core


def extract_raw(raw, download=False):
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'CMAPSS_NASA_archive.zip'
    if download and not archive.exists():
        urllib.request.urlretrieve(CONFIG['archive_url'], archive)
    if archive.exists():
        with zipfile.ZipFile(archive) as handle:
            assert handle.testzip() is None
            archives = [handle]
            for info in handle.infolist():
                if Path(info.filename).name.lower() == 'cmapssdata.zip':
                    archives.append(zipfile.ZipFile(io.BytesIO(handle.read(info))))
            for nested in archives:
                for info in nested.infolist():
                    name = Path(info.filename).name
                    if name in [f'train_{d}.txt' for d in CONFIG['datasets']] or name.lower() == 'readme.txt':
                        (raw / name).write_bytes(nested.read(info))
    paths = [raw / f'train_{d}.txt' for d in CONFIG['datasets']]
    assert all(p.is_file() for p in paths), 'Supply the four official train files, or use --download.'
    return {'archive_sha256': sha(archive) if archive.exists() else None,
            'archive_bytes': archive.stat().st_size if archive.exists() else None,
            'raw_files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths},
            'official_source': CONFIG['official_source'], 'download_url': CONFIG['archive_url']}


def load_dataset(path):
    array = np.loadtxt(path)
    assert array.shape[1] == 26 and np.isfinite(array).all()
    frame = pd.DataFrame({'unit_id': array[:, 0].astype(int), 'cycle': array[:, 1].astype(int),
                          'regime': np.rint(array[:, 2]).astype(int)})
    for j, sensor_number in enumerate(CONFIG['selected_original_sensor_numbers'], 1):
        frame[f's{j}'] = array[:, 4 + sensor_number]
    frame['max_cycle'] = frame.groupby('unit_id').cycle.transform('max')
    frame['true_rul_raw'] = frame.max_cycle - frame.cycle
    frame['true_rul'] = np.minimum(CONFIG['rul_cap'], frame.true_rul_raw)
    for _, group in frame.groupby('unit_id', sort=False):
        assert np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1))
    assert frame.max_cycle.min() > CONFIG['lead_time'] + 1
    return frame


def split_frame(frame, seed):
    units = np.sort(frame.unit_id.unique())
    shuffled = np.random.default_rng(seed).permutation(units)
    n = len(units)
    sizes = [int(math.floor(f * n)) for f in CONFIG['split_fractions_fit_tune_cal_test'][:3]]
    sizes.append(n - sum(sizes))
    parts, records, offset = {}, [], 0
    for name, size in zip(['fit', 'tune', 'cal', 'test'], sizes):
        selected = shuffled[offset:offset + size]
        offset += size
        parts[name] = frame[frame.unit_id.isin(selected)].copy().reset_index(drop=True)
        records.extend({'unit_id': int(u), 'phase': name} for u in sorted(selected))
    assert sum(len(x) for x in parts.values()) == len(frame)
    assert len(parts['cal'].unit_id.unique()) >= 19
    return parts, records


def unit_sequences(frame, point, scale):
    return [dict(unit_id=int(uid), T=float(frame.iloc[pos].max_cycle.iloc[0]),
                 cycles=frame.iloc[pos].cycle.to_numpy(dtype=float), point=point[pos], scale=scale[pos])
            for uid, pos in frame.groupby('unit_id', sort=False).indices.items()]


def at_margin(sequences, margin):
    rows = []
    ell, rho, waste = CONFIG['lead_time'], CONFIG['rho'], CONFIG['waste_cost']
    for unit in sequences:
        if margin is None:
            tau = 0.0
        else:
            # Compare crossings directly to ensure consistent floating-point equality.
            crossing = (unit['point'] - ell) / unit['scale']
            eligible = np.flatnonzero(crossing <= float(margin))
            tau = float(unit['cycles'][eligible[0]]) if len(eligible) else np.inf
        fail = int(tau + ell >= unit['T'])
        life = min(unit['T'], tau + ell)
        unused = max(0.0, unit['T'] - (tau + ell)) if not fail else 0.0
        event_cost = 1.0 + (rho - 1.0) * fail
        rows.append({'unit_id': unit['unit_id'], 'T_u': unit['T'],
                     'tau': tau if np.isfinite(tau) else -1.0, 'fail': fail,
                     'service_cycles': life, 'waste_cycles': unused,
                     'event_cost': event_cost, 'total_cost': event_cost + waste * unused})
    return pd.DataFrame(rows)


def summary(units):
    n, fails = len(units), int(units.fail.sum())
    upper = float(stats.beta.ppf(.95, fails + 1, n - fails)) if fails < n else 1.0
    return dict(n_test=n, failures=fails, fail_pct=100.0 * fails / n,
                descriptive_cp_ucb95_pct=100.0 * upper,
                j_life=100.0 * units.total_cost.sum() / units.service_cycles.sum(),
                j_event=100.0 * units.event_cost.sum() / units.service_cycles.sum(),
                utilization_pct=100.0 * units.service_cycles.sum() / units.T_u.sum(),
                waste_cycles=units.waste_cycles.mean())


def exact_margin_frontier(sequences, lower=0.0):
    """Global empirical ratio optimum; O(M + K log K), K prefix-record crossings.

    Earlier time indices dominate later crossings at equal thresholds. Group tied
    events before evaluating the ratio. The bound q can be data dependent; only
    the pathwise inequality margin >= q is needed for the static risk bound.
    """
    if lower is None:
        units = at_margin(sequences, None)
        return None, summary(units)['j_life'], []
    lower = float(max(0.0, lower))
    units = at_margin(sequences, lower)
    costs = units.total_cost.to_numpy(copy=True)
    life = units.service_cycles.to_numpy(copy=True)
    events = []
    for index, unit in enumerate(sequences):
        record = np.inf
        for t, crossing in zip(unit['cycles'], (unit['point'] - CONFIG['lead_time']) / unit['scale']):
            value = float(crossing)
            if value < record:
                record = value
                if value > lower:
                    events.append((value, index, float(t)))
    events.sort()
    total_cost, total_life = float(costs.sum()), float(life.sum())
    best_margin, best_cost = lower, 100.0 * total_cost / total_life
    frontier = [{'margin': lower, 'j_life': best_cost}]
    offset = 0
    while offset < len(events):
        value = events[offset][0]
        while offset < len(events) and events[offset][0] == value:
            _, index, tau = events[offset]
            unit = sequences[index]
            fail = int(tau + CONFIG['lead_time'] >= unit['T'])
            new_life = min(unit['T'], tau + CONFIG['lead_time'])
            unused = max(0.0, unit['T'] - (tau + CONFIG['lead_time'])) if not fail else 0.0
            new_cost = 1.0 + (CONFIG['rho'] - 1.0) * fail + CONFIG['waste_cost'] * unused
            total_cost += new_cost - costs[index]
            total_life += new_life - life[index]
            costs[index], life[index] = new_cost, new_life
            offset += 1
        objective = 100.0 * total_cost / total_life
        frontier.append({'margin': value, 'j_life': objective})
        if objective < best_cost - 1e-12:
            best_margin, best_cost = value, objective
    checked = summary(at_margin(sequences, best_margin))['j_life']
    assert abs(checked - best_cost) < 1e-9, (checked, best_cost)
    return float(best_margin), float(checked), frontier


def calibration_threshold(sequences):
    scores = [float(np.min(((u['point'] - CONFIG['lead_time']) / u['scale'])[
        u['cycles'] + CONFIG['lead_time'] < u['T']])) for u in sequences]
    rank = int(math.ceil((len(scores) + 1) * (1 - CONFIG['epsilon'])))
    return (max(0.0, sorted(scores)[rank - 1]) if rank <= len(scores) else None), scores, rank


def solver_tests():
    rng = np.random.default_rng(39122)
    largest = 0.0
    cases = 0
    for _ in range(24):
        seqs = [dict(unit_id=j, T=float(rng.integers(28, 50)), cycles=np.arange(1., 26.),
                     point=rng.uniform(5, 60, 25), scale=rng.uniform(.3, 7, 25)) for j in range(4)]
        # Deliberately create ties, nonmonotone predictions, negative crossings.
        seqs[0]['point'][2:5] = 20.0
        seqs[0]['scale'][2:5] = 2.0
        for bound in (0.0, 1.125, 8.0):
            margin, objective, _ = exact_margin_frontier(seqs, bound)
            candidates = sorted(set([bound] + [float(v) for s in seqs for v in
                                (s['point'] - CONFIG['lead_time']) / s['scale'] if v >= bound]))
            brute = min(summary(at_margin(seqs, c))['j_life'] for c in candidates)
            largest = max(largest, abs(objective - brute))
            assert margin >= bound and abs(objective - brute) < 1e-9
            cases += 1
    return {'exact_frontier_vs_brute_force_cases': cases, 'max_objective_difference': largest}


def paired_bootstrap(units, output):
    primary = units[units.seed == CONFIG['primary_seed']]
    methods = primary.method.unique().tolist()
    by_dataset = {}
    for dataset, group in primary.groupby('dataset', sort=True):
        order = sorted(group.unit_id.unique())
        by_dataset[dataset] = {m: group[group.method == m].set_index('unit_id').loc[order] for m in methods}
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    ratios = {m: np.zeros(CONFIG['bootstrap_replicates']) for m in methods}
    numerator = {m: np.zeros(CONFIG['bootstrap_replicates']) for m in methods}
    denominator = {m: np.zeros(CONFIG['bootstrap_replicates']) for m in methods}
    for dataset, records in by_dataset.items():
        count = len(next(iter(records.values())))
        draws = rng.integers(0, count, (CONFIG['bootstrap_replicates'], count))
        for method in methods:
            numerator[method] += records[method].total_cost.to_numpy()[draws].sum(axis=1)
            denominator[method] += records[method].service_cycles.to_numpy()[draws].sum(axis=1)
    for method in methods:
        ratios[method] = 100.0 * numerator[method] / denominator[method]
    comparisons = [('Ridge_Floor', 'Ridge_ERM_Tune'), ('Ridge_Floor', 'Ridge_ERM_EqualBudget'),
                   ('Ridge_Constrained', 'Ridge_Floor'), ('Ridge_Constrained', 'Ridge_ERM_EqualBudget'),
                   ('Linex_Floor', 'Linex_ERM_Tune'), ('Linex_Floor', 'Linex_ERM_EqualBudget'),
                   ('Linex_Constrained', 'Linex_Floor'), ('Linex_Constrained', 'Linex_ERM_EqualBudget'),
                   ('Linex_Constrained', 'Ridge_Constrained'), ('Ridge_UnitCP', 'Cycle_CP')]
    rows = []
    for treatment, comparator in comparisons:
        diff = ratios[treatment] - ratios[comparator]
        def point(method):
            g = primary[primary.method == method]
            return 100.0 * g.total_cost.sum() / g.service_cycles.sum()
        lo, hi = np.quantile(diff, [.025, .975])
        rows.append({'treatment': treatment, 'comparator': comparator,
                     'delta_j_life': point(treatment) - point(comparator),
                     'conditional_bootstrap_ci95_low': lo, 'conditional_bootstrap_ci95_high': hi,
                     'bootstrap_replicates': CONFIG['bootstrap_replicates'],
                     'interpretation': 'Stratified paired unit bootstrap conditional on fixed primary fitted/calibrated rules; not a retraining or simultaneous interval'})
    pd.DataFrame(rows).to_csv(output / 'cmapss_paired_uncertainty.csv', index=False, lineterminator='\n')


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Use a fresh empty output directory.'
    # This protocol is written before any outcomes are calculated. It is not an external preregistration.
    write_json(out / 'cmapss_protocol.json', CONFIG)
    core = load_core(args.core.resolve())
    provenance = extract_raw(args.raw.resolve(), args.download)
    write_json(out / 'cmapss_source_provenance.json', provenance)
    tests = solver_tests()
    seed_rows, all_units, split_rows, calibration_rows, frontier_rows, equivalence_rows = [], [], [], [], [], []
    forecast_rows, parameters = [], {}
    start = time.perf_counter()
    seeds = [CONFIG['primary_seed']] + ([] if args.primary_only else CONFIG['sensitivity_seeds'])
    for dataset in CONFIG['datasets']:
        frame = load_dataset(args.raw / f'train_{dataset}.txt')
        # FD004's bundled README swaps train/test counts; the actual train file
        # contains engine IDs 1..249. Do not discard engine 249 to match the typo.
        expected = {'FD001': 100, 'FD002': 260, 'FD003': 100, 'FD004': 249}[dataset]
        assert frame.unit_id.nunique() == expected
        for seed in seeds:
            print(f'[{dataset} seed {seed}] fit/tune/cal/test', flush=True)
            parts, splits = split_frame(frame, seed)
            split_rows.extend(dict(dataset=dataset, seed=seed, **row) for row in splits)
            enc = core.CausalMonotoneEncoder(True, True).fit(parts['fit'])
            features = {key: enc.transform(df)[0] for key, df in parts.items()}
            if seed == CONFIG['primary_seed']:
                u = parts['test'].unit_id.iloc[0]
                complete = parts['test'][parts['test'].unit_id == u].copy()
                prefix = complete.iloc[:max(1, len(complete) // 2)].copy()
                altered = complete.copy()
                altered.loc[altered.index[len(prefix):], core.SENSORS] += 173.0
                x0, x1, x2 = enc.transform(complete)[0], enc.transform(prefix)[0], enc.transform(altered)[0]
                error = max(float(np.max(np.abs(x0[:len(prefix)] - x1))),
                            float(np.max(np.abs(x0[:len(prefix)] - x2[:len(prefix)]))))
                assert error < 1e-12
                tests[f'{dataset}_prefix_future_perturbation_max_feature_error'] = error
            heads = {}
            for head_name, kind, focus in [('Ridge', 'ridge', False), ('Linex', 'linex', True)]:
                head = core.PrognosticHead().fit(features['fit'], parts['fit'].true_rul.to_numpy(),
                                                kind=kind, reg=CONFIG['regularization'], focus_window=focus)
                predictions = {key: head.predict(x, parts[key], causal_smooth=head_name == 'Linex')
                               for key, x in features.items()}
                if head_name == 'Linex':
                    bias = float(np.mean(predictions['tune'] - parts['tune'].true_rul.to_numpy()))
                    predictions = {key: np.clip(p - .85 * bias, 0, CONFIG['rul_cap']) for key, p in predictions.items()}
                else:
                    bias = 0.0
                sd = max(1e-6, float(np.std(predictions['tune'] - parts['tune'].true_rul.to_numpy())))
                scales = {key: (sd * (.30 + .70 * expit((37 - p) / 9)) if head_name == 'Linex'
                                else np.full(len(p), sd)) for key, p in predictions.items()}
                sequences = {key: unit_sequences(parts[key], predictions[key], scales[key]) for key in parts}
                q, scores, rank = calibration_threshold(sequences['cal'])
                tune_margin, tune_cost, curve = exact_margin_frontier(sequences['tune'])
                equal_margin, _, _ = exact_margin_frontier(sequences['tune'] + sequences['cal'])
                floor = None if q is None else max(q, tune_margin)
                constrained, constrained_cost, constrained_curve = exact_margin_frontier(sequences['tune'], q)
                max_floor_cost = summary(at_margin(sequences['tune'], floor))['j_life']
                assert constrained_cost <= max_floor_cost + 1e-9
                if seed == CONFIG['primary_seed']:
                    for typ, values in [('unconstrained', curve), ('constrained', constrained_curve)]:
                        frontier_rows.extend(dict(dataset=dataset, head=head_name, constraint=typ, **row) for row in values)
                    for unit, score in zip(sequences['cal'], scores):
                        calibration_rows.append(dict(dataset=dataset, head=head_name, unit_id=unit['unit_id'],
                            critical_score=score, rank=rank, n_cal=len(scores), q=q))
                    # Independent CRC evaluation at each candidate, using binary whole-unit loss.
                    crc_candidates = sorted(set(scores))
                    feasible = [v for v in crc_candidates if
                        (sum(s > v for s in scores) + 1) / (len(scores) + 1) <= CONFIG['epsilon'] + 1e-15]
                    crc = max(0.0, feasible[0]) if feasible else None
                    assert q == crc
                    equivalence_rows.append(dict(dataset=dataset, head=head_name, q=q, direct_binary_crc=crc,
                                                  absolute_difference=abs(q - crc)))
                    parameters[f'{dataset}_{head_name}'] = dict(
                        head_mu=head.mu.tolist(), head_sd=head.sd.tolist(), head_w=head.w.tolist(),
                        tune_bias=bias, residual_sd=sd, q=q, tune_margin=tune_margin,
                        equal_budget_margin=equal_margin, constrained_margin=constrained,
                        encoder_global_mu=enc.global_mu.tolist(), encoder_global_sd=enc.global_sd.tolist(),
                        encoder_polarities=enc.polarities.tolist(),
                        encoder_norms={str(k): [a.tolist(), b.tolist()] for k, (a, b) in enc.norms.items()})
                    for phase in ['tune', 'cal', 'test']:
                        piece = parts[phase][['unit_id', 'cycle', 'max_cycle', 'true_rul']].copy()
                        piece.insert(0, 'dataset', dataset)
                        piece.insert(1, 'head', head_name)
                        piece.insert(2, 'phase', phase)
                        piece['point'] = predictions[phase]
                        piece['scale'] = scales[phase]
                        forecast_rows.append(piece)
                policies = [('ERM_Tune', tune_margin), ('UnitCP', q), ('Floor', floor),
                            ('Constrained', constrained), ('ERM_EqualBudget', equal_margin)]
                if head_name == 'Ridge':
                    residuals = predictions['cal'] - parts['cal'].true_rul.to_numpy()
                    k = math.ceil((len(residuals) + 1) * (1 - CONFIG['epsilon']))
                    residual_q = max(0.0, np.sort(residuals)[min(k - 1, len(residuals) - 1)])
                    policies.append(('Cycle_CP', residual_q / sd))
                for policy, margin in policies:
                    name = 'Cycle_CP' if policy == 'Cycle_CP' else f'{head_name}_{policy}'
                    units = at_margin(sequences['test'], margin)
                    for column, value in [('dataset', dataset), ('seed', seed), ('method', name)]:
                        units.insert(0, column, value)
                    all_units.append(units)
                    rmse = np.sqrt(np.mean([np.mean((predictions['test'][pos] - parts['test'].iloc[pos].true_rul.to_numpy()) ** 2)
                                        for pos in parts['test'].groupby('unit_id', sort=False).indices.values()]))
                    seed_rows.append(dict(dataset=dataset, seed=seed, method=name, margin=margin,
                        rmse_unit_balanced_capped=rmse, n_fit=parts['fit'].unit_id.nunique(),
                        n_tune=parts['tune'].unit_id.nunique(), n_cal=parts['cal'].unit_id.nunique(),
                        tune_optimal_cost=tune_cost, constrained_tune_cost=constrained_cost,
                        max_floor_tune_cost=max_floor_cost, **summary(units)))
                heads[head_name] = (head, predictions, scales)
    per_split = pd.DataFrame(seed_rows)
    units = pd.concat(all_units, ignore_index=True)
    per_split.to_csv(out / 'cmapss_per_split.csv', index=False, lineterminator='\n')
    units.to_csv(out / 'cmapss_per_unit.csv', index=False, lineterminator='\n')
    pd.DataFrame(split_rows).to_csv(out / 'cmapss_unit_splits.csv', index=False, lineterminator='\n')
    pd.DataFrame(calibration_rows).to_csv(out / 'cmapss_calibration_scores.csv', index=False, lineterminator='\n')
    pd.DataFrame(frontier_rows).to_csv(out / 'cmapss_exact_frontiers.csv', index=False, lineterminator='\n')
    pd.DataFrame(equivalence_rows).to_csv(out / 'cmapss_crc_equivalence.csv', index=False, lineterminator='\n')
    pd.concat(forecast_rows, ignore_index=True).to_csv(out / 'cmapss_primary_forecasts.csv', index=False, lineterminator='\n')
    write_json(out / 'cmapss_fitted_parameters.json', parameters)
    primary = per_split[per_split.seed == CONFIG['primary_seed']]
    primary.to_csv(out / 'cmapss_primary_summary.csv', index=False, lineterminator='\n')
    paired_bootstrap(units, out)
    tests['unit_score_vs_direct_binary_crc_max_difference'] = max(r['absolute_difference'] for r in equivalence_rows)
    tests['all_constrained_tune_costs_no_higher_than_max_floor'] = bool((per_split.constrained_tune_cost <= per_split.max_floor_tune_cost + 1e-9).all())
    write_json(out / 'cmapss_self_tests.json', tests)
    write_json(out / 'cmapss_runtime.json', dict(python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, elapsed_seconds=time.perf_counter() - start,
        primary_unique_test_units=int(units[units.seed == CONFIG['primary_seed']].drop_duplicates(['dataset', 'unit_id']).shape[0]),
        split_seeds=seeds, methods=int(per_split.method.nunique()), no_original_results_modified=True))
    print(primary[['dataset', 'method', 'j_life', 'failures', 'n_test']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--primary-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
