#!/usr/bin/env python3
"""Independently re-aggregate C-MAPSS results and check frozen-prefix inference."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import hashlib
import math
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: pd.read_csv(args.results / name, float_precision='round_trip')
    protocol = json.loads((args.results / 'cmapss_protocol.json').read_text(encoding='utf-8'))
    provenance = json.loads((args.results / 'cmapss_source_provenance.json').read_text(encoding='utf-8'))
    for name, record in provenance['raw_files'].items():
        file = args.raw / name
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256']
    assert hashlib.sha256(args.core.read_bytes()).hexdigest() == protocol['core_sha256']
    units, split, summaries = read('cmapss_per_unit.csv'), read('cmapss_unit_splits.csv'), read('cmapss_per_split.csv')
    assert not split.duplicated(['dataset', 'seed', 'unit_id']).any()
    assert split.groupby(['dataset', 'seed']).size().to_dict() == {
        (dataset, seed): count for dataset, count in [('FD001', 100), ('FD002', 260), ('FD003', 100), ('FD004', 249)]
        for seed in summaries.seed.unique()}
    assert len(summaries.method.unique()) == 11
    event_errors, cost_error, rate_error = 0, 0.0, 0.0
    for (dataset, seed, method), group in units.groupby(['dataset', 'seed', 'method']):
        expected_test = split[(split.dataset == dataset) & (split.seed == seed) & (split.phase == 'test')]
        assert set(expected_test.unit_id) == set(group.unit_id)
        tau = np.where(group.tau.to_numpy() < 0, np.inf, group.tau)
        fail = (tau + protocol['lead_time'] >= group.T_u).astype(int)
        life = np.minimum(group.T_u, tau + protocol['lead_time'])
        waste = np.maximum(0, group.T_u - (tau + protocol['lead_time']))
        cost = 1 + (protocol['rho'] - 1) * fail + protocol['waste_cost'] * waste
        event_errors += int(np.sum(fail != group.fail))
        assert np.max(np.abs(life - group.service_cycles)) < 1e-12
        assert np.max(np.abs(waste - group.waste_cycles)) < 1e-12
        cost_error = max(cost_error, float(np.max(np.abs(cost - group.total_cost))))
        record = summaries[(summaries.dataset == dataset) & (summaries.seed == seed) & (summaries.method == method)].iloc[0]
        rate_error = max(rate_error, abs(float(100 * np.sum(cost) / np.sum(life)) - record.j_life))
        assert record.failures == int(fail.sum()) and record.n_test == len(group)
    assert event_errors == 0 and cost_error < 1e-12 and rate_error < 1e-10
    forecasts = read('cmapss_primary_forecasts.csv')
    scores = read('cmapss_calibration_scores.csv')
    max_score_error, replay_errors = 0.0, 0
    for (dataset, head, phase, uid), group in forecasts.groupby(['dataset', 'head', 'phase', 'unit_id']):
        a = (group.point.to_numpy() - protocol['lead_time']) / group.scale.to_numpy()
        if phase == 'cal':
            eligible = group.cycle.to_numpy() + protocol['lead_time'] < group.max_cycle.iloc[0]
            b = float(a[eligible].min())
            row = scores[(scores.dataset == dataset) & (scores['head'] == head) & (scores.unit_id == uid)].iloc[0]
            max_score_error = max(max_score_error, abs(b - row.critical_score))
        if phase == 'test':
            records = units[(units.dataset == dataset) & (units.seed == protocol['primary_seed']) & (units.unit_id == uid)]
            for _, row in records.iterrows():
                if row.method == 'Cycle_CP' and head != 'Ridge':
                    continue
                if row.method != 'Cycle_CP' and not row.method.startswith(head + '_'):
                    continue
                margin = summaries[(summaries.dataset == dataset) & (summaries.seed == protocol['primary_seed']) & (summaries.method == row.method)].margin.iloc[0]
                # CSV decimal serialization can place a boundary within one ulp;
                # require the reported crossing and all earlier noncrossings.
                if row.tau >= 0:
                    position = np.flatnonzero(group.cycle.to_numpy() == row.tau)[0]
                    okay = a[position] <= margin + 1e-11 and np.all(a[:position] > margin - 1e-11)
                else:
                    okay = bool(np.all(a > margin - 1e-11))
                replay_errors += int(not okay)
    assert max_score_error < 1e-10 and replay_errors == 0

    spec = importlib.util.spec_from_file_location('verification_core', args.core)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core._C_PAVA_LIB = False
    params = json.loads((args.results / 'cmapss_fitted_parameters.json').read_text(encoding='utf-8'))
    feature_error, forecast_error = 0.0, 0.0
    for dataset in protocol['datasets']:
        raw = np.loadtxt(args.raw / f'train_{dataset}.txt')
        primary_test = split[(split.dataset == dataset) & (split.seed == protocol['primary_seed']) & (split.phase == 'test')]
        uid = int(primary_test.unit_id.min())
        a = raw[raw[:, 0] == uid]
        frame = pd.DataFrame({'unit_id': a[:, 0].astype(int), 'cycle': a[:, 1].astype(int), 'regime': np.rint(a[:, 2]).astype(int)})
        for j, number in enumerate(protocol['selected_original_sensor_numbers'], 1):
            frame[f's{j}'] = a[:, 4 + number]
        # Neither the original T nor future RUL labels are supplied to inference.
        cut = len(frame) // 2
        for head_name in ['Ridge', 'Linex']:
            parameter = params[f'{dataset}_{head_name}']
            encoder = core.CausalMonotoneEncoder(True, True)
            for attribute in ['global_mu', 'global_sd', 'polarities']:
                setattr(encoder, attribute, np.asarray(parameter['encoder_' + attribute]))
            encoder.norms = {int(k): (np.asarray(v[0]), np.asarray(v[1])) for k, v in parameter['encoder_norms'].items()}
            head = core.PrognosticHead()
            for attribute in ['mu', 'sd', 'w']:
                setattr(head, attribute, np.asarray(parameter['head_' + attribute]))
            altered = frame.copy()
            altered.loc[cut:, core.SENSORS] -= 219.0
            matrices = [encoder.transform(piece)[0] for piece in [frame, frame.iloc[:cut], altered]]
            feature_error = max(feature_error, float(np.max(np.abs(matrices[0][:cut] - matrices[1]))), float(np.max(np.abs(matrices[0][:cut] - matrices[2][:cut]))))
            predictions = [head.predict(x, d, causal_smooth=head_name == 'Linex') for x, d in zip(matrices, [frame, frame.iloc[:cut], altered])]
            predictions = [np.clip(p - .85 * parameter['tune_bias'], 0, 125) for p in predictions]
            forecast_error = max(forecast_error, float(np.max(np.abs(predictions[0][:cut] - predictions[1]))), float(np.max(np.abs(predictions[0][:cut] - predictions[2][:cut]))))
    assert feature_error < 1e-12 and forecast_error < 1e-10
    report = {'passed': True, 'primary_unique_test_engines': int(units[units.seed == protocol['primary_seed']].drop_duplicates(['dataset', 'unit_id']).shape[0]),
              'distinct_source_engines': 709, 'policy_replay_records_checked': len(units),
              'event_rule_mismatches': event_errors, 'forecast_to_alarm_mismatches': replay_errors,
              'max_cost_error': cost_error, 'max_summary_rate_error': rate_error,
              'max_score_recomputation_error': max_score_error,
              'max_prefix_feature_error': feature_error, 'max_prefix_forecast_error': forecast_error,
              'future_measurements_perturbed': True, 'inference_without_failure_or_rul_labels': True,
              'raw_sha256_verified': True, 'original_core_sha256_verified': True}
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
