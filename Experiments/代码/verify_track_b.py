"""Numerical and finite-path audit of the complete-history replay."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import numpy as np
import pandas as pd
import run_public_benchmark as rpb


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root/'公开基准/results_full_pool_rerun')
    parser.add_argument('--reference', type=Path, default=root/'公开基准/results_full_pool')
    parser.add_argument('--report', type=Path, default=root.parent/'Rerun_Reports/full_pool.json')
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    read=lambda name: pd.read_csv(args.results/name, float_precision='round_trip')
    units=read('cmapss_upgraded_per_unit.csv')
    summaries=read('cmapss_upgraded_per_split.csv')
    rc=read('cmapss_right_censored_verification.csv')
    assert len(units)==25740 and len(summaries)==360 and len(rc)==144
    assert len(units.method.unique())==15
    maximum_error=0.0
    for (dataset, seed, method), g in units.groupby(['dataset','seed','method']):
        tau=np.where(g.tau<0,np.inf,g.tau)
        fail=(tau+rpb.CONFIG['lead_time']>=g.T_u).astype(int)
        life=np.minimum(g.T_u,tau+rpb.CONFIG['lead_time'])
        waste=np.maximum(0,g.T_u-tau-rpb.CONFIG['lead_time'])
        cost=1+(rpb.CONFIG['rho']-1)*fail+rpb.CONFIG['waste_cost']*waste
        assert np.array_equal(fail,g.fail)
        assert np.array_equal(life,g.service_cycles)
        assert np.array_equal(waste,g.waste_cycles)
        maximum_error=max(maximum_error,float(np.max(np.abs(cost-g.total_cost))))
        row=summaries[(summaries.dataset==dataset)&(summaries.seed==seed)&(summaries.method==method)].iloc[0]
        assert abs(float(100*cost.sum()/life.sum())-row.j_life)<1e-10
    assert maximum_error<1e-12
    assert rc.max_dominance_violation.max()<=1e-12
    assert rc.upper_tail_identifiability_error.max()<=1e-12
    assert rc.envelope_threshold_error.max()<=1e-12
    primary=rc[rc.seed==rpb.CONFIG['primary_seed']]
    percentile=primary[primary.scenario.isin(['Historical_Q70_Tune','Historical_Q85_Tune'])]
    assert percentile.exact_q_diff.max()<1e-12
    comparisons=[]
    for p in sorted(args.results.glob('*.csv')):
        ref=args.reference/p.name
        if not ref.exists():continue
        a=pd.read_csv(p,float_precision='round_trip'); b=pd.read_csv(ref,float_precision='round_trip')
        assert a.shape==b.shape and list(a.columns)==list(b.columns)
        max_diff=0.0
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                diff=np.abs(a[col].to_numpy()-b[col].to_numpy())
                if len(diff) and not np.isnan(diff).all():max_diff=max(max_diff,float(np.nanmax(diff)))
                # Cross-platform numerical differences are reported, not hidden
                # by increasing a tolerance. Within-run identities are checked above.
            else:
                assert a[col].fillna('').equals(b[col].fillna('')),(p.name,col)
        comparisons.append(dict(file=p.name,rows=len(a),maximum_numeric_difference=max_diff,
                                byte_identical=hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(ref.read_bytes()).digest()))
    old_units=pd.read_csv(args.reference/'cmapss_upgraded_per_unit.csv',float_precision='round_trip')
    assert np.array_equal(units.fail,old_units.fail)
    changed=units.tau.to_numpy()!=old_units.tau.to_numpy()
    primary_mask=units.seed.to_numpy()==rpb.CONFIG['primary_seed']
    assert not changed[primary_mask].any()
    changed_units=units.loc[changed].copy()
    changed_units['archived_tau']=old_units.loc[changed,'tau'].to_numpy()
    changed_units.to_csv(args.report.with_name('changed_replay_decisions.csv'),index=False,lineterminator='\n')
    # Enumerate tied, oscillating and constant finite paths. This checks the
    # identities used by RC-CRC; the manuscript supplies their general proofs.
    cases=0
    values=(0.0,.5,1.0,2.0)
    thresholds=(0.0,.25,.5,1.0,1.5,2.0,3.0)
    for a in itertools.product(values, repeat=5):
        a=np.asarray(a); b=float(a[:4].min())
        for h in thresholds:
            matches=np.flatnonzero(a<=h)
            tau=int(matches[0])+1 if len(matches) else math.inf
            fail=int(tau+15>=20)
            tilde=b if fail else float(a[:int(tau)].min())
            assert fail==int(b>h)
            assert b<=tilde
            for lam in thresholds:
                if lam>=h:assert int(b>lam)==int(tilde>lam)
            assert max(h,b)==max(h,tilde)
            cases+=1
    # The historical policy retires both paths at age two; their observed
    # prefixes coincide but their lower-margin counterfactual risks differ.
    path1=np.array([2.,3.,0.,3.,3.]);path2=np.array([2.,3.,1.,3.,3.])
    assert np.array_equal(path1[:2],path2[:2])
    assert int(path1[:4].min()>.5)!=int(path2[:4].min()>.5)
    # Exact small-sample rank check with a data-dependent feasible selector.
    rank_events=0
    for perm in itertools.permutations([0.,1.,2.,3.]):
        q=max(perm[:3]);selected=q+(sum(perm[:3])%2)
        rank_events+=int(perm[3]>selected)
    assert rank_events/24<=.25
    solver=rpb.solver_tests()
    report=dict(passed=True,policy_unit_records=len(units),subset_seed_policy_summaries=len(summaries),
                retirement_configurations=len(rc),finite_path_cases=cases,
                cost_identity_max_error=maximum_error,
                dominance_max_violation=float(max(0,rc.max_dominance_violation.max())),
                upper_tail_max_error=float(rc.upper_tail_identifiability_error.max()),
                envelope_max_error=float(rc.envelope_threshold_error.max()),
                primary_percentile_history_max_quantile_error=float(percentile.exact_q_diff.max()),
                nonidentification_counterexample_checked=True,permutation_rank_events=rank_events,
                solver_tests=solver,primary_decisions_identical_to_archive=True,
                changed_alarm_times_outside_primary=int(changed.sum()),
                changed_failure_indicators=0,archived_vs_local_csvs=comparisons)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
