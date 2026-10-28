"""Verify the information-audit summaries and random finite-fleet consequences."""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd
import run_retirement_event_optimization as ro
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=ROOT/'Experiments'/'公开基准'/'results_information_audit')
    ap.add_argument('--report',type=Path,default=ROOT/'Rerun_Reports'/'information_audit.json')
    args=ap.parse_args();d=pd.read_csv(args.results/'information_audit_configurations.csv',float_precision='round_trip')
    assert len(d)==144 and not d.duplicated(['dataset','seed','head','scenario']).any()
    assert np.allclose(d.identified_policy_gap,np.maximum(0,d.observed_objective-d.complete_safe_objective),atol=1e-12,rtol=0)
    assert np.all(d.observed_margin>=d.observed_floor-1e-12)
    assert np.allclose(d.observed_floor,np.maximum(d.historical_margin,d.complete_floor),atol=1e-12,rtol=0)
    assert np.all(d.complete_safe_margin>=d.complete_floor-1e-12)
    assert np.all(d.loc[d.sufficient_zero_gap_condition,'identified_policy_gap']<=1e-10)
    assert int(d.zero_gap.sum())==130
    curve=pd.read_csv(args.results/'historical_threshold_curve.csv',float_precision='round_trip')
    assert np.all(np.diff(curve.historical_margin)>0) and np.all(np.diff(curve.identified_objective)>=-1e-10)
    original=pd.read_csv(ROOT/'Experiments'/'公开基准'/'results_retirement_event'/'retirement_event_identification_audit.csv',float_precision='round_trip')
    joined=d.merge(original,on=['dataset','seed','head','scenario'],suffixes=('_new','_old'),validate='one_to_one')
    assert np.allclose(joined.observed_objective,joined.event_objective,atol=1e-12,rtol=0)
    assert np.allclose(joined.observed_margin,joined.selected_margin,atol=1e-12,rtol=0)
    rng=np.random.default_rng(471906);oldell=ro.rpb.CONFIG['lead_time'];ro.rpb.CONFIG['lead_time']=2.
    checked=0;sufficient=0
    try:
        for _ in range(200):
            units=[]
            for i in range(8):
                T=float(rng.integers(7,16));cycles=np.arange(1.,T)
                a=rng.choice([-.5,0,.5,1.,2.,3.,4.],len(cycles))
                units.append(dict(unit_id=i,T=T,cycles=cycles,point=2+a,scale=np.ones(len(a))))
            B=[float(np.min(((s['point']-2)/s['scale'])[s['cycles']+2<s['T']])) for s in units]
            q=max(0.,sorted(B)[6]);margin,v0=ro.complete_event_opt(units,q);last=-math.inf
            for h in np.linspace(0,4,9):
                records=[ro.observe(s,float(h)) for s in units]
                qr=max(0.,sorted(r['score'] for r in records)[6]);L=max(0.,h,qr)
                assert abs(L-max(h,q))<1e-12
                chosen,vh,_=ro.observed_sweep(records,L)
                assert vh>=v0-1e-10 and vh>=last-1e-10
                if h<=margin:
                    assert abs(vh-v0)<1e-10;sufficient+=1
                last=vh;checked+=1
    finally:ro.rpb.CONFIG['lead_time']=oldell
    report=dict(passed=True,archived_configurations=144,zero_gap_configurations=130,
        high_censoring_cases=int((d.censor_pct>=90).sum()),high_censoring_zero_gap=int(((d.censor_pct>=90)&d.zero_gap).sum()),
        random_fleets=200,random_threshold_checks=checked,sufficient_condition_random_checks=sufficient,
        threshold_curve_cases=len(curve),original_replay_objectives_identical=True)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
