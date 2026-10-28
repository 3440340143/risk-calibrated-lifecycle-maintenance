"""Quantify the empirical policy-range cost of preventive censoring.

Complete lifetimes enter the oracle audit only. The observed-record optimizer
receives exactly the same retirement records as the original implementation.
"""
from pathlib import Path
import os,sys,argparse,json,math
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
import run_retirement_event_optimization as ro
rpb=ro.rpb
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'Experiments'/'公开基准'/'results_information_audit_rerun')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=rpb.CONFIG;core=rpb.load_core(ROOT/'Experiments'/'代码'/'decision_calibrated_pdm_suite.py')
    archive=pd.read_csv(ROOT/'Experiments'/'公开基准'/'results_retirement_event'/'retirement_event_identification_audit.csv',float_precision='round_trip')
    rows=[];curves=[];largest_replay_error=0.
    for ds in cfg['datasets']:
        frame=rpb.load_dataset(ROOT/'Experiments'/'公开基准'/'raw'/f'train_{ds}.txt')
        for seed in [cfg['primary_seed']]+cfg['sensitivity_seeds']:
            parts,_=rpb.split_frame(frame,seed)
            enc=core.CausalMonotoneEncoder(True,True).fit(parts['fit'])
            features={k:enc.transform(v)[0] for k,v in parts.items()}
            for head,kind,focus in [('Ridge','ridge',False),('Linex','linex',True)]:
                model=core.PrognosticHead().fit(features['fit'],parts['fit'].true_rul.to_numpy(),kind=kind,reg=cfg['regularization'],focus_window=focus)
                pred={k:model.predict(v,parts[k],causal_smooth=head=='Linex') for k,v in features.items()}
                if head=='Linex':
                    bias=float(np.mean(pred['tune']-parts['tune'].true_rul.to_numpy()))
                    pred={k:np.clip(v-.85*bias,0,cfg['rul_cap']) for k,v in pred.items()}
                sd=max(1e-6,float(np.std(pred['tune']-parts['tune'].true_rul.to_numpy())))
                scale={k:sd*(.3+.7*expit((37-v)/9)) if head=='Linex' else np.full(len(v),sd) for k,v in pred.items()}
                seq={k:rpb.unit_sequences(parts[k],pred[k],scale[k]) for k in parts}
                q,_,rank=rpb.calibration_threshold(seq['cal'])
                complete_margin,complete_value=ro.complete_event_opt(seq['cal'],q)
                baseline=archive[(archive.dataset==ds)&(archive.seed==seed)&(archive['head']==head)]
                for _,old in baseline.iterrows():
                    h=float(old.historical_margin);records=[ro.observe(u,h) for u in seq['cal']]
                    qret=max(0.,float(sorted(r['score'] for r in records)[rank-1]));L=max(0.,h,qret)
                    chosen,value,_=ro.observed_sweep(records,L)
                    error=max(abs(value-old.event_objective),abs(chosen-old.selected_margin))
                    largest_replay_error=max(largest_replay_error,error)
                    assert error<1e-9
                    gap=value-complete_value
                    assert gap>=-1e-10
                    sufficient=h<=complete_margin
                    assert not sufficient or abs(gap)<1e-10
                    rows.append(dict(dataset=ds,seed=seed,head=head,scenario=old.scenario,n_cal=len(records),
                        n_censored=sum(1-r['failed'] for r in records),censor_pct=100*np.mean([1-r['failed'] for r in records]),
                        historical_margin=h,complete_floor=q,observed_floor=L,
                        complete_safe_margin=complete_margin,observed_margin=chosen,
                        complete_safe_objective=complete_value,observed_objective=value,
                        identified_policy_gap=max(0.,gap),relative_gap_pct=100*max(0.,gap)/complete_value,
                        sufficient_zero_gap_condition=sufficient,zero_gap=abs(gap)<1e-10))
                if ds=='FD002' and seed==cfg['primary_seed'] and head=='Linex':
                    tune_scores=np.array([np.min(((u['point']-15)/u['scale'])[u['cycles']+15<u['T']]) for u in seq['tune']])
                    hs=np.unique(np.r_[np.linspace(0,max(complete_margin*1.8,5.),45),baseline.historical_margin,complete_margin,q])
                    for h in hs:
                        records=[ro.observe(u,float(h)) for u in seq['cal']]
                        qr=max(0,float(sorted(r['score'] for r in records)[rank-1]));L=max(0,h,qr)
                        selected,value,_=ro.observed_sweep(records,L)
                        curves.append(dict(historical_margin=h,censor_pct=100*np.mean([1-r['failed'] for r in records]),
                            risk_floor=L,selected_margin=selected,identified_objective=value,complete_safe_objective=complete_value,
                            identified_policy_gap=max(0.,value-complete_value),complete_safe_margin=complete_margin,complete_floor=q))
            print(ds,seed,flush=True)
    result=pd.DataFrame(rows);curve=pd.DataFrame(curves).sort_values('historical_margin')
    assert len(result)==144 and np.all(np.diff(curve.identified_objective)>=-1e-10)
    result.to_csv(args.output/'information_audit_configurations.csv',index=False)
    curve.to_csv(args.output/'historical_threshold_curve.csv',index=False)
    report=dict(passed=True,configurations=len(result),zero_gap_configurations=int(result.zero_gap.sum()),
        sufficient_condition_configurations=int(result.sufficient_zero_gap_condition.sum()),
        sufficient_condition_violations=int((result.sufficient_zero_gap_condition&~result.zero_gap).sum()),
        high_censoring_configurations=int((result.censor_pct>=90).sum()),
        high_censoring_zero_gap=int(((result.censor_pct>=90)&result.zero_gap).sum()),
        maximum_gap=float(result.identified_policy_gap.max()),maximum_relative_gap_pct=float(result.relative_gap_pct.max()),
        max_existing_replay_error=largest_replay_error,threshold_curve_cases=len(curve),curve_monotonicity_passed=True,
        interpretation='Empirical optimization on Cal. Complete latent lifetimes are used only by the reference oracle; repeated configurations reuse source engines.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(result.groupby(['head','scenario']).agg(configurations=('zero_gap','size'),zero_gap=('zero_gap','sum'),mean_gap=('identified_policy_gap','mean'),max_gap=('identified_policy_gap','max')).to_string())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
