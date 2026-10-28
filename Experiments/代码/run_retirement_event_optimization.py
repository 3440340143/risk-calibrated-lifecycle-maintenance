"""Audit observable event-cost optimization from historical retirement records.

The predictor and history policy are frozen before calibration. No complete
lifetime or hidden sensor suffix of a preventive retirement enters selection.
"""
from pathlib import Path
import sys, os, math, json, itertools, argparse
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[n]='1'
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'Experiments'/'代码'))
import numpy as np
import pandas as pd
from scipy.special import expit
import run_public_benchmark as rpb

def observe(seq,h):
    ell=rpb.CONFIG['lead_time']; a=(seq['point']-ell)/seq['scale']
    cross=np.flatnonzero(a<=h)
    tau=float(seq['cycles'][cross[0]]) if len(cross) else math.inf
    z=int(tau+ell>=seq['T']); age=min(seq['T'],tau+ell)
    keep=np.ones(len(a),bool) if z else seq['cycles']<=tau
    c=seq['cycles'][keep].copy(); a=a[keep].copy()
    score=float(a[c+ell<age].min()) if z else float(a.min())
    return dict(unit_id=seq['unit_id'],cycles=c,a=a,age=float(age),failed=z,score=score,h=float(h))

def observed_replay(records,lam):
    ell=rpb.CONFIG['lead_time']; costs=[]; services=[]; fails=[]
    for r in records:
        if lam<r['h']-1e-12:raise ValueError('Outside identifiable ray')
        ix=np.flatnonzero(r['a']<=lam)
        tau=float(r['cycles'][ix[0]]) if len(ix) else math.inf
        z=int(r['failed'] and tau+ell>=r['age'])
        service=min(tau+ell,r['age'])
        if not r['failed']: assert tau+ell<=r['age']
        costs.append(1+9*z); services.append(service); fails.append(z)
    return np.asarray(costs,float),np.asarray(services,float),np.asarray(fails,int)

def observed_sweep(records,lower):
    """One sort and constant work per strict prefix record crossing."""
    ell=rpb.CONFIG['lead_time']; candidates=[]
    for i,r in enumerate(records):
        best=math.inf
        for a,t in zip(r['a'],r['cycles']):
            if a<best:
                if a>lower:candidates.append((float(a),i,float(t)))
                best=float(a)
    costs,services,_=observed_replay(records,lower)
    nc=float(costs.sum()); nd=float(services.sum()); opt=lower; best=100*nc/nd
    candidates.sort(key=lambda x:x[0])
    for val,grp in itertools.groupby(candidates,key=lambda x:x[0]):
        for _,i,t in grp:
            r=records[i]; z=int(r['failed'] and t+ell>=r['age'])
            cost=1+9*z; service=min(t+ell,r['age'])
            nc+=cost-costs[i]; nd+=service-services[i]
            costs[i]=cost;services[i]=service
        now=100*nc/nd
        if now<best-1e-14:best=now;opt=val
    return opt,best,len(candidates)

def complete_event_opt(seqs,lower):
    old=rpb.CONFIG['waste_cost'];rpb.CONFIG['waste_cost']=0
    try:return rpb.exact_margin_frontier(seqs,lower)[:2]
    finally:rpb.CONFIG['waste_cost']=old

def event_test(seqs,lam):
    g=rpb.at_margin(seqs,lam)
    return dict(event_cost=float(100*(1+9*g.fail).sum()/g.service_cycles.sum()),
                failures=int(g.fail.sum()),n_test=len(g),
                service_cycles=float(g.service_cycles.sum()),
                total_event_cost=float((1+9*g.fail).sum()),
                fail_pct=float(100*g.fail.mean()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'Experiments'/'公开基准'/'results_retirement_event_rerun');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    core=rpb.load_core(BASE/'Experiments'/'代码'/'decision_calibrated_pdm_suite.py')
    rows=[]; units=[]; audit=[]; cfg=rpb.CONFIG
    for ds in cfg['datasets']:
        frame=rpb.load_dataset(BASE/'Experiments'/'公开基准'/'raw'/f'train_{ds}.txt')
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
                tune_b=np.array([np.min(((u['point']-15)/u['scale'])[u['cycles']+15<u['T']]) for u in seq['tune']])
                hist,_j,_=rpb.exact_margin_frontier(seq['tune'])
                q,_,rank=rpb.calibration_threshold(seq['cal'])
                histories=[('Tune_Q70',float(np.quantile(tune_b,.70))),('Tune_Q85',float(np.quantile(tune_b,.85))),('Tune_ERM',float(hist))]
                for scenario,h in histories:
                    records=[observe(u,h) for u in seq['cal']]
                    qret=max(0,float(sorted(r['score'] for r in records)[rank-1]))
                    lower=max(0,h,qret)
                    margin,obj,k=observed_sweep(records,lower)
                    oracle,oracle_obj=complete_event_opt(seq['cal'],lower)
                    oc,ou,of=observed_replay(records,margin)
                    full=rpb.at_margin(seq['cal'],margin)
                    assert np.array_equal(of,full.fail)
                    assert np.array_equal(ou,full.service_cycles)
                    assert abs(obj-oracle_obj)<1e-10
                    assert abs(margin-oracle)<1e-10
                    # An informative natural comparator fills every censored
                    # score with its known threshold bound instead of its prefix.
                    sentinel=[r['score'] if r['failed'] else h for r in records]
                    sentinel_q=max(0,float(sorted(sentinel)[rank-1]))
                    assert abs(max(0,h,sentinel_q)-lower)<1e-12
                    audit.append(dict(dataset=ds,seed=seed,head=head,scenario=scenario,
                      n_cal=len(records),n_censored=sum(1-r['failed'] for r in records),
                      historical_margin=h,q_complete=q,q_retirement=qret,q_threshold_bound=sentinel_q,
                      risk_floor=lower,selected_margin=margin,oracle_margin=oracle,
                      event_objective=obj,oracle_event_objective=oracle_obj,
                      threshold_bound_envelope_error=abs(max(0,h,sentinel_q)-lower),
                      observed_service_error=float(np.max(np.abs(ou-full.service_cycles))),
                      observed_event_error=int(np.sum(of!=full.fail)),record_crossings=k))
                    for name,lam in [('Historical',h),('Retirement_floor',lower),('Retirement_event_opt',margin),('Complete_event_opt_same_ray',oracle)]:
                        row=dict(dataset=ds,seed=seed,head=head,scenario=scenario,method=name,margin=lam,**event_test(seq['test'],lam))
                        rows.append(row)
                        g=rpb.at_margin(seq['test'],lam)
                        for _,u in g.iterrows():units.append(dict(dataset=ds,seed=seed,head=head,scenario=scenario,method=name,unit_id=int(u.unit_id),fail=int(u.fail),service_cycles=float(u.service_cycles),event_cost=float(1+9*u.fail)))
            print(ds,seed,flush=True)
    per=pd.DataFrame(rows);au=pd.DataFrame(audit);pu=pd.DataFrame(units)
    per.to_csv(args.out/'retirement_event_per_split.csv',index=False)
    au.to_csv(args.out/'retirement_event_identification_audit.csv',index=False)
    pu.to_csv(args.out/'retirement_event_per_unit.csv',index=False)
    pooled=[]
    for key,g in per.groupby(['seed','head','scenario','method'],sort=False):
        pooled.append(dict(zip(['seed','head','scenario','method'],key)),)
        pooled[-1].update(event_cost=100*g.total_event_cost.sum()/g.service_cycles.sum(),
                         fail_pct=100*g.failures.sum()/g.n_test.sum(),failures=int(g.failures.sum()),n_test=int(g.n_test.sum()))
    pooled=pd.DataFrame(pooled);pooled.to_csv(args.out/'retirement_event_pooled_by_seed.csv',index=False)
    primary=pooled[pooled.seed==cfg['primary_seed']]
    primary.to_csv(args.out/'retirement_event_primary.csv',index=False)
    print(primary.to_string(index=False))
    report=dict(passed=True,configurations=len(au),test_policy_records=len(pu),
                max_oracle_objective_error=float(np.max(abs(au.event_objective-au.oracle_event_objective))),
                max_oracle_margin_error=float(np.max(abs(au.selected_margin-au.oracle_margin))),
                max_observed_service_error=float(au.observed_service_error.max()),
                event_mismatches=int(au.observed_event_error.sum()),
                sentinel_envelopes_identical=bool((au.threshold_bound_envelope_error==0).all()),
                score_map='same frozen map used in historical operation and selection',
                cost='event cost only; unused natural life is not an observed label')
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
