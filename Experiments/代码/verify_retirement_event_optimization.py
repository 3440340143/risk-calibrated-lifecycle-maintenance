from pathlib import Path
import sys, math, json, itertools
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_retirement_event_optimization as ro

import argparse
ap=argparse.ArgumentParser(description='Independent finite-path, hidden-continuation and paired-bootstrap checks.')
ap.add_argument('--results',type=Path,default=ro.BASE/'Experiments'/'公开基准'/'results_retirement_event')
ap.add_argument('--report',type=Path,default=ro.BASE/'Review'/'retirement_extra_audit_rerun.json')
args=ap.parse_args();out=args.results;args.report.parent.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(471905)
cases=0; hidden_cases=0; max_error=0.
oldell=ro.rpb.CONFIG['lead_time'];ro.rpb.CONFIG['lead_time']=2.
for _ in range(300):
    seq=[]
    for u in range(5):
        T=float(rng.integers(7,15));cycles=np.arange(1.,T)
        a=rng.choice([-.5,0.,.5,1.,2.,3.,4.],size=len(cycles))
        seq.append(dict(unit_id=u,T=T,cycles=cycles,point=2.+a,scale=np.ones(len(a))))
    h=float(rng.choice([0.,.5,1.,2.,3.]));records=[ro.observe(u,h) for u in seq]
    L=max(h,float(sorted(r['score'] for r in records)[3]))
    lam,value,_=ro.observed_sweep(records,L)
    cand=sorted({L}|{float(a) for u in seq for a in (u['point']-2.)/u['scale'] if a>L})
    vals=[]
    for c in cand:
        g=ro.rpb.at_margin(seq,c);v=100*(1+9*g.fail).sum()/g.service_cycles.sum();vals.append(float(v))
    assert abs(value-min(vals))<1e-10
    for c in cand:
        cost,life,fail=ro.observed_replay(records,c);g=ro.rpb.at_margin(seq,c)
        assert np.array_equal(life,g.service_cycles) and np.array_equal(fail,g.fail)
        cases+=1
    altered=[]
    for u,r in zip(seq,records):
        if r['failed']:altered.append(u);continue
        T=r['age']+float(rng.integers(1,8));cy=np.arange(1.,T)
        a=rng.uniform(-100,100,len(cy));mask=cy<=r['cycles'][-1]
        a[mask]=r['a'];altered.append(dict(unit_id=u['unit_id'],T=T,cycles=cy,point=2.+a,scale=np.ones(len(cy))))
    for c in cand:
        g=ro.rpb.at_margin(altered,c);co,li,fa=ro.observed_replay(records,c)
        assert np.array_equal(li,g.service_cycles) and np.array_equal(fa,g.fail)
        hidden_cases+=1
ro.rpb.CONFIG['lead_time']=oldell

units=pd.read_csv(out/'retirement_event_per_unit.csv')
units=units[(units.seed==ro.rpb.CONFIG['primary_seed'])&(units.scenario=='Tune_ERM')]
methods=['Historical','Retirement_floor','Retirement_event_opt']
results=[]
for head in ['Ridge','Linex']:
    d=units[units['head']==head];rng=np.random.default_rng(471902)
    numer={m:np.zeros(4000) for m in methods};denom={m:np.zeros(4000) for m in methods};fail={m:np.zeros(4000) for m in methods}
    for ds,g in d.groupby('dataset',sort=True):
        ids=sorted(g.unit_id.unique());ix=rng.integers(0,len(ids),(4000,len(ids)))
        for m in methods:
            x=g[g.method==m].set_index('unit_id').loc[ids]
            numer[m]+=x.event_cost.to_numpy()[ix].sum(1);denom[m]+=x.service_cycles.to_numpy()[ix].sum(1);fail[m]+=x.fail.to_numpy()[ix].sum(1)
    for comp in ['Historical','Retirement_floor']:
        treat='Retirement_event_opt';dj=100*numer[treat]/denom[treat]-100*numer[comp]/denom[comp]
        df=100*(fail[treat]-fail[comp])/286
        t=d[d.method==treat];c=d[d.method==comp]
        point=100*t.event_cost.sum()/t.service_cycles.sum()-100*c.event_cost.sum()/c.service_cycles.sum()
        jlo,jhi=np.quantile(dj,[.025,.975]);flo,fhi=np.quantile(df,[.025,.975])
        results.append(dict(head=head,treatment=treat,comparator=comp,delta_event_cost=point,ci95_low=jlo,ci95_high=jhi,delta_fail_pct=100*(t.fail.mean()-c.fail.mean()),ci95_fail_low=flo,ci95_fail_high=fhi))
pd.DataFrame(results).to_csv(args.report.with_name('retirement_event_paired_bootstrap_rerun.csv'),index=False)
report=dict(passed=True,finite_path_replays=cases,hidden_suffix_and_lifetime_perturbation_replays=hidden_cases,random_fleets=300,complete_enumeration_matches=300,bootstrap_replicates=4000,bootstrap_seed=471902)
args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2));print(pd.DataFrame(results).to_string(index=False))
