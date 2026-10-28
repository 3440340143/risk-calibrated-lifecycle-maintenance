"""Compare a fresh lifecycle rerun with the archived numerical records."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=BASE/'Experiments'/'公开基准'/'results_censored_lifecycle')
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args()
    failures=[];checks=[]
    for name,keys in [
        ('lifecycle_identification.csv',['dataset','seed','head','scenario']),
        ('lifecycle_primary.csv',['seed','head','scenario','method']),
        ('lifecycle_pooled_by_seed.csv',['seed','head','scenario','method']),
        ('lifecycle_paired_bootstrap.csv',['head','scenario','treatment','comparator']),
        ('lifecycle_per_unit.csv',['dataset','seed','head','scenario','method','unit_id'])]:
        x=pd.read_csv(a.results/name).sort_values(keys).reset_index(drop=True)
        y=pd.read_csv(a.reference/name).sort_values(keys).reset_index(drop=True)
        errors=[]
        if x.shape!=y.shape:errors.append('shape')
        elif list(x.columns)!=list(y.columns):errors.append('columns')
        else:
            for col in x:
                if pd.api.types.is_numeric_dtype(x[col]):
                    same=np.allclose(x[col],y[col],atol=1e-9,rtol=1e-10,equal_nan=True)
                else:same=x[col].equals(y[col])
                if not same:errors.append(col)
        checks.append(dict(file=name,rows=len(x),passed=not errors,errors=errors))
        failures.extend(errors)
    audit=json.loads((a.results/'lifecycle_audit.json').read_text(encoding='utf-8'))
    for key,value in [('configurations',144),('event_matches_ray',125),('low_matches_ray',143),
                      ('mean_matches_ray',144),('minimax_matches_ray',128),('finite_interval_invariance',122),
                      ('unbounded_invariance',95),('max_event_error',0),('max_service_error',0),
                      ('monotonic_violations',0),('endpoint_regret_violations',0)]:
        if audit[key]!=value:failures.append(key)
    if audit['max_scalar_identity_error']>=1e-10:failures.append('scalar identity')
    report=dict(passed=not failures,comparisons=checks,failures=failures)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
