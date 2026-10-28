#!/usr/bin/env python3
"""Verify supplied snapshot hashes and re-aggregate event/cost identities (stdlib).

Verifies:
1. All 60 CSV SHA-256 hashes in snapshot_identity.csv (51 original + 9 empirical/economic CSVs).
2. Unchanged SHA-256 of decision_calibrated_pdm_suite.py.
3. Unit-level event/cost identities and seed/summary aggregations for B1-B10 (1,000 test units).
4. Official NIST MPC61.DAT temperature corrections (Row 80 = 22.97 C, Row 256 = 22.82 C) and
   unit-level event/cost identities and split/summary aggregations across all new NIST and
   economic attribution CSVs (empirical_and_economic_unit_audit.csv, nist_empirical_benchmark_*,
   tuning_sample_scarcity_*, commissioning_to_field_*).
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,math,statistics

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'数据')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args();root=Path(__file__).resolve().parent;r=args.results
    identities=read(root.parent/'数据/snapshot_identity.csv')
    hash_errors=[row['file'] for row in identities if hashlib.sha256((r/row['file']).read_bytes()).hexdigest()!=row['sha256']]

    # 1. Verify B1-B10 1,000-unit event/cost identities
    units=read(r/'unit_policy_b1_b10_per_unit.csv');seeds=read(r/'unit_policy_b1_b10_per_seed.csv')
    summaries=read(r/'unit_policy_b1_b10_summary.csv');fails=defaultdict(int);records=defaultdict(list)
    event_errors=[];cost_error=0.;rate_error=0.
    assert len(units)==1000 and len(seeds)==200
    assert len({(x['scenario'],x['seed'],x['unit_id']) for x in units})==1000
    for row in units:
        T=float(row['T_u'])
        for i in range(1,11):
            b='B'+str(i);alarm=float(row[b+'_tau']);tau=math.inf if alarm<0 else alarm
            failed=int(tau+15>=T);observed=int(row[b+'_fail']);fails[b]+=observed
            if failed!=observed:event_errors.append((row['scenario'],row['seed'],row['unit_id'],b))
            service=min(T,tau+15);waste=max(0,T-(tau+15))
            cost=10. if failed else 1.+.045*waste
            if b+'_cost' in row:cost_error=max(cost_error,abs(cost-float(row[b+'_cost'])))
            records[(row['scenario'],row['seed'],b)].append((service,cost,observed))
    for row in seeds:
        b=row['method'].split(':',1)[0];key=(row['scenario'],row['seed'],b);data=records[key]
        rate=100*sum(x[1] for x in data)/sum(x[0] for x in data)
        rate_error=max(rate_error,abs(rate-float(row['j_life_rate'])))
        assert sum(x[2] for x in data)==int(row['n_fails'])
    summary_errors=[]
    for row in summaries:
        b=row['method'].split(':',1)[0]
        block=[float(x['j_life_rate']) for x in seeds if x['method'].startswith(b+':')]
        if abs(statistics.mean(block)-float(row['j_life_mean']))>1e-10:summary_errors.append(b)
        if fails[b]!=int(row['total_fails']):summary_errors.append(b+' failures')

    # 2. Verify NIST raw data & unit-level audit identities
    nist_arr=read(r/'nist_apr423_arrhenius_degradation_raw.csv')
    nist_mpc=read(r/'nist_mpc61_wafer_resistivity_raw.csv')
    assert len(nist_arr)==15 and len(nist_mpc)==300
    # Verify official NIST MPC61 temperature corrections on rows 80 and 256 (0-indexed 79 and 255):
    assert abs(float(nist_mpc[79]['temp_C'])-22.97)<1e-9, f"Row 80 temp error: {nist_mpc[79]['temp_C']}"
    assert abs(float(nist_mpc[255]['temp_C'])-22.82)<1e-9, f"Row 256 temp error: {nist_mpc[255]['temp_C']}"

    emp_units=read(r/'empirical_and_economic_unit_audit.csv')
    nist_splits=read(r/'nist_empirical_benchmark_per_split.csv')
    nist_sum=read(r/'nist_empirical_benchmark_summary.csv')
    emp_groups=defaultdict(list)
    for row in emp_units:
        T=float(row['T_u']);alarm=float(row['tau']);tau=math.inf if alarm<0 else alarm
        failed=int(tau+15>=T);obs=int(row['Z_fail'])
        if failed!=obs:event_errors.append(('NIST_unit',row['benchmark'],row['split'],row['policy'],row['unit_id']))
        service=min(T,tau+15);waste=0. if failed else max(0.,T-(tau+15))
        cost=10. if failed else 1.+.045*waste
        cost_error=max(cost_error,abs(cost-float(row['c_tot'])),abs(service-float(row['U_u'])),abs(waste-float(row['W_u'])))
        emp_groups[(row['benchmark'],int(row['split']),row['policy'])].append((service,cost,obs,waste))

    for row in nist_splits:
        key=(row['benchmark'],int(row['split']),row['policy']);g=emp_groups[key]
        rate=100.*sum(x[1] for x in g)/sum(x[0] for x in g)
        fail_p=100.*sum(x[2] for x in g)/len(g)
        waste_m=sum(x[3] for x in g)/len(g)
        rate_error=max(rate_error,abs(rate-float(row['j_life_rate'])),abs(fail_p-float(row['fail_pct'])),abs(waste_m-float(row['waste_cycles'])))

    for row in nist_sum:
        sub=[x for x in nist_splits if x['benchmark']==row['benchmark'] and x['policy']==row['policy']]
        if abs(statistics.mean(float(x['j_life_rate']) for x in sub)-float(row['j_life_mean']))>1e-9:
            summary_errors.append('NIST_summary:'+row['policy'])
        if abs(statistics.mean(float(x['fail_pct']) for x in sub)-float(row['fail_pct_mean']))>1e-9:
            summary_errors.append('NIST_fail:'+row['policy'])

    # 3. Verify H1 & H2 seed-to-summary consistency
    h1_seeds=read(r/'tuning_sample_scarcity_per_seed.csv')
    h1_sum=read(r/'tuning_sample_scarcity_summary.csv')
    for row in h1_sum:
        nt=row['n_tune']
        sub=[x for x in h1_seeds if x['n_tune']==nt]
        if abs(statistics.mean(float(x['j_erm']) for x in sub)-float(row['j_erm_mean']))>1e-9:
            summary_errors.append(f'H1_erm_N{nt}')
        if abs(statistics.mean(float(x['j_cp']) for x in sub)-float(row['j_cp_mean']))>1e-9:
            summary_errors.append(f'H1_cp_N{nt}')

    h2_seeds=read(r/'commissioning_to_field_per_seed.csv')
    h2_sum=read(r/'commissioning_to_field_summary.csv')
    code_map=[
        ('erm','Cost-Tuned Margin Only (ERM on Commissioning D_tune, n=30)'),
        ('b7','B7: Smooth-Cost Tuning (Commissioning D_tune, n=30)'),
        ('erm_field','Equal-Budget Field-Retuned ERM (40 Field Units, Grid [0,6])'),
        ('b7_field','Equal-Budget Field-Retuned B7 (40 Field Units, Smooth-Cost)'),
        ('cyc','B5: Field Cycle-Level Residual CP (40 Field Units, alpha=0.05)'),
        ('pure_cp','Pure Field Unit-CP (40 Field Units, No Tune Floor, alpha=0.05)'),
        ('b9','B9: DC-PdM Static (Commissioning Tune + 40 Field Unit-CP Floor)'),
        ('b10','B10: DC-PdM Online (Commissioning Tune + Field Event-ACI)'),
    ]
    for code,pname in code_map:
        row=next(x for x in h2_sum if x['policy']==pname)
        if abs(statistics.mean(float(x['j_'+code]) for x in h2_seeds)-float(row['j_life_mean']))>1e-9:
            summary_errors.append('H2_'+code)
        if abs(statistics.mean(float(x['waste_'+code]) for x in h2_seeds)-float(row['waste_cycles_mean']))>1e-9:
            summary_errors.append('H2_waste_'+code)

    module_hash=hashlib.sha256((root/'decision_calibrated_pdm_suite.py').read_bytes()).hexdigest()
    expected_module_hash='1392a8f83cc277d96d2fddfb436883b987385454acea60b357f6d87d61790300'
    ok=not hash_errors and not event_errors and cost_error<1e-9 and rate_error<1e-9 and not summary_errors and module_hash==expected_module_hash
    report={'passed':ok,'snapshot_csvs_checked':len(identities),'hash_errors':hash_errors,
      'test_units':len(units),'empirical_unit_evaluations_checked':len(emp_units),
      'scenario_seed_method_records':len(seeds),'event_rule_mismatches':len(event_errors),
      'max_cost_identity_error':cost_error,'max_seed_cost_rate_error':rate_error,
      'summary_errors':summary_errors,'numerical_core_sha256':module_hash,
      'numerical_core_unchanged':module_hash==expected_module_hash,'failure_counts_out_of_1000':dict(sorted(fails.items())),
      'interpretation':'Verifies all 60 CSV hashes and re-aggregates unit-level event/cost identities across both B1-B10 and the semi-empirical/economic extensions.'}
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
