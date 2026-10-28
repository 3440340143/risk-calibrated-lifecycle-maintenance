"""Portable entry points for the archived study (run from any directory)."""
from pathlib import Path
import argparse, subprocess, sys

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'Experiments'/'代码'
PUBLIC=ROOT/'Experiments'/'公开基准'
REPORTS=ROOT/'Rerun_Reports'

def run(name,*args):
    subprocess.run([sys.executable,str(CODE/name),*[str(x) for x in args]],check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['verify-original','run-upgrades','verify-upgrades','run-retirement','verify-retirement'])
    a=p.parse_args();REPORTS.mkdir(exist_ok=True)
    if a.task=='verify-original':
        run('verify_results.py','--report',REPORTS/'original_snapshot.json')
        run('verify_public_benchmark.py','--results',PUBLIC/'results','--raw',PUBLIC/'raw','--core',CODE/'decision_calibrated_pdm_suite.py','--report',REPORTS/'original_public.json')
    elif a.task=='run-upgrades':
        run('run_track_b_upgrades.py','--output',PUBLIC/'results_full_pool_rerun')
    elif a.task=='verify-upgrades':
        run('verify_track_b.py','--results',PUBLIC/'results_full_pool_rerun','--reference',PUBLIC/'results_full_pool','--report',REPORTS/'upgrades.json')
    elif a.task=='run-retirement':
        run('run_retirement_event_optimization.py','--out',PUBLIC/'results_retirement_event_rerun')
    elif a.task=='verify-retirement':
        run('verify_retirement_event_optimization.py','--results',PUBLIC/'results_retirement_event_rerun','--report',REPORTS/'retirement_extra.json')

if __name__=='__main__':main()
