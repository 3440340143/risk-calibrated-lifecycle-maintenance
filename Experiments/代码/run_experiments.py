#!/usr/bin/env python3
"""Portable entry point for the unchanged, verified experimental core and empirical/economic extensions.

Never runs the original __main__, never deletes existing output, and defaults to
the portable Python PAVA implementation. Published Linux+C latency measurements
are not reproduced by this wrapper's Python backend.
"""
from pathlib import Path
import argparse,importlib.util,importlib.metadata,json,os,platform,sys,time
sys.dont_write_bytecode = True

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*args, **kwargs):
        return nullcontext()

def _pkg_ver(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return 'not_installed'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['self-tests','full','public-benchmark'],default='self-tests')
    ap.add_argument('--output',type=Path,required=True,help='A new or empty result directory')
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty. Choose a new directory; no files were removed.')
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='public-benchmark':
        import subprocess
        script=Path(__file__).resolve().parent
        subprocess.run([sys.executable,str(script/'run_public_benchmark.py'),'--core',str(script/'decision_calibrated_pdm_suite.py'),'--raw',str(script.parent/'公开基准'/'raw'),'--output',str(out)],check=True)
        return
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    root=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('verified_dc_pdm_core',root/'decision_calibrated_pdm_suite.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    core._C_PAVA_LIB=False
    spec_emp=importlib.util.spec_from_file_location('emp_econ_val',root/'run_empirical_and_economic_validation.py')
    emp_mod=importlib.util.module_from_spec(spec_emp);spec_emp.loader.exec_module(emp_mod)
    metadata={'stage':args.stage,'python':sys.version,'platform':platform.platform(),
      'backend':'Python stack-PAVA (C compilation explicitly disabled)',
      'threads':1,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'packages':{p:_pkg_ver(p) for p in ['numpy','pandas','scipy','matplotlib','threadpoolctl']},
      'timings_refer_to_this_run_only':True}
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        print('[1/8] Self-tests, gradients, prefix causality, score inversion',flush=True)
        core.run_all_self_tests(str(out))
        if args.stage=='full':
            print('[2/8] D0-D5 forensic controls',flush=True)
            core.run_forensic_d0_d5(str(out))
            print('[3/8] B1-B10 and exchangeable benchmarks',flush=True)
            _,_,_,_,cache=core.run_unit_policy_benchmark_b1_b10(str(out))
            print('[4/8] Censoring and mechanism mismatch',flush=True)
            core.run_censoring_and_mismatch_experiments(cache,str(out))
            print('[5/8] Attribution, matched controls, HI and timing',flush=True)
            core.run_controlled_attribution_and_efficiency(cache,str(out))
            print('[6/8] Fair sensitivity, recovery and transfer diagnostics',flush=True)
            core.run_revision5_diagnostics_and_fair_sensitivity(cache,str(out))
            print('[7/8] Empirical NIST physical benchmarks and economic-gain regimes',flush=True)
            emp_mod.run_all(str(out))
            import subprocess
            print('[8/8] Independent public C-MAPSS benchmark',flush=True)
            subprocess.run([sys.executable,str(root/'run_public_benchmark.py'),'--core',str(root/'decision_calibrated_pdm_suite.py'),'--raw',str(root.parent/'公开基准'/'raw'),'--output',str(out/'public_benchmark')],check=True)
    metadata['elapsed_s']=time.perf_counter()-start
    (out/'runtime_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'stage':args.stage,'elapsed_s':round(metadata['elapsed_s'],2)},indent=2))

if __name__=='__main__':main()
