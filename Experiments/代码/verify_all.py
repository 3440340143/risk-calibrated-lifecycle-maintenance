#!/usr/bin/env python3
"""Verify the original snapshot and independent public benchmark together."""
from pathlib import Path
import subprocess,sys

root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'verify_results.py')],check=True)
subprocess.run([sys.executable,str(root/'verify_public_benchmark.py'),'--results',str(root.parent/'公开基准'/'results'),'--raw',str(root.parent/'公开基准'/'raw'),'--core',str(root/'decision_calibrated_pdm_suite.py'),'--report',str(root.parent/'公开基准'/'results'/'verification_report.json')],check=True)
print('All snapshot and public-benchmark verification checks passed.')
