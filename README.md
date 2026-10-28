# Risk-calibrated lifecycle optimization from censored maintenance records

This repository contains the code, archived numerical results, and NASA C-MAPSS simulator data used in the manuscript. The study reconstructs maintenance outcomes from censored records and evaluates risk-calibrated lifecycle decisions.

## Contents

- `Experiments/代码/`: experiment, analysis, plotting, and verification scripts.
- `Experiments/公开基准/raw/`: NASA C-MAPSS training trajectories and the NASA readme.
- `Experiments/公开基准/results*/`: archived benchmark results, including lifecycle and retirement-event audits.
- `Experiments/数据/` and `Experiments/derived/`: synthetic and empirical experiment records and derived figure data.

The four C-MAPSS training subsets comprise 709 simulated engine trajectories. The primary test split contains 286 engines; other splits reuse the same source engines. See `Experiments/公开基准/README.md` and the manuscript's supplementary material for the study design and interpretation.

## Run and verify

From the repository root, with Python 3.13 and the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python reproduce.py verify-original
python reproduce.py run-upgrades
python reproduce.py verify-upgrades
python reproduce.py run-retirement
python reproduce.py verify-retirement
```

The original verification checks archived data hashes and re-aggregates policy replay and cost calculations. The rerun commands create new output folders rather than replace the archived results. For a full lifecycle rerun and comparison:

```bash
python Experiments/代码/run_censored_lifecycle.py --out Experiments/公开基准/results_censored_lifecycle_rerun
python Experiments/代码/verify_censored_lifecycle.py --results Experiments/公开基准/results_censored_lifecycle_rerun --report Rerun_Reports/lifecycle.json
```

The full experiments are CPU-based and may take substantial time. Runtime and numerical behavior depend on the environment.

## Data provenance

The C-MAPSS data were provided by NASA's Prognostics Center of Excellence: [Turbofan Engine Degradation Simulation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The supplied NASA readme and source provenance record document the benchmark. NIST measurements and transformations are described in the manuscript's supplementary material. Third-party data retain their original terms.
