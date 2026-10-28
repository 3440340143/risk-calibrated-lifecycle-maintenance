# NASA C-MAPSS policy-replay benchmark

The four official complete training files contain 709 engine simulator trajectories. The primary engine-level Fit/Tune/Cal/Test partition contains 212/70/141/286 assets. Five additional fixed splits use the same source engines. The official truncated test records are not used for stopping-policy replay because the future sensor sequences are unavailable.

## Archived results

- `results`: original public replay, forecasts, calibration scores, decisions, and NASA source provenance.
- `results_full_pool`: the final 15-policy complete-history results, fixed-split sensitivity, and primary paired bootstrap used by this paper.
- `results_retirement_event`: retirement-only event-cost optimization, complete-data comparison on the same threshold ray, per-unit records, and the identification audit.

The final full-pool archive is the verified rerun version. The author's review note records its comparison with the earlier supplied version. New runs write separate `_rerun` folders.

From the repository root, run `python reproduce.py run-upgrades`, `python reproduce.py verify-upgrades`, `python reproduce.py run-retirement`, and `python reproduce.py verify-retirement`. See the root README for dependency installation and the lifecycle commands. Archived numerical records are in the results directories above.

Costs are per 100 service cycles. The lifecycle rate includes unused natural life, whereas the event rate uses maintenance events and service time. The retirement-only optimization targets the latter. The primary bootstrap resamples engines within subset and conditions on the fitted policies.

Source: A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository. Official source links, archive provenance, and raw-file hashes are in `results/cmapss_source_provenance.json`. The original NASA readme and training files are supplied unchanged.
