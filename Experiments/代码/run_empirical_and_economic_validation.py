#!/usr/bin/env python3
"""
Semi-Empirical Trace-Driven Policy Evaluation on Official NIST Physical Datasets
(NIST Section 8.4.2.3 & NIST Section 2.6.1.1.1) and Controlled Economic Attribution
(Tuning-Sample Scarcity H1 & Commissioning-to-Field Shift H2 with Equal-Label-Budget Controls).

Closes all P1/P2 findings from the Independent Audit:
1. Exact official NIST MPC61.DAT temperatures verified across all 300 rows (2,700 tokens):
   - Row 80 (Run 1, Wafer 140, Probe 2062, Month 3, Day 17, Op 1): 22.97 C (corrected from 23.07 C).
   - Row 256 (Run 2, Wafer 141, Probe 283, Month 4, Day 19, Op 2): 22.82 C (corrected from 23.03 C).
2. Retrospective offline-interpolated trajectory construction and prefix replay:
   - Complete published readouts are used offline to construct piecewise-linear trajectories;
     within each interval (t_{m-1}, t_m], linear interpolation uses the right-endpoint readout
     offline, so the replay does not establish online causality under the original measurement
     schedule. Knot-boundary checks verify only that scaling segments after a knot t_m leaves
     the constructed segment on [0, t_m] unchanged (OLS projected failure times 241.84-14,461.88h).
3. Strictly disjoint (mutually exclusive) Fit / Tune / Cal / Test partitions & unified risk target:
   - APR423 (15 physical units): 3 Fit / 3 Tune / 6 Cal / 3 Test per split (zero overlap; all prior
     rates and residual scales frozen exclusively on Fit/Tune). Reports both exact alpha=0.05
     (which triggers the Proposition 1 backup rule tau_u=0 since k=ceil(7*0.95)=7 > n_cal=6) and
     the finest finite order-statistic resolution on n_cal=6 (k=6, alpha_eff=1/7=14.29%).
   - MPC61 (50 physical 6-day series): 15 Run-1 Fit / 10 Run-1 Tune / 19 Run-2 Cal / 6 Run-2 Test
     (all wafer reference baselines, failure thresholds, biases, and residual scales frozen
     exclusively on Run-1 Fit; unified alpha=0.05 -> k=19 on Run-2 Cal), plus an equal-budget
     Run-2 ERM control tuned on the same 19 Run-2 trajectories.
   - Repeated-split differences are reported purely as descriptive finite-sample split summaries
     (with unit-grouped / wafer-grouped dispersion), never as independent cross-population p-values.
4. H1 (Tuning-Sample Scarcity) & H2 (Commissioning-to-Field Shift):
   - H1 uses unified alpha=0.05 (plus alpha=0.025 sensitivity), exports Pure Unit-CP, and reports
     5-seed-block 95% t-intervals and p-values as the primary inference unit alongside 20-run
     exploratory summaries.
   - H2 adds equal-field-label-budget controls (`Field-Retuned ERM` on all 40 field trajectories
     over unclipped grid [0, 6] and `Field-Retuned B7`) to separate field recalibration from
     conformal-specific advantage over equal-budget field cost tuning.
5. Portable `run_all(out_dir)` entry point compatible with `run_experiments.py` and full unit-level
   event/cost identity export (`empirical_and_economic_unit_audit.csv`) verified by `verify_results.py`.

Sources:
1. NIST/SEMATECH e-Handbook of Statistical Methods, Section 8.4.2.3:
   https://www.itl.nist.gov/div898/handbook/apr/section4/apr423.htm
2. NIST/SEMATECH e-Handbook of Statistical Methods, Section 2.6.1.1.1 (MPC61.DAT):
   https://www.itl.nist.gov/div898/handbook/mpc/section6/mpc6111.htm
"""

import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

for _env_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env_var] = '1'

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*args, **kwargs):
        return nullcontext()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_calibrated_pdm_suite as core
core._C_PAVA_LIB = False

# ==============================================================================
# 1. VERBATIM NIST SECTION 8.4.2.3 MULTI-REGIME ARRHENIUS DEGRADATION DATASET
# ==============================================================================
NIST_APR423_ROWS = [
    (1, 65, 1, 0.87, 1.48, 2.81),
    (2, 65, 2, 0.33, 0.96, 2.13),
    (3, 65, 3, 0.94, 2.91, 5.67),
    (4, 65, 4, 0.72, 1.98, 4.28),
    (5, 65, 5, 0.66, 0.99, 2.14),
    (6, 85, 1, 1.41, 2.47, 5.71),
    (7, 85, 2, 3.61, 8.99, 17.69),
    (8, 85, 3, 2.13, 5.72, 11.54),
    (9, 85, 4, 4.36, 9.82, 19.55),
    (10, 85, 5, 6.91, 17.37, 34.84),
    (11, 105, 1, 24.58, 62.02, 124.10),
    (12, 105, 2, 9.73, 24.07, 48.06),
    (13, 105, 3, 4.74, 11.53, 23.72),
    (14, 105, 4, 23.61, 58.21, 117.20),
    (15, 105, 5, 10.90, 27.85, 54.97),
]

# ==============================================================================
# 2. VERBATIM NIST SECTION 2.6.1.1.1 SILICON WAFER RESISTIVITY DATABASE (MPC61)
#    All 300 rows (2,700 tokens) verified against official NIST mpc6111.htm / MPC61.DAT
# ==============================================================================
NIST_MPC61_TEXT = """
1 138. 1. 3. 15. 1. 22.98 95.1772 0.1191
1 138. 1. 3. 17. 1. 23.02 95.1567 0.0183
1 138. 1. 3. 18. 1. 22.79 95.1937 0.1282
1 138. 1. 3. 21. 1. 23.17 95.1959 0.0398
1 138. 1. 3. 23. 2. 23.25 95.1442 0.0346
1 138. 1. 3. 23. 1. 23.20 95.0610 0.1539
1 138. 281. 3. 16. 1. 22.99 95.1591 0.0963
1 138. 281. 3. 17. 1. 22.97 95.1195 0.0606
1 138. 281. 3. 18. 1. 22.83 95.1065 0.0842
1 138. 281. 3. 21. 1. 23.28 95.0925 0.0973
1 138. 281. 3. 23. 2. 23.14 95.1990 0.1062
1 138. 281. 3. 23. 1. 23.16 95.1682 0.1090
1 138. 283. 3. 16. 1. 22.95 95.1252 0.0531
1 138. 283. 3. 17. 1. 23.08 95.1600 0.0998
1 138. 283. 3. 18. 1. 23.13 95.0818 0.1108
1 138. 283. 3. 21. 1. 23.28 95.1620 0.0408
1 138. 283. 3. 22. 1. 23.36 95.1735 0.0501
1 138. 283. 3. 24. 2. 22.97 95.1932 0.0287
1 138. 2062. 3. 16. 1. 22.97 95.1311 0.1066
1 138. 2062. 3. 17. 1. 22.98 95.1132 0.0415
1 138. 2062. 3. 18. 1. 23.16 95.0432 0.0491
1 138. 2062. 3. 21. 1. 23.16 95.1254 0.0603
1 138. 2062. 3. 22. 1. 23.28 95.1322 0.0561
1 138. 2062. 3. 24. 2. 23.19 95.1299 0.0349
1 138. 2362. 3. 15. 1. 23.08 95.1162 0.0480
1 138. 2362. 3. 17. 1. 23.01 95.0569 0.0577
1 138. 2362. 3. 18. 1. 22.97 95.0598 0.0516
1 138. 2362. 3. 22. 1. 23.23 95.1487 0.0386
1 138. 2362. 3. 23. 2. 23.28 95.0743 0.0256
1 138. 2362. 3. 24. 2. 23.10 95.1010 0.0420
1 139. 1. 3. 15. 1. 23.01 99.3528 0.1424
1 139. 1. 3. 17. 1. 23.00 99.2940 0.0660
1 139. 1. 3. 17. 1. 23.01 99.2340 0.1179
1 139. 1. 3. 21. 1. 23.20 99.3489 0.0506
1 139. 1. 3. 23. 2. 23.22 99.2625 0.1111
1 139. 1. 3. 23. 1. 23.22 99.3787 0.1103
1 139. 281. 3. 16. 1. 22.95 99.3244 0.1134
1 139. 281. 3. 17. 1. 22.98 99.3378 0.0949
1 139. 281. 3. 18. 1. 22.86 99.3424 0.0847
1 139. 281. 3. 22. 1. 23.17 99.4033 0.0801
1 139. 281. 3. 23. 2. 23.10 99.3717 0.0630
1 139. 281. 3. 23. 1. 23.14 99.3493 0.1157
1 139. 283. 3. 16. 1. 22.94 99.3065 0.0381
1 139. 283. 3. 17. 1. 23.09 99.3280 0.1153
1 139. 283. 3. 18. 1. 23.11 99.3000 0.0818
1 139. 283. 3. 21. 1. 23.25 99.3347 0.0972
1 139. 283. 3. 22. 1. 23.36 99.3929 0.1189
1 139. 283. 3. 23. 1. 23.18 99.2644 0.0622
1 139. 2062. 3. 16. 1. 22.94 99.3324 0.1531
1 139. 2062. 3. 17. 1. 23.08 99.3254 0.0543
1 139. 2062. 3. 18. 1. 23.15 99.2555 0.1024
1 139. 2062. 3. 18. 1. 23.18 99.1946 0.0851
1 139. 2062. 3. 22. 1. 23.27 99.3542 0.1227
1 139. 2062. 3. 24. 2. 23.23 99.2365 0.1218
1 139. 2362. 3. 15. 1. 23.08 99.2939 0.0818
1 139. 2362. 3. 17. 1. 23.02 99.3234 0.0723
1 139. 2362. 3. 18. 1. 22.93 99.2748 0.0756
1 139. 2362. 3. 22. 1. 23.29 99.3512 0.0475
1 139. 2362. 3. 23. 2. 23.25 99.2350 0.0517
1 139. 2362. 3. 24. 2. 23.05 99.3574 0.0485
1 140. 1. 3. 15. 1. 23.07 96.1334 0.1052
1 140. 1. 3. 17. 1. 23.08 96.1250 0.0916
1 140. 1. 3. 18. 1. 22.77 96.0665 0.0836
1 140. 1. 3. 21. 1. 23.18 96.0725 0.0620
1 140. 1. 3. 23. 2. 23.20 96.1006 0.0582
1 140. 1. 3. 23. 1. 23.21 96.1131 0.1757
1 140. 281. 3. 16. 1. 22.94 96.0467 0.0565
1 140. 281. 3. 17. 1. 22.99 96.1081 0.1293
1 140. 281. 3. 18. 1. 22.91 96.0578 0.1148
1 140. 281. 3. 22. 1. 23.15 96.0700 0.0495
1 140. 281. 3. 22. 1. 23.33 96.1052 0.1722
1 140. 281. 3. 23. 1. 23.19 96.0952 0.1786
1 140. 283. 3. 16. 1. 22.89 96.0650 0.1301
1 140. 283. 3. 17. 1. 23.07 96.0870 0.0881
1 140. 283. 3. 18. 1. 23.07 95.8906 0.1842
1 140. 283. 3. 21. 1. 23.24 96.0842 0.1008
1 140. 283. 3. 22. 1. 23.34 96.0189 0.0865
1 140. 283. 3. 23. 1. 23.19 96.1047 0.0923
1 140. 2062. 3. 16. 1. 22.95 96.0379 0.2190
1 140. 2062. 3. 17. 1. 22.97 96.0671 0.0991
1 140. 2062. 3. 18. 1. 23.15 96.0206 0.0648
1 140. 2062. 3. 21. 1. 23.14 96.0207 0.1410
1 140. 2062. 3. 22. 1. 23.32 96.0587 0.1634
1 140. 2062. 3. 24. 2. 23.17 96.0903 0.0406
1 140. 2362. 3. 15. 1. 23.08 96.0771 0.1024
1 140. 2362. 3. 17. 1. 23.00 95.9976 0.0943
1 140. 2362. 3. 18. 1. 23.01 96.0148 0.0622
1 140. 2362. 3. 22. 1. 23.27 96.0397 0.0702
1 140. 2362. 3. 23. 2. 23.24 96.0407 0.0627
1 140. 2362. 3. 24. 2. 23.13 96.0445 0.0622
1 141. 1. 3. 15. 1. 23.01 101.2124 0.0900
1 141. 1. 3. 17. 1. 23.08 101.1018 0.0820
1 141. 1. 3. 18. 1. 22.75 101.1119 0.0500
1 141. 1. 3. 21. 1. 23.21 101.1072 0.0641
1 141. 1. 3. 23. 2. 23.25 101.0802 0.0704
1 141. 1. 3. 23. 1. 23.19 101.1350 0.0699
1 141. 281. 3. 16. 1. 22.93 101.0287 0.0520
1 141. 281. 3. 17. 1. 23.00 101.0131 0.0710
1 141. 281. 3. 18. 1. 22.90 101.1329 0.0800
1 141. 281. 3. 22. 1. 23.19 101.0562 0.1594
1 141. 281. 3. 23. 2. 23.18 101.0891 0.1252
1 141. 281. 3. 23. 1. 23.17 101.1283 0.1151
1 141. 283. 3. 16. 1. 22.85 101.1597 0.0990
1 141. 283. 3. 17. 1. 23.09 101.0784 0.0810
1 141. 283. 3. 18. 1. 23.08 101.0715 0.0460
1 141. 283. 3. 21. 1. 23.27 101.0910 0.0880
1 141. 283. 3. 22. 1. 23.34 101.0967 0.0901
1 141. 283. 3. 24. 2. 23.00 101.1627 0.0888
1 141. 2062. 3. 16. 1. 22.97 101.1077 0.0970
1 141. 2062. 3. 17. 1. 22.96 101.0245 0.1210
1 141. 2062. 3. 18. 1. 23.19 100.9650 0.0700
1 141. 2062. 3. 18. 1. 23.18 101.0319 0.1070
1 141. 2062. 3. 22. 1. 23.34 101.0849 0.0960
1 141. 2062. 3. 24. 2. 23.21 101.1302 0.0505
1 141. 2362. 3. 15. 1. 23.08 101.0471 0.0320
1 141. 2362. 3. 17. 1. 23.01 101.0224 0.1020
1 141. 2362. 3. 18. 1. 23.05 101.0702 0.0580
1 141. 2362. 3. 22. 1. 23.22 101.0904 0.1049
1 141. 2362. 3. 23. 2. 23.29 101.0626 0.0702
1 141. 2362. 3. 24. 2. 23.15 101.0686 0.0661
1 142. 1. 3. 15. 1. 23.02 94.3160 0.1372
1 142. 1. 3. 17. 1. 23.04 94.2808 0.0999
1 142. 1. 3. 18. 1. 22.73 94.2478 0.0803
1 142. 1. 3. 21. 1. 23.19 94.2862 0.0700
1 142. 1. 3. 23. 2. 23.25 94.1859 0.0899
1 142. 1. 3. 23. 1. 23.21 94.2389 0.0686
1 142. 281. 3. 16. 1. 22.98 94.2640 0.0862
1 142. 281. 3. 17. 1. 23.00 94.3333 0.1330
1 142. 281. 3. 18. 1. 22.88 94.2994 0.0908
1 142. 281. 3. 21. 1. 23.28 94.2873 0.0846
1 142. 281. 3. 23. 2. 23.07 94.2576 0.0795
1 142. 281. 3. 23. 1. 23.12 94.3027 0.0389
1 142. 283. 3. 16. 1. 22.92 94.2846 0.1021
1 142. 283. 3. 17. 1. 23.08 94.2197 0.0627
1 142. 283. 3. 18. 1. 23.09 94.2119 0.0785
1 142. 283. 3. 21. 1. 23.29 94.2536 0.0712
1 142. 283. 3. 22. 1. 23.34 94.2280 0.0692
1 142. 283. 3. 24. 2. 22.92 94.2944 0.0958
1 142. 2062. 3. 16. 1. 22.96 94.2238 0.0492
1 142. 2062. 3. 17. 1. 22.95 94.3061 0.2194
1 142. 2062. 3. 18. 1. 23.16 94.1868 0.0474
1 142. 2062. 3. 21. 1. 23.11 94.2645 0.0697
1 142. 2062. 3. 22. 1. 23.31 94.3101 0.0532
1 142. 2062. 3. 24. 2. 23.24 94.2204 0.1023
1 142. 2362. 3. 15. 1. 23.08 94.2437 0.0503
1 142. 2362. 3. 17. 1. 23.00 94.2115 0.0919
1 142. 2362. 3. 18. 1. 22.99 94.2348 0.0282
1 142. 2362. 3. 22. 1. 23.26 94.2124 0.0513
1 142. 2362. 3. 23. 2. 23.27 94.2214 0.0627
1 142. 2362. 3. 24. 2. 23.08 94.1651 0.1010
2 138. 1. 4. 13. 1. 23.12 95.1996 0.0645
2 138. 1. 4. 15. 1. 22.73 95.1315 0.1192
2 138. 1. 4. 18. 2. 22.76 95.1845 0.0452
2 138. 1. 4. 19. 1. 22.73 95.1359 0.1498
2 138. 1. 4. 20. 2. 22.73 95.1435 0.0629
2 138. 1. 4. 21. 2. 22.93 95.1839 0.0563
2 138. 281. 4. 14. 2. 22.46 95.2106 0.1049
2 138. 281. 4. 18. 2. 22.80 95.2505 0.0771
2 138. 281. 4. 18. 2. 22.77 95.2648 0.1046
2 138. 281. 4. 20. 2. 22.80 95.2197 0.1779
2 138. 281. 4. 20. 2. 22.87 95.2003 0.1376
2 138. 281. 4. 21. 2. 22.95 95.0982 0.1611
2 138. 283. 4. 18. 2. 22.83 95.1211 0.0794
2 138. 283. 4. 13. 1. 23.17 95.1327 0.0409
2 138. 283. 4. 18. 1. 22.67 95.2053 0.1525
2 138. 283. 4. 19. 2. 23.00 95.1292 0.0655
2 138. 283. 4. 21. 2. 22.91 95.1669 0.0619
2 138. 283. 4. 21. 2. 22.96 95.1401 0.0831
2 138. 2062. 4. 15. 1. 22.64 95.2479 0.2867
2 138. 2062. 4. 15. 1. 22.67 95.2224 0.1945
2 138. 2062. 4. 19. 2. 22.99 95.2810 0.1960
2 138. 2062. 4. 19. 1. 22.75 95.1869 0.1571
2 138. 2062. 4. 21. 2. 22.84 95.3053 0.2012
2 138. 2062. 4. 21. 2. 22.92 95.1432 0.1532
2 138. 2362. 4. 12. 1. 22.74 95.1687 0.0785
2 138. 2362. 4. 18. 2. 22.75 95.1564 0.0430
2 138. 2362. 4. 19. 2. 22.88 95.1354 0.0983
2 138. 2362. 4. 19. 1. 22.73 95.0422 0.0773
2 138. 2362. 4. 20. 2. 22.86 95.1354 0.0587
2 138. 2362. 4. 21. 2. 22.94 95.1075 0.0776
2 139. 1. 4. 13. 2. 23.14 99.3274 0.0220
2 139. 1. 4. 15. 2. 22.77 99.5020 0.0997
2 139. 1. 4. 18. 2. 22.80 99.4016 0.0704
2 139. 1. 4. 19. 1. 22.68 99.3181 0.1245
2 139. 1. 4. 20. 2. 22.78 99.3858 0.0903
2 139. 1. 4. 21. 2. 22.93 99.3141 0.0255
2 139. 281. 4. 14. 2. 23.05 99.2915 0.0859
2 139. 281. 4. 15. 2. 22.71 99.4032 0.1322
2 139. 281. 4. 18. 2. 22.79 99.4612 0.1765
2 139. 281. 4. 20. 2. 22.74 99.4001 0.0889
2 139. 281. 4. 20. 2. 22.91 99.3765 0.1041
2 139. 281. 4. 21. 2. 22.92 99.3507 0.0717
2 139. 283. 4. 13. 2. 23.11 99.3848 0.0792
2 139. 283. 4. 18. 2. 22.84 99.4952 0.1122
2 139. 283. 4. 18. 2. 22.76 99.3220 0.0915
2 139. 283. 4. 19. 2. 23.03 99.4165 0.0503
2 139. 283. 4. 21. 2. 22.87 99.3791 0.1138
2 139. 283. 4. 21. 2. 22.98 99.3985 0.0661
2 139. 2062. 4. 14. 2. 22.43 99.4283 0.0891
2 139. 2062. 4. 15. 2. 22.70 99.4139 0.2147
2 139. 2062. 4. 19. 2. 22.97 99.3813 0.1143
2 139. 2062. 4. 19. 1. 22.77 99.4314 0.1685
2 139. 2062. 4. 21. 2. 22.79 99.4166 0.2080
2 139. 2062. 4. 21. 2. 22.94 99.4052 0.2400
2 139. 2362. 4. 12. 1. 22.82 99.3408 0.1279
2 139. 2362. 4. 18. 2. 22.77 99.3116 0.1131
2 139. 2362. 4. 19. 2. 22.82 99.3241 0.0519
2 139. 2362. 4. 19. 1. 22.74 99.2991 0.0903
2 139. 2362. 4. 20. 2. 22.88 99.3049 0.0783
2 139. 2362. 4. 21. 2. 22.94 99.2782 0.0718
2 140. 1. 4. 13. 1. 23.10 96.0811 0.0463
2 140. 1. 4. 15. 2. 22.75 96.1460 0.0725
2 140. 1. 4. 18. 2. 22.78 96.1582 0.1428
2 140. 1. 4. 19. 1. 22.70 96.1039 0.1056
2 140. 1. 4. 20. 2. 22.75 96.1262 0.0672
2 140. 1. 4. 21. 2. 22.93 96.1478 0.0562
2 140. 281. 4. 15. 2. 22.71 96.1153 0.1097
2 140. 281. 4. 14. 2. 22.49 96.1297 0.1202
2 140. 281. 4. 18. 2. 22.81 96.1233 0.1331
2 140. 281. 4. 20. 2. 22.78 96.1731 0.1484
2 140. 281. 4. 20. 2. 22.89 96.0872 0.0857
2 140. 281. 4. 21. 2. 22.91 96.1331 0.0944
2 140. 283. 4. 13. 2. 23.22 96.1135 0.0983
2 140. 283. 4. 18. 2. 22.85 96.1111 0.1210
2 140. 283. 4. 18. 2. 22.78 96.1221 0.0644
2 140. 283. 4. 19. 2. 23.01 96.1063 0.0921
2 140. 283. 4. 21. 2. 22.91 96.1155 0.0704
2 140. 283. 4. 21. 2. 22.94 96.1308 0.0258
2 140. 2062. 4. 15. 2. 22.60 95.9767 0.2225
2 140. 2062. 4. 15. 2. 22.66 96.1277 0.1792
2 140. 2062. 4. 19. 2. 22.96 96.1858 0.1312
2 140. 2062. 4. 19. 1. 22.75 96.1912 0.1936
2 140. 2062. 4. 21. 2. 22.82 96.1650 0.1902
2 140. 2062. 4. 21. 2. 22.92 96.1603 0.1777
2 140. 2362. 4. 12. 1. 22.88 96.0793 0.0996
2 140. 2362. 4. 18. 2. 22.76 96.1115 0.0533
2 140. 2362. 4. 19. 2. 22.79 96.0803 0.0364
2 140. 2362. 4. 19. 1. 22.71 96.0411 0.0768
2 140. 2362. 4. 20. 2. 22.84 96.0988 0.1042
2 140. 2362. 4. 21. 1. 22.94 96.0482 0.0868
2 141. 1. 4. 13. 1. 23.07 101.1984 0.0803
2 141. 1. 4. 15. 2. 22.72 101.1645 0.0914
2 141. 1. 4. 18. 2. 22.75 101.2454 0.1109
2 141. 1. 4. 19. 1. 22.69 101.1096 0.1376
2 141. 1. 4. 20. 2. 22.83 101.2066 0.0717
2 141. 1. 4. 21. 2. 22.93 101.0645 0.1205
2 141. 281. 4. 15. 2. 22.72 101.1615 0.1272
2 141. 281. 4. 14. 2. 22.40 101.1650 0.0595
2 141. 281. 4. 18. 2. 22.78 101.1815 0.1393
2 141. 281. 4. 20. 2. 22.73 101.1106 0.1189
2 141. 281. 4. 20. 2. 22.86 101.1420 0.0713
2 141. 281. 4. 21. 2. 22.94 101.0116 0.1088
2 141. 283. 4. 13. 2. 23.26 101.1554 0.0429
2 141. 283. 4. 18. 2. 22.85 101.1267 0.0751
2 141. 283. 4. 18. 2. 22.76 101.1227 0.0826
2 141. 283. 4. 19. 2. 22.82 101.0635 0.1715
2 141. 283. 4. 21. 2. 22.89 101.1264 0.1447
2 141. 283. 4. 21. 2. 22.96 101.0853 0.1189
2 141. 2062. 4. 15. 2. 22.65 101.1332 0.2532
2 141. 2062. 4. 15. 1. 22.68 101.1487 0.1413
2 141. 2062. 4. 19. 2. 22.95 101.1778 0.1772
2 141. 2062. 4. 19. 1. 22.77 101.0988 0.0884
2 141. 2062. 4. 21. 2. 22.87 101.1686 0.2940
2 141. 2062. 4. 21. 2. 22.94 101.3289 0.2072
2 141. 2362. 4. 12. 1. 22.83 101.1353 0.0585
2 141. 2362. 4. 18. 2. 22.83 101.1201 0.0868
2 141. 2362. 4. 19. 2. 22.91 101.0946 0.0855
2 141. 2362. 4. 19. 1. 22.71 100.9977 0.0645
2 141. 2362. 4. 20. 2. 22.87 101.0963 0.0638
2 141. 2362. 4. 21. 2. 22.94 101.0300 0.0549
2 142. 1. 4. 13. 1. 23.07 94.3049 0.1197
2 142. 1. 4. 15. 2. 22.73 94.3153 0.0566
2 142. 1. 4. 18. 2. 22.77 94.3073 0.0875
2 142. 1. 4. 19. 1. 22.67 94.2803 0.0376
2 142. 1. 4. 20. 2. 22.80 94.3008 0.0703
2 142. 1. 4. 21. 2. 22.93 94.2916 0.0604
2 142. 281. 4. 14. 2. 22.90 94.2557 0.0619
2 142. 281. 4. 18. 2. 22.83 94.3542 0.1027
2 142. 281. 4. 18. 2. 22.80 94.3007 0.1492
2 142. 281. 4. 20. 2. 22.76 94.3351 0.1059
2 142. 281. 4. 20. 2. 22.88 94.3406 0.1508
2 142. 281. 4. 21. 2. 22.92 94.2621 0.0946
2 142. 283. 4. 13. 2. 23.25 94.3124 0.0534
2 142. 283. 4. 18. 2. 22.85 94.3680 0.1643
2 142. 283. 4. 18. 1. 22.67 94.3442 0.0346
2 142. 283. 4. 19. 2. 22.80 94.3391 0.0616
2 142. 283. 4. 21. 2. 22.91 94.2238 0.0721
2 142. 283. 4. 21. 2. 22.95 94.2721 0.0998
2 142. 2062. 4. 14. 2. 22.49 94.2915 0.2189
2 142. 2062. 4. 15. 2. 22.69 94.2803 0.0690
2 142. 2062. 4. 19. 2. 22.94 94.2818 0.0987
2 142. 2062. 4. 19. 1. 22.76 94.2227 0.2628
2 142. 2062. 4. 21. 2. 22.74 94.4109 0.1230
2 142. 2062. 4. 21. 2. 22.94 94.2616 0.0929
2 142. 2362. 4. 12. 1. 22.86 94.2052 0.0813
2 142. 2362. 4. 18. 2. 22.83 94.2824 0.0605
2 142. 2362. 4. 19. 2. 22.85 94.2396 0.0882
2 142. 2362. 4. 19. 1. 22.75 94.2087 0.0702
2 142. 2362. 4. 20. 2. 22.86 94.2937 0.0591
2 142. 2362. 4. 21. 1. 22.93 94.2330 0.0556
"""


def build_nist_apr423_dataframe():
    df_wide = pd.DataFrame(
        NIST_APR423_ROWS,
        columns=['unit_id', 'temp_C', 'cell_unit', 'deg_200h', 'deg_500h', 'deg_1000h']
    )
    long_rows = []
    ols_tf_list = []
    for _, r in df_wide.iterrows():
        t_vec = np.array([200.0, 500.0, 1000.0])
        d_vec = np.array([r['deg_200h'], r['deg_500h'], r['deg_1000h']])
        rate_u = float(np.dot(t_vec, d_vec) / np.dot(t_vec, t_vec))
        tf_hr = 30.0 / rate_u
        ols_tf_list.append(tf_hr)
        for t_hr, col in [(200.0, 'deg_200h'), (500.0, 'deg_500h'), (1000.0, 'deg_1000h')]:
            deg_val = float(r[col])
            y_ijk = math.log(30.0) - (math.log(deg_val) - math.log(t_hr))
            x_ijk = 1.0 / (8.617e-5 * (float(r['temp_C']) + 273.16))
            long_rows.append({
                'unit_id': int(r['unit_id']),
                'temp_C': int(r['temp_C']),
                'cell_unit': int(r['cell_unit']),
                'time_hr': t_hr,
                'deg_pct': deg_val,
                'y_log_tf': y_ijk,
                'x_arrhenius': x_ijk,
                'unit_ols_rate_pct_per_hr': rate_u,
                'unit_projected_failure_hr': tf_hr,
                'rel_log_residual': math.log(deg_val / (rate_u * t_hr))
            })
    df_long = pd.DataFrame(long_rows)
    slope, intercept, _, _, _ = stats.linregress(df_long['x_arrhenius'], df_long['y_log_tf'])
    resid_sd = float(np.sqrt(np.sum((df_long['y_log_tf'] - (intercept + slope * df_long['x_arrhenius'])) ** 2) / 43.0))
    assert abs(intercept - (-18.94337)) < 1e-4, f"NIST intercept mismatch: {intercept}"
    assert abs(slope - 0.81877) < 1e-4, f"NIST slope mismatch: {slope}"
    assert abs(resid_sd - 0.5611) < 1e-3, f"NIST residual SD mismatch: {resid_sd}"
    return df_wide, df_long, {
        'ln_A': float(intercept),
        'delta_H_eV': float(slope),
        'sigma_lognormal': resid_sd,
        'ols_projected_tf_min_hr': float(min(ols_tf_list)),
        'ols_projected_tf_max_hr': float(max(ols_tf_list)),
    }


def build_nist_mpc61_dataframe():
    rows = []
    for line in NIST_MPC61_TEXT.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        rows.append({
            'run': int(float(parts[0])),
            'wafer': int(float(parts[1])),
            'probe': int(float(parts[2])),
            'month': int(float(parts[3])),
            'day': int(float(parts[4])),
            'operator': int(float(parts[5])),
            'temp_C': float(parts[6]),
            'resistivity_ohm_cm': float(parts[7]),
            'within_wafer_sd': float(parts[8]),
        })
    df = pd.DataFrame(rows)
    assert len(df) == 300, f"Expected 300 rows in NIST MPC61, got {len(df)}"
    # Verify the two audited official NIST temperature values explicitly:
    assert abs(float(df.iloc[79]['temp_C']) - 22.97) < 1e-9, f"Row 80 temp mismatch: {df.iloc[79]['temp_C']}"
    assert abs(float(df.iloc[255]['temp_C']) - 22.82) < 1e-9, f"Row 256 temp mismatch: {df.iloc[255]['temp_C']}"
    return df


def construct_causal_apr423_trace(d200, d500, d1000, dt_hr, prior_rate_pct_per_hr):
    """Construct a retrospective, offline-interpolated APR423 replay trajectory.

    Complete 200h/500h/1000h readouts define the path offline. Each interpolation
    interval uses its right-endpoint readout, including before its physical arrival.
    The prefix PAVA forecast is non-anticipative only with respect to the constructed
    replay sequence, not the original physical measurement schedule. The function
    name is retained for API compatibility. dt_hr and the prior rate are fit-split
    quantities; after 1000h the constructed path continues at the 0--1000h OLS rate.
    """
    t_vec = np.array([200.0, 500.0, 1000.0])
    d_vec = np.array([float(d200), float(d500), float(d1000)])
    ols_rate_1000 = max(1e-5, float(np.dot(t_vec, d_vec) / np.dot(t_vec, t_vec)))

    d_list, t_hr_list = [], []
    c = 0
    while True:
        c += 1
        t_hr = c * dt_hr
        if t_hr <= 200.0:
            d_val = float(d200) * (t_hr / 200.0)
        elif t_hr <= 500.0:
            d_val = float(d200) + (float(d500) - float(d200)) * ((t_hr - 200.0) / 300.0)
        elif t_hr <= 1000.0:
            d_val = float(d500) + (float(d1000) - float(d500)) * ((t_hr - 500.0) / 500.0)
        else:
            d_val = float(d1000) + ols_rate_1000 * (t_hr - 1000.0)
        d_list.append(d_val)
        t_hr_list.append(t_hr)
        if d_val >= 30.0 or c >= 2000:
            break

    d_arr = np.asarray(d_list, dtype=float)
    t_hr_arr = np.asarray(t_hr_list, dtype=float)
    T_u = float(len(d_arr))
    cycles = np.arange(1, len(d_arr) + 1, dtype=float)

    secant_rate = np.maximum(1e-6, d_arr / t_hr_arr)
    pava_rate = core.prefix_pava_endpoints(secant_rate)
    run_max_d = np.maximum.accumulate(np.maximum(0.0, d_arr))
    w_prior = np.exp(-cycles / 15.0)
    eff_rate = w_prior * prior_rate_pct_per_hr + (1.0 - w_prior) * (0.60 * pava_rate + 0.40 * secant_rate)
    rem_pred_raw = np.maximum(0.0, 30.0 - run_max_d) / (eff_rate * dt_hr)
    return {
        'T_u': T_u, 'cycles': cycles, 't_hr': t_hr_arr, 'd_curve': d_arr,
        'true_rul': T_u - cycles, 'rem_pred_raw': rem_pred_raw
    }


def construct_causal_mpc61_trace(res_6, sd_6, temp_6, wafer_base, wafer_sd, d_f_mpc, prior_rate_fit):
    """Construct a retrospective, offline-interpolated wafer-probe replay trajectory.

    Six ordered measurements define six 30-cycle intervals. Each interval uses its
    right-endpoint measurement offline. Prefix forecasts are non-anticipative only
    with respect to this constructed replay sequence, not the original physical
    measurement schedule. Fit-split wafer references, threshold, and prior rate
    remain frozen. The function name is retained for API compatibility.
    """
    dev_res = np.abs(np.asarray(res_6, dtype=float) - wafer_base) / max(1e-4, wafer_sd)
    sd_rat = np.asarray(sd_6, dtype=float) / max(1e-4, wafer_sd)
    temp_shift = np.abs(np.asarray(temp_6, dtype=float) - 23.10) * 3.2
    step_inc = 0.45 * dev_res + 0.35 * sd_rat + 0.20 * temp_shift
    cum_w = np.cumsum(step_inc)
    t_knots = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    w_knots = np.r_[0.0, cum_w]
    rate_180 = max(1e-5, float(w_knots[-1] / 180.0))

    w_list = []
    c = 0
    while True:
        c += 1
        if c <= 180:
            w_val = float(np.interp(float(c), t_knots, w_knots))
        else:
            w_val = float(w_knots[-1] + rate_180 * (c - 180.0))
        w_list.append(w_val)
        if (w_val >= d_f_mpc and c >= 25) or c >= 600:
            break

    w_curve = np.asarray(w_list, dtype=float)
    T_u = float(len(w_curve))
    cyc = np.arange(1, len(w_curve) + 1, dtype=float)

    inst_r = np.maximum(1e-5, w_curve / cyc)
    pava_r = core.prefix_pava_endpoints(inst_r)
    rmax_w = np.maximum.accumulate(np.maximum(0.0, w_curve))
    w_e = np.exp(-cyc / 45.0)
    eff_r = w_e * prior_rate_fit + (1.0 - w_e) * (0.60 * pava_r + 0.40 * inst_r)
    rem_pred = np.clip(np.maximum(0.0, d_f_mpc - rmax_w) / np.maximum(1e-5, eff_r), 0.0, 300.0)
    return {
        'T_u': T_u, 'cycles': cyc, 'w_curve': w_curve, 'w_6_end': float(w_knots[-1]),
        'true_rul': T_u - cyc, 'rem_pred_raw': rem_pred
    }


def verify_future_readout_perturbation_invariance():
    """Knot-boundary invariance check: verifies that scaling segments after a readout knot t_m leaves the offline-interpolated segment on [0, t_m] unchanged (does not test interior points within (t_{m-1}, t_m], which use the right-endpoint readout offline)."""
    tr0 = construct_causal_apr423_trace(0.87, 1.48, 2.81, 50.0, 0.003)
    tr_p500_1000 = construct_causal_apr423_trace(0.87, 1.48 * 1.5, 2.81 * 1.5, 50.0, 0.003)
    tr_p1000 = construct_causal_apr423_trace(0.87, 1.48, 2.81 * 1.5, 50.0, 0.003)
    n200 = int(np.sum(tr0['t_hr'] <= 200.0))
    n500 = int(np.sum(tr0['t_hr'] <= 500.0))
    diff_apr_200 = float(np.max(np.abs(tr0['rem_pred_raw'][:n200] - tr_p500_1000['rem_pred_raw'][:n200])))
    diff_apr_500 = float(np.max(np.abs(tr0['rem_pred_raw'][:n500] - tr_p1000['rem_pred_raw'][:n500])))
    assert diff_apr_200 == 0.0 and diff_apr_500 == 0.0, f"APR423 future leak detected: {diff_apr_200}, {diff_apr_500}"

    r0 = np.array([95.18, 95.16, 95.19, 95.20, 95.14, 95.06])
    s0 = np.array([0.12, 0.02, 0.13, 0.04, 0.03, 0.15])
    t0 = np.array([22.98, 23.02, 22.79, 23.17, 23.25, 23.20])
    tr_m0 = construct_causal_mpc61_trace(r0, s0, t0, 95.15, 0.08, 12.0, 0.06)
    s_pert = s0.copy()
    s_pert[1:] *= 1.5
    tr_mp = construct_causal_mpc61_trace(r0, s_pert, t0, 95.15, 0.08, 12.0, 0.06)
    diff_mpc_day1 = float(np.max(np.abs(tr_m0['rem_pred_raw'][:30] - tr_mp['rem_pred_raw'][:30])))
    assert diff_mpc_day1 == 0.0, f"MPC61 future leak detected: {diff_mpc_day1}"
    return {
        'apr423_t_le_200h_max_diff': diff_apr_200,
        'apr423_t_le_500h_max_diff': diff_apr_500,
        'mpc61_day1_max_diff': diff_mpc_day1,
    }


def evaluate_nist_physical_benchmarks(out_dir):
    verify_future_readout_perturbation_invariance()
    df_apr_wide, df_apr_long, apr_params = build_nist_apr423_dataframe()
    df_mpc61 = build_nist_mpc61_dataframe()

    df_apr_wide.to_csv(f'{out_dir}/nist_apr423_arrhenius_degradation_raw.csv', index=False)
    df_mpc61.to_csv(f'{out_dir}/nist_mpc61_wafer_resistivity_raw.csv', index=False)

    # --------------------------------------------------------------------------
    # Benchmark G1: NIST Section 8.4.2.3 Semi-Empirical Trace-Driven Policy Evaluation
    # Strictly disjoint partitions across the 15 physical units (5 per regime):
    #   3 Fit (1/regime) + 3 Tune (1/regime) + 6 Cal (2/regime) + 3 Test (1/regime) = 15 units.
    #   Zero overlap between Fit, Tune, Cal, and Test!
    # --------------------------------------------------------------------------
    apr_dict = {int(r['unit_id']): r for _, r in df_apr_wide.iterrows()}
    u_65 = [1, 2, 3, 4, 5]
    u_85 = [6, 7, 8, 9, 10]
    u_105 = [11, 12, 13, 14, 15]
    grid_lam = np.linspace(0.0, 4.0, 81)
    g1_direct_rows = []
    unit_audit_rows = []
    split_idx = 0

    for t65 in u_65:
        rem65 = [u for u in u_65 if u != t65]
        for t85 in u_85:
            rem85 = [u for u in u_85 if u != t85]
            for t105 in u_105:
                rem105 = [u for u in u_105 if u != t105]
                split_idx += 1
                rot = split_idx % 4
                fit_ids = [rem65[rot], rem85[rot], rem105[rot]]
                tune_ids = [rem65[(rot + 1) % 4], rem85[(rot + 1) % 4], rem105[(rot + 1) % 4]]
                cal_ids = [
                    rem65[(rot + 2) % 4], rem65[(rot + 3) % 4],
                    rem85[(rot + 2) % 4], rem85[(rot + 3) % 4],
                    rem105[(rot + 2) % 4], rem105[(rot + 3) % 4],
                ]
                test_ids = [t65, t85, t105]
                assert len(set(fit_ids) | set(tune_ids) | set(cal_ids) | set(test_ids)) == 15

                prior_rates = {}
                dt_hrs = {}
                for uid_f in fit_ids:
                    rf = apr_dict[uid_f]
                    tc = int(rf['temp_C'])
                    t_v = np.array([200.0, 500.0, 1000.0])
                    d_v = np.array([float(rf['deg_200h']), float(rf['deg_500h']), float(rf['deg_1000h'])])
                    r_f = max(1e-5, float(np.dot(t_v, d_v) / np.dot(t_v, t_v)))
                    prior_rates[tc] = r_f
                    dt_hrs[tc] = (30.0 / r_f) / 120.0

                traces = {}
                for uid_all, r_all in apr_dict.items():
                    tc = int(r_all['temp_C'])
                    traces[uid_all] = construct_causal_apr423_trace(
                        r_all['deg_200h'], r_all['deg_500h'], r_all['deg_1000h'], dt_hrs[tc], prior_rates[tc]
                    )

                ft_errs = np.concatenate([traces[u]['rem_pred_raw'] - traces[u]['true_rul'] for u in (fit_ids + tune_ids)])
                sd_ft = max(1.0, float(np.std(ft_errs)))

                def _get_u_ps(u_id):
                    tr = traces[u_id]
                    p_u = tr['rem_pred_raw']
                    s_u = sd_ft * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - p_u) / 9.0))
                    return tr['cycles'], p_u, s_u, tr['T_u'], tr['true_rul']

                cyc_res, B_cal_list = [], []
                for u_c in cal_ids:
                    cy_c, p_c, s_c, T_c, y_c = _get_u_ps(u_c)
                    cyc_res.append((p_c - y_c) / s_c)
                    B_cal_list.append(core.unit_critical_score(cy_c, p_c, s_c, T_c, core.LEAD_TIME))
                cyc_scores = np.sort(np.concatenate(cyc_res))
                lam_cyc = max(0.0, float(cyc_scores[min(len(cyc_scores) - 1, math.ceil((len(cyc_scores) + 1) * 0.95) - 1)]))
                lam_unit_k6, _ = core.choose_static_margin(B_cal_list, alpha=1.0 / 7.0, tuned_margin=0.0)
                lam_unit_005, _ = core.choose_static_margin(B_cal_list, alpha=0.05, tuned_margin=0.0)

                def _eval_uids(uids, lam_v, record_audit=False, b_name='', pol_name=''):
                    tot_c, tot_u, fails, wastes = 0.0, 0.0, 0, []
                    for u_k in uids:
                        cy_k, p_k, s_k, T_k, _ = _get_u_ps(u_k)
                        out = core.replay_unit_stopping_policy(cy_k, p_k, s_k, T_k, core.LEAD_TIME, lam_v)
                        c_ev = 1.0 * (1 - out['Z_fail']) + 10.0 * out['Z_fail']
                        c_tot = c_ev + 0.045 * out['W_u']
                        tot_c += c_tot
                        tot_u += out['U_u']
                        fails += out['Z_fail']
                        wastes.append(out['W_u'])
                        if record_audit:
                            unit_audit_rows.append({
                                'benchmark': b_name, 'split': split_idx, 'policy': pol_name,
                                'unit_id': int(u_k), 'T_u': float(T_k),
                                'margin': np.nan if lam_v is None else float(lam_v),
                                'tau': -1 if not np.isfinite(out['tau']) else int(out['tau']),
                                'Z_fail': int(out['Z_fail']), 'U_u': float(out['U_u']),
                                'W_u': float(out['W_u']), 'c_tot': float(c_tot)
                            })
                    return 100.0 * tot_c / max(1.0, tot_u), 100.0 * fails / len(uids), float(np.mean(wastes))

                tune_costs = [_eval_uids(tune_ids, g)[0] for g in grid_lam]
                lam_erm = float(grid_lam[int(np.argmin(tune_costs))])
                lam_dc_k6, _ = core.choose_static_margin(B_cal_list, alpha=1.0 / 7.0, tuned_margin=lam_erm)
                lam_dc_005, _ = core.choose_static_margin(B_cal_list, alpha=0.05, tuned_margin=lam_erm)

                for pol_lbl, l_v in [
                    ('Cycle-CP (Step-Level Residual CP, alpha=0.05)', lam_cyc),
                    ('Pure Unit-CP (Disjoint n_cal=6, k=6, alpha_eff=1/7)', lam_unit_k6),
                    ('Cost-Tuned Margin Only (ERM on Disjoint Tune, n_tune=3)', lam_erm),
                    ('DC-PdM Static (Tune + Unit-CP k=6, alpha_eff=1/7)', lam_dc_k6),
                    ('DC-PdM Backup (alpha=0.05, k=7>n_cal=6 -> tau=0)', lam_dc_005),
                ]:
                    jv, fv, wv = _eval_uids(
                        test_ids, l_v, record_audit=True,
                        b_name='NIST_APR423_SemiEmpirical_Disjoint_125Splits', pol_name=pol_lbl
                    )
                    g1_direct_rows.append({
                        'benchmark': 'NIST_APR423_SemiEmpirical_Disjoint_125Splits',
                        'split': split_idx, 'policy': pol_lbl,
                        'margin': np.nan if l_v is None else float(l_v),
                        'j_life_rate': jv, 'fail_pct': fv, 'waste_cycles': wv
                    })

    # --------------------------------------------------------------------------
    # Benchmark G2: Chronological Month-3 (Run 1) -> Month-4 (Run 2) on NIST MPC61.DAT
    # Strictly disjoint partitions:
    #   Run 1: 15 Fit (3 probes/wafer) + 10 Tune (2 probes/wafer) = 25 Run-1 trajectories.
    #   Run 2: 19 Cal + 6 Test = 25 Run-2 trajectories.
    #   Wafer baselines, SDs, threshold D_F, bias, and residual scale are estimated
    #   EXCLUSIVELY on Run-1 Fit! Includes equal-budget Run-2 ERM (n=19) control.
    # --------------------------------------------------------------------------
    raw_series = []
    uid_m = 0
    for (run_id, wafer_id, probe_id), grp in df_mpc61.groupby(['run', 'wafer', 'probe'], sort=False):
        uid_m += 1
        g_s = grp.sort_values(['month', 'day']).reset_index(drop=True)
        raw_series.append({
            'traj_id': uid_m, 'run': int(run_id), 'wafer': int(wafer_id), 'probe': int(probe_id),
            'res_6': g_s['resistivity_ohm_cm'].to_numpy(),
            'sd_6': g_s['within_wafer_sd'].to_numpy(),
            'temp_6': g_s['temp_C'].to_numpy(),
        })

    run1_raw = [m for m in raw_series if m['run'] == 1]
    run2_raw = [m for m in raw_series if m['run'] == 2]
    rng_mpc = np.random.RandomState(2028)
    g2_direct_rows = []

    for fold_id in range(1, 21):
        fit_idx, tune_idx = [], []
        for w_id in [138, 139, 140, 141, 142]:
            w_idxs = [i for i, m in enumerate(run1_raw) if m['wafer'] == w_id]
            perm_w = rng_mpc.permutation(w_idxs)
            fit_idx.extend(perm_w[:3])
            tune_idx.extend(perm_w[3:5])
        perm2 = rng_mpc.permutation(len(run2_raw))
        cal_idx = perm2[:19]
        test_idx = perm2[19:25]
        assert len(set(fit_idx) & set(tune_idx)) == 0
        assert len(set(cal_idx) & set(test_idx)) == 0

        fit_wafers = {}
        for w_id in [138, 139, 140, 141, 142]:
            f_w = [run1_raw[i] for i in fit_idx if run1_raw[i]['wafer'] == w_id]
            all_res = np.concatenate([x['res_6'] for x in f_w])
            all_sd = np.concatenate([x['sd_6'] for x in f_w])
            fit_wafers[w_id] = (float(np.median(all_res)), max(1e-4, float(np.median(all_sd))))

        fit_w6_ends = []
        for i in fit_idx:
            x = run1_raw[i]
            wb, wsd = fit_wafers[x['wafer']]
            dev_r = np.abs(x['res_6'] - wb) / wsd
            sd_r = x['sd_6'] / wsd
            t_sh = np.abs(x['temp_6'] - 23.10) * 3.2
            fit_w6_ends.append(float(np.sum(0.45 * dev_r + 0.35 * sd_r + 0.20 * t_sh)))
        d_f_fit = float(np.median(fit_w6_ends)) * 0.95
        prior_rate_fit = d_f_fit / 180.0

        def _build_tr(x):
            wb, wsd = fit_wafers[x['wafer']]
            tr = construct_causal_mpc61_trace(x['res_6'], x['sd_6'], x['temp_6'], wb, wsd, d_f_fit, prior_rate_fit)
            tr['traj_id'] = x['traj_id']
            return tr

        fit_m = [_build_tr(run1_raw[i]) for i in fit_idx]
        tune_m = [_build_tr(run1_raw[i]) for i in tune_idx]
        cal_m = [_build_tr(run2_raw[i]) for i in cal_idx]
        test_m = [_build_tr(run2_raw[i]) for i in test_idx]

        fit_errs = np.concatenate([m['rem_pred_raw'] - m['true_rul'] for m in fit_m])
        bias_m = float(np.mean(fit_errs))
        sd_m = max(1.0, float(np.std(fit_errs)))

        def _get_m_ps(m_obj):
            p_u = np.clip(m_obj['rem_pred_raw'] - 0.85 * bias_m, 0.0, 300.0)
            s_u = sd_m * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - p_u) / 9.0))
            return m_obj['cycles'], p_u, s_u, m_obj['T_u'], m_obj['true_rul']

        cyc_res, B_cal_g2 = [], []
        for m_c in cal_m:
            cy_c, p_c, s_c, T_c, y_c = _get_m_ps(m_c)
            cyc_res.append((p_c - y_c) / s_c)
            B_cal_g2.append(core.unit_critical_score(cy_c, p_c, s_c, T_c, core.LEAD_TIME))
        cyc_scores = np.sort(np.concatenate(cyc_res))
        lam_cyc = max(0.0, float(cyc_scores[min(len(cyc_scores) - 1, math.ceil((len(cyc_scores) + 1) * 0.95) - 1)]))
        lam_unit_pure, _ = core.choose_static_margin(B_cal_g2, alpha=0.05, tuned_margin=0.0)

        def _eval_mlist(mlist, lam_v, record_audit=False, b_name='', pol_name=''):
            tot_c, tot_u, fails, wastes = 0.0, 0.0, 0, []
            for m_k in mlist:
                cy_k, p_k, s_k, T_k, _ = _get_m_ps(m_k)
                out = core.replay_unit_stopping_policy(cy_k, p_k, s_k, T_k, core.LEAD_TIME, lam_v)
                c_ev = 1.0 * (1 - out['Z_fail']) + 10.0 * out['Z_fail']
                c_tot = c_ev + 0.045 * out['W_u']
                tot_c += c_tot
                tot_u += out['U_u']
                fails += out['Z_fail']
                wastes.append(out['W_u'])
                if record_audit:
                    unit_audit_rows.append({
                        'benchmark': b_name, 'split': fold_id, 'policy': pol_name,
                        'unit_id': int(m_k['traj_id']), 'T_u': float(T_k),
                        'margin': np.nan if lam_v is None else float(lam_v),
                        'tau': -1 if not np.isfinite(out['tau']) else int(out['tau']),
                        'Z_fail': int(out['Z_fail']), 'U_u': float(out['U_u']),
                        'W_u': float(out['W_u']), 'c_tot': float(c_tot)
                    })
            return 100.0 * tot_c / max(1.0, tot_u), 100.0 * fails / len(mlist), float(np.mean(wastes))

        tune_costs = [_eval_mlist(tune_m, g)[0] for g in grid_lam]
        lam_erm = float(grid_lam[int(np.argmin(tune_costs))])
        # Equal-Run2-Label-Budget ERM: tuned on the same 19 Run-2 trajectories used by Unit-CP
        run2_cal_costs = [_eval_mlist(cal_m, g)[0] for g in grid_lam]
        lam_erm_run2 = float(grid_lam[int(np.argmin(run2_cal_costs))])
        lam_dc, _ = core.choose_static_margin(B_cal_g2, alpha=0.05, tuned_margin=lam_erm)

        for pol_lbl, l_v in [
            ('Cycle-CP (Step-Level Residual CP, alpha=0.05)', lam_cyc),
            ('Pure Unit-CP (Run-2 Cal, n_cal=19, k=19, alpha=0.05)', lam_unit_pure),
            ('Cost-Tuned Margin Only (ERM on Disjoint Run-1 Tune, n=10)', lam_erm),
            ('Equal-Budget Run-2 ERM (Tuned on Run-2 Cal, n=19)', lam_erm_run2),
            ('DC-PdM Static (Run-1 Tune + Run-2 Unit-CP Floor, alpha=0.05)', lam_dc),
        ]:
            jv, fv, wv = _eval_mlist(
                test_m, l_v, record_audit=True,
                b_name='NIST_MPC61_SemiEmpirical_Run1_to_Run2_20Folds', pol_name=pol_lbl
            )
            g2_direct_rows.append({
                'benchmark': 'NIST_MPC61_SemiEmpirical_Run1_to_Run2_20Folds',
                'split': fold_id, 'policy': pol_lbl,
                'margin': np.nan if l_v is None else float(l_v),
                'j_life_rate': jv, 'fail_pct': fv, 'waste_cycles': wv
            })

    df_all_direct = pd.DataFrame(g1_direct_rows + g2_direct_rows)
    df_all_direct.to_csv(f'{out_dir}/nist_empirical_benchmark_per_split.csv', index=False)
    pd.DataFrame(unit_audit_rows).to_csv(f'{out_dir}/empirical_and_economic_unit_audit.csv', index=False)

    sum_rows = []
    for (b_name, pol_name), grp in df_all_direct.groupby(['benchmark', 'policy'], sort=False):
        erm_pol = [p for p in df_all_direct[df_all_direct.benchmark == b_name]['policy'].unique() if 'ERM on Disjoint' in p][0]
        cyc_pol = 'Cycle-CP (Step-Level Residual CP, alpha=0.05)'
        sub_erm = df_all_direct[(df_all_direct.benchmark == b_name) & (df_all_direct.policy == erm_pol)].sort_values('split')
        sub_cyc = df_all_direct[(df_all_direct.benchmark == b_name) & (df_all_direct.policy == cyc_pol)].sort_values('split')
        sub_cur = grp.sort_values('split')
        diff_erm_vec = sub_cur['j_life_rate'].to_numpy() - sub_erm['j_life_rate'].to_numpy()
        diff_cyc_vec = sub_cur['j_life_rate'].to_numpy() - sub_cyc['j_life_rate'].to_numpy()
        sum_rows.append({
            'benchmark': b_name, 'policy': pol_name,
            'n_splits': len(grp),
            'mean_margin': float(grp['margin'].dropna().mean()) if grp['margin'].notna().any() else np.nan,
            'j_life_mean': float(grp['j_life_rate'].mean()),
            'j_life_sd': float(grp['j_life_rate'].std(ddof=1)),
            'fail_pct_mean': float(grp['fail_pct'].mean()),
            'waste_cycles_mean': float(grp['waste_cycles'].mean()),
            'descriptive_split_diff_vs_erm_mean': float(np.mean(diff_erm_vec)),
            'descriptive_split_diff_vs_erm_sd': float(np.std(diff_erm_vec, ddof=1)),
            'descriptive_split_diff_vs_cycle_cp_mean': float(np.mean(diff_cyc_vec)),
            'descriptive_split_diff_vs_cycle_cp_sd': float(np.std(diff_cyc_vec, ddof=1)),
            'inference_note': 'Descriptive summary across repeated splits of a fixed finite physical dataset; not an independent cross-population p-value test.'
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(f'{out_dir}/nist_empirical_benchmark_summary.csv', index=False)
    return df_sum, apr_params


def _fast_fleet_j_life(unit_slices, point, scale, margin, ell=core.LEAD_TIME, rho=core.RHO_DEFAULT, c_waste=core.C_WASTE_PER_CYC):
    tot_c, tot_u = 0.0, 0.0
    m_val = float(margin)
    for cyc, T_u, pos in unit_slices:
        alarm_idx = np.flatnonzero((point[pos] - m_val * scale[pos]) <= ell)
        tau = float(cyc[alarm_idx[0]]) if len(alarm_idx) > 0 else np.inf
        z_fail = 1 if (tau + ell >= T_u) else 0
        u_life = float(min(T_u, tau + ell))
        w_life = float(max(0.0, T_u - (tau + ell))) if z_fail == 0 else 0.0
        c_ev = 1.0 * (1 - z_fail) + rho * z_fail
        tot_c += c_ev + c_waste * w_life
        tot_u += u_life
    return 100.0 * tot_c / max(1.0, tot_u)


def evaluate_economic_gain_regimes(out_dir):
    grid_emp_3 = np.linspace(0.0, 3.0, 61)
    grid_emp_6 = np.linspace(0.0, 6.0, 121)
    h1_rows = []
    h2_rows = []

    for sc in core.SCENARIOS:
        for sd in core.SEEDS:
            df = core.generate_synthetic_fleet(sc, n_units=180, seed=sd, disturbance_mode='neg_dips')
            df_fit = df[df.unit_id <= 60].copy()
            df_tune_full = df[(df.unit_id > 60) & (df.unit_id <= 90)].copy()
            df_cal = df[(df.unit_id > 90) & (df.unit_id <= 130)].copy()
            df_test = df[df.unit_id > 130].copy()

            enc = core.CausalMonotoneEncoder(True, True).fit(df_fit)
            Xf, _ = enc.transform(df_fit)
            Xt_full, _ = enc.transform(df_tune_full)
            Xc, _ = enc.transform(df_cal)
            Xte, _ = enc.transform(df_test)
            yf = df_fit.true_rul.to_numpy()
            yc = df_cal.true_rul.to_numpy()

            h_dc = core.PrognosticHead().fit(Xf, yf, kind='linex', rho=10.0, reg=14.0, focus_window=True)
            pt_full_raw = h_dc.predict(Xt_full, df_tune_full, True)
            pc_raw = h_dc.predict(Xc, df_cal, True)
            pte_raw = h_dc.predict(Xte, df_test, True)

            tune_uids_all = df_tune_full.unit_id.unique()
            for n_tune_sub in [6, 10, 15, 20, 30]:
                sub_uids = tune_uids_all[:n_tune_sub]
                m_sub = df_tune_full.unit_id.isin(sub_uids).to_numpy()
                df_tune = df_tune_full.loc[m_sub].copy()
                yt = df_tune.true_rul.to_numpy()
                pt_raw = pt_full_raw[m_sub]

                b_dc = float(np.mean(pt_raw - yt))
                pt = np.clip(pt_raw - 0.85 * b_dc, 0.0, 125.0)
                pc = np.clip(pc_raw - 0.85 * b_dc, 0.0, 125.0)
                pte = np.clip(pte_raw - 0.85 * b_dc, 0.0, 125.0)
                sd_dc = float(np.std(pt - yt))
                sct = sd_dc * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - pt) / 9.0))
                scc = sd_dc * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - pc) / 9.0))
                scte = sd_dc * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - pte) / 9.0))

                u_slices = [
                    (df_tune.iloc[np.asarray(pos)].cycle.to_numpy(), float(df_tune.iloc[np.asarray(pos)].max_cycle.iloc[0]), np.asarray(pos))
                    for _, pos in df_tune.groupby('unit_id', sort=False).indices.items()
                ]
                costs_t = [_fast_fleet_j_life(u_slices, pt, sct, g) for g in grid_emp_3]
                lam_erm = float(grid_emp_3[int(np.argmin(costs_t))])

                B_cal = [
                    core.unit_critical_score(
                        df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                        pc[np.asarray(pos)], scc[np.asarray(pos)],
                        float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                    )
                    for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
                ]
                # Unified default risk target alpha = 0.05 (core.EPSILON) plus secondary alpha = 0.025 record
                lam_pure_cp, _ = core.choose_static_margin(B_cal, alpha=core.EPSILON, tuned_margin=0.0)
                lam_cp, _ = core.choose_static_margin(B_cal, alpha=core.EPSILON, tuned_margin=lam_erm)
                lam_cp_0025, _ = core.choose_static_margin(B_cal, alpha=0.025, tuned_margin=lam_erm)

                ev_erm = core.evaluate_fleet(df_test, pte, scte, lam_erm)
                ev_pure_cp = core.evaluate_fleet(df_test, pte, scte, lam_pure_cp)
                ev_cp = core.evaluate_fleet(df_test, pte, scte, lam_cp)
                ev_cp_0025 = core.evaluate_fleet(df_test, pte, scte, lam_cp_0025)
                h1_rows.append({
                    'n_tune': n_tune_sub, 'scenario': sc, 'seed': sd,
                    'lam_erm': lam_erm, 'lam_pure_cp': lam_pure_cp, 'lam_cp': lam_cp, 'lam_cp_0025': lam_cp_0025,
                    'j_erm': ev_erm['j_life_rate'], 'fail_erm_pct': ev_erm['fail_pct'], 'waste_erm': ev_erm['waste_cycles'],
                    'j_pure_cp': ev_pure_cp['j_life_rate'], 'fail_pure_cp_pct': ev_pure_cp['fail_pct'], 'waste_pure_cp': ev_pure_cp['waste_cycles'],
                    'j_cp': ev_cp['j_life_rate'], 'fail_cp_pct': ev_cp['fail_pct'], 'waste_cp': ev_cp['waste_cycles'],
                    'j_cp_0025': ev_cp_0025['j_life_rate'], 'fail_cp_0025_pct': ev_cp_0025['fail_pct'], 'waste_cp_0025': ev_cp_0025['waste_cycles'],
                    'diff_cp_minus_erm': ev_cp['j_life_rate'] - ev_erm['j_life_rate'],
                    'diff_cp0025_minus_erm': ev_cp_0025['j_life_rate'] - ev_erm['j_life_rate']
                })

            # --- H2: Commissioning-to-Field Shift (Fit+Tune on clean 'no_shocks', Cal+Test on field 'neg_dips') ---
            # Includes Equal-Field-Label-Budget controls: Field-Retuned ERM (40 field units, grid [0,6]) and Field-Retuned B7!
            df_comm = core.generate_synthetic_fleet(sc, n_units=90, seed=sd, disturbance_mode='no_shocks')
            df_fit_c = df_comm[df_comm.unit_id <= 60].copy()
            df_tune_c = df_comm[(df_comm.unit_id > 60) & (df_comm.unit_id <= 90)].copy()

            enc_c = core.CausalMonotoneEncoder(True, True).fit(df_fit_c)
            Xf_c, _ = enc_c.transform(df_fit_c)
            Xt_c, _ = enc_c.transform(df_tune_c)
            Xc_f, _ = enc_c.transform(df_cal)
            Xte_f, _ = enc_c.transform(df_test)
            yf_c = df_fit_c.true_rul.to_numpy()
            yt_c = df_tune_c.true_rul.to_numpy()

            h_dc_c = core.PrognosticHead().fit(Xf_c, yf_c, kind='linex', rho=10.0, reg=14.0, focus_window=True)
            pt_c = h_dc_c.predict(Xt_c, df_tune_c, True)
            pc_f = h_dc_c.predict(Xc_f, df_cal, True)
            pte_f = h_dc_c.predict(Xte_f, df_test, True)
            b_c = float(np.mean(pt_c - yt_c))
            pt_c, pc_f, pte_f = [np.clip(p - 0.85 * b_c, 0.0, 125.0) for p in (pt_c, pc_f, pte_f)]
            sd_c = float(np.std(pt_c - yt_c))
            sct_c, scc_f, scte_f = [sd_c * (0.30 + 0.70 * expit((core.LEAD_TIME + 22.0 - p) / 9.0)) for p in (pt_c, pc_f, pte_f)]

            cyc_scores = np.sort((pc_f - yc) / scc_f)
            k_c = math.ceil((len(cyc_scores) + 1) * (1.0 - core.EPSILON))
            lam_cyc_f = max(0.0, float(cyc_scores[min(len(cyc_scores) - 1, k_c - 1)]))

            u_slices_c = [
                (df_tune_c.iloc[np.asarray(pos)].cycle.to_numpy(), float(df_tune_c.iloc[np.asarray(pos)].max_cycle.iloc[0]), np.asarray(pos))
                for _, pos in df_tune_c.groupby('unit_id', sort=False).indices.items()
            ]
            costs_tc = [_fast_fleet_j_life(u_slices_c, pt_c, sct_c, g) for g in grid_emp_3]
            lam_erm_c = float(grid_emp_3[int(np.argmin(costs_tc))])
            lam_b7_c = core.tune_differentiable_policy_margin(df_tune_c, pt_c, sct_c)

            # Equal-Field-Label-Budget Controls using the exact same 40 field calibration trajectories (df_cal):
            u_slices_field = [
                (df_cal.iloc[np.asarray(pos)].cycle.to_numpy(), float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0]), np.asarray(pos))
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            costs_tfield = [_fast_fleet_j_life(u_slices_field, pc_f, scc_f, g) for g in grid_emp_6]
            lam_erm_field = float(grid_emp_6[int(np.argmin(costs_tfield))])
            lam_b7_field = core.tune_differentiable_policy_margin(df_cal, pc_f, scc_f)

            B_cal_f = [
                core.unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pc_f[np.asarray(pos)], scc_f[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_pure_cp_f, _ = core.choose_static_margin(B_cal_f, alpha=core.EPSILON, tuned_margin=0.0)
            lam_b9_f, _ = core.choose_static_margin(B_cal_f, alpha=core.EPSILON, tuned_margin=lam_b7_c)

            B_sorted_f = np.sort(np.asarray(B_cal_f, dtype=float))
            a_dc_f = core.ONLINE_ALPHA_TARGET
            margins_b10_f = []
            for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
                pos = np.asarray(pos)
                rk = min(len(B_sorted_f), max(1, math.ceil((len(B_sorted_f) + 1) * (1.0 - np.clip(a_dc_f, 0.005, 0.25)))))
                lam_u = max(0.0, lam_b7_c, float(B_sorted_f[rk - 1]))
                margins_b10_f.append(lam_u)
                out_u = core.replay_unit_stopping_policy(
                    df_test.iloc[pos].cycle.to_numpy(), pte_f[pos], scte_f[pos],
                    float(df_test.iloc[pos].max_cycle.iloc[0]), core.LEAD_TIME, lam_u
                )
                a_dc_f = float(np.clip(a_dc_f + core.ONLINE_GAMMA * (core.ONLINE_ALPHA_TARGET - out_u['Z_fail']), core.ONLINE_ALPHA_MIN, core.ONLINE_ALPHA_MAX))

            ev_cyc_f = core.evaluate_fleet(df_test, pte_f, scte_f, lam_cyc_f)
            ev_erm_f = core.evaluate_fleet(df_test, pte_f, scte_f, lam_erm_c)
            ev_b7_f = core.evaluate_fleet(df_test, pte_f, scte_f, lam_b7_c)
            ev_erm_field = core.evaluate_fleet(df_test, pte_f, scte_f, lam_erm_field)
            ev_b7_field = core.evaluate_fleet(df_test, pte_f, scte_f, lam_b7_field)
            ev_pcp_f = core.evaluate_fleet(df_test, pte_f, scte_f, lam_pure_cp_f)
            ev_b9_f = core.evaluate_fleet(df_test, pte_f, scte_f, lam_b9_f)
            ev_b10_f = core.evaluate_fleet(df_test, pte_f, scte_f, margins_b10_f)

            h2_rows.append({
                'scenario': sc, 'seed': sd,
                'j_cyc': ev_cyc_f['j_life_rate'], 'fail_cyc_pct': ev_cyc_f['fail_pct'], 'waste_cyc': ev_cyc_f['waste_cycles'],
                'j_erm': ev_erm_f['j_life_rate'], 'fail_erm_pct': ev_erm_f['fail_pct'], 'waste_erm': ev_erm_f['waste_cycles'],
                'j_b7': ev_b7_f['j_life_rate'], 'fail_b7_pct': ev_b7_f['fail_pct'], 'waste_b7': ev_b7_f['waste_cycles'],
                'j_erm_field': ev_erm_field['j_life_rate'], 'fail_erm_field_pct': ev_erm_field['fail_pct'], 'waste_erm_field': ev_erm_field['waste_cycles'],
                'j_b7_field': ev_b7_field['j_life_rate'], 'fail_b7_field_pct': ev_b7_field['fail_pct'], 'waste_b7_field': ev_b7_field['waste_cycles'],
                'j_pure_cp': ev_pcp_f['j_life_rate'], 'fail_pure_cp_pct': ev_pcp_f['fail_pct'], 'waste_pure_cp': ev_pcp_f['waste_cycles'],
                'j_b9': ev_b9_f['j_life_rate'], 'fail_b9_pct': ev_b9_f['fail_pct'], 'waste_b9': ev_b9_f['waste_cycles'],
                'j_b10': ev_b10_f['j_life_rate'], 'fail_b10_pct': ev_b10_f['fail_pct'], 'waste_b10': ev_b10_f['waste_cycles'],
            })

    df_h1 = pd.DataFrame(h1_rows)
    df_h1.to_csv(f'{out_dir}/tuning_sample_scarcity_per_seed.csv', index=False)
    h1_sum_rows = []
    for nt, grp in df_h1.groupby('n_tune', sort=False):
        st20 = core._paired_stats(grp['diff_cp_minus_erm'])
        st5 = core._paired_stats(grp.groupby('seed')['diff_cp_minus_erm'].mean())
        st20_0025 = core._paired_stats(grp['diff_cp0025_minus_erm'])
        st5_0025 = core._paired_stats(grp.groupby('seed')['diff_cp0025_minus_erm'].mean())
        h1_sum_rows.append({
            'n_tune': int(nt),
            'j_erm_mean': float(grp['j_erm'].mean()), 'j_erm_sd': float(grp['j_erm'].std(ddof=1)),
            'fail_erm_pct': float(grp['fail_erm_pct'].mean()), 'waste_erm_cycles': float(grp['waste_erm'].mean()),
            'j_pure_cp_mean': float(grp['j_pure_cp'].mean()), 'j_pure_cp_sd': float(grp['j_pure_cp'].std(ddof=1)),
            'fail_pure_cp_pct': float(grp['fail_pure_cp_pct'].mean()), 'waste_pure_cp_cycles': float(grp['waste_pure_cp'].mean()),
            'j_cp_mean': float(grp['j_cp'].mean()), 'j_cp_sd': float(grp['j_cp'].std(ddof=1)),
            'fail_cp_pct': float(grp['fail_cp_pct'].mean()), 'waste_cp_cycles': float(grp['waste_cp'].mean()),
            'diff_mean': st5['mean_difference'],
            'diff_5seed_ci95_low': st5['ci95_low'], 'diff_5seed_ci95_high': st5['ci95_high'],
            'p_t_5seed': st5['p_t'], 'p_wilcoxon_5seed': st5['p_wilcoxon'],
            'diff_20run_ci95_low': st20['ci95_low'], 'diff_20run_ci95_high': st20['ci95_high'],
            'p_t_20run': st20['p_t'], 'p_wilcoxon_20run': st20['p_wilcoxon'],
            'j_cp_0025_mean': float(grp['j_cp_0025'].mean()), 'j_cp_0025_sd': float(grp['j_cp_0025'].std(ddof=1)),
            'fail_cp_0025_pct': float(grp['fail_cp_0025_pct'].mean()),
            'diff_0025_mean': st5_0025['mean_difference'],
            'diff_0025_5seed_ci95_low': st5_0025['ci95_low'], 'diff_0025_5seed_ci95_high': st5_0025['ci95_high'],
            'p_t_0025_5seed': st5_0025['p_t'], 'p_t_0025_20run': st20_0025['p_t']
        })
    df_h1_sum = pd.DataFrame(h1_sum_rows)
    df_h1_sum.to_csv(f'{out_dir}/tuning_sample_scarcity_summary.csv', index=False)

    df_h2 = pd.DataFrame(h2_rows)
    df_h2.to_csv(f'{out_dir}/commissioning_to_field_per_seed.csv', index=False)
    h2_sum_rows = []
    for code, name_p in [
        ('erm', 'Cost-Tuned Margin Only (ERM on Commissioning D_tune, n=30)'),
        ('b7', 'B7: Smooth-Cost Tuning (Commissioning D_tune, n=30)'),
        ('erm_field', 'Equal-Budget Field-Retuned ERM (40 Field Units, Grid [0,6])'),
        ('b7_field', 'Equal-Budget Field-Retuned B7 (40 Field Units, Smooth-Cost)'),
        ('cyc', 'B5: Field Cycle-Level Residual CP (40 Field Units, alpha=0.05)'),
        ('pure_cp', 'Pure Field Unit-CP (40 Field Units, No Tune Floor, alpha=0.05)'),
        ('b9', 'B9: DC-PdM Static (Commissioning Tune + 40 Field Unit-CP Floor)'),
        ('b10', 'B10: DC-PdM Online (Commissioning Tune + Field Event-ACI)'),
    ]:
        d20_vs_b7 = df_h2[f'j_{code}'] - df_h2['j_b7']
        st20 = core._paired_stats(d20_vs_b7) if code != 'b7' else {'mean_difference': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'p_t': np.nan, 'p_wilcoxon': np.nan}
        st5 = core._paired_stats(d20_vs_b7.groupby(df_h2['seed']).mean()) if code != 'b7' else {'ci95_low': 0.0, 'ci95_high': 0.0, 'p_t': np.nan}

        d20_vs_ferm = df_h2[f'j_{code}'] - df_h2['j_erm_field']
        st20_ferm = core._paired_stats(d20_vs_ferm) if code != 'erm_field' else {'mean_difference': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'p_t': np.nan}
        st5_ferm = core._paired_stats(d20_vs_ferm.groupby(df_h2['seed']).mean()) if code != 'erm_field' else {'ci95_low': 0.0, 'ci95_high': 0.0, 'p_t': np.nan}

        h2_sum_rows.append({
            'policy': name_p,
            'j_life_mean': float(df_h2[f'j_{code}'].mean()),
            'j_life_sd': float(df_h2[f'j_{code}'].std(ddof=1)),
            'fail_pct_mean': float(df_h2[f'fail_{code}_pct'].mean()),
            'waste_cycles_mean': float(df_h2[f'waste_{code}'].mean()),
            'diff_vs_comm_b7_mean': st20['mean_difference'],
            'diff_vs_comm_b7_5seed_ci95_low': st5['ci95_low'],
            'diff_vs_comm_b7_5seed_ci95_high': st5['ci95_high'],
            'p_t_5seed_vs_comm_b7': st5['p_t'],
            'diff_vs_comm_b7_20run_ci95_low': st20['ci95_low'],
            'diff_vs_comm_b7_20run_ci95_high': st20['ci95_high'],
            'p_t_20run_vs_comm_b7': st20['p_t'],
            'diff_vs_field_erm_mean': st20_ferm['mean_difference'],
            'diff_vs_field_erm_5seed_ci95_low': st5_ferm['ci95_low'],
            'diff_vs_field_erm_5seed_ci95_high': st5_ferm['ci95_high'],
            'p_t_5seed_vs_field_erm': st5_ferm['p_t'],
            'p_t_20run_vs_field_erm': st20_ferm['p_t']
        })
    df_h2_sum = pd.DataFrame(h2_sum_rows)
    df_h2_sum.to_csv(f'{out_dir}/commissioning_to_field_summary.csv', index=False)
    return df_h1_sum, df_h2_sum


def run_all(out_dir):
    """Portable entry point invoked by run_experiments.py (Stage 7/7) or standalone."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    core._C_PAVA_LIB = False
    with threadpool_limits(limits=1):
        df_nist_sum, apr_p = evaluate_nist_physical_benchmarks(str(out_path))
        df_h1_sum, df_h2_sum = evaluate_economic_gain_regimes(str(out_path))
    return df_nist_sum, apr_p, df_h1_sum, df_h2_sum


if __name__ == '__main__':
    import time
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '数据'),
        help='Output directory for CSV records'
    )
    args = ap.parse_args()
    t_start = time.perf_counter()
    run_all(args.output)
    print(f"Completed in {time.perf_counter() - t_start:.2f}s!")
