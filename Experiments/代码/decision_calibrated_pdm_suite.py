#!/usr/bin/env python3
"""
Complete End-to-End Reproducible Experimental, Attribution & Audit Suite (Revision 4)
Manuscript Title (EAAI Target):
  "Unit-Level Calibration of Lead-Time Predictive Maintenance with Causal Prognostic States"

Closes all Third-Round Review items (R3-01 through R3-19):
- R3-01 & R3-04: 4-way margin attribution (Cycle-CP, Pure Unit-CP [Binary CRC], Cost-Tuned Margin Only,
  and Cost-Tuned + Unit-CP Floor) on 250 S-FD004 trajectories (`margin_controls_summary.csv`) AND 1,000
  4-scenario trajectories, plus margin floor activity (`attribution_margin_floor_activity.csv`).
- R3-02: Progressively increasing prediction-bias stress test comparing Original Static (alpha=0.05),
  Matched-Initial Static (alpha=0.0325 frozen), and Online Event-ACI (alpha_0=0.0325, unified gamma=0.035,
  bounds [0.005, 0.20]), with per-seed export (`drift_scaling_per_seed.csv`) and 5-seed paired t/Wilcoxon tests.
- R3-03: Cache-free deterministic execution (`--recompute-all` default) with SHA-256 input/output manifest,
  synchronized fresh Table 2 (`forensic_d0_d5_summary.csv`), and byte-identical loose vs. ZIP CSVs.
- R3-05 & R3-14 & R3-17: True single-module removal for A2 (`A2_static` vs `A1` and `A2_online` vs `A0`),
  pre- vs. post-processing residual analysis (`loss_head_pre_post_residuals.csv`), 5-seed paired 95% t-CIs
  (`attribution_paired_tests.csv`), and 5-seed-block vs. 20-run paired tests (`main_paired_tests_with_seed_blocks.csv`).
- R3-06 & R3-07: Strictly exchangeable zero-drift benchmark (`exchangeable_zero_drift_summary.csv`) where
  Proposition 1 holds exactly, and algebraic/numerical verification (`unit_score_vs_binary_crc.csv`) that
  unit-score quantile calibration equals Binary Unit-Loss Conformal Risk Control (Angelopoulos et al., 2024).
- R3-08: Accurate baseline naming (`B3: Smooth One-Sided Pinball + Residual CP`, `B5: Cycle-Level One-Sided
  Residual CP`, `B7: In-House Smooth-Cost Margin Tuning`).
- R3-10: Exact Moradi et al. (2023) HI metrics (strict signed-difference monotonicity, inter-trajectory
  resampled pairwise trendability, prognosability), counterexample verification (`hi_metric_counterexamples.csv`),
  and separation of prefix-PAVA endpoint vs. post-running-maximum state.
- R3-11: Named-configuration CPU timing profile separating single-step streaming latency (p50/p95 us),
  amortized batch time (us/cycle), measured trajectory replay (ms/unit, no fixed 0.195 multiplier), and
  complete end-to-end offline Fit+Tune+Cal wall-clock time including encoder fitting/transform.
- R3-12: Verified non-monotone physical recovery trajectories (`non_monotone_latent_check.csv`, 100% of units
  exhibit negative latent degradation increments during recovery) and recovery-segment evaluation.
- R3-13: End-to-end support for `margin=None` backup in `evaluate_fleet` and `replay_unit_stopping_policy`
  with exact physical event semantics `Z_fail = 1{ell >= T_u}` when `tau = 0` (tested for T_u=10, ell=15),
  and explicit piecewise vs. C2-smoothed NASA loss derivative verification at r=0.
"""

import os
import sys
for _env_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env_var] = '1'

import time
import json
import math
import ctypes
import hashlib
import shutil
import zipfile
import platform
import subprocess
import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Noto Sans'],
    'font.size': 9.5,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.2,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linestyle': '--'
})

PALETTE = {
    'navy': '#1B365D',
    'red': '#C0392B',
    'teal': '#2A9D8F',
    'gold': '#D4A017',
    'slate': '#6C757D',
    'purple': '#6D597A',
    'orange': '#E76F51',
    'light_bg': '#F8F9FA'
}

SEEDS = [11, 22, 33, 44, 55]
SCENARIOS = ['S-FD001', 'S-FD002', 'S-FD003', 'S-FD004']
SENSORS = [f's{i}' for i in range(1, 15)]
LEAD_TIME = 15.0
EPSILON = 0.05
ONLINE_INIT_FACTOR = 0.65
ONLINE_ALPHA_TARGET = EPSILON * ONLINE_INIT_FACTOR  # 0.0325
ONLINE_GAMMA = 0.035
ONLINE_ALPHA_MIN = 0.005
ONLINE_ALPHA_MAX = 0.20
RHO_DEFAULT = 10.0
C_WASTE_PER_CYC = 0.045

OUT_DIR = 'artifacts/file_generation/ttl=63d/output'
INTER_DIR = 'artifacts/file_generation/ttl=63d/intermediate'


# ==============================================================================
# 0. C-ACCELERATED STACK PAVA WITH PURE-PYTHON FALLBACK
# ==============================================================================
_C_PAVA_LIB = None


def _init_c_pava():
    global _C_PAVA_LIB
    if _C_PAVA_LIB is not None:
        return _C_PAVA_LIB
    so_path = '/tmp/dc_pdm_pava.so'
    c_path = '/tmp/dc_pdm_pava.c'
    c_code = r"""
    void prefix_pava_2d(const double* x, double* out, int T, int D) {
        double sums[2048];
        int counts[2048];
        for (int j = 0; j < D; ++j) {
            int top = 0;
            for (int t = 0; t < T; ++t) {
                sums[top] = x[t * D + j];
                counts[top] = 1;
                top++;
                while (top >= 2 && (sums[top - 2] / counts[top - 2]) > (sums[top - 1] / counts[top - 1])) {
                    sums[top - 2] += sums[top - 1];
                    counts[top - 2] += counts[top - 1];
                    top--;
                }
                out[t * D + j] = sums[top - 1] / counts[top - 1];
            }
        }
    }
    void full_pava_2d(const double* x, double* out, int T, int D) {
        double sums[2048];
        int counts[2048];
        for (int j = 0; j < D; ++j) {
            int top = 0;
            for (int t = 0; t < T; ++t) {
                sums[top] = x[t * D + j];
                counts[top] = 1;
                top++;
                while (top >= 2 && (sums[top - 2] / counts[top - 2]) > (sums[top - 1] / counts[top - 1])) {
                    sums[top - 2] += sums[top - 1];
                    counts[top - 2] += counts[top - 1];
                    top--;
                }
            }
            int idx = 0;
            for (int b = 0; b < top; ++b) {
                double val = sums[b] / counts[b];
                for (int c = 0; c < counts[b]; ++c) {
                    out[idx * D + j] = val;
                    idx++;
                }
            }
        }
    }
    """
    try:
        if not os.path.exists(so_path):
            with open(c_path, 'w') as f:
                f.write(c_code)
            subprocess.run(['gcc', '-O3', '-shared', '-fPIC', c_path, '-o', so_path], check=True)
        lib = ctypes.CDLL(so_path)
        lib.prefix_pava_2d.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
        ]
        lib.full_pava_2d.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
        ]
        _C_PAVA_LIB = lib
    except Exception:
        _C_PAVA_LIB = False
    return _C_PAVA_LIB


def prefix_pava_endpoints(x):
    """Strictly causal O(T) stack-based PAVA returning only the prefix endpoint at each t."""
    sums, counts, endpoints = [], [], np.empty(len(x))
    for t, value in enumerate(x):
        sums.append(float(value))
        counts.append(1)
        while len(sums) >= 2 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            last_sum, last_count = sums.pop(), counts.pop()
            sums[-1] += last_sum
            counts[-1] += last_count
        endpoints[t] = sums[-1] / counts[-1]
    return endpoints


def prefix_pava_matrix(mat):
    """Applies causal prefix PAVA column-by-column to a 2D array (T, D)."""
    lib = _init_c_pava()
    if lib:
        mat_c = np.ascontiguousarray(mat, dtype=np.float64)
        out = np.empty_like(mat_c)
        T, D = mat_c.shape
        lib.prefix_pava_2d(
            mat_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            T, D
        )
        return out
    return np.column_stack([prefix_pava_endpoints(mat[:, j]) for j in range(mat.shape[1])])


def full_pava_matrix(mat):
    """Applies non-causal full-trajectory PAVA column-by-column (used ONLY in D0 forensic audit)."""
    lib = _init_c_pava()
    if lib:
        mat_c = np.ascontiguousarray(mat, dtype=np.float64)
        out = np.empty_like(mat_c)
        T, D = mat_c.shape
        lib.full_pava_2d(
            mat_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            T, D
        )
        return out
    out = np.empty_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        y = mat[:, j]
        blocks = [[float(y[i]), 1] for i in range(len(y))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
                blocks[i][0] += blocks[i + 1][0]
                blocks[i][1] += blocks[i + 1][1]
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        idx = 0
        for total, count in blocks:
            out[idx:idx + count, j] = total / count
            idx += count
    return out


# ==============================================================================
# 1. SYNTHETIC MULTI-REGIME DEGRADATION GENERATORS
# ==============================================================================
def generate_original_protocol_dataset(dataset_id='FD001', n_units=90, seed=42):
    """Exact generator from original PAC-DFL script for forensic D0-D5 audit."""
    rng = np.random.RandomState(seed)
    base_id = dataset_id.replace('S-', '')
    configs = {
        'FD001': {'n_regimes': 1, 'n_modes': 1, 'noise_std': 0.22, 'drift_rate': 0.08},
        'FD002': {'n_regimes': 6, 'n_modes': 1, 'noise_std': 0.28, 'drift_rate': 0.22},
        'FD003': {'n_regimes': 1, 'n_modes': 2, 'noise_std': 0.25, 'drift_rate': 0.15},
        'FD004': {'n_regimes': 6, 'n_modes': 2, 'noise_std': 0.32, 'drift_rate': 0.28},
    }
    cfg = configs[base_id]
    n_sensors = 14
    regime_centers = rng.uniform(-1.5, 1.5, size=(cfg['n_regimes'], 3))
    regime_gains = rng.uniform(0.75, 1.35, size=(cfg['n_regimes'], n_sensors))
    regime_offsets = rng.uniform(-1.8, 1.8, size=(cfg['n_regimes'], n_sensors))
    sensor_loadings_m1 = rng.uniform(0.6, 1.4, size=(n_sensors,)) * np.array(
        [1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
    )
    sensor_loadings_m2 = rng.uniform(0.5, 1.3, size=(n_sensors,)) * np.array(
        [-1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
    )
    records = []
    rul_cap = 125.0
    for unit_id in range(1, n_units + 1):
        fleet_progress = max(0.0, (unit_id - 0.65 * n_units) / (0.35 * n_units))
        T_life = int(rng.uniform(150, 260) * (1.0 - 0.14 * cfg['drift_rate'] * fleet_progress))
        t_grid = np.arange(1, T_life + 1, dtype=float)
        true_rul_raw = T_life - t_grid
        true_rul = np.minimum(true_rul_raw, rul_cap)
        onset = int(rng.uniform(0.22 * T_life, 0.40 * T_life))
        tau = np.maximum(0.0, (t_grid - onset) / max(1.0, (T_life - onset)))
        alpha_wear = rng.uniform(1.75, 2.35)
        weibull_shocks = np.cumsum(rng.gamma(shape=1.2, scale=0.004, size=T_life))
        h1 = (tau ** alpha_wear) + 0.40 * np.expm1(1.35 * tau) + weibull_shocks
        h1 = h1 / max(1e-6, h1[-1])
        if cfg['n_modes'] == 2 and (unit_id % 2 == 0):
            h2 = (tau ** (alpha_wear * 0.85)) + 0.45 * (tau ** 3)
            h2 = h2 / max(1e-6, h2[-1])
            mode_flag = 1
        else:
            h2 = np.zeros_like(h1)
            mode_flag = 0
        if cfg['n_regimes'] > 1:
            regimes = rng.randint(0, cfg['n_regimes'], size=T_life)
        else:
            regimes = np.zeros(T_life, dtype=int)
        op_cond = regime_centers[regimes] + rng.normal(0, 0.04, size=(T_life, 3))
        sensors = np.zeros((T_life, n_sensors))
        dip_shocks = (rng.rand(T_life) < 0.14).astype(float) * rng.exponential(0.50, size=T_life)
        for s in range(n_sensors):
            pol = np.sign(sensor_loadings_m1[s])
            base_sig = (
                sensor_loadings_m1[s] * h1
                + 0.60 * sensor_loadings_m2[s] * h2
                - pol * dip_shocks * (0.35 + 0.65 * tau)
            )
            reg_mod = regime_gains[regimes, s] * base_sig + regime_offsets[regimes, s]
            noise = rng.normal(0, cfg['noise_std'], size=T_life)
            sensors[:, s] = reg_mod + noise
        unit_df = pd.DataFrame(sensors, columns=SENSORS)
        unit_df['unit_id'] = unit_id
        unit_df['cycle'] = t_grid.astype(int)
        unit_df['max_cycle'] = T_life
        unit_df['regime'] = regimes
        unit_df['fault_mode'] = mode_flag
        unit_df['latent_deg'] = 0.72 * h1 + 0.28 * h2
        unit_df['true_rul_raw'] = true_rul_raw
        unit_df['true_rul'] = true_rul
        unit_df['op1'] = op_cond[:, 0]
        unit_df['op2'] = op_cond[:, 1]
        unit_df['op3'] = op_cond[:, 2]
        records.append(unit_df)
    return pd.concat(records, ignore_index=True)


def generate_synthetic_fleet(
    scenario='S-FD001', n_units=180, seed=42, disturbance_mode='neg_dips',
    custom_drift_rate=None, exchangeable_mode=False
):
    """
    180-unit synthetic fleet generator supporting:
    - Default stress-test mode (exchangeable_mode=False): preserves exact legacy random stream
      for B1-B10 and Experiment F replication.
    - Strictly exchangeable mode (exchangeable_mode=True): sets drift_rate=0.0 and assigns
      fault modes i.i.d. so Fit/Tune/Cal/Test trajectories are strictly exchangeable (Proposition 1).
    - True non-monotone recovery ('non_monotone_true'): creates a verified physical recovery dip
      where latent degradation decreases mid-life before terminal wear resumes.
    """
    rng = np.random.RandomState(seed)
    base_id = scenario.replace('S-', '')
    configs = {
        'FD001': {'n_regimes': 1, 'n_modes': 1, 'noise_std': 0.22, 'drift_rate': 0.08},
        'FD002': {'n_regimes': 6, 'n_modes': 1, 'noise_std': 0.28, 'drift_rate': 0.20},
        'FD003': {'n_regimes': 1, 'n_modes': 2, 'noise_std': 0.25, 'drift_rate': 0.14},
        'FD004': {'n_regimes': 6, 'n_modes': 2, 'noise_std': 0.32, 'drift_rate': 0.25},
    }
    cfg = configs[base_id].copy()
    if exchangeable_mode:
        cfg['drift_rate'] = 0.0
    elif custom_drift_rate is not None:
        cfg['drift_rate'] = float(custom_drift_rate)
    n_sensors = 14
    n_reg_total = cfg['n_regimes'] + (1 if disturbance_mode == 'unseen_regime' else 0)

    regime_centers = rng.uniform(-1.5, 1.5, size=(n_reg_total, 3))
    regime_gains = rng.uniform(0.75, 1.35, size=(n_reg_total, n_sensors))
    regime_offsets = rng.uniform(-1.8, 1.8, size=(n_reg_total, n_sensors))

    # For 'unseen_regime_isolated', keep the base 6-regime RNG stream 100% identical to 'neg_dips'
    # and draw Regime 7 parameters and test-fleet Regime 7 transitions from an independent RNG stream.
    iso_rng = np.random.RandomState(seed + 77777) if disturbance_mode == 'unseen_regime_isolated' else None
    if disturbance_mode == 'unseen_regime_isolated':
        r7_center = iso_rng.uniform(-1.5, 1.5, size=(1, 3))
        r7_gain = iso_rng.uniform(0.75, 1.35, size=(1, n_sensors))
        r7_offset = iso_rng.uniform(-1.8, 1.8, size=(1, n_sensors))
        regime_centers = np.vstack([regime_centers, r7_center])
        regime_gains = np.vstack([regime_gains, r7_gain])
        regime_offsets = np.vstack([regime_offsets, r7_offset])

    sensor_loadings_m1 = rng.uniform(0.6, 1.4, size=(n_sensors,)) * np.array(
        [1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
    )
    sensor_loadings_m2 = rng.uniform(0.5, 1.3, size=(n_sensors,)) * np.array(
        [-1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
    )

    # For exchangeable_mode, use a separate RNG for i.i.d. mode assignment so base parameters stay aligned
    mode_rng = np.random.RandomState(seed + 10000) if exchangeable_mode else None
    records = []
    rul_cap = 125.0

    for unit_id in range(1, n_units + 1):
        fleet_progress = 0.0 if exchangeable_mode else max(0.0, (unit_id - 130.0) / max(1.0, n_units - 130.0))
        T_life = int(rng.uniform(155, 255) * (1.0 - 0.14 * cfg['drift_rate'] * fleet_progress))
        t_grid = np.arange(1, T_life + 1, dtype=float)
        true_rul_raw = T_life - t_grid
        true_rul = np.minimum(true_rul_raw, rul_cap)

        onset = int(rng.uniform(0.22 * T_life, 0.40 * T_life))
        tau = np.maximum(0.0, (t_grid - onset) / max(1.0, (T_life - onset)))
        alpha_wear = rng.uniform(1.75, 2.35)
        gamma_inc = np.cumsum(rng.gamma(shape=1.2, scale=0.004, size=T_life))
        h1 = (tau ** alpha_wear) + 0.40 * np.expm1(1.35 * tau) + gamma_inc
        h1 = h1 / max(1e-6, h1[-1])

        use_m2 = (mode_rng.rand() < 0.5) if (exchangeable_mode and cfg['n_modes'] == 2) else (cfg['n_modes'] == 2 and (unit_id % 2 == 0))
        if use_m2:
            h2 = (tau ** (alpha_wear * 0.85)) + 0.45 * (tau ** 3)
            h2 = h2 / max(1e-6, h2[-1])
            mode_flag = 1
        else:
            h2 = np.zeros_like(h1)
            mode_flag = 0

        if cfg['n_regimes'] > 1:
            if disturbance_mode == 'unseen_regime' and unit_id > 130:
                regimes = rng.randint(0, n_reg_total, size=T_life)
            else:
                regimes = rng.randint(0, cfg['n_regimes'], size=T_life)
                if disturbance_mode == 'unseen_regime_isolated' and unit_id > 130:
                    swap_m = iso_rng.rand(T_life) < (1.0 / 7.0)
                    regimes = np.where(swap_m, cfg['n_regimes'], regimes)
        else:
            regimes = np.zeros(T_life, dtype=int)

        sensors = np.zeros((T_life, n_sensors))
        if disturbance_mode in ('neg_dips', 'unseen_regime', 'unseen_regime_isolated'):
            shocks = -(rng.rand(T_life) < 0.16).astype(float) * rng.exponential(0.55, size=T_life)
        elif disturbance_mode in ('no_shocks', 'non_monotone', 'non_monotone_true'):
            shocks = np.zeros(T_life)
        elif disturbance_mode == 'pos_spikes':
            shocks = +(rng.rand(T_life) < 0.16).astype(float) * rng.exponential(0.55, size=T_life)
        elif disturbance_mode == 'bidirectional':
            shocks = (rng.rand(T_life) < 0.16).astype(float) * rng.standard_t(df=3, size=T_life) * 0.40
        else:
            shocks = np.zeros(T_life)

        # True non-monotone recovery bump: creates genuine negative latent increments in tau in [0.55, 0.72]
        recovery_bump = 0.28 * np.exp(-((tau - 0.68) / 0.08) ** 2)
        non_mono_true_curve = h1 - recovery_bump

        for s in range(n_sensors):
            pol = np.sign(sensor_loadings_m1[s])
            if disturbance_mode == 'non_monotone' and s in [0, 3, 6, 9, 12]:
                non_mono_curve = h1 - 0.55 * np.maximum(0.0, tau - 0.65) ** 1.5
                base_sig = sensor_loadings_m1[s] * non_mono_curve + 0.60 * sensor_loadings_m2[s] * h2
            elif disturbance_mode == 'non_monotone_true':
                base_sig = sensor_loadings_m1[s] * non_mono_true_curve + 0.60 * sensor_loadings_m2[s] * h2
            else:
                base_sig = (
                    sensor_loadings_m1[s] * h1
                    + 0.60 * sensor_loadings_m2[s] * h2
                    + pol * shocks * (0.35 + 0.65 * tau)
                )
            reg_mod = regime_gains[regimes, s] * base_sig + regime_offsets[regimes, s]
            noise = rng.normal(0, cfg['noise_std'], size=T_life)
            sensors[:, s] = reg_mod + noise

        unit_df = pd.DataFrame(sensors, columns=SENSORS)
        unit_df['unit_id'] = unit_id
        unit_df['cycle'] = t_grid.astype(int)
        unit_df['max_cycle'] = T_life
        unit_df['regime'] = regimes
        unit_df['fault_mode'] = mode_flag
        unit_df['latent_deg'] = (0.72 * non_mono_true_curve + 0.28 * h2) if disturbance_mode == 'non_monotone_true' else (0.72 * h1 + 0.28 * h2)
        unit_df['true_rul_raw'] = true_rul_raw
        unit_df['true_rul'] = true_rul
        records.append(unit_df)

    return pd.concat(records, ignore_index=True)


# ==============================================================================
# 2. CAUSAL MONOTONE ENCODER, EXACT HI METRICS & FORENSIC FEATURE EXTRACTOR
# ==============================================================================
def ema_causal(x, gain):
    zi = ((1.0 - gain) * x[0])[None, :]
    return signal.lfilter([gain], [1.0, -(1.0 - gain)], x, axis=0, zi=zi)[0]


def compute_hi_quality_metrics(df_part, hi_vec):
    """
    Computes exact Health Indicator (HI) quality metrics (Moradi et al., 2023 EAAI; Coble & Hines, 2009):
      1. Strict Absolute Monotonicity: mean_u | (#{dh > 1e-12} - #{dh < -1e-12}) / (T_u - 1) |
         (Constant HI -> 0.0; Strictly increasing -> 1.0; Strictly decreasing -> 1.0)
      2. Strict Non-Decreasing Monotonicity: mean_u max(0, (#{dh > 1e-12} - #{dh < -1e-12}) / (T_u - 1))
         (Constant HI -> 0.0; Strictly increasing -> 1.0; Strictly decreasing -> 0.0)
      3. Inter-Trajectory Trendability: Resamples each unit's HI to K=50 normalized lifetime points
         and computes min_{u != v} |corr(HI_u_resampled, HI_v_resampled)| (and mean pairwise |corr|)
         when >= 2 units exist, plus single-unit time correlation |corr(HI_u, t_u)|.
      4. Prognosability: exp( - std(HI_terminal) / mean(|HI_terminal - HI_initial|) )
         (0.0 if mean span < 1e-12 so a constant HI gets 0.0).
    """
    abs_mono_list = []
    pos_mono_list = []
    time_corr_list = []
    term_vals = []
    span_vals = []
    resampled_curves = []
    grid_norm = np.linspace(0.0, 1.0, 50)

    for _, pos in df_part.groupby('unit_id', sort=False).indices.items():
        pos = np.asarray(pos)
        h_u = np.asarray(hi_vec[pos], dtype=float)
        t_u = df_part.iloc[pos].cycle.to_numpy(dtype=float)
        if len(h_u) >= 2:
            dh = np.diff(h_u)
            n_pos = int(np.sum(dh > 1e-12))
            n_neg = int(np.sum(dh < -1e-12))
            signed_m = float(n_pos - n_neg) / float(len(h_u) - 1)
            abs_mono_list.append(abs(signed_m))
            pos_mono_list.append(max(0.0, signed_m))
            if np.std(h_u) > 1e-12 and np.std(t_u) > 1e-12:
                c_ut = float(np.corrcoef(h_u, t_u)[0, 1])
                time_corr_list.append(abs(0.0 if np.isnan(c_ut) else c_ut))
            else:
                time_corr_list.append(0.0)
            term_vals.append(float(h_u[-1]))
            span_vals.append(abs(float(h_u[-1] - h_u[0])))
            u_norm = np.linspace(0.0, 1.0, len(h_u))
            resampled_curves.append(np.interp(grid_norm, u_norm, h_u))

    abs_monotonicity = float(np.mean(abs_mono_list)) if abs_mono_list else 0.0
    pos_monotonicity = float(np.mean(pos_mono_list)) if pos_mono_list else 0.0
    time_trendability = float(np.mean(time_corr_list)) if time_corr_list else 0.0

    if len(resampled_curves) >= 2:
        mat_r = np.vstack(resampled_curves)
        stds = np.std(mat_r, axis=1)
        if np.all(stds > 1e-12):
            corr_m = np.corrcoef(mat_r)
            iu = np.triu_indices(len(resampled_curves), k=1)
            pair_corrs = np.abs(np.nan_to_num(corr_m[iu], nan=0.0))
            inter_trend_min = float(np.min(pair_corrs))
            inter_trend_mean = float(np.mean(pair_corrs))
        else:
            inter_trend_min = 0.0
            inter_trend_mean = 0.0
    else:
        inter_trend_min = time_trendability
        inter_trend_mean = time_trendability

    mean_span = float(np.mean(span_vals)) if span_vals else 0.0
    if mean_span < 1e-12:
        prognosability = 0.0
    else:
        std_term = float(np.std(term_vals)) if len(term_vals) >= 2 else 0.0
        prognosability = float(np.exp(-std_term / mean_span))

    return {
        'monotonicity': abs_monotonicity,
        'pos_monotonicity': pos_monotonicity,
        'trendability': inter_trend_mean,
        'inter_trendability_min': inter_trend_min,
        'time_corr_mean': time_trendability,
        'prognosability': prognosability
    }


def extract_forensic_features(df, causal_prefix=False, fit_on_train_only=False, train_mask=None):
    """Feature extractor supporting single-factor switches between D0, D1, and D2."""
    df_proc = df.copy()
    ref_df = df_proc.loc[train_mask] if (fit_on_train_only and train_mask is not None) else df_proc

    global_early = ref_df[ref_df['cycle'] <= 25]
    g_mu = (global_early[SENSORS] if len(global_early) >= 5 else ref_df[SENSORS]).mean(axis=0)
    g_sd = (global_early[SENSORS] if len(global_early) >= 5 else ref_df[SENSORS]).std(axis=0).replace(0, 1.0)

    reg_stats = {}
    for reg, grp in ref_df.groupby('regime'):
        early_m = grp['cycle'] <= 25
        r_grp = grp[early_m] if early_m.sum() >= 5 else grp
        reg_stats[int(reg)] = (r_grp[SENSORS].mean(axis=0), r_grp[SENSORS].std(axis=0).replace(0, 1.0))

    for reg, grp in df_proc.groupby('regime'):
        mu, sd = reg_stats.get(int(reg), (g_mu, g_sd))
        df_proc.loc[grp.index, SENSORS] = (grp[SENSORS] - mu) / sd

    ref_proc = df_proc.loc[train_mask] if (fit_on_train_only and train_mask is not None) else df_proc
    corrs = np.array([
        np.corrcoef(ref_proc[c].values, ref_proc['cycle'].values)[0, 1]
        for c in SENSORS
    ])
    polarities = np.sign(np.nan_to_num(corrs, nan=1.0))

    n_total = len(df_proc)
    X_all = np.zeros((n_total, 46))
    monotonic_hi = np.zeros(n_total)

    for _, grp in df_proc.groupby('unit_id', sort=False):
        idx = grp.index.values
        S = grp[SENSORS].values * polarities
        T = S.shape[0]

        if not causal_prefix:
            ema_fast = signal.lfilter([0.32], [1.0, -0.68], S, axis=0)
            ema_slow = signal.lfilter([0.10], [1.0, -0.90], S, axis=0)
            ema_fast[0] = S[0]
            ema_slow[0] = S[0]
            hf_diff = S - ema_fast
            spec_energy = signal.lfilter([0.20], [1.0, -0.80], hf_diff ** 2, axis=0)
            running_peak = np.maximum.accumulate(0.65 * ema_slow + 0.35 * ema_fast, axis=0)
            mono_slow = 0.55 * full_pava_matrix(ema_slow) + 0.45 * running_peak
            hi_curve = np.mean(mono_slow, axis=1)
            hi_rate = np.gradient(hi_curve) if T > 1 else np.zeros(T)
            anchor_excess = np.maximum(0.0, hi_curve - np.median(hi_curve)) ** 2
        else:
            ema_fast = ema_causal(S, 0.32)
            ema_slow = ema_causal(S, 0.10)
            hf_diff = S - ema_fast
            spec_energy = ema_causal(hf_diff ** 2, 0.20)
            running_peak = np.maximum.accumulate(0.65 * ema_slow + 0.35 * ema_fast, axis=0)
            mono_slow = 0.55 * prefix_pava_matrix(ema_slow) + 0.45 * running_peak
            hi_curve = np.mean(mono_slow, axis=1)
            hi_rate = np.r_[0.0, np.diff(hi_curve)]
            hi_base = ema_causal(hi_curve[:, None], 0.04)[:, 0]
            anchor_excess = np.maximum(0.0, hi_curve - hi_base) ** 2

        monotonic_hi[idx] = hi_curve
        cycle_norm = grp['cycle'].values / 200.0
        X_all[idx] = np.column_stack([
            mono_slow,
            ema_fast - mono_slow,
            np.log1p(spec_energy),
            cycle_norm,
            hi_rate,
            hi_curve ** 2,
            anchor_excess
        ])

    return X_all, monotonic_hi


class CausalMonotoneEncoder:
    def __init__(self, regime_norm=True, monotone_state=True, enforce_running_max=True):
        self.regime_norm = regime_norm
        self.monotone_state = monotone_state
        self.enforce_running_max = enforce_running_max

    def fit(self, df_train):
        refs = df_train.loc[df_train.cycle <= 25, SENSORS] if self.regime_norm else df_train[SENSORS]
        self.global_mu = refs.mean().to_numpy()
        self.global_sd = np.maximum(refs.std(ddof=1).to_numpy(), 1e-6)
        self.norms = {}
        if self.regime_norm:
            for reg, grp in df_train.groupby('regime'):
                ref = grp.loc[grp.cycle <= 25, SENSORS]
                if len(ref) >= 5:
                    self.norms[int(reg)] = (
                        ref.mean().to_numpy(),
                        np.maximum(ref.std(ddof=1).to_numpy(), 1e-6)
                    )
        whitened = self.whiten(df_train)
        corrs = [np.corrcoef(whitened[:, j], df_train.cycle)[0, 1] for j in range(14)]
        self.polarities = np.where(np.nan_to_num(corrs) < 0, -1.0, 1.0)
        return self

    def whiten(self, df):
        x = df[SENSORS].to_numpy().copy()
        regimes = df.regime.to_numpy()
        for reg in np.unique(regimes):
            mask = regimes == reg
            mu, sd = self.norms.get(int(reg), (self.global_mu, self.global_sd))
            x[mask] = (x[mask] - mu) / sd
        return x

    def transform(self, df, return_pava_pre_accum=False):
        values = self.whiten(df) * self.polarities
        n_cols = 60 if self.monotone_state else 46
        features = np.zeros((len(df), n_cols))
        hi_out = np.zeros(len(df))
        pava_pre_out = np.zeros(len(df))

        for _, positions in df.groupby('unit_id', sort=False).indices.items():
            positions = np.asarray(positions)
            cycles = df.iloc[positions].cycle.to_numpy()
            s = values[positions]

            fast_raw = ema_causal(s, 0.32)
            slow_raw = ema_causal(s, 0.10)
            energy = ema_causal((s - fast_raw) ** 2, 0.20)

            if self.monotone_state:
                s_clipped = np.maximum(s, slow_raw - 0.55)
                slow_damped = ema_causal(s_clipped, 0.10)
                wear_exc = np.maximum(0.0, slow_damped - 0.20)
                pava_end = prefix_pava_matrix(wear_exc)
                pre_accum = np.mean(0.60 * pava_end + 0.40 * wear_exc, axis=1)
                mono_h = np.maximum.accumulate(pre_accum) if self.enforce_running_max else pre_accum.copy()
                rate = np.r_[0.0, np.diff(mono_h)]
                features[positions] = np.column_stack([
                    slow_raw,
                    pava_end,
                    fast_raw - slow_raw,
                    np.log1p(energy),
                    cycles / 200.0,
                    rate,
                    mono_h ** 2,
                    np.maximum(0.0, mono_h - 0.8) ** 2
                ])
                hi_out[positions] = mono_h
                pava_pre_out[positions] = pre_accum
            else:
                h_raw = np.mean(slow_raw, axis=1)
                rate = np.r_[0.0, np.diff(h_raw)]
                features[positions] = np.column_stack([
                    slow_raw,
                    fast_raw - slow_raw,
                    np.log1p(energy),
                    cycles / 200.0,
                    rate,
                    h_raw ** 2,
                    np.maximum(0.0, h_raw - 0.8) ** 2
                ])
                hi_out[positions] = h_raw
                pava_pre_out[positions] = h_raw

        if return_pava_pre_accum:
            return features, hi_out, pava_pre_out
        return features, hi_out


# ==============================================================================
# 3. REGRESSION LOSS SURROGATES & PROGNOSTIC HEADS
# ==============================================================================
def safe_exp_value_derivative(x):
    x = np.asarray(x)
    z = np.maximum(0.0, x - 30.0)
    base = np.exp(np.minimum(x, 30.0))
    return base * (1.0 + z + 0.5 * z * z), base * (1.0 + z)


def surrogate_value_gradient(w, xb, y, kind='linex', rho=10.0, reg=10.0, sw=None):
    """
    Evaluates objective and gradient for:
      - 'ridge': C2 quadratic loss
      - 'linex': C2 centered Linex loss
      - 'quantile': C2 softplus-smoothed pinball loss (tau=0.05, delta=0.5)
      - 'nasa': Piecewise NASA exponential loss (C1-discontinuous at r=0: left deriv -1/13, right deriv +1/10)
    """
    r = xb @ w - y
    n = len(y)
    if sw is None:
        sw = np.ones(n)
    if kind == 'ridge':
        loss = 0.5 * sw * (r ** 2)
        dr = sw * r
    elif kind == 'linex':
        a = min(0.065, 0.020 * np.log1p(rho))
        lam = 3.5 * np.log1p(rho)
        val, deriv = safe_exp_value_derivative(a * r)
        loss = sw * (0.5 * r * r + lam * (val - a * r - 1.0))
        dr = sw * (r + lam * a * (deriv - 1.0))
    elif kind == 'nasa':
        scale = np.where(r < 0, -1.0 / 13.0, 1.0 / 10.0)
        val, deriv = safe_exp_value_derivative(scale * r)
        loss = sw * (val - 1.0)
        dr = sw * deriv * scale
        dr[r == 0] = 0.0
    elif kind == 'quantile':
        tau, delta = 0.05, 0.5
        u = -r
        loss = sw * (tau * u + delta * np.logaddexp(0.0, -u / delta))
        dr = sw * (expit(-u / delta) - tau)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    obj = float(loss.mean() + 0.5 * reg / n * np.dot(w[1:], w[1:]))
    grad = (xb.T @ dr) / n
    grad[1:] += (reg / n) * w[1:]
    return obj, grad


class PrognosticHead:
    def fit(self, x, y, kind='ridge', rho=10.0, reg=14.0, focus_window=False, ell=LEAD_TIME):
        self.mu = x.mean(axis=0)
        self.sd = np.maximum(x.std(axis=0), 1e-6)
        xb = self.design(x)
        if focus_window:
            sw = 1.0 + 2.2 * expit((ell + 30.0 - y) / 10.0)
            sw = sw / np.mean(sw)
        else:
            sw = np.ones(len(y))

        W_col = sw[:, None]
        penalty = reg * np.eye(xb.shape[1])
        penalty[0, 0] = 0.0
        self.w = np.linalg.solve(xb.T @ (W_col * xb) + penalty, xb.T @ (sw * y))

        if kind != 'ridge':
            res = optimize.minimize(
                surrogate_value_gradient, self.w, args=(xb, y, kind, rho, reg, sw),
                jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-5}
            )
            self.w = res.x
        return self

    def design(self, x):
        return np.column_stack([np.ones(len(x)), (x - self.mu) / self.sd])

    def predict(self, x, df_part=None, causal_smooth=False):
        raw_p = np.clip(self.design(x) @ self.w, 0.0, 125.0)
        if not causal_smooth or df_part is None:
            return raw_p
        out_p = np.zeros_like(raw_p)
        for _, pos in df_part.groupby('unit_id', sort=False).indices.items():
            pos = np.asarray(pos)
            rp = raw_p[pos]
            sp = np.zeros_like(rp)
            for t in range(len(rp)):
                if t == 0:
                    sp[t] = rp[0]
                else:
                    step_prior = max(0.0, sp[t - 1] - 0.90)
                    sp[t] = 0.62 * rp[t] + 0.38 * step_prior
            out_p[pos] = np.clip(sp, 0.0, 125.0)
        return out_p


# ==============================================================================
# 4. UNIT-LEVEL CRITICAL SCORE CALIBRATION & STOPPING POLICY REPLAY
# ==============================================================================
def unit_critical_score(cycles, point, scale, failure_time, lead_time=LEAD_TIME):
    eligible = (cycles + lead_time) < failure_time
    if not np.any(eligible):
        return np.inf
    return float(np.min((point[eligible] - lead_time) / scale[eligible]))


def choose_static_margin(cal_scores, alpha=EPSILON, tuned_margin=0.0):
    scores = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n or not np.isfinite(scores[rank - 1]):
        return None, {'rank': rank, 'n': n, 'backup': True}
    q = float(scores[rank - 1])
    return float(max(0.0, tuned_margin, q)), {'rank': rank, 'n': n, 'q': q, 'backup': False}


def replay_unit_stopping_policy(cycles, point, scale, T_u, ell=LEAD_TIME, margin=0.0):
    """
    Replays the lead-time stopping policy on a single unit trajectory.
    If margin is None (backup triggered when k > n or B_(k) = +inf), an immediate
    preventive replacement order is issued at t = 0 (requiring lead time ell):
      - If ell < T_u: preventive replacement completes at U_u = ell (Z_fail = 0, W_u = T_u - ell).
      - If ell >= T_u: unit fails at U_u = T_u before lead time ell completes (Z_fail = 1, W_u = 0).
    """
    if margin is None or (isinstance(margin, float) and np.isnan(margin)):
        tau = 0.0
        z_fail = 1 if (tau + ell >= T_u) else 0
        u_life = float(min(T_u, tau + ell))
        w_life = float(max(0.0, T_u - (tau + ell))) if z_fail == 0 else 0.0
        return {'tau': tau, 'Z_fail': z_fail, 'U_u': u_life, 'W_u': w_life, 'backup': 1}
    alarm_idx = np.flatnonzero((point - float(margin) * scale) <= ell)
    tau = float(cycles[alarm_idx[0]]) if len(alarm_idx) > 0 else np.inf
    z_fail = 1 if (tau + ell >= T_u) else 0
    u_life = float(min(T_u, tau + ell))
    w_life = float(max(0.0, T_u - (tau + ell))) if z_fail == 0 else 0.0
    return {'tau': tau, 'Z_fail': z_fail, 'U_u': u_life, 'W_u': w_life, 'backup': 0}


def evaluate_fleet(df_part, point, scale, margin_per_unit, ell=LEAD_TIME, rho=RHO_DEFAULT, c_waste=C_WASTE_PER_CYC):
    c_pm = 1.0
    c_cm = rho * c_pm
    rows = []
    indices_map = df_part.groupby('unit_id', sort=False).indices
    for k_idx, (uid, pos) in enumerate(indices_map.items()):
        pos = np.asarray(pos)
        cycles = df_part.iloc[pos].cycle.to_numpy()
        T_u = float(df_part.iloc[pos].max_cycle.iloc[0])
        if isinstance(margin_per_unit, (list, np.ndarray)):
            raw_m = margin_per_unit[k_idx]
            m_u = None if raw_m is None else float(raw_m)
        else:
            m_u = None if margin_per_unit is None else float(margin_per_unit)
        out = replay_unit_stopping_policy(cycles, point[pos], scale[pos], T_u, ell, m_u)
        c_ev = c_pm * (1 - out['Z_fail']) + c_cm * out['Z_fail']
        c_tot = c_ev + c_waste * out['W_u']
        term_err = float(point[pos][-1] - df_part.iloc[pos].true_rul.iloc[-1])
        rows.append({
            'unit_id': int(uid), 'T_u': T_u, 'tau': out['tau'],
            'Z_fail': out['Z_fail'], 'U_u': out['U_u'], 'W_u': out['W_u'],
            'c_ev': c_ev, 'c_tot': c_tot, 'margin': np.nan if m_u is None else m_u,
            'terminal_rul_err': term_err, 'backup': out['backup']
        })
    udf = pd.DataFrame(rows)
    sum_u = max(1.0, udf['U_u'].sum())
    return {
        'j_life_rate': float(100.0 * udf['c_tot'].sum() / sum_u),
        'j_event_rate': float(100.0 * udf['c_ev'].sum() / sum_u),
        'unit_cost': float(udf['c_tot'].mean()),
        'fail_pct': float(100.0 * udf['Z_fail'].mean()),
        'avail_pct': float(100.0 * sum_u / udf['T_u'].sum()),
        'waste_cycles': float(udf['W_u'].mean()),
        'n_units': len(udf),
        'n_fails': int(udf['Z_fail'].sum()),
        'unit_df': udf
    }


def tune_differentiable_policy_margin(df_tune, point_tune, scale_tune, ell=LEAD_TIME, rho=RHO_DEFAULT, eps=EPSILON, c_waste=C_WASTE_PER_CYC):
    """In-house smooth-cost margin tuning over a 53-point grid on D_tune."""
    indices_map = df_tune.groupby('unit_id', sort=False).indices
    kappa = 0.50
    c_pm, c_cm = 1.0, rho * 1.0

    def smooth_obj(lam_val):
        tot_c, tot_u, tot_f = 0.0, 0.0, 0.0
        for _, pos in indices_map.items():
            pos = np.asarray(pos)
            cycles = df_tune.iloc[pos].cycle.to_numpy()
            T_u = float(df_tune.iloc[pos].max_cycle.iloc[0])
            g_t = point_tune[pos] - lam_val * scale_tune[pos]
            h_t = expit(kappa * (ell - g_t))
            log_surv = np.log(np.maximum(1e-12, 1.0 - h_t))
            cum_surv = np.r_[0.0, np.cumsum(log_surv)[:-1]]
            p_stop = h_t * np.exp(cum_surv)

            safe_m = (cycles + ell) < T_u
            p_pm = float(np.sum(p_stop[safe_m]))
            p_cm = max(0.0, 1.0 - p_pm)
            u_exp = float(np.sum(p_stop[safe_m] * (cycles[safe_m] + ell))) + p_cm * T_u
            w_exp = float(np.sum(p_stop[safe_m] * np.maximum(0.0, T_u - (cycles[safe_m] + ell))))
            c_exp = c_pm * p_pm + c_cm * p_cm + c_waste * w_exp
            tot_c += c_exp
            tot_u += max(ell, u_exp)
            tot_f += p_cm
        mean_f = tot_f / len(indices_map)
        pen = 80.0 * max(0.0, mean_f - eps * 0.70) ** 2
        return (100.0 * tot_c / tot_u) + pen

    grid = np.linspace(0.2, 2.8, 53)
    objs = [smooth_obj(g) for g in grid]
    return float(grid[int(np.argmin(objs))])


# ==============================================================================
# 5. SELF-TESTS & COUNTEREXAMPLE AUDITS (`self_test_report.json`)
# ==============================================================================
def run_all_self_tests(out_dir=OUT_DIR):
    rng = np.random.default_rng(17)
    df = generate_synthetic_fleet('S-FD004', n_units=30, seed=7)
    df_tr = df[df.unit_id <= 15].copy()
    df_u = df[df.unit_id == 20].copy()

    enc = CausalMonotoneEncoder(regime_norm=True, monotone_state=True).fit(df_tr)
    full_x, full_h = enc.transform(df_u)
    pref_x, pref_h = enc.transform(df_u.iloc[:70].copy())
    max_pref_diff = float(np.max(np.abs(full_x[:70] - pref_x)))
    assert max_pref_diff < 1e-12
    assert np.all(np.diff(full_h) >= -1e-12)

    # Verify prefix PAVA endpoint alone is NOT monotone for [2.0, 0.0] -> [2.0, 1.0]
    pava_2_0 = prefix_pava_endpoints([2.0, 0.0]).tolist()
    assert np.allclose(pava_2_0, [2.0, 1.0])

    perturbed = df_u.copy()
    perturbed.loc[perturbed.index[70:], SENSORS] += 1000.0
    pert_x, _ = enc.transform(perturbed)
    max_pert_diff = float(np.max(np.abs(pert_x[:70] - full_x[:70])))
    assert max_pert_diff < 1e-12

    xb = np.column_stack([np.ones(35), rng.normal(size=(35, 6))])
    y = rng.normal(size=35)
    w = rng.normal(size=7)
    grad_errs = {}
    for k_loss in ['ridge', 'linex', 'nasa', 'quantile']:
        err = float(optimize.check_grad(
            lambda ww: surrogate_value_gradient(ww, xb, y, kind=k_loss)[0],
            lambda ww: surrogate_value_gradient(ww, xb, y, kind=k_loss)[1],
            w
        ))
        assert err < 1e-4
        grad_errs[k_loss] = err

    # Check piecewise NASA derivative kink at r = 0
    xx_1 = np.ones((1, 1))
    h_step = 1e-7
    f_nasa_0 = lambda v: surrogate_value_gradient(np.array([v]), xx_1, np.array([0.0]), kind='nasa', reg=0.0)[0]
    nasa_left_deriv = float((f_nasa_0(0.0) - f_nasa_0(-h_step)) / h_step)
    nasa_right_deriv = float((f_nasa_0(h_step) - f_nasa_0(0.0)) / h_step)

    t_arr = np.arange(1, 121, dtype=float)
    p_arr = 121.0 - t_arr + rng.normal(0, 3.0, size=120)
    s_arr = np.ones(120) * 4.0
    B_val = unit_critical_score(t_arr, p_arr, s_arr, 120.0, lead_time=15.0)
    for lam_test in np.linspace(0.0, 5.0, 51):
        z_ev = replay_unit_stopping_policy(t_arr, p_arr, s_arr, 120.0, 15.0, lam_test)['Z_fail']
        assert z_ev <= int(B_val > lam_test)

    # Backup policy checks (including evaluate_fleet(margin=None) and T_u=10 < ell=15)
    m18, info18 = choose_static_margin(np.arange(18, dtype=float), alpha=0.05)
    m19, info19 = choose_static_margin(np.arange(19, dtype=float), alpha=0.05)
    m40, info40 = choose_static_margin(np.arange(40, dtype=float), alpha=0.05)
    assert m18 is None and info18['backup'] is True
    assert m19 == 18.0 and info19['backup'] is False
    assert m40 == 38.0 and info40['rank'] == 39

    b_t10 = replay_unit_stopping_policy(np.arange(1, 11), np.ones(10), np.ones(10), 10.0, 15.0, None)
    b_t30 = replay_unit_stopping_policy(np.arange(1, 31), np.ones(30), np.ones(30), 30.0, 15.0, None)
    assert b_t10['Z_fail'] == 1 and b_t10['U_u'] == 10.0 and b_t10['backup'] == 1
    assert b_t30['Z_fail'] == 0 and b_t30['U_u'] == 15.0 and b_t30['W_u'] == 15.0 and b_t30['backup'] == 1
    ev_none = evaluate_fleet(
        pd.DataFrame({'unit_id': [1, 1], 'cycle': [1, 2], 'max_cycle': [10, 10], 'true_rul': [9, 8]}),
        np.ones(2), np.ones(2), None, ell=15.0
    )
    assert ev_none['n_fails'] == 1 and ev_none['fail_pct'] == 100.0

    # HI metric counterexample verification (constant -> 0.0, increasing -> 1.0, decreasing -> 1.0 abs / 0.0 pos)
    hi_ce_rows = []
    for lbl, vec in [('constant', np.ones(8)), ('increasing', np.arange(8.0)), ('decreasing', -np.arange(8.0))]:
        df_ce = pd.DataFrame({'unit_id': np.ones(8, dtype=int), 'cycle': np.arange(1, 9)})
        m_ce = compute_hi_quality_metrics(df_ce, vec)
        hi_ce_rows.append({'example': lbl, **m_ce})
    df_hi_ce = pd.DataFrame(hi_ce_rows)
    df_hi_ce.to_csv(f'{out_dir}/hi_metric_counterexamples.csv', index=False)
    assert df_hi_ce.loc[df_hi_ce.example == 'constant', 'monotonicity'].iloc[0] == 0.0
    assert df_hi_ce.loc[df_hi_ce.example == 'constant', 'prognosability'].iloc[0] == 0.0
    assert df_hi_ce.loc[df_hi_ce.example == 'increasing', 'monotonicity'].iloc[0] == 1.0
    assert df_hi_ce.loc[df_hi_ce.example == 'decreasing', 'pos_monotonicity'].iloc[0] == 0.0

    # Binary Unit-Loss Conformal Risk Control (CRC) equivalence verification across (n, epsilon)
    crc_rows = []
    for n_val in [19, 20, 40, 60, 100]:
        b_scores = np.sort(np.random.default_rng(n_val).normal(size=n_val))
        for eps_val in [0.05, 0.10, 0.20]:
            k_rk = int(math.ceil((n_val + 1) * (1.0 - eps_val)))
            if k_rk > n_val:
                continue
            q_rk = float(b_scores[k_rk - 1])
            passing = [float(x) for x in b_scores if (np.sum(b_scores > x) + 1.0) / (n_val + 1.0) <= eps_val + 1e-14]
            crc_q = min(passing) if passing else np.inf
            eq_flag = bool(np.isclose(q_rk, crc_q))
            assert eq_flag
            crc_rows.append({
                'n': n_val, 'epsilon': eps_val, 'rank': k_rk,
                'unit_score_quantile': q_rk, 'binary_crc_threshold': crc_q, 'equal': eq_flag
            })
    pd.DataFrame(crc_rows).to_csv(f'{out_dir}/unit_score_vs_binary_crc.csv', index=False)

    target, gamma, a_min, a_max, T_hor = 0.04, 0.065, 0.002, 0.18, 1000
    a_cur, corr_sum = target, 0.0
    for _ in range(T_hor):
        unproj = a_cur + gamma * (target - 1.0)
        proj = float(np.clip(unproj, a_min, a_max))
        corr_sum += (proj - unproj)
        a_cur = proj
    exact_lhs = 1.0 - target
    exact_rhs = (target - a_cur + corr_sum) / (gamma * T_hor)
    naive_rhs = (a_max - a_min) / (gamma * T_hor)
    assert abs(exact_lhs - exact_rhs) < 1e-12

    report = {
        'causal_prefix_invariance': True,
        'max_prefix_diff': max_pref_diff,
        'future_perturbation_invariance': True,
        'max_perturbation_diff': max_pert_diff,
        'prefix_pava_endpoint_2_0_example': pava_2_0,
        'monotone_health_index_via_running_max': True,
        'gradient_errors_away_from_zero': grad_errs,
        'max_gradient_error': float(max(grad_errs.values())),
        'nasa_piecewise_left_right_deriv_at_zero': [nasa_left_deriv, nasa_right_deriv],
        'unit_score_event_inclusion': True,
        'unit_score_equals_binary_crc': True,
        'n18_fallback_verified': True,
        'backup_T10_ell15_verified': b_t10,
        'backup_T30_ell15_verified': b_t30,
        'hi_counterexamples_verified': True,
        'n40_rank39_verified': True,
        'projected_identity_verified': True,
        'projected_identity_residual': float(abs(exact_lhs - exact_rhs)),
        'sum_d_t': float(corr_sum),
        'naive_unprojected_bound_violation_ratio': float(exact_lhs / naive_rhs)
    }
    with open(f'{out_dir}/self_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    return report


# ==============================================================================
# 6. EXPERIMENT A: CONTROLLED SINGLE-FACTOR FORENSIC AUDIT (D0–D5)
# ==============================================================================
def eval_cycle_proxy_cost(y_true, L_bound, rho=10.0, c_early=0.35, mult=10.0):
    c_fail = rho * c_early * mult
    err = L_bound - y_true
    unsafe = err > 0.0
    cost = np.where(unsafe, c_fail * (1.0 + 0.22 * err), c_early * np.abs(err))
    return float(np.mean(cost)), float(np.mean(unsafe) * 100.0)


def fit_original_linex(X, y, rho=10.0, l2_reg=6.0):
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    I = np.eye(d + 1)
    I[0, 0] = 0.0
    w0 = np.linalg.solve(Xb.T @ Xb + l2_reg * I, Xb.T @ y)
    alpha_linex = min(0.095, 0.028 * np.log1p(rho))
    lam_linex = 14.0 * np.log1p(rho)

    def loss_and_grad(w):
        r = Xb @ w - y
        val, deriv = safe_exp_value_derivative(alpha_linex * r)
        quad_loss = 0.5 * (r ** 2)
        linex_loss = lam_linex * (val - alpha_linex * r - 1.0)
        reg_val = 0.5 * l2_reg * np.sum(w[1:] ** 2) / n
        total_loss = np.mean(quad_loss + linex_loss) + reg_val
        d_r = r + lam_linex * alpha_linex * (deriv - 1.0)
        grad_w = (Xb.T @ d_r) / n
        grad_w[1:] += (l2_reg / n) * w[1:]
        return float(total_loss), grad_w

    res = optimize.minimize(loss_and_grad, w0, method='L-BFGS-B', jac=True, options={'maxiter': 80, 'ftol': 1e-6})
    return res.x


def calibrate_forensic_da_acrc(cal_y, cal_pred, test_y, test_pred, cal_hi, test_hi, test_df,
                               feedback_mode='instant', delay_D=25, alpha_target=0.05, rho=10.0, gamma_lr=0.065):
    hi_min, hi_max = np.percentile(cal_hi, 5), np.percentile(cal_hi, 95)
    cal_hi_norm = np.clip((cal_hi - hi_min) / max(1e-6, hi_max - hi_min), 0.0, 1.0)
    test_hi_norm = np.clip((test_hi - hi_min) / max(1e-6, hi_max - hi_min), 0.0, 1.0)

    pos_res = np.maximum(1.0, np.abs(cal_pred - cal_y))
    slope, intercept = np.polyfit(cal_hi_norm, pos_res, 1)
    raw_cal_scale = np.clip(intercept + slope * cal_hi_norm, 4.0, 25.0)
    raw_test_scale = np.clip(intercept + slope * test_hi_norm, 4.0, 25.0)
    mean_s = float(np.mean(raw_cal_scale))
    cal_scale = raw_cal_scale / mean_s
    test_scale = raw_test_scale / mean_s

    cal_scores = np.sort((cal_pred - cal_y) / cal_scale)
    n_cal = len(cal_scores)
    p_grid = np.linspace(0.0, 1.0, n_cal)

    best_alpha_star = float(np.clip(0.28 / (rho + 2.0), 0.010, alpha_target * 0.75))
    best_cost = np.inf
    for a_try in np.linspace(0.008, alpha_target * 0.85, 25):
        q_try = float(np.interp(1.0 - a_try, p_grid, cal_scores))
        L_try = np.maximum(0.0, cal_pred - q_try * cal_scale)
        c_try, f_try = eval_cycle_proxy_cost(cal_y, L_try, rho=rho, mult=10.0)
        if f_try <= alpha_target * 100.0 and c_try < best_cost:
            best_cost = c_try
            best_alpha_star = float(a_try)

    n_test = len(test_y)
    lower_bounds = np.zeros(n_test)
    alpha_t = best_alpha_star

    if feedback_mode in ('instant', 'static', 'delayed'):
        errs = np.zeros(n_test)
        for t in range(n_test):
            q_curr = float(np.interp(np.clip(1.0 - alpha_t, 0.50, 0.998), p_grid, cal_scores))
            L_t = max(0.0, test_pred[t] - max(0.0, q_curr * test_scale[t]))
            lower_bounds[t] = L_t
            errs[t] = 1.0 if (L_t > test_y[t]) else 0.0
            if feedback_mode == 'instant':
                alpha_t = np.clip(alpha_t + gamma_lr * (best_alpha_star - errs[t]), 0.002, 0.18)
            elif feedback_mode == 'delayed' and t >= delay_D:
                alpha_t = np.clip(alpha_t + gamma_lr * (best_alpha_star - errs[t - delay_D]), 0.002, 0.18)
    elif feedback_mode == 'unit_retire':
        pos_cursor = 0
        for _, grp in test_df.groupby('unit_id', sort=False):
            m_len = len(grp)
            sl = slice(pos_cursor, pos_cursor + m_len)
            q_curr = float(np.interp(np.clip(1.0 - alpha_t, 0.50, 0.998), p_grid, cal_scores))
            L_u = np.maximum(0.0, test_pred[sl] - np.maximum(0.0, q_curr * test_scale[sl]))
            lower_bounds[sl] = L_u
            u_err_rate = float(np.mean(L_u > test_y[sl]))
            alpha_t = np.clip(alpha_t + gamma_lr * (best_alpha_star - u_err_rate), 0.002, 0.18)
            pos_cursor += m_len

    return lower_bounds


def run_forensic_d0_d5(out_dir=OUT_DIR):
    rows = []
    for sc in SCENARIOS:
        base_ds = sc.replace('S-', '')
        for sd in SEEDS:
            df = generate_original_protocol_dataset(base_ds, n_units=90, seed=sd)
            units = df['unit_id'].unique()
            n_train = int(0.55 * len(units))
            n_cal = int(0.20 * len(units))
            tr_m = df['unit_id'].isin(units[:n_train]).values
            cal_m = df['unit_id'].isin(units[n_train:n_train + n_cal]).values
            te_m = df['unit_id'].isin(units[n_train + n_cal:]).values
            te_df = df.loc[te_m].copy()

            y_tr = df.loc[tr_m, 'true_rul'].values
            y_cal = df.loc[cal_m, 'true_rul'].values
            y_te = df.loc[te_m, 'true_rul'].values

            X_d0, hi_d0 = extract_forensic_features(df, causal_prefix=False, fit_on_train_only=False)
            X_d1, hi_d1 = extract_forensic_features(df, causal_prefix=True, fit_on_train_only=False)
            X_d2, hi_d2 = extract_forensic_features(df, causal_prefix=True, fit_on_train_only=True, train_mask=tr_m)

            fitted_cache = {}
            for key_f, X_use in [('d0', X_d0), ('d1', X_d1), ('d2', X_d2)]:
                w = fit_original_linex(X_use[tr_m], y_tr, rho=10.0, l2_reg=6.0)
                p_cal = np.clip(np.column_stack([np.ones(cal_m.sum()), X_use[cal_m]]) @ w, 0, 125)
                p_te = np.clip(np.column_stack([np.ones(te_m.sum()), X_use[te_m]]) @ w, 0, 125)
                bias_c = np.mean(p_cal - y_cal)
                pt_cal = np.clip(p_cal - 0.92 * bias_c, 0, 125)
                pt_te = np.clip(p_te - 0.92 * bias_c, 0, 125)
                rmse = float(np.sqrt(np.mean((pt_te - y_te) ** 2)))
                d_err = pt_te - y_te
                nasa_uncapped = float(np.mean(np.where(d_err < 0, np.expm1(-d_err / 13.0), np.expm1(d_err / 10.0))))
                fitted_cache[key_f] = (pt_cal, pt_te, rmse, nasa_uncapped)

            for stage_code, key_f, hi_use, fb_mode in [
                ('D0: Original Protocol (Non-Causal + Transductive + Instant Oracle)', 'd0', hi_d0, 'instant'),
                ('D1: Strictly Causal Prefix Only (Instant Oracle Retained)', 'd1', hi_d1, 'instant'),
                ('D2: D1 + Train-Only Preprocessing (Instant Oracle Retained)', 'd2', hi_d2, 'instant'),
                ('D3a: D2 + Fixed 25-Cycle Delayed Feedback (No Instant Oracle)', 'd2', hi_d2, 'delayed'),
                ('D3b: D2 + Unit-Retirement Feedback (Active Oracle Removed)', 'd2', hi_d2, 'unit_retire'),
                ('D3c: D2 + Static Frozen Calibration (Zero Online Update)', 'd2', hi_d2, 'static'),
            ]:
                pt_cal, pt_te, rmse, nasa_uncapped = fitted_cache[key_f]
                L_b = calibrate_forensic_da_acrc(
                    y_cal, pt_cal, y_te, pt_te, hi_use[cal_m], hi_use[te_m], te_df, feedback_mode=fb_mode
                )
                c100, v_pct = eval_cycle_proxy_cost(y_te, L_b, rho=10.0, mult=10.0)
                c10, _ = eval_cycle_proxy_cost(y_te, L_b, rho=10.0, mult=1.0)
                rows.append({
                    'scenario': sc, 'seed': sd, 'stage': stage_code,
                    'rmse': rmse, 'nasa_uncapped': nasa_uncapped,
                    'cost_eff100': c100, 'cost_direct10': c10, 'cycle_viol_pct': v_pct
                })

            n_tr, d_tr = X_d2[tr_m].shape
            Xb_tr = np.column_stack([np.ones(n_tr), X_d2[tr_m]])
            I_mat = np.eye(d_tr + 1)
            I_mat[0, 0] = 0.0
            w_r = np.linalg.solve(Xb_tr.T @ Xb_tr + 8.0 * I_mat, Xb_tr.T @ y_tr)
            pr_cal = np.clip(np.column_stack([np.ones(cal_m.sum()), X_d2[cal_m]]) @ w_r, 0, 125)
            pr_te = np.clip(np.column_stack([np.ones(te_m.sum()), X_d2[te_m]]) @ w_r, 0, 125)
            rmse_r = float(np.sqrt(np.mean((pr_te - y_te) ** 2)))
            dr_err = pr_te - y_te
            nasa_r = float(np.mean(np.where(dr_err < 0, np.expm1(-dr_err / 13.0), np.expm1(dr_err / 10.0))))

            q_m1 = float(np.quantile(np.abs(pr_cal - y_cal), 0.95))
            L_m1 = np.maximum(0.0, pr_te - q_m1)
            c100_m1, v_m1 = eval_cycle_proxy_cost(y_te, L_m1, rho=10.0, mult=10.0)
            c10_m1, _ = eval_cycle_proxy_cost(y_te, L_m1, rho=10.0, mult=1.0)

            abs_res_cal = np.abs(pr_cal - y_cal)
            a_m3 = 0.05
            L_m3 = np.zeros_like(pr_te)
            cur = 0
            for _, grp in te_df.groupby('unit_id', sort=False):
                m_len = len(grp)
                sl = slice(cur, cur + m_len)
                q_c = float(np.quantile(abs_res_cal, np.clip(1.0 - a_m3, 0.50, 0.995)))
                L_m3[sl] = np.maximum(0.0, pr_te[sl] - q_c)
                err_u = float(np.mean(np.abs(pr_te[sl] - y_te[sl]) > q_c))
                a_m3 = np.clip(a_m3 + 0.05 * (0.05 - err_u), 0.005, 0.20)
                cur += m_len
            c100_m3, v_m3 = eval_cycle_proxy_cost(y_te, L_m3, rho=10.0, mult=10.0)
            c10_m3, _ = eval_cycle_proxy_cost(y_te, L_m3, rho=10.0, mult=1.0)

            L_m5 = calibrate_forensic_da_acrc(
                y_cal, pr_cal, y_te, pr_te, hi_d2[cal_m], hi_d2[te_m], te_df, feedback_mode='unit_retire'
            )
            c100_m5, v_m5 = eval_cycle_proxy_cost(y_te, L_m5, rho=10.0, mult=10.0)
            c10_m5, _ = eval_cycle_proxy_cost(y_te, L_m5, rho=10.0, mult=1.0)

            for st_name, c100_v, c10_v, vp in [
                ('D5 (M1): Matched Causal-TMS + Ridge + Two-Sided Split CP', c100_m1, c10_m1, v_m1),
                ('D5 (M3): Matched Causal-TMS + Ridge + Two-Sided Retire-ACI', c100_m3, c10_m3, v_m3),
                ('D5 (M5): Matched Causal-TMS + Ridge + Hazard DA-ACRC (Retire)', c100_m5, c10_m5, v_m5),
            ]:
                rows.append({
                    'scenario': sc, 'seed': sd, 'stage': st_name,
                    'rmse': rmse_r, 'nasa_uncapped': nasa_r,
                    'cost_eff100': c100_v, 'cost_direct10': c10_v, 'cycle_viol_pct': vp
                })

    df_d0_d5 = pd.DataFrame(rows)
    df_d0_d5.to_csv(f'{out_dir}/forensic_d0_d5_per_seed.csv', index=False)

    summary_d0_d5 = df_d0_d5.groupby('stage', sort=False).agg(
        rmse_mean=('rmse', 'mean'),
        rmse_sd=('rmse', 'std'),
        nasa_mean=('nasa_uncapped', 'mean'),
        cost_eff100_mean=('cost_eff100', 'mean'),
        cost_eff100_sd=('cost_eff100', 'std'),
        cost_direct10_mean=('cost_direct10', 'mean'),
        cost_direct10_sd=('cost_direct10', 'std'),
        viol_pct_mean=('cycle_viol_pct', 'mean'),
        viol_pct_sd=('cycle_viol_pct', 'std')
    ).reset_index()
    summary_d0_d5.to_csv(f'{out_dir}/forensic_d0_d5_summary.csv', index=False)
    return df_d0_d5, summary_d0_d5


# ==============================================================================
# 7. EXPERIMENT B & C: 1,000-UNIT POLICY BENCHMARK (B1–B10) & EXCHANGEABLE BENCHMARK
# ==============================================================================
def _paired_stats(diff_vec):
    diff_vec = np.asarray(diff_vec, dtype=float)
    n = len(diff_vec)
    se = float(diff_vec.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1) * se)
    mean_d = float(diff_vec.mean())
    p_t = float(stats.ttest_1samp(diff_vec, 0.0).pvalue)
    p_w = float(stats.wilcoxon(diff_vec).pvalue) if np.any(diff_vec != 0) else 1.0
    return {
        'n': n, 'mean_difference': mean_d,
        'ci95_low': mean_d - half, 'ci95_high': mean_d + half,
        'p_t': p_t, 'p_wilcoxon': p_w
    }


def run_unit_policy_benchmark_b1_b10(out_dir=OUT_DIR):
    seed_rows = []
    unit_rows = []
    sample_replay_data = None
    fd004_cache = {}
    exch_rows = []

    for sc in SCENARIOS:
        for sd in SEEDS:
            df = generate_synthetic_fleet(sc, n_units=180, seed=sd, disturbance_mode='neg_dips')
            m_fit = (df.unit_id <= 60).to_numpy()
            m_tune = ((df.unit_id > 60) & (df.unit_id <= 90)).to_numpy()
            m_cal = ((df.unit_id > 90) & (df.unit_id <= 130)).to_numpy()
            m_test = (df.unit_id > 130).to_numpy()

            df_fit, df_tune = df.loc[m_fit].copy(), df.loc[m_tune].copy()
            df_cal, df_test = df.loc[m_cal].copy(), df.loc[m_test].copy()

            enc_mono = CausalMonotoneEncoder(regime_norm=True, monotone_state=True).fit(df_fit)
            X_fit, _ = enc_mono.transform(df_fit)
            X_tune, _ = enc_mono.transform(df_tune)
            X_cal, _ = enc_mono.transform(df_cal)
            X_test, _ = enc_mono.transform(df_test)

            y_fit = df_fit.true_rul.to_numpy()
            y_tune = df_tune.true_rul.to_numpy()
            y_cal = df_cal.true_rul.to_numpy()
            y_test = df_test.true_rul.to_numpy()

            h_ridge = PrognosticHead().fit(X_fit, y_fit, kind='ridge', reg=14.0)
            h_qr = PrognosticHead().fit(X_fit, y_fit, kind='quantile', reg=14.0)
            h_nasa = PrognosticHead().fit(X_fit, y_fit, kind='nasa', reg=14.0)
            h_dc = PrognosticHead().fit(X_fit, y_fit, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)

            pr_tune = h_ridge.predict(X_tune)
            pr_cal = h_ridge.predict(X_cal)
            pr_test = h_ridge.predict(X_test)

            pqr_cal = h_qr.predict(X_cal)
            pqr_test = h_qr.predict(X_test)

            pnasa_cal = h_nasa.predict(X_cal)
            pnasa_test = h_nasa.predict(X_test)

            pdc_tune = h_dc.predict(X_tune, df_tune, causal_smooth=True)
            pdc_cal = h_dc.predict(X_cal, df_cal, causal_smooth=True)
            pdc_test = h_dc.predict(X_test, df_test, causal_smooth=True)
            dc_b = float(np.mean(pdc_tune - y_tune))
            pdc_tune = np.clip(pdc_tune - 0.85 * dc_b, 0.0, 125.0)
            pdc_cal = np.clip(pdc_cal - 0.85 * dc_b, 0.0, 125.0)
            pdc_test = np.clip(pdc_test - 0.85 * dc_b, 0.0, 125.0)

            sd_ridge = float(np.std(pr_tune - y_tune))
            sc_ridge_t = np.full(len(df_tune), sd_ridge)
            sc_ridge_c = np.full(len(df_cal), sd_ridge)
            sc_ridge_te = np.full(len(df_test), sd_ridge)

            sd_dc = float(np.std(pdc_tune - y_tune))
            sc_dc_t = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdc_tune) / 9.0))
            sc_dc_c = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdc_cal) / 9.0))
            sc_dc_te = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdc_test) / 9.0))

            if sc == 'S-FD004':
                fd004_cache[sd] = (
                    df, df_fit, df_tune, df_cal, df_test,
                    X_fit, X_tune, X_cal, X_test,
                    y_fit, y_tune, y_cal, y_test,
                    pr_tune, pr_cal, pr_test,
                    pdc_tune, pdc_cal, pdc_test,
                    sc_ridge_c, sc_ridge_te, sc_dc_t, sc_dc_c, sc_dc_te
                )

            m_b1 = 0.0
            grid_b2 = np.linspace(0.0, 3.0, 31)
            costs_b2 = [evaluate_fleet(df_tune, pr_tune, sc_ridge_t, g)['j_life_rate'] for g in grid_b2]
            m_b2 = float(grid_b2[int(np.argmin(costs_b2))])

            cqr_scores = np.sort(pqr_cal - y_cal)
            k_cqr = math.ceil((len(cqr_scores) + 1) * (1.0 - EPSILON))
            q_cqr = float(cqr_scores[min(len(cqr_scores) - 1, k_cqr - 1)])

            nasa_scores = np.sort(np.abs(pnasa_cal - y_cal))
            k_nasa = math.ceil((len(nasa_scores) + 1) * (1.0 - EPSILON))
            q_nasa = float(nasa_scores[min(len(nasa_scores) - 1, k_nasa - 1)])

            rc_scores = np.sort(pr_cal - y_cal)
            k_rc = math.ceil((len(rc_scores) + 1) * (1.0 - EPSILON))
            q_rc = max(0.0, float(rc_scores[min(len(rc_scores) - 1, k_rc - 1)]))

            a_aci = EPSILON
            margins_b6 = []
            for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
                pos = np.asarray(pos)
                q_u = max(0.0, float(np.quantile(rc_scores, np.clip(1.0 - a_aci, 0.50, 0.995))))
                margins_b6.append(q_u)
                out_u = replay_unit_stopping_policy(
                    df_test.iloc[pos].cycle.to_numpy(), pr_test[pos], np.ones(len(pos)),
                    float(df_test.iloc[pos].max_cycle.iloc[0]), LEAD_TIME, q_u
                )
                obs_m = df_test.iloc[pos].cycle.to_numpy() <= out_u['U_u']
                err_u = float(np.mean((pr_test[pos][obs_m] - q_u) > y_test[pos][obs_m]))
                a_aci = float(np.clip(a_aci + 0.04 * (EPSILON - err_u), 0.005, 0.20))

            lam_dfpdm = tune_differentiable_policy_margin(df_tune, pdc_tune, sc_dc_t)

            B_cal_ridge = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pr_cal[np.asarray(pos)], sc_ridge_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b8, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=0.0)

            B_cal_dc = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pdc_cal[np.asarray(pos)], sc_dc_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b9, _ = choose_static_margin(B_cal_dc, alpha=EPSILON, tuned_margin=lam_dfpdm)

            B_sorted_dc = np.sort(np.asarray(B_cal_dc, dtype=float))
            a_dc = ONLINE_ALPHA_TARGET
            margins_b10 = []
            for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
                pos = np.asarray(pos)
                rk = min(len(B_sorted_dc), max(1, math.ceil((len(B_sorted_dc) + 1) * (1.0 - np.clip(a_dc, 0.005, 0.25)))))
                q_dc_u = float(B_sorted_dc[rk - 1])
                lam_u = max(0.0, lam_dfpdm, q_dc_u)
                margins_b10.append(lam_u)
                out_u = replay_unit_stopping_policy(
                    df_test.iloc[pos].cycle.to_numpy(), pdc_test[pos], sc_dc_te[pos],
                    float(df_test.iloc[pos].max_cycle.iloc[0]), LEAD_TIME, lam_u
                )
                a_dc = float(np.clip(a_dc + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - out_u['Z_fail']), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))

            if sc == 'S-FD004' and sd == 33:
                sample_replay_data = (df_test, pdc_test, sc_dc_te, lam_b9, pr_test, q_rc)

            ones_te = np.ones(len(df_test))
            method_specs = [
                ('B1: Uncalibrated Point Threshold', pr_test, ones_te, m_b1),
                ('B2: Cost-Tuned PTO Margin', pr_test, sc_ridge_te, m_b2),
                ('B3: Smooth One-Sided Pinball (0.05) + Residual CP', pqr_test, ones_te, q_cqr),
                ('B4: Piecewise NASA-Loss + Split CP', pnasa_test, ones_te, q_nasa),
                ('B5: Cycle-Level One-Sided Residual CP', pr_test, ones_te, q_rc),
                ('B6: Cycle-Level Online ACI', pr_test, ones_te, margins_b6),
                ('B7: In-House Smooth-Cost Margin Tuning (Uncal.)', pdc_test, sc_dc_te, lam_dfpdm),
                ('B8: Simple-Combo (Ridge + Pure Unit-CP / Binary CRC)', pr_test, sc_ridge_te, lam_b8),
                ('B9: DC-PdM Static (Cost-Tuned + Unit-CP Floor)', pdc_test, sc_dc_te, lam_b9),
                ('B10: DC-PdM Online (Event-ACI Z_u)', pdc_test, sc_dc_te, margins_b10),
            ]

            for m_name, pt_arr, sc_arr, marg in method_specs:
                ev = evaluate_fleet(df_test, pt_arr, sc_arr, marg, ell=LEAD_TIME, rho=RHO_DEFAULT)
                rmse_val = float(np.sqrt(np.mean((pt_arr - y_test) ** 2)))
                seed_rows.append({
                    'scenario': sc, 'seed': sd, 'method': m_name,
                    'rmse': rmse_val,
                    'j_life_rate': ev['j_life_rate'],
                    'j_event_rate': ev['j_event_rate'],
                    'unit_cost': ev['unit_cost'],
                    'fail_pct': ev['fail_pct'],
                    'n_fails': ev['n_fails'],
                    'n_units': ev['n_units'],
                    'avail_pct': ev['avail_pct'],
                    'waste_cycles': ev['waste_cycles']
                })
                udf = ev['unit_df'].copy()
                udf.insert(0, 'method', m_name)
                udf.insert(0, 'seed', sd)
                udf.insert(0, 'scenario', sc)
                unit_rows.append(udf)

            # Strictly Exchangeable Zero-Drift Protocol (`exchangeable_mode=True`)
            df_ex = generate_synthetic_fleet(sc, n_units=180, seed=sd, disturbance_mode='neg_dips', exchangeable_mode=True)
            df_ex_fit = df_ex[df_ex.unit_id <= 60].copy()
            df_ex_tune = df_ex[(df_ex.unit_id > 60) & (df_ex.unit_id <= 90)].copy()
            df_ex_cal = df_ex[(df_ex.unit_id > 90) & (df_ex.unit_id <= 130)].copy()
            df_ex_test = df_ex[df_ex.unit_id > 130].copy()

            enc_ex = CausalMonotoneEncoder(True, True).fit(df_ex_fit)
            Xf_ex, _ = enc_ex.transform(df_ex_fit)
            Xt_ex, _ = enc_ex.transform(df_ex_tune)
            Xc_ex, _ = enc_ex.transform(df_ex_cal)
            Xte_ex, _ = enc_ex.transform(df_ex_test)

            yf_ex = df_ex_fit.true_rul.to_numpy()
            yt_ex = df_ex_tune.true_rul.to_numpy()
            yc_ex = df_ex_cal.true_rul.to_numpy()
            yte_ex = df_ex_test.true_rul.to_numpy()

            hr_ex = PrognosticHead().fit(Xf_ex, yf_ex, kind='ridge', reg=14.0)
            hdc_ex = PrognosticHead().fit(Xf_ex, yf_ex, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)

            prt_ex, prc_ex, prte_ex = hr_ex.predict(Xt_ex), hr_ex.predict(Xc_ex), hr_ex.predict(Xte_ex)
            pdt_ex = hdc_ex.predict(Xt_ex, df_ex_tune, True)
            pdc_ex = hdc_ex.predict(Xc_ex, df_ex_cal, True)
            pdte_ex = hdc_ex.predict(Xte_ex, df_ex_test, True)
            b_ex = float(np.mean(pdt_ex - yt_ex))
            pdt_ex = np.clip(pdt_ex - 0.85 * b_ex, 0.0, 125.0)
            pdc_ex = np.clip(pdc_ex - 0.85 * b_ex, 0.0, 125.0)
            pdte_ex = np.clip(pdte_ex - 0.85 * b_ex, 0.0, 125.0)

            sdr_ex = float(np.std(prt_ex - yt_ex))
            scr_c_ex = np.full(len(df_ex_cal), sdr_ex)
            scr_te_ex = np.full(len(df_ex_test), sdr_ex)

            sddc_ex = float(np.std(pdt_ex - yt_ex))
            scdc_t_ex = sddc_ex * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdt_ex) / 9.0))
            scdc_c_ex = sddc_ex * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdc_ex) / 9.0))
            scdc_te_ex = sddc_ex * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pdte_ex) / 9.0))

            B_r_ex = [
                unit_critical_score(df_ex_cal.iloc[np.asarray(pos)].cycle.to_numpy(), prc_ex[np.asarray(pos)], scr_c_ex[np.asarray(pos)], float(df_ex_cal.iloc[np.asarray(pos)].max_cycle.iloc[0]))
                for _, pos in df_ex_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_r_pure_ex, _ = choose_static_margin(B_r_ex, alpha=EPSILON, tuned_margin=0.0)

            B_dc_ex = [
                unit_critical_score(df_ex_cal.iloc[np.asarray(pos)].cycle.to_numpy(), pdc_ex[np.asarray(pos)], scdc_c_ex[np.asarray(pos)], float(df_ex_cal.iloc[np.asarray(pos)].max_cycle.iloc[0]))
                for _, pos in df_ex_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_dc_pure_ex, _ = choose_static_margin(B_dc_ex, alpha=EPSILON, tuned_margin=0.0)
            lam_tu_ex = tune_differentiable_policy_margin(df_ex_tune, pdt_ex, scdc_t_ex)
            lam_dc_floor_ex, _ = choose_static_margin(B_dc_ex, alpha=EPSILON, tuned_margin=lam_tu_ex)

            for ex_lbl, pt_e, sc_e, m_e in [
                ('Exchangeable B8: Ridge + Pure Unit-CP (Prop. 1 Certified)', prte_ex, scr_te_ex, lam_r_pure_ex),
                ('Exchangeable Pure Unit-CP (Linex, No Tune Floor, Prop. 1 Certified)', pdte_ex, scdc_te_ex, lam_dc_pure_ex),
                ('Exchangeable B7: Cost-Tuned Margin Only (Uncal.)', pdte_ex, scdc_te_ex, lam_tu_ex),
                ('Exchangeable B9: DC-PdM Static (Cost-Tuned + Unit-CP Floor, Prop. 1 Certified)', pdte_ex, scdc_te_ex, lam_dc_floor_ex),
            ]:
                ev_ex = evaluate_fleet(df_ex_test, pt_e, sc_e, m_e)
                exch_rows.append({
                    'scenario': sc, 'seed': sd, 'policy': ex_lbl,
                    'j_life_rate': ev_ex['j_life_rate'], 'fail_pct': ev_ex['fail_pct'],
                    'n_fails': ev_ex['n_fails'], 'n_units': ev_ex['n_units'],
                    'avail_pct': ev_ex['avail_pct'], 'waste_cycles': ev_ex['waste_cycles']
                })

    df_per_unit_long = pd.concat(unit_rows, ignore_index=True)
    df_per_seed = pd.DataFrame(seed_rows)

    df_per_unit_long.to_csv(f'{out_dir}/unit_policy_b1_b10_per_unit_long.csv', index=False)
    df_per_seed.to_csv(f'{out_dir}/unit_policy_b1_b10_per_seed.csv', index=False)

    # Deterministic wide-format `unit_policy_b1_b10_per_unit.csv` (1,000 rows x 30 cols)
    # Note: if tau == +inf (no alarm before T_u), encode as -1 so `tau < 0 -> +inf` is unambiguous.
    method_short = {
        'B1: Uncalibrated Point Threshold': 'B1',
        'B2: Cost-Tuned PTO Margin': 'B2',
        'B3: Smooth One-Sided Pinball (0.05) + Residual CP': 'B3',
        'B4: Piecewise NASA-Loss + Split CP': 'B4',
        'B5: Cycle-Level One-Sided Residual CP': 'B5',
        'B6: Cycle-Level Online ACI': 'B6',
        'B7: In-House Smooth-Cost Margin Tuning (Uncal.)': 'B7',
        'B8: Simple-Combo (Ridge + Pure Unit-CP / Binary CRC)': 'B8',
        'B9: DC-PdM Static (Cost-Tuned + Unit-CP Floor)': 'B9',
        'B10: DC-PdM Online (Event-ACI Z_u)': 'B10',
    }
    df_per_unit_long['m_code'] = df_per_unit_long['method'].map(method_short)
    base_u = df_per_unit_long[df_per_unit_long['m_code'] == 'B1'][['scenario', 'seed', 'unit_id', 'T_u']].copy()
    base_u = base_u.set_index(['scenario', 'seed', 'unit_id'])
    for m_c in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        sub_m = df_per_unit_long[df_per_unit_long['m_code'] == m_c].set_index(['scenario', 'seed', 'unit_id'])
        base_u[f'{m_c}_tau'] = np.where(np.isfinite(sub_m['tau']), sub_m['tau'], -1).astype(int)
    for m_c in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        sub_m = df_per_unit_long[df_per_unit_long['m_code'] == m_c].set_index(['scenario', 'seed', 'unit_id'])
        base_u[f'{m_c}_fail'] = sub_m['Z_fail'].astype(int)
    for m_c in ['B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        sub_m = df_per_unit_long[df_per_unit_long['m_code'] == m_c].set_index(['scenario', 'seed', 'unit_id'])
        base_u[f'{m_c}_cost'] = sub_m['c_tot'].round(6)
    df_per_unit_wide = base_u.reset_index()
    df_per_unit_wide.to_csv(f'{out_dir}/unit_policy_b1_b10_per_unit.csv', index=False)

    # Save strictly exchangeable zero-drift summary
    df_exch = pd.DataFrame(exch_rows)
    df_exch.to_csv(f'{out_dir}/exchangeable_zero_drift_per_seed.csv', index=False)
    exch_sum = []
    for p_name, grp in df_exch.groupby('policy', sort=False):
        kf = int(grp['n_fails'].sum())
        nt = int(grp['n_units'].sum())
        ucb = float(stats.beta.ppf(0.95, kf + 1, nt - kf) * 100.0) if kf < nt else 100.0
        exch_sum.append({
            'policy': p_name,
            'j_life_mean': float(grp['j_life_rate'].mean()),
            'j_life_sd': float(grp['j_life_rate'].std(ddof=1)),
            'total_fails': kf, 'total_units': nt,
            'fail_pct': float(100.0 * kf / nt),
            'clopper_pearson_ucb95_pct': ucb,
            'avail_pct_mean': float(grp['avail_pct'].mean()),
            'waste_cycles_mean': float(grp['waste_cycles'].mean())
        })
    pd.DataFrame(exch_sum).to_csv(f'{out_dir}/exchangeable_zero_drift_summary.csv', index=False)

    # Paired tests across both 20 scenario-seed runs and 5 independent seed blocks
    piv = df_per_seed.pivot(index=['scenario', 'seed'], columns='method', values='j_life_rate')
    b10_col = 'B10: DC-PdM Online (Event-ACI Z_u)'
    paired_test_rows = []
    comp_methods = [
        'B2: Cost-Tuned PTO Margin',
        'B3: Smooth One-Sided Pinball (0.05) + Residual CP',
        'B5: Cycle-Level One-Sided Residual CP',
        'B6: Cycle-Level Online ACI',
        'B7: In-House Smooth-Cost Margin Tuning (Uncal.)',
        'B8: Simple-Combo (Ridge + Pure Unit-CP / Binary CRC)',
        'B9: DC-PdM Static (Cost-Tuned + Unit-CP Floor)'
    ]
    raw_p_20 = {}
    raw_p_5 = {}
    for cm in comp_methods:
        d20 = piv[cm] - piv[b10_col]
        d5 = d20.groupby('seed').mean()
        st20 = _paired_stats(d20)
        st5 = _paired_stats(d5)
        raw_p_20[cm] = (st20['p_t'], st20['p_wilcoxon'])
        raw_p_5[cm] = (st5['p_t'], st5['p_wilcoxon'])
        m_short = cm.split(':')[0]
        paired_test_rows.append({'comparison': f'{m_short} minus B10', 'aggregation': 'scenario_seed_20', **st20})
        paired_test_rows.append({'comparison': f'{m_short} minus B10', 'aggregation': 'seed_block_5', **st5})
    pd.DataFrame(paired_test_rows).to_csv(f'{out_dir}/main_paired_tests_with_seed_blocks.csv', index=False)

    sorted_cms = sorted(comp_methods[:6], key=lambda m: raw_p_20[m][0])
    holm_map = {}
    running_max = 0.0
    m_total = len(sorted_cms)
    for rank_idx, cm in enumerate(sorted_cms):
        adj_p = min(1.0, raw_p_20[cm][0] * (m_total - rank_idx))
        running_max = max(running_max, adj_p)
        holm_map[cm] = running_max

    summary_list = []
    for m_name, grp in df_per_seed.groupby('method', sort=False):
        k_fails = int(grp['n_fails'].sum())
        n_tot = int(grp['n_units'].sum())
        ucb_95 = float(stats.beta.ppf(0.95, k_fails + 1, n_tot - k_fails) * 100.0) if k_fails < n_tot else 100.0
        p_t20, p_w20 = raw_p_20.get(m_name, (np.nan, np.nan))
        p_t5, p_w5 = raw_p_5.get(m_name, (np.nan, np.nan))
        p_holm_val = holm_map.get(m_name, np.nan)
        summary_list.append({
            'method': m_name,
            'rmse_mean': float(grp['rmse'].mean()),
            'j_life_mean': float(grp['j_life_rate'].mean()),
            'j_life_sd': float(grp['j_life_rate'].std(ddof=1)),
            'j_event_mean': float(grp['j_event_rate'].mean()),
            'total_fails': k_fails,
            'total_units': n_tot,
            'fail_pct': float(100.0 * k_fails / n_tot),
            'clopper_pearson_ucb95_pct': ucb_95,
            'avail_pct_mean': float(grp['avail_pct'].mean()),
            'waste_cycles_mean': float(grp['waste_cycles'].mean()),
            'p_ttest_20_vs_b10': p_t20,
            'p_holm_20_vs_b10': p_holm_val,
            'p_wilcoxon_20_vs_b10': p_w20,
            'p_ttest_5seedblock_vs_b10': p_t5,
            'p_wilcoxon_5seedblock_vs_b10': p_w5
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(f'{out_dir}/unit_policy_b1_b10_summary.csv', index=False)
    return df_per_unit_wide, df_per_seed, df_summary, sample_replay_data, fd004_cache


# ==============================================================================
# 8. EXPERIMENT D & E: RIGHT-CENSORING, RHO SENSITIVITY & MISMATCH BOUNDARIES
# ==============================================================================
def run_censoring_and_mismatch_experiments(fd004_cache, out_dir=OUT_DIR):
    censor_unit_rows = []
    rho_rows = []
    rhos = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]

    for sd in SEEDS:
        (
            df, df_fit, df_tune, df_cal, df_test,
            X_fit, X_tune, X_cal, X_test,
            y_fit, y_tune, y_cal, y_test,
            pr_tune, pr_cal, pr_test,
            pdc_tune, pdc_cal, pdc_test,
            sc_ridge_c, sc_ridge_te, sc_dc_t, sc_dc_c, sc_dc_te
        ) = fd004_cache[sd]

        rc_scores = np.sort(pr_cal - y_cal)

        for c_mode in ['Cycle-ACI (Oracle Uncensored)', 'Cycle-ACI (Censored Imputation)', 'Cycle-ACI (Uncensored-Only Bias)']:
            a_aci = EPSILON
            margins_m = []
            for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
                pos = np.asarray(pos)
                q_u = max(0.0, float(np.quantile(rc_scores, np.clip(1.0 - a_aci, 0.50, 0.995))))
                margins_m.append(q_u)
                cyc_u = df_test.iloc[pos].cycle.to_numpy()
                T_u = float(df_test.iloc[pos].max_cycle.iloc[0])
                out_u = replay_unit_stopping_policy(cyc_u, pr_test[pos], np.ones(len(pos)), T_u, LEAD_TIME, q_u)
                obs_m = cyc_u <= out_u['U_u']
                if c_mode == 'Cycle-ACI (Oracle Uncensored)':
                    err_u = float(np.mean((pr_test[pos][obs_m] - q_u) > y_test[pos][obs_m]))
                    a_aci = float(np.clip(a_aci + 0.04 * (EPSILON - err_u), 0.005, 0.20))
                elif c_mode == 'Cycle-ACI (Censored Imputation)':
                    y_cens = np.minimum(125.0, out_u['U_u'] - cyc_u[obs_m])
                    err_u = float(np.mean((pr_test[pos][obs_m] - q_u) > y_cens))
                    a_aci = float(np.clip(a_aci + 0.04 * (EPSILON - err_u), 0.005, 0.20))
                elif c_mode == 'Cycle-ACI (Uncensored-Only Bias)':
                    if out_u['Z_fail'] == 1:
                        err_u = float(np.mean((pr_test[pos][obs_m] - q_u) > y_test[pos][obs_m]))
                        a_aci = float(np.clip(a_aci + 0.04 * (EPSILON - err_u), 0.005, 0.20))
                    else:
                        a_aci = float(np.clip(a_aci + 0.04 * EPSILON, 0.005, 0.25))
            ev_m = evaluate_fleet(df_test, pr_test, np.ones(len(df_test)), margins_m, ell=LEAD_TIME, rho=RHO_DEFAULT)
            udf_m = ev_m['unit_df'].copy()
            udf_m.insert(0, 'mode', c_mode)
            udf_m.insert(0, 'seed', sd)
            censor_unit_rows.append(udf_m)

        B_cal_dc = [
            unit_critical_score(
                df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                pdc_cal[np.asarray(pos)], sc_dc_c[np.asarray(pos)],
                float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
            )
            for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
        ]
        lam_dfpdm = tune_differentiable_policy_margin(df_tune, pdc_tune, sc_dc_t)
        lam_b9, _ = choose_static_margin(B_cal_dc, alpha=EPSILON, tuned_margin=lam_dfpdm)
        ev_b9 = evaluate_fleet(df_test, pdc_test, sc_dc_te, lam_b9, ell=LEAD_TIME, rho=RHO_DEFAULT)
        udf_b9 = ev_b9['unit_df'].copy()
        udf_b9.insert(0, 'mode', 'DC-PdM Static (Zero Online Labels)')
        udf_b9.insert(0, 'seed', sd)
        censor_unit_rows.append(udf_b9)

        B_sorted_dc = np.sort(np.asarray(B_cal_dc, dtype=float))
        a_dc = ONLINE_ALPHA_TARGET
        margins_b10 = []
        for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
            pos = np.asarray(pos)
            rk = min(len(B_sorted_dc), max(1, math.ceil((len(B_sorted_dc) + 1) * (1.0 - np.clip(a_dc, 0.005, 0.25)))))
            lam_u = max(0.0, lam_dfpdm, float(B_sorted_dc[rk - 1]))
            margins_b10.append(lam_u)
            out_u = replay_unit_stopping_policy(
                df_test.iloc[pos].cycle.to_numpy(), pdc_test[pos], sc_dc_te[pos],
                float(df_test.iloc[pos].max_cycle.iloc[0]), LEAD_TIME, lam_u
            )
            a_dc = float(np.clip(a_dc + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - out_u['Z_fail']), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))
        ev_b10 = evaluate_fleet(df_test, pdc_test, sc_dc_te, margins_b10, ell=LEAD_TIME, rho=RHO_DEFAULT)
        udf_b10 = ev_b10['unit_df'].copy()
        udf_b10.insert(0, 'mode', 'DC-PdM Event-ACI (Censoring-Invariant)')
        udf_b10.insert(0, 'seed', sd)
        censor_unit_rows.append(udf_b10)

        B_cal_ridge = [
            unit_critical_score(
                df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                pr_cal[np.asarray(pos)], sc_ridge_c[np.asarray(pos)],
                float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
            )
            for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
        ]
        lam_b8, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=0.0)
        q_rc = max(0.0, float(rc_scores[math.ceil((len(rc_scores) + 1) * (1.0 - EPSILON)) - 1]))

        for r_val in rhos:
            h_dc_r = PrognosticHead().fit(X_fit, y_fit, kind='linex', rho=r_val, reg=14.0, focus_window=True)
            pt_t = h_dc_r.predict(X_tune, df_tune, causal_smooth=True)
            pt_c = h_dc_r.predict(X_cal, df_cal, causal_smooth=True)
            pt_te = h_dc_r.predict(X_test, df_test, causal_smooth=True)
            b_r = float(np.mean(pt_t - y_tune))
            pt_t = np.clip(pt_t - 0.85 * b_r, 0.0, 125.0)
            pt_c = np.clip(pt_c - 0.85 * b_r, 0.0, 125.0)
            pt_te = np.clip(pt_te - 0.85 * b_r, 0.0, 125.0)
            sd_r = float(np.std(pt_t - y_tune))
            sc_t_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_t) / 9.0))
            sc_c_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_c) / 9.0))
            sc_te_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_te) / 9.0))
            B_c_r = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pt_c[np.asarray(pos)], sc_c_r[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_t_r = tune_differentiable_policy_margin(df_tune, pt_t, sc_t_r, rho=r_val)
            lam_dc_r, _ = choose_static_margin(B_c_r, alpha=EPSILON, tuned_margin=lam_t_r)

            ev_rc_r = evaluate_fleet(df_test, pr_test, np.ones(len(df_test)), q_rc, rho=r_val)
            ev_b8_r = evaluate_fleet(df_test, pr_test, sc_ridge_te, lam_b8, rho=r_val)
            ev_dc_r = evaluate_fleet(df_test, pt_te, sc_te_r, lam_dc_r, rho=r_val)
            rho_rows.append({
                'seed': sd, 'rho': r_val,
                'j_life_rc': ev_rc_r['j_life_rate'],
                'j_life_b8': ev_b8_r['j_life_rate'],
                'j_life_dc': ev_dc_r['j_life_rate']
            })

    df_censor_units = pd.concat(censor_unit_rows, ignore_index=True)
    df_censor_units.to_csv(f'{out_dir}/censoring_observability_per_unit.csv', index=False)

    censor_sum = []
    for m_name, grp in df_censor_units.groupby('mode', sort=False):
        sum_u = max(1.0, grp['U_u'].sum())
        censor_sum.append({
            'mode': m_name,
            'j_life_rate': float(100.0 * grp['c_tot'].sum() / sum_u),
            'fail_pct': float(100.0 * grp['Z_fail'].mean()),
            'avail_pct': float(100.0 * sum_u / grp['T_u'].sum()),
            'waste_cycles': float(grp['W_u'].mean())
        })
    df_censor_summary = pd.DataFrame(censor_sum)
    df_censor_summary.to_csv(f'{out_dir}/censoring_observability_summary.csv', index=False)

    df_rho = pd.DataFrame(rho_rows).groupby('rho', as_index=False).mean().drop(columns=['seed'], errors='ignore')
    df_rho.to_csv(f'{out_dir}/rho_sensitivity_summary.csv', index=False)

    # Include both legacy 'non_monotone' and verified 'non_monotone_true' (with genuine negative latent increments)
    mismatch_modes = ['neg_dips', 'no_shocks', 'pos_spikes', 'bidirectional', 'non_monotone_true', 'unseen_regime']
    mismatch_unit_rows = []
    mismatch_seed_rows = []
    non_mono_check_rows = []

    for d_mode in mismatch_modes:
        for sd in SEEDS:
            if d_mode == 'neg_dips':
                (
                    df, df_fit, df_tune, df_cal, df_test,
                    X_fit_m, X_tune_m, X_cal_m, X_te_m,
                    y_fit, y_tune, y_cal, y_test,
                    _, _, p_mono_te,
                    pt_t, pt_c, pt_te,
                    _, _, sc_t, sc_c, sc_te
                ) = fd004_cache[sd]
                enc_unc = CausalMonotoneEncoder(True, False).fit(df_fit)
                X_fit_u, _ = enc_unc.transform(df_fit)
                X_te_u, _ = enc_unc.transform(df_test)
                h_unc = PrognosticHead().fit(X_fit_u, y_fit, kind='ridge', reg=14.0)
                p_unc_te = h_unc.predict(X_te_u)
            else:
                df = generate_synthetic_fleet('S-FD004', n_units=180, seed=sd, disturbance_mode=d_mode)
                if d_mode == 'non_monotone_true':
                    for uid_chk, g_chk in df.groupby('unit_id'):
                        lat_v = g_chk.latent_deg.to_numpy()
                        d_lat = np.diff(lat_v)
                        non_mono_check_rows.append({
                            'seed': sd, 'unit_id': int(uid_chk), 'T_u': int(g_chk.max_cycle.iloc[0]),
                            'has_negative_latent_increment': bool(np.any(d_lat < -1e-12)),
                            'minimum_latent_increment': float(np.min(d_lat))
                        })
                df_fit = df[df.unit_id <= 60].copy()
                df_tune = df[(df.unit_id > 60) & (df.unit_id <= 90)].copy()
                df_cal = df[(df.unit_id > 90) & (df.unit_id <= 130)].copy()
                df_test = df[df.unit_id > 130].copy()

                enc_unc = CausalMonotoneEncoder(True, False).fit(df_fit)
                X_fit_u, _ = enc_unc.transform(df_fit)
                X_te_u, _ = enc_unc.transform(df_test)

                enc_mono = CausalMonotoneEncoder(True, True).fit(df_fit)
                X_fit_m, _ = enc_mono.transform(df_fit)
                X_tune_m, _ = enc_mono.transform(df_tune)
                X_cal_m, _ = enc_mono.transform(df_cal)
                X_te_m, _ = enc_mono.transform(df_test)

                y_fit = df_fit.true_rul.to_numpy()
                y_tune = df_tune.true_rul.to_numpy()
                y_cal = df_cal.true_rul.to_numpy()
                y_test = df_test.true_rul.to_numpy()

                h_unc = PrognosticHead().fit(X_fit_u, y_fit, kind='ridge', reg=14.0)
                p_unc_te = h_unc.predict(X_te_u)

                h_mono_r = PrognosticHead().fit(X_fit_m, y_fit, kind='ridge', reg=14.0)
                p_mono_te = h_mono_r.predict(X_te_m)

                h_dc = PrognosticHead().fit(X_fit_m, y_fit, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)
                pt_t = h_dc.predict(X_tune_m, df_tune, causal_smooth=True)
                pt_c = h_dc.predict(X_cal_m, df_cal, causal_smooth=True)
                pt_te = h_dc.predict(X_te_m, df_test, causal_smooth=True)
                b_m = float(np.mean(pt_t - y_tune))
                pt_t = np.clip(pt_t - 0.85 * b_m, 0.0, 125.0)
                pt_c = np.clip(pt_c - 0.85 * b_m, 0.0, 125.0)
                pt_te = np.clip(pt_te - 0.85 * b_m, 0.0, 125.0)

                sd_dc = float(np.std(pt_t - y_tune))
                sc_t = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_t) / 9.0))
                sc_c = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_c) / 9.0))
                sc_te = sd_dc * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_te) / 9.0))

            B_cal_m = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pt_c[np.asarray(pos)], sc_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_t = tune_differentiable_policy_margin(df_tune, pt_t, sc_t)
            lam_dc, _ = choose_static_margin(B_cal_m, alpha=EPSILON, tuned_margin=lam_t)
            ev_dc = evaluate_fleet(df_test, pt_te, sc_te, lam_dc, ell=LEAD_TIME, rho=RHO_DEFAULT)

            udf = ev_dc['unit_df'].copy()
            udf.insert(0, 'seed', sd)
            udf.insert(0, 'disturbance_mode', d_mode)
            mismatch_unit_rows.append(udf)

            # Recovery window [0.55 T_u, 0.78 T_u] RMSE check
            rec_mask = (df_test.cycle.to_numpy() >= 0.55 * df_test.max_cycle.to_numpy()) & (df_test.cycle.to_numpy() <= 0.78 * df_test.max_cycle.to_numpy())
            rmse_unc_rec = float(np.sqrt(np.mean((p_unc_te[rec_mask] - y_test[rec_mask]) ** 2)))
            rmse_mono_rec = float(np.sqrt(np.mean((p_mono_te[rec_mask] - y_test[rec_mask]) ** 2)))

            mismatch_seed_rows.append({
                'disturbance_mode': d_mode, 'seed': sd,
                'rmse_unconstrained_ridge': float(np.sqrt(np.mean((p_unc_te - y_test) ** 2))),
                'rmse_monotone_ridge': float(np.sqrt(np.mean((p_mono_te - y_test) ** 2))),
                'rmse_dc_pdm': float(np.sqrt(np.mean((pt_te - y_test) ** 2))),
                'rmse_recovery_window_unconstrained': rmse_unc_rec,
                'rmse_recovery_window_monotone': rmse_mono_rec,
                'j_life_dc_pdm': ev_dc['j_life_rate'],
                'fail_pct_dc_pdm': ev_dc['fail_pct']
            })

    df_nm_chk = pd.DataFrame(non_mono_check_rows)
    df_nm_chk.to_csv(f'{out_dir}/non_monotone_latent_check.csv', index=False)
    assert df_nm_chk['has_negative_latent_increment'].all(), "All non_monotone_true trajectories must have negative latent increments!"

    df_mismatch_units = pd.concat(mismatch_unit_rows, ignore_index=True)
    df_mismatch_units.to_csv(f'{out_dir}/mismatch_stress_per_unit.csv', index=False)

    df_c1 = df_censor_units.copy().rename(columns={'mode': 'condition'})
    df_c1.insert(0, 'experiment_group', 'censoring_observability')
    df_m1 = df_mismatch_units.copy().rename(columns={'disturbance_mode': 'condition'})
    df_m1.insert(0, 'experiment_group', 'mismatch_stress')
    pd.concat([df_c1, df_m1], ignore_index=True).to_csv(f'{out_dir}/censoring_and_mismatch_per_unit.csv', index=False)

    df_mismatch_summary = pd.DataFrame(mismatch_seed_rows).groupby('disturbance_mode', sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore')
    df_mismatch_summary.to_csv(f'{out_dir}/mismatch_stress_summary.csv', index=False)

    return df_censor_summary, df_rho, df_mismatch_summary


# ==============================================================================
# 9. EXPERIMENT F: 4-WAY MARGIN ATTRIBUTION, MATCHED LOSSES, ABLATION, HI & CPU
# ==============================================================================
def run_controlled_attribution_and_efficiency(fd004_cache, out_dir=OUT_DIR):
    attr_rows = []
    margin_ctrl_rows = []
    floor_rows = []
    pre_post_loss_rows = []
    hi_rows = []
    timing_rows = []

    for sd in SEEDS:
        (
            df, df_fit, df_tune, df_cal, df_test,
            X_fit, X_tune, X_cal, X_test,
            y_fit, y_tune, y_cal, y_test,
            pr_tune, pr_cal, pr_test,
            pdc_tune, pdc_cal, pdc_test,
            sc_ridge_c, sc_ridge_te, sc_dc_t, sc_dc_c, sc_dc_te
        ) = fd004_cache[sd]

        # 1. HI Quality Metrics (Unconstrained EMA vs. Prefix-PAVA Pre-Running-Max vs. Post-Running-Max)
        t_enc_raw_fit0 = time.perf_counter()
        enc_raw = CausalMonotoneEncoder(regime_norm=True, monotone_state=False).fit(df_fit)
        X_fit_r_only, _ = enc_raw.transform(df_fit)
        X_tune_raw, _ = enc_raw.transform(df_tune)
        X_cal_raw, _ = enc_raw.transform(df_cal)
        t_enc_raw_fit_ms = (time.perf_counter() - t_enc_raw_fit0) * 1000.0

        t_enc_raw_te0 = time.perf_counter()
        X_te_raw, hi_te_raw = enc_raw.transform(df_test)
        enc_raw_batch_us = (time.perf_counter() - t_enc_raw_te0) * 1e6 / len(df_test)

        t_enc_mono_fit0 = time.perf_counter()
        enc_mono = CausalMonotoneEncoder(regime_norm=True, monotone_state=True).fit(df_fit)
        _ = enc_mono.transform(df_fit)
        _ = enc_mono.transform(df_tune)
        _ = enc_mono.transform(df_cal)
        t_enc_mono_fit_ms = (time.perf_counter() - t_enc_mono_fit0) * 1000.0

        t_enc_mono_te0 = time.perf_counter()
        _, hi_te_mono, hi_te_pava_pre = enc_mono.transform(df_test, return_pava_pre_accum=True)
        enc_mono_batch_us = (time.perf_counter() - t_enc_mono_te0) * 1e6 / len(df_test)

        hi_rows.append({'seed': sd, 'representation': 'Unconstrained Causal EMA State', **compute_hi_quality_metrics(df_test, hi_te_raw)})
        hi_rows.append({'seed': sd, 'representation': 'Prefix-PAVA Endpoint Composite (Pre Running-Max)', **compute_hi_quality_metrics(df_test, hi_te_pava_pre)})
        hi_rows.append({'seed': sd, 'representation': 'Prefix-PAVA + Running-Max Monotone State (Ours)', **compute_hi_quality_metrics(df_test, hi_te_mono)})

        # Helper to run matched pipeline for any head & calibration configuration
        def eval_matched_pipeline(
            X_f, X_tu, X_ca, X_te_in, kind='linex', focus_win=True, smooth=True,
            cal_mode='unit_plus_tune', online_upd=False, label=''
        ):
            t_train_0 = time.perf_counter()
            head = PrognosticHead().fit(X_f, y_fit, kind=kind, rho=RHO_DEFAULT, reg=14.0, focus_window=focus_win)
            p_tu_raw = head.predict(X_tu, df_tune, causal_smooth=False)
            p_te_raw = head.predict(X_te_in, df_test, causal_smooth=False)

            p_tu = head.predict(X_tu, df_tune, causal_smooth=smooth)
            p_ca = head.predict(X_ca, df_cal, causal_smooth=smooth)
            b_tu = float(np.mean(p_tu - y_tune))
            p_tu = np.clip(p_tu - 0.85 * b_tu, 0.0, 125.0)
            p_ca = np.clip(p_ca - 0.85 * b_tu, 0.0, 125.0)

            sd_tu = float(np.std(p_tu - y_tune))
            sc_tu = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_tu) / 9.0))
            sc_ca = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_ca) / 9.0))

            lam_tu = tune_differentiable_policy_margin(df_tune, p_tu, sc_tu)
            B_ca = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    p_ca[np.asarray(pos)], sc_ca[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_unit_pure, _ = choose_static_margin(B_ca, alpha=EPSILON, tuned_margin=0.0)
            lam_unit_plus_tune, _ = choose_static_margin(B_ca, alpha=EPSILON, tuned_margin=lam_tu)

            cyc_scores = np.sort((p_ca - y_cal) / sc_ca)
            k_c = math.ceil((len(cyc_scores) + 1) * (1.0 - EPSILON))
            lam_cyc = max(0.0, float(cyc_scores[min(len(cyc_scores) - 1, k_c - 1)]))

            if cal_mode == 'cycle':
                lam_static = lam_cyc
                B_sorted = cyc_scores
                floor_for_online = 0.0
            elif cal_mode == 'unit_no_tune':
                lam_static = lam_unit_pure
                B_sorted = np.sort(np.asarray(B_ca, dtype=float))
                floor_for_online = 0.0
            elif cal_mode == 'tune_only':
                lam_static = lam_tu
                B_sorted = np.sort(np.asarray(B_ca, dtype=float))
                floor_for_online = lam_tu
            else:  # 'unit_plus_tune'
                lam_static = lam_unit_plus_tune
                B_sorted = np.sort(np.asarray(B_ca, dtype=float))
                floor_for_online = lam_tu

            t_head_train_ms = (time.perf_counter() - t_train_0) * 1000.0

            t_inf_0 = time.perf_counter()
            p_te_out = head.predict(X_te_in, df_test, causal_smooth=smooth)
            p_te_out = np.clip(p_te_out - 0.85 * b_tu, 0.0, 125.0)
            sc_te_out = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_te_out) / 9.0))
            t_head_batch_us = (time.perf_counter() - t_inf_0) * 1e6 / len(df_test)

            t_rep_0 = time.perf_counter()
            if not online_upd:
                margins_use = lam_static
            else:
                a_cur = ONLINE_ALPHA_TARGET
                margins_use = []
                for _, pos in df_test.groupby('unit_id', sort=False).indices.items():
                    pos = np.asarray(pos)
                    rk = min(len(B_sorted), max(1, math.ceil((len(B_sorted) + 1) * (1.0 - np.clip(a_cur, 0.005, 0.25)))))
                    q_u = max(0.0, float(B_sorted[rk - 1]))
                    lam_u = max(0.0, floor_for_online, q_u)
                    margins_use.append(lam_u)
                    out_u = replay_unit_stopping_policy(
                        df_test.iloc[pos].cycle.to_numpy(), p_te_out[pos], sc_te_out[pos],
                        float(df_test.iloc[pos].max_cycle.iloc[0]), LEAD_TIME, lam_u
                    )
                    a_cur = float(np.clip(a_cur + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - out_u['Z_fail']), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))
            ev = evaluate_fleet(df_test, p_te_out, sc_te_out, margins_use, ell=LEAD_TIME, rho=RHO_DEFAULT)
            t_replay_only_ms_per_unit = (time.perf_counter() - t_rep_0) * 1000.0 / ev['n_units']

            rmse_te = float(np.sqrt(np.mean((p_te_out - y_test) ** 2)))
            near_m = (y_test <= 35.0)
            rmse_near = float(np.sqrt(np.mean((p_te_out[near_m] - y_test[near_m]) ** 2)))
            over_near = float(np.mean(np.maximum(0.0, p_te_out[near_m] - y_test[near_m])))

            res_raw = p_te_raw - y_test
            res_post = p_te_out - y_test
            return {
                'seed': sd, 'variant': label,
                'cal_mode': cal_mode, 'online': online_upd,
                'lambda_tuned': lam_tu, 'lambda_unit_pure': lam_unit_pure, 'lambda_static': lam_static,
                'tuning_floor_active': bool(np.isclose(lam_static, lam_tu)) if cal_mode == 'unit_plus_tune' else False,
                'rmse_all': rmse_te, 'rmse_near35': rmse_near, 'overest_near35': over_near,
                'raw_mean_err': float(np.mean(res_raw)), 'raw_sd_err': float(np.std(res_raw)), 'raw_q95_err': float(np.quantile(res_raw, 0.95)),
                'post_mean_err': float(np.mean(res_post)), 'post_sd_err': float(np.std(res_post)), 'post_q95_err': float(np.quantile(res_post, 0.95)),
                'j_life_rate': ev['j_life_rate'], 'fail_pct': ev['fail_pct'],
                'n_fails': ev['n_fails'], 'n_units': ev['n_units'],
                'avail_pct': ev['avail_pct'], 'waste_cycles': ev['waste_cycles'],
                'head_train_ms': t_head_train_ms, 'head_batch_us': t_head_batch_us, 'replay_only_ms_per_unit': t_replay_only_ms_per_unit
            }

        # Q1: 4-Way Margin Attribution for Fixed Ridge and Fixed Linex
        q1_ridge_cyc = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=False, smooth=False, cal_mode='cycle', online_upd=False, label='Q1a: Fixed Ridge Head + Cycle-CP')
        q1_ridge_u_pure = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=False, smooth=False, cal_mode='unit_no_tune', online_upd=False, label='Q1b_pure: Fixed Ridge Head + Unit-CP Only (No Tune Floor)')
        q1_ridge_tune_only = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=False, smooth=False, cal_mode='tune_only', online_upd=False, label='Q1b_tune: Fixed Ridge Head + Cost-Tuned Margin Only')
        q1_ridge_u_floor = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=False, smooth=False, cal_mode='unit_plus_tune', online_upd=False, label='Q1b: Fixed Ridge Head + Unit-CP + Tune Floor')

        q1_linex_cyc = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='cycle', online_upd=False, label='Q1c: Fixed Linex Head + Cycle-CP')
        q1_linex_u_pure = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='unit_no_tune', online_upd=False, label='Q1d_pure: Fixed Linex Head + Unit-CP Only (No Tune Floor)')
        q1_linex_tune_only = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='tune_only', online_upd=False, label='Q1d_tune: Fixed Linex Head + Cost-Tuned Margin Only')
        q1_linex_u_floor = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='Q1d: Fixed Linex Head + Unit-CP + Tune Floor')

        for pred_lbl, r_c, r_up, r_to, r_uf in [
            ('ridge', q1_ridge_cyc, q1_ridge_u_pure, q1_ridge_tune_only, q1_ridge_u_floor),
            ('linex', q1_linex_cyc, q1_linex_u_pure, q1_linex_tune_only, q1_linex_u_floor)
        ]:
            for pol_lbl, rr in [
                ('cycle_CP', r_c),
                ('unit_CP_no_tune_floor', r_up),
                ('tuned_margin_only', r_to),
                ('unit_CP_plus_tune_floor', r_uf)
            ]:
                margin_ctrl_rows.append({
                    'seed': sd, 'predictor': pred_lbl, 'policy': pol_lbl,
                    'margin': rr['lambda_static'], 'j_life_rate': rr['j_life_rate'],
                    'fail_pct': rr['fail_pct'], 'waste_cycles': rr['waste_cycles'],
                    'n_units': rr['n_units'], 'n_fails': rr['n_fails']
                })

        # Q2: Strictly Matched Loss Heads (L1..L4)
        r_l1 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='L1: Matched Quadratic (Ridge) + Unit-CP')
        r_l2 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='quantile', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='L2: Matched Pinball (tau=0.05) + Unit-CP')
        r_l3 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='nasa', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='L3: Matched NASA Exponential + Unit-CP')
        r_l4 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='L4: Matched Centered Linex + Unit-CP')

        for r_lx in [r_l1, r_l2, r_l3, r_l4]:
            pre_post_loss_rows.append({
                'seed': sd, 'variant': r_lx['variant'],
                'raw_mean_err': r_lx['raw_mean_err'], 'raw_sd_err': r_lx['raw_sd_err'], 'raw_q95_err': r_lx['raw_q95_err'],
                'post_mean_err': r_lx['post_mean_err'], 'post_sd_err': r_lx['post_sd_err'], 'post_q95_err': r_lx['post_q95_err'],
                'j_life_rate': r_lx['j_life_rate'], 'fail_pct': r_lx['fail_pct']
            })

        # Q3: Single-Module Removal Ablations (with both A2_static [vs A1] and A2_online [vs A0])
        r_a0 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=True, label='A0: Full DC-PdM (All Modules Active)')
        r_a1 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=False, label='A1: -Online Event Update (Static Unit-CP)')
        r_a2_stat = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='cycle', online_upd=False, label='A2: -Unit-Level Score (Static Cycle-CP vs A1)')
        r_a2_on = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=True, cal_mode='cycle', online_upd=True, label='A2_online: -Unit-Level Score (Online Cycle-CP vs A0)')
        r_a3 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='ridge', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=True, label='A3: -Asymmetric Linex Loss (Use Matched Ridge)')
        r_a4 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=False, smooth=True, cal_mode='unit_plus_tune', online_upd=True, label='A4: -Decision-Window Weighting (Uniform Weights)')
        r_a5 = eval_matched_pipeline(X_fit, X_tune, X_cal, X_test, kind='linex', focus_win=True, smooth=False, cal_mode='unit_plus_tune', online_upd=True, label='A5: -Causal Step Smoothing (Raw Predictions)')
        r_a6 = eval_matched_pipeline(X_fit_r_only, X_tune_raw, X_cal_raw, X_te_raw, kind='linex', focus_win=True, smooth=True, cal_mode='unit_plus_tune', online_upd=True, label='A6: -Monotone Prefix-PAVA State (Raw EMA State)')

        # Raw EMA + Ridge + Cycle-CP explicitly evaluated for accurate CPU timing (R3-11)
        r_raw_ridge_cyc = eval_matched_pipeline(X_fit_r_only, X_tune_raw, X_cal_raw, X_te_raw, kind='ridge', focus_win=False, smooth=False, cal_mode='cycle', online_upd=False, label='Raw_EMA_Ridge_Cycle_CP')

        seed_variants = [
            q1_ridge_cyc, q1_ridge_u_pure, q1_ridge_tune_only, q1_ridge_u_floor,
            q1_linex_cyc, q1_linex_u_pure, q1_linex_tune_only, q1_linex_u_floor,
            r_l1, r_l2, r_l3, r_l4,
            r_a0, r_a1, r_a2_stat, r_a2_on, r_a3, r_a4, r_a5, r_a6
        ]
        attr_rows.extend(seed_variants)
        for sv in seed_variants:
            floor_rows.append({
                'seed': sd, 'id': sv['variant'].split(':')[0], 'variant': sv['variant'],
                'cal_mode': sv['cal_mode'], 'lambda_tuned': sv['lambda_tuned'],
                'lambda_unit_pure': sv['lambda_unit_pure'], 'lambda_static': sv['lambda_static'],
                'tuning_floor_active': sv['tuning_floor_active']
            })

        # Named-configuration CPU timing profile (including true single-step streaming vs. batch & full offline time)
        u_first_df = df_test[df_test.unit_id == df_test.unit_id.unique()[0]].copy()
        step_times_mono = []
        for t_step in range(20, min(60, len(u_first_df))):
            sub_pref = u_first_df.iloc[:t_step].copy()
            t_s0 = time.perf_counter()
            x_s, _ = enc_mono.transform(sub_pref)
            _ = np.dot(x_s[-1], np.ones(x_s.shape[1]))
            step_times_mono.append((time.perf_counter() - t_s0) * 1e6)
        stream_p50_us = float(np.percentile(step_times_mono, 50))
        stream_p95_us = float(np.percentile(step_times_mono, 95))

        mean_cycles_per_unit = float(len(df_test)) / 50.0
        for arch_name, enc_b_us, enc_fit_ms, r_ref in [
            ('Unconstrained EMA + Ridge + Cycle-CP', enc_raw_batch_us, t_enc_raw_fit_ms, r_raw_ridge_cyc),
            ('Causal Monotone + Ridge + Unit-CP (Q1b)', enc_mono_batch_us, t_enc_mono_fit_ms, q1_ridge_u_floor),
            ('DC-PdM Static (Causal Monotone + Linex, B9)', enc_mono_batch_us, t_enc_mono_fit_ms, r_a1),
            ('DC-PdM Online (Causal Monotone + Event-ACI, B10)', enc_mono_batch_us, t_enc_mono_fit_ms, r_a0),
        ]:
            tot_batch_us = enc_b_us + r_ref['head_batch_us']
            measured_unit_total_ms = (tot_batch_us * mean_cycles_per_unit) / 1000.0 + r_ref['replay_only_ms_per_unit']
            timing_rows.append({
                'seed': sd, 'architecture': arch_name,
                'encoder_batch_us_per_cycle': enc_b_us,
                'head_batch_us_per_cycle': r_ref['head_batch_us'],
                'amortized_batch_us_per_cycle': tot_batch_us,
                'streaming_prefix_p50_us': stream_p50_us,
                'streaming_prefix_p95_us': stream_p95_us,
                'measured_unit_replay_ms': measured_unit_total_ms,
                'full_offline_fit_tune_cal_ms': enc_fit_ms + r_ref['head_train_ms']
            })

    df_attr = pd.DataFrame(attr_rows)
    df_attr.to_csv(f'{out_dir}/controlled_attribution_per_seed.csv', index=False)
    df_attr_sum = df_attr.groupby('variant', sort=False).agg(
        rmse_all=('rmse_all', 'mean'),
        rmse_near35=('rmse_near35', 'mean'),
        overest_near35=('overest_near35', 'mean'),
        j_life_mean=('j_life_rate', 'mean'),
        j_life_sd=('j_life_rate', 'std'),
        fail_pct=('fail_pct', 'mean'),
        avail_pct=('avail_pct', 'mean'),
        waste_cycles=('waste_cycles', 'mean')
    ).reset_index()
    df_attr_sum.to_csv(f'{out_dir}/controlled_attribution_summary.csv', index=False)

    df_mctrl = pd.DataFrame(margin_ctrl_rows)
    df_mctrl.to_csv(f'{out_dir}/margin_controls_per_seed.csv', index=False)
    df_mctrl_sum = df_mctrl.groupby(['predictor', 'policy'], sort=False).agg(
        j_life_mean=('j_life_rate', 'mean'),
        j_life_sd=('j_life_rate', 'std'),
        fail_pct=('fail_pct', 'mean'),
        waste_cycles=('waste_cycles', 'mean')
    ).reset_index()
    df_mctrl_sum.to_csv(f'{out_dir}/margin_controls_summary.csv', index=False)

    df_fl = pd.DataFrame(floor_rows)
    df_fl_sum = df_fl.groupby('id', sort=False).agg(
        seeds=('seed', 'size'),
        seeds_tuning_floor_active=('tuning_floor_active', 'sum'),
        mean_tuned_margin=('lambda_tuned', 'mean'),
        mean_unit_pure_margin=('lambda_unit_pure', 'mean'),
        mean_final_margin=('lambda_static', 'mean')
    ).reset_index()
    df_fl_sum.to_csv(f'{out_dir}/attribution_margin_floor_activity.csv', index=False)

    pd.DataFrame(pre_post_loss_rows).groupby('variant', sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore').to_csv(
        f'{out_dir}/loss_head_pre_post_residuals.csv', index=False
    )

    # Paired tests across the 5 S-FD004 seeds for Experiment F
    df_attr['id'] = df_attr['variant'].str.split(':').str[0]
    piv_a = df_attr.pivot(index='seed', columns='id', values='j_life_rate')
    attr_test_rows = []
    for a_id, b_id in [
        ('Q1a', 'Q1b_pure'), ('Q1b_pure', 'Q1b_tune'), ('Q1b', 'Q1b_tune'),
        ('Q1c', 'Q1d_pure'), ('Q1d_pure', 'Q1d_tune'), ('Q1d', 'Q1d_tune'),
        ('L1', 'L4'), ('L2', 'L4'), ('L3', 'L4'),
        ('A0', 'A1'), ('A2', 'A1'), ('A2_online', 'A0'),
        ('A3', 'A0'), ('A4', 'A0'), ('A5', 'A0'), ('A6', 'A0')
    ]:
        attr_test_rows.append({'comparison': f'{a_id} minus {b_id}', **_paired_stats(piv_a[a_id] - piv_a[b_id])})
    df_attr_tests = pd.DataFrame(attr_test_rows)
    df_attr_tests.to_csv(f'{out_dir}/attribution_paired_tests.csv', index=False)

    df_hi_sum = pd.DataFrame(hi_rows).groupby('representation', sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore')
    df_hi_sum.to_csv(f'{out_dir}/hi_quality_metrics_summary.csv', index=False)

    df_time_sum = pd.DataFrame(timing_rows).groupby('architecture', sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore')
    df_time_sum.to_csv(f'{out_dir}/cpu_resource_timing_summary.csv', index=False)

    # Q3 Part 2: Progressively Increasing Prediction-Bias Stress Test (with Matched-Initial Static Control)
    drift_rates = [0.00, 0.20, 0.40, 0.60, 0.80]
    drift_rows = []
    for dr in drift_rates:
        for sd in SEEDS:
            (
                _, df_fit, df_tune, df_cal, df_test_base,
                _, _, _, _,
                _, y_tune, y_cal, y_test_base,
                _, _, _,
                pdc_tune, pdc_cal, pdc_test_base,
                _, _, sc_dc_t, sc_dc_c, sc_dc_te
            ) = fd004_cache[sd]

            pdc_test_dr = pdc_test_base.copy()
            for k_u, (_, pos) in enumerate(df_test_base.groupby('unit_id', sort=False).indices.items()):
                pos = np.asarray(pos)
                prog_u = (k_u + 1.0) / 50.0
                pdc_test_dr[pos] = np.clip(pdc_test_base[pos] + dr * 12.5 * prog_u, 0.0, 125.0)

            B_cal_dc = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pdc_cal[np.asarray(pos)], sc_dc_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_tu = tune_differentiable_policy_margin(df_tune, pdc_tune, sc_dc_t)
            lam_stat_005, _ = choose_static_margin(B_cal_dc, alpha=EPSILON, tuned_margin=lam_tu)
            lam_stat_matched_00325, _ = choose_static_margin(B_cal_dc, alpha=ONLINE_ALPHA_TARGET, tuned_margin=lam_tu)

            ev_stat_orig = evaluate_fleet(df_test_base, pdc_test_dr, sc_dc_te, lam_stat_005, ell=LEAD_TIME, rho=RHO_DEFAULT)
            ev_stat_match = evaluate_fleet(df_test_base, pdc_test_dr, sc_dc_te, lam_stat_matched_00325, ell=LEAD_TIME, rho=RHO_DEFAULT)

            B_sorted = np.sort(np.asarray(B_cal_dc, dtype=float))
            a_cur = ONLINE_ALPHA_TARGET
            margins_on = []
            for _, pos in df_test_base.groupby('unit_id', sort=False).indices.items():
                pos = np.asarray(pos)
                rk = min(len(B_sorted), max(1, math.ceil((len(B_sorted) + 1) * (1.0 - np.clip(a_cur, 0.005, 0.25)))))
                lam_u = max(0.0, lam_tu, float(B_sorted[rk - 1]))
                margins_on.append(lam_u)
                out_u = replay_unit_stopping_policy(
                    df_test_base.iloc[pos].cycle.to_numpy(), pdc_test_dr[pos], sc_dc_te[pos],
                    float(df_test_base.iloc[pos].max_cycle.iloc[0]), LEAD_TIME, lam_u
                )
                a_cur = float(np.clip(a_cur + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - out_u['Z_fail']), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))
            ev_on = evaluate_fleet(df_test_base, pdc_test_dr, sc_dc_te, margins_on, ell=LEAD_TIME, rho=RHO_DEFAULT)

            drift_rows.append({
                'drift_rate': dr, 'seed': int(sd),
                'j_life_static_alpha005': ev_stat_orig['j_life_rate'],
                'fail_pct_static_alpha005': ev_stat_orig['fail_pct'],
                'j_life_static_matched00325': ev_stat_match['j_life_rate'],
                'fail_pct_static_matched00325': ev_stat_match['fail_pct'],
                'j_life_online': ev_on['j_life_rate'],
                'fail_pct_online': ev_on['fail_pct']
            })

    df_drift_seed = pd.DataFrame(drift_rows)
    df_drift_seed.to_csv(f'{out_dir}/drift_scaling_per_seed.csv', index=False)

    df_drift = df_drift_seed.groupby('drift_rate', as_index=False).agg(
        j_life_static=('j_life_static_alpha005', 'mean'),
        fail_pct_static=('fail_pct_static_alpha005', 'mean'),
        j_life_matched_static=('j_life_static_matched00325', 'mean'),
        fail_pct_matched_static=('fail_pct_static_matched00325', 'mean'),
        j_life_online=('j_life_online', 'mean'),
        fail_pct_online=('fail_pct_online', 'mean')
    )
    df_drift.to_csv(f'{out_dir}/drift_scaling_summary.csv', index=False)

    drift_paired_rows = []
    for dr_val, grp in df_drift_seed.groupby('drift_rate'):
        diff_m_on = grp['j_life_static_matched00325'] - grp['j_life_online']
        drift_paired_rows.append({'drift_rate': dr_val, **_paired_stats(diff_m_on)})
    pd.DataFrame(drift_paired_rows).to_csv(f'{out_dir}/drift_matched_initialization_paired.csv', index=False)

    return df_attr_sum, df_mctrl_sum, df_attr_tests, df_hi_sum, df_time_sum, df_drift


# ==============================================================================
# 10. RECONSTRUCTED FIGURES 1–4 (VECTOR PDF + 600 DPI PNG)
# ==============================================================================


def _get_cpu_model_name():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    val = line.split(':', 1)[1].strip()
                    if val and val.lower() != 'unknown':
                        return val
    except Exception:
        pass
    return platform.processor() or 'unavailable (CPU model detection failed)'


def run_revision5_diagnostics_and_fair_sensitivity(fd004_cache, out_dir=OUT_DIR):
    """
    Executes all Revision 5 & Revision 6 supplementary verifications and reviewer closure experiments:
      1. Finite exchangeable non-i.i.d. Beta counterexample (`exchangeable_non_iid_beta_counterexample.csv`)
         and Beta-binomial conditional risk diagnostics.
      2. Paired incremental cost difference `Tune+Unit-CP minus Tune-only` on the 1,000-unit
         exchangeable benchmark (`exchangeable_incremental_cost_paired.csv`) and descriptive bounds
         (`exchangeable_risk_bounds_descriptive.csv`).
      3. Fixed-policy waste-weight rescoring (`fixed_policy_waste_weight_rescoring.csv`) AND
         Fairly Re-Optimized One-at-a-Time Waste-Weight & Cost-Ratio Sensitivity (`fair_reoptimized_cost_sensitivity.csv`,
         `fair_reoptimized_cost_sensitivity_per_seed.csv`, `fair_reoptimized_rho_sensitivity.csv`,
         `fair_reoptimized_rho_sensitivity_per_seed.csv`, `fair_sensitivity_independent_paired.csv`).
      4. Exact recovery-window negative-increment diagnostics (`recovery_window_diagnostics.csv`),
         Running-Max Isolation (`running_max_isolation_per_seed.csv`, `running_max_isolation_summary.csv`,
         `running_max_paired_uncertainty.csv` including both refitted branches and fixed-parameter test-time
         toggle), and Frozen Source-Model Transfer across all disturbance modes including both the legacy
         compound unseen-regime shift (`unseen_regime_generator_shift.csv`) and the strictly isolated
         Regime-7 transfer (`mismatch_frozen_transfer_per_seed.csv`, `mismatch_frozen_transfer_summary.csv`).
      5. Executable Full-Pipeline Single-Step Streaming Latency across the 25-unit full-lifetime sample
         (5,251 steps per architecture, exported in `cpu_timing_step_level_raw.csv`) with exact `g_t = f_t - lam_s * s_t`
         stopping verification (`cpu_stream_vs_reference_per_unit.csv`, `cpu_prefix_forecast_equivalence.csv`:
         0/25 alarm or failure discrepancies across all 4 architectures), Direct Per-Unit Batch Replay across
         all 250 test units (`cpu_timing_unit_level_raw.csv`), and Direct Offline Fit+Tune+Cal Wall-Clock Time.
    """
    # --- 1. Exchangeable Non-IID Beta Counterexample ---
    fixed_scores = np.linspace(0.0, 1.0, 41)
    ce_rows = []
    for missing in range(41):
        cal = np.delete(fixed_scores, missing)
        q = float(np.sort(cal)[38])  # k = 39 at n = 40, eps = 0.05
        f_ind = int(fixed_scores[missing] > q)
        ce_rows.append({
            'missing_rank': missing + 1,
            'calibration_threshold': q,
            'conditional_failure_risk': f_ind
        })
    pd.DataFrame(ce_rows).to_csv(f'{out_dir}/exchangeable_non_iid_beta_counterexample.csv', index=False)

    # --- 2. Exchangeable Incremental Cost Paired & Descriptive Risk Bounds ---
    ex_df = pd.read_csv(f'{out_dir}/exchangeable_zero_drift_per_seed.csv')
    ex_df['id'] = ex_df['policy'].map(
        lambda p: 'ridge_pure' if 'B8:' in p else (
            'linex_pure' if 'Pure Unit-CP (Linex' in p else (
                'tune_only' if 'B7:' in p else 'tune_plus_cp'
            )
        )
    )
    ep = ex_df.pivot(index=['scenario', 'seed'], columns='id', values='j_life_rate')
    ed = ep['tune_plus_cp'] - ep['tune_only']
    st_20 = _paired_stats(ed)
    st_5 = _paired_stats(ed.groupby('seed').mean())
    pd.DataFrame([
        {
            'comparison': 'exchangeable tune+CP minus tune-only', 'grouping': '20 scenario-seed',
            'n': st_20['n'], 'mean_difference': st_20['mean_difference'],
            'ci_low': st_20['ci95_low'], 'ci_high': st_20['ci95_high'],
            'p_t': st_20['p_t'], 'p_wilcoxon': st_20['p_wilcoxon']
        },
        {
            'comparison': 'exchangeable tune+CP minus tune-only', 'grouping': '5 seed blocks',
            'n': st_5['n'], 'mean_difference': st_5['mean_difference'],
            'ci_low': st_5['ci95_low'], 'ci_high': st_5['ci95_high'],
            'p_t': st_5['p_t'], 'p_wilcoxon': st_5['p_wilcoxon']
        }
    ]).to_csv(f'{out_dir}/exchangeable_incremental_cost_paired.csv', index=False)

    risk_rows = []
    for name, grp in ex_df.groupby('policy', sort=False):
        k_f = int(grp['n_fails'].sum())
        n_u = int(grp['n_units'].sum())
        lo = 0.0 if k_f == 0 else float(stats.beta.ppf(0.05, k_f, n_u - k_f + 1))
        hi = 1.0 if k_f == n_u else float(stats.beta.ppf(0.95, k_f + 1, n_u - k_f))
        risk_rows.append({
            'policy': name, 'n_fail': k_f, 'n_units': n_u,
            'fail_pct': 100.0 * k_f / n_u,
            'iid_binomial_lower95_pct': 100.0 * lo,
            'iid_binomial_upper95_pct': 100.0 * hi,
            'inference_note': 'Descriptive iid-binomial calculation, not exact pooled confidence under shared calibration/fleet dependence.'
        })
    pd.DataFrame(risk_rows).to_csv(f'{out_dir}/exchangeable_risk_bounds_descriptive.csv', index=False)

    # --- 3A. Fixed-Policy Waste-Weight Rescoring (No Retuning) ---
    long_df = pd.read_csv(f'{out_dir}/unit_policy_b1_b10_per_unit_long.csv')
    rescore_rows = []
    for w_val in [0.0, 0.01, 0.025, 0.045, 0.075, 0.10]:
        for (sc_k, sd_k, m_k), grp in long_df.groupby(['scenario', 'seed', 'method'], sort=False):
            j_val = 100.0 * (grp['c_ev'].sum() + w_val * grp['W_u'].sum()) / grp['U_u'].sum()
            rescore_rows.append({'waste_weight': w_val, 'scenario': sc_k, 'seed': sd_k, 'method': m_k, 'j_life_rate': j_val})
    pd.DataFrame(rescore_rows).groupby(['waste_weight', 'method'], sort=False).agg(
        j_mean=('j_life_rate', 'mean'), j_sd=('j_life_rate', 'std')
    ).reset_index().to_csv(f'{out_dir}/fixed_policy_waste_weight_rescoring.csv', index=False)

    # --- 3B. Fairly Re-Optimized One-at-a-Time Waste-Weight & Cost-Ratio Sensitivity on S-FD004 ---
    waste_grid = [0.000, 0.010, 0.025, 0.045, 0.075, 0.100]
    fair_waste_seed_rows = []
    grid_b2 = np.linspace(0.0, 3.0, 31)
    indep_paired_rows = []

    for w_val in waste_grid:
        for sd in SEEDS:
            (
                df, df_fit, df_tune, df_cal, df_test,
                X_fit, X_tune, X_cal, X_test,
                y_fit, y_tune, y_cal, y_test,
                pr_tune, pr_cal, pr_test,
                pdc_tune, pdc_cal, pdc_test,
                sc_ridge_c, sc_ridge_te, sc_dc_t, sc_dc_c, sc_dc_te
            ) = fd004_cache[sd]
            sd_ridge = float(np.std(pr_tune - y_tune))
            sc_ridge_t = np.full(len(df_tune), sd_ridge)

            rc_scores = np.sort(pr_cal - y_cal)
            q_rc = max(0.0, float(rc_scores[math.ceil((len(rc_scores) + 1) * (1.0 - EPSILON)) - 1]))

            B_cal_ridge = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pr_cal[np.asarray(pos)], sc_ridge_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b8, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=0.0)

            costs_b2_w = [evaluate_fleet(df_tune, pr_tune, sc_ridge_t, g, rho=RHO_DEFAULT, c_waste=w_val)['j_life_rate'] for g in grid_b2]
            m_b2_w = float(grid_b2[int(np.argmin(costs_b2_w))])
            lam_q1b_w, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=m_b2_w)

            lam_b7_w = tune_differentiable_policy_margin(df_tune, pdc_tune, sc_dc_t, rho=RHO_DEFAULT, c_waste=w_val)
            B_cal_dc = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pdc_cal[np.asarray(pos)], sc_dc_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b9_w, _ = choose_static_margin(B_cal_dc, alpha=EPSILON, tuned_margin=lam_b7_w)

            ev_b5 = evaluate_fleet(df_test, pr_test, np.ones(len(df_test)), q_rc, rho=RHO_DEFAULT, c_waste=w_val)
            ev_b8 = evaluate_fleet(df_test, pr_test, sc_ridge_te, lam_b8, rho=RHO_DEFAULT, c_waste=w_val)
            ev_b2 = evaluate_fleet(df_test, pr_test, sc_ridge_te, m_b2_w, rho=RHO_DEFAULT, c_waste=w_val)
            ev_q1b = evaluate_fleet(df_test, pr_test, sc_ridge_te, lam_q1b_w, rho=RHO_DEFAULT, c_waste=w_val)
            ev_b7 = evaluate_fleet(df_test, pdc_test, sc_dc_te, lam_b7_w, rho=RHO_DEFAULT, c_waste=w_val)
            ev_b9 = evaluate_fleet(df_test, pdc_test, sc_dc_te, lam_b9_w, rho=RHO_DEFAULT, c_waste=w_val)

            fair_waste_seed_rows.append({
                'c_waste': w_val, 'seed': sd,
                'lam_b2_retuned': m_b2_w, 'lam_b7_retuned': lam_b7_w, 'lam_b9_retuned': lam_b9_w,
                'j_b5_cycle_cp': ev_b5['j_life_rate'], 'fail_b5': ev_b5['fail_pct'],
                'j_b8_pure_unit_cp': ev_b8['j_life_rate'], 'fail_b8': ev_b8['fail_pct'],
                'j_b2_ridge_retuned': ev_b2['j_life_rate'], 'fail_b2': ev_b2['fail_pct'],
                'j_q1b_ridge_tune_plus_cp': ev_q1b['j_life_rate'], 'fail_q1b': ev_q1b['fail_pct'],
                'j_b7_linex_retuned': ev_b7['j_life_rate'], 'fail_b7': ev_b7['fail_pct'],
                'j_b9_linex_tune_plus_cp': ev_b9['j_life_rate'], 'fail_b9': ev_b9['fail_pct'],
            })

    df_fw_seed = pd.DataFrame(fair_waste_seed_rows)
    df_fw_seed.to_csv(f'{out_dir}/fair_reoptimized_cost_sensitivity_per_seed.csv', index=False)
    fw_sum_rows = []
    for w_val, grp in df_fw_seed.groupby('c_waste', sort=False):
        st_b9_b7 = _paired_stats(grp['j_b9_linex_tune_plus_cp'] - grp['j_b7_linex_retuned'])
        st_b9_b2 = _paired_stats(grp['j_b9_linex_tune_plus_cp'] - grp['j_b2_ridge_retuned'])
        indep_paired_rows.append({'parameter': 'c_waste', 'value': w_val, 'comparison': 'B9 minus j_b7_linex_retuned', **st_b9_b7})
        indep_paired_rows.append({'parameter': 'c_waste', 'value': w_val, 'comparison': 'B9 minus j_b2_ridge_retuned', **st_b9_b2})
        fw_sum_rows.append({
            'c_waste': w_val,
            'j_b5_cycle_cp': float(grp['j_b5_cycle_cp'].mean()),
            'j_b8_pure_unit_cp': float(grp['j_b8_pure_unit_cp'].mean()),
            'j_b2_ridge_retuned': float(grp['j_b2_ridge_retuned'].mean()),
            'fail_b2_pct': float(grp['fail_b2'].mean()),
            'j_q1b_ridge_tune_plus_cp': float(grp['j_q1b_ridge_tune_plus_cp'].mean()),
            'fail_q1b_pct': float(grp['fail_q1b'].mean()),
            'j_b7_linex_retuned': float(grp['j_b7_linex_retuned'].mean()),
            'fail_b7_pct': float(grp['fail_b7'].mean()),
            'j_b9_linex_tune_plus_cp': float(grp['j_b9_linex_tune_plus_cp'].mean()),
            'fail_b9_pct': float(grp['fail_b9'].mean()),
            'diff_b9_minus_b7_mean': st_b9_b7['mean_difference'],
            'diff_b9_minus_b7_ci95_low': st_b9_b7['ci95_low'],
            'diff_b9_minus_b7_ci95_high': st_b9_b7['ci95_high'],
            'diff_b9_minus_b2_mean': st_b9_b2['mean_difference'],
            'diff_b9_minus_b2_ci95_low': st_b9_b2['ci95_low'],
            'diff_b9_minus_b2_ci95_high': st_b9_b2['ci95_high'],
        })
    pd.DataFrame(fw_sum_rows).to_csv(f'{out_dir}/fair_reoptimized_cost_sensitivity.csv', index=False)

    rho_grid = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    fair_rho_rows = []
    for r_val in rho_grid:
        for sd in SEEDS:
            (
                df, df_fit, df_tune, df_cal, df_test,
                X_fit, X_tune, X_cal, X_test,
                y_fit, y_tune, y_cal, y_test,
                pr_tune, pr_cal, pr_test,
                _, _, _,
                sc_ridge_c, sc_ridge_te, _, _, _
            ) = fd004_cache[sd]
            sd_ridge = float(np.std(pr_tune - y_tune))
            sc_ridge_t = np.full(len(df_tune), sd_ridge)
            rc_scores = np.sort(pr_cal - y_cal)
            q_rc = max(0.0, float(rc_scores[math.ceil((len(rc_scores) + 1) * (1.0 - EPSILON)) - 1]))
            B_cal_ridge = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pr_cal[np.asarray(pos)], sc_ridge_c[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b8, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=0.0)
            costs_b2_r = [evaluate_fleet(df_tune, pr_tune, sc_ridge_t, g, rho=r_val)['j_life_rate'] for g in grid_b2]
            m_b2_r = float(grid_b2[int(np.argmin(costs_b2_r))])
            lam_q1b_r, _ = choose_static_margin(B_cal_ridge, alpha=EPSILON, tuned_margin=m_b2_r)

            h_dc_r = PrognosticHead().fit(X_fit, y_fit, kind='linex', rho=r_val, reg=14.0, focus_window=True)
            pt_t = h_dc_r.predict(X_tune, df_tune, causal_smooth=True)
            pt_c = h_dc_r.predict(X_cal, df_cal, causal_smooth=True)
            pt_te = h_dc_r.predict(X_test, df_test, causal_smooth=True)
            b_r = float(np.mean(pt_t - y_tune))
            pt_t = np.clip(pt_t - 0.85 * b_r, 0.0, 125.0)
            pt_c = np.clip(pt_c - 0.85 * b_r, 0.0, 125.0)
            pt_te = np.clip(pt_te - 0.85 * b_r, 0.0, 125.0)
            sd_r = float(np.std(pt_t - y_tune))
            sc_t_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_t) / 9.0))
            sc_c_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_c) / 9.0))
            sc_te_r = sd_r * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_te) / 9.0))
            B_c_r = [
                unit_critical_score(
                    df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                    pt_c[np.asarray(pos)], sc_c_r[np.asarray(pos)],
                    float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                )
                for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
            ]
            lam_b7_r = tune_differentiable_policy_margin(df_tune, pt_t, sc_t_r, rho=r_val)
            lam_b9_r, _ = choose_static_margin(B_c_r, alpha=EPSILON, tuned_margin=lam_b7_r)

            ev_b5 = evaluate_fleet(df_test, pr_test, np.ones(len(df_test)), q_rc, rho=r_val)
            ev_b8 = evaluate_fleet(df_test, pr_test, sc_ridge_te, lam_b8, rho=r_val)
            ev_b2 = evaluate_fleet(df_test, pr_test, sc_ridge_te, m_b2_r, rho=r_val)
            ev_q1b = evaluate_fleet(df_test, pr_test, sc_ridge_te, lam_q1b_r, rho=r_val)
            ev_b7 = evaluate_fleet(df_test, pt_te, sc_te_r, lam_b7_r, rho=r_val)
            ev_b9 = evaluate_fleet(df_test, pt_te, sc_te_r, lam_b9_r, rho=r_val)

            fair_rho_rows.append({
                'rho': r_val, 'seed': sd,
                'j_b5_cycle_cp': ev_b5['j_life_rate'],
                'j_b8_pure_unit_cp': ev_b8['j_life_rate'],
                'j_b2_ridge_retuned': ev_b2['j_life_rate'], 'fail_b2_pct': ev_b2['fail_pct'],
                'j_q1b_ridge_tune_plus_cp': ev_q1b['j_life_rate'], 'fail_q1b_pct': ev_q1b['fail_pct'],
                'j_b7_linex_retuned': ev_b7['j_life_rate'], 'fail_b7_pct': ev_b7['fail_pct'],
                'j_b9_linex_tune_plus_cp': ev_b9['j_life_rate'], 'fail_b9_pct': ev_b9['fail_pct'],
            })
    df_fr_seed = pd.DataFrame(fair_rho_rows)
    df_fr_seed.to_csv(f'{out_dir}/fair_reoptimized_rho_sensitivity_per_seed.csv', index=False)
    for r_val, grp in df_fr_seed.groupby('rho', sort=False):
        st_b9_b7 = _paired_stats(grp['j_b9_linex_tune_plus_cp'] - grp['j_b7_linex_retuned'])
        st_b9_b2 = _paired_stats(grp['j_b9_linex_tune_plus_cp'] - grp['j_b2_ridge_retuned'])
        indep_paired_rows.append({'parameter': 'rho', 'value': r_val, 'comparison': 'B9 minus j_b7_linex_retuned', **st_b9_b7})
        indep_paired_rows.append({'parameter': 'rho', 'value': r_val, 'comparison': 'B9 minus j_b2_ridge_retuned', **st_b9_b2})
    df_fr_seed.groupby('rho', as_index=False).mean().drop(columns=['seed'], errors='ignore').to_csv(
        f'{out_dir}/fair_reoptimized_rho_sensitivity.csv', index=False
    )
    pd.DataFrame(indep_paired_rows).to_csv(f'{out_dir}/fair_sensitivity_independent_paired.csv', index=False)

    # --- 4. Recovery Window Diagnostics, Running-Max Isolation (Refitted & Fixed-Head) & Frozen Transfer ---
    rec_diag_rows = []
    rmax_iso_rows = []
    frozen_transfer_rows = []
    gen_shift_rows = []

    def _global_params_for_check(seed_val, n_reg):
        rng_chk = np.random.RandomState(seed_val)
        c_chk = rng_chk.uniform(-1.5, 1.5, size=(n_reg, 3))
        g_chk = rng_chk.uniform(0.75, 1.35, size=(n_reg, 14))
        o_chk = rng_chk.uniform(-1.8, 1.8, size=(n_reg, 14))
        l1_chk = rng_chk.uniform(0.6, 1.4, size=14)
        l2_chk = rng_chk.uniform(0.5, 1.3, size=14)
        return c_chk, g_chk, o_chk, l1_chk, l2_chk

    for sd in SEEDS:
        # Audit legacy unseen_regime vs neg_dips generator parameter shift
        src_gp = _global_params_for_check(sd, 6)
        tgt_gp = _global_params_for_check(sd, 7)
        ds_base = generate_synthetic_fleet('S-FD004', seed=sd, disturbance_mode='neg_dips')
        dt_comp = generate_synthetic_fleet('S-FD004', seed=sd, disturbance_mode='unseen_regime')
        dt_iso = generate_synthetic_fleet('S-FD004', seed=sd, disturbance_mode='unseen_regime_isolated')
        ts_b = ds_base.groupby('unit_id').max_cycle.first()
        tt_c = dt_comp.groupby('unit_id').max_cycle.first()
        tt_i = dt_iso.groupby('unit_id').max_cycle.first()
        assert int(np.sum(ts_b.loc[131:] != tt_i.loc[131:])) == 0, "Isolated unseen regime must preserve 100% of test lifetimes!"
        gen_shift_rows.append({
            'seed': sd,
            'known_regime_centers_changed': int(np.sum(src_gp[0] != tgt_gp[0][:6])),
            'known_regime_gains_changed': int(np.sum(src_gp[1] != tgt_gp[1][:6])),
            'known_regime_offsets_changed': int(np.sum(src_gp[2] != tgt_gp[2][:6])),
            'sensor_loading1_magnitudes_changed': int(np.sum(src_gp[3] != tgt_gp[3])),
            'sensor_loading2_magnitudes_changed': int(np.sum(src_gp[4] != tgt_gp[4])),
            'test_unit_lifetimes_changed': int(np.sum(ts_b.loc[131:] != tt_c.loc[131:])),
            'isolated_mode_test_lifetimes_changed': int(np.sum(ts_b.loc[131:] != tt_i.loc[131:])),
            'test_units': 50
        })

        # Prepare frozen source models trained on default 'neg_dips'
        (
            _, df_fit_src, df_tune_src, df_cal_src, _,
            X_fit_m_src, X_tune_m_src, X_cal_m_src, _,
            y_fit_src, y_tune_src, y_cal_src, _,
            _, _, _,
            pdc_tune_src, pdc_cal_src, _,
            _, _, sc_dc_t_src, sc_dc_c_src, _
        ) = fd004_cache[sd]

        enc_unc_src = CausalMonotoneEncoder(True, False, True).fit(df_fit_src)
        X_fit_u_src, _ = enc_unc_src.transform(df_fit_src)
        h_unc_src = PrognosticHead().fit(X_fit_u_src, y_fit_src, kind='ridge', reg=14.0)

        enc_mono_on_src = CausalMonotoneEncoder(True, True, enforce_running_max=True).fit(df_fit_src)
        h_mono_r_on_src = PrognosticHead().fit(X_fit_m_src, y_fit_src, kind='ridge', reg=14.0)
        h_dc_on_src = PrognosticHead().fit(X_fit_m_src, y_fit_src, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)
        b_dc_on_src = float(np.mean(h_dc_on_src.predict(X_tune_m_src, df_tune_src, causal_smooth=True) - y_tune_src))
        sd_dc_on_src = float(np.std(pdc_tune_src - y_tune_src))
        B_cal_on_src = [
            unit_critical_score(
                df_cal_src.iloc[np.asarray(pos)].cycle.to_numpy(),
                pdc_cal_src[np.asarray(pos)], sc_dc_c_src[np.asarray(pos)],
                float(df_cal_src.iloc[np.asarray(pos)].max_cycle.iloc[0])
            )
            for _, pos in df_cal_src.groupby('unit_id', sort=False).indices.items()
        ]
        lam_t_on_src = tune_differentiable_policy_margin(df_tune_src, pdc_tune_src, sc_dc_t_src)
        lam_dc_on_src, _ = choose_static_margin(B_cal_on_src, alpha=EPSILON, tuned_margin=lam_t_on_src)

        enc_mono_off_src = CausalMonotoneEncoder(True, True, enforce_running_max=False).fit(df_fit_src)
        X_fit_moff_src, _ = enc_mono_off_src.transform(df_fit_src)
        X_tune_moff_src, _ = enc_mono_off_src.transform(df_tune_src)
        X_cal_moff_src, _ = enc_mono_off_src.transform(df_cal_src)
        h_mono_r_off_src = PrognosticHead().fit(X_fit_moff_src, y_fit_src, kind='ridge', reg=14.0)
        h_dc_off_src = PrognosticHead().fit(X_fit_moff_src, y_fit_src, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)
        pt_t_off_src = h_dc_off_src.predict(X_tune_moff_src, df_tune_src, causal_smooth=True)
        pt_c_off_src = h_dc_off_src.predict(X_cal_moff_src, df_cal_src, causal_smooth=True)
        b_dc_off_src = float(np.mean(pt_t_off_src - y_tune_src))
        pt_t_off_src = np.clip(pt_t_off_src - 0.85 * b_dc_off_src, 0.0, 125.0)
        pt_c_off_src = np.clip(pt_c_off_src - 0.85 * b_dc_off_src, 0.0, 125.0)
        sd_dc_off_src = float(np.std(pt_t_off_src - y_tune_src))
        sc_t_off_src = sd_dc_off_src * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_t_off_src) / 9.0))
        sc_c_off_src = sd_dc_off_src * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_c_off_src) / 9.0))
        B_cal_off_src = [
            unit_critical_score(
                df_cal_src.iloc[np.asarray(pos)].cycle.to_numpy(),
                pt_c_off_src[np.asarray(pos)], sc_c_off_src[np.asarray(pos)],
                float(df_cal_src.iloc[np.asarray(pos)].max_cycle.iloc[0])
            )
            for _, pos in df_cal_src.groupby('unit_id', sort=False).indices.items()
        ]
        lam_t_off_src = tune_differentiable_policy_margin(df_tune_src, pt_t_off_src, sc_t_off_src)
        lam_dc_off_src, _ = choose_static_margin(B_cal_off_src, alpha=EPSILON, tuned_margin=lam_t_off_src)

        # Evaluate frozen source-model transfer across all 7 disturbance modes (including isolated unseen regime 7)
        for d_mode in ['neg_dips', 'no_shocks', 'pos_spikes', 'bidirectional', 'non_monotone_true', 'unseen_regime', 'unseen_regime_isolated']:
            df_m = generate_synthetic_fleet('S-FD004', n_units=180, seed=sd, disturbance_mode=d_mode)
            df_te_m = df_m[df_m.unit_id > 130].copy()
            y_te_m = df_te_m.true_rul.to_numpy()

            X_te_u_tr, _ = enc_unc_src.transform(df_te_m)
            p_unc_tr = h_unc_src.predict(X_te_u_tr)

            X_te_mon_tr, hi_mon_tr = enc_mono_on_src.transform(df_te_m)
            p_mon_r_tr = h_mono_r_on_src.predict(X_te_mon_tr)
            p_dc_tr = np.clip(h_dc_on_src.predict(X_te_mon_tr, df_te_m, causal_smooth=True) - 0.85 * b_dc_on_src, 0.0, 125.0)
            sc_dc_tr = sd_dc_on_src * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_dc_tr) / 9.0))
            ev_dc_tr = evaluate_fleet(df_te_m, p_dc_tr, sc_dc_tr, lam_dc_on_src, ell=LEAD_TIME, rho=RHO_DEFAULT)

            frozen_transfer_rows.append({
                'disturbance_mode': d_mode, 'seed': sd,
                'rmse_unconstrained_frozen': float(np.sqrt(np.mean((p_unc_tr - y_te_m) ** 2))),
                'rmse_monotone_ridge_frozen': float(np.sqrt(np.mean((p_mon_r_tr - y_te_m) ** 2))),
                'rmse_dc_pdm_frozen': float(np.sqrt(np.mean((p_dc_tr - y_te_m) ** 2))),
                'j_life_dc_pdm_frozen': ev_dc_tr['j_life_rate'],
                'fail_pct_dc_pdm_frozen': ev_dc_tr['fail_pct']
            })

            if d_mode == 'non_monotone_true':
                for uid, g_u in df_m.groupby('unit_id', sort=False):
                    dh = np.diff(g_u.latent_deg.to_numpy())
                    pos_frac = g_u.cycle.to_numpy()[1:] / float(g_u.max_cycle.iloc[0])
                    neg_m = dh < -1e-12
                    in_win = (pos_frac >= 0.55) & (pos_frac <= 0.78)
                    rec_diag_rows.append({
                        'seed': sd, 'unit_id': int(uid),
                        'has_negative_increment': bool(neg_m.any()),
                        'first_negative_time_fraction': float(pos_frac[neg_m].min()) if neg_m.any() else np.nan,
                        'last_negative_time_fraction': float(pos_frac[neg_m].max()) if neg_m.any() else np.nan,
                        'negative_steps_inside_claimed_window': int(np.sum(neg_m & in_win)),
                        'negative_steps_outside_claimed_window': int(np.sum(neg_m & (~in_win)))
                    })

                true_neg_mask = np.zeros(len(df_te_m), dtype=bool)
                for _, pos_u in df_te_m.groupby('unit_id', sort=False).indices.items():
                    pos_u = np.asarray(pos_u)
                    lat_u = df_te_m.iloc[pos_u].latent_deg.to_numpy()
                    dh_u = np.r_[0.0, np.diff(lat_u)]
                    true_neg_mask[pos_u] = (dh_u < -1e-12)
                mid_win_mask = (df_te_m.cycle.to_numpy() >= 0.55 * df_te_m.max_cycle.to_numpy()) & (
                    df_te_m.cycle.to_numpy() <= 0.78 * df_te_m.max_cycle.to_numpy()
                )

                # Frozen transfer: Refitted Running-Max ON vs OFF
                X_te_moff_tr, hi_moff_tr = enc_mono_off_src.transform(df_te_m)
                p_moff_r_tr = h_mono_r_off_src.predict(X_te_moff_tr)
                p_dc_off_tr = np.clip(h_dc_off_src.predict(X_te_moff_tr, df_te_m, causal_smooth=True) - 0.85 * b_dc_off_src, 0.0, 125.0)
                sc_dc_off_tr = sd_dc_off_src * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_dc_off_tr) / 9.0))
                ev_dc_off_tr = evaluate_fleet(df_te_m, p_dc_off_tr, sc_dc_off_tr, lam_dc_off_src, ell=LEAD_TIME, rho=RHO_DEFAULT)

                # Also evaluate Fixed-Head Test-Time Intervention: exact ON head weights + test-time Running-Max OFF features!
                p_fixed_r_off = h_mono_r_on_src.predict(X_te_moff_tr)
                p_fixed_dc_off = np.clip(h_dc_on_src.predict(X_te_moff_tr, df_te_m, causal_smooth=True) - 0.85 * b_dc_on_src, 0.0, 125.0)
                sc_fixed_dc_off = sd_dc_on_src * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_fixed_dc_off) / 9.0))
                ev_fixed_dc_off = evaluate_fleet(df_te_m, p_fixed_dc_off, sc_fixed_dc_off, lam_dc_on_src, ell=LEAD_TIME, rho=RHO_DEFAULT)

                for reg_mode, r_max_flag, hi_v, p_r_v, p_dc_v, ev_v in [
                    ('Frozen Source Transfer (Trained on neg_dips)', True, hi_mon_tr, p_mon_r_tr, p_dc_tr, ev_dc_tr),
                    ('Frozen Source Transfer (Trained on neg_dips)', False, hi_moff_tr, p_moff_r_tr, p_dc_off_tr, ev_dc_off_tr),
                    ('Fixed ON Head Coefficients (Test-Time Clamping Toggle Only)', False, hi_moff_tr, p_fixed_r_off, p_fixed_dc_off, ev_fixed_dc_off),
                ]:
                    dh_est = []
                    for _, pos_u in df_te_m.groupby('unit_id', sort=False).indices.items():
                        pos_u = np.asarray(pos_u)
                        dh_est.append(np.r_[0.0, np.diff(hi_v[pos_u])])
                    dh_est_vec = np.concatenate(dh_est)
                    clamped_frac_on_neg = float(np.mean(dh_est_vec[true_neg_mask] >= -1e-12))

                    rmax_iso_rows.append({
                        'protocol': reg_mode, 'enforce_running_max': r_max_flag, 'seed': sd,
                        'clamped_non_decreasing_frac_on_true_recovery': clamped_frac_on_neg,
                        'ridge_rmse_all': float(np.sqrt(np.mean((p_r_v - y_te_m) ** 2))),
                        'ridge_rmse_true_neg_mask': float(np.sqrt(np.mean((p_r_v[true_neg_mask] - y_te_m[true_neg_mask]) ** 2))),
                        'ridge_rmse_mid_window_055_078': float(np.sqrt(np.mean((p_r_v[mid_win_mask] - y_te_m[mid_win_mask]) ** 2))),
                        'dc_pdm_rmse_all': float(np.sqrt(np.mean((p_dc_v - y_te_m) ** 2))),
                        'dc_pdm_rmse_true_neg_mask': float(np.sqrt(np.mean((p_dc_v[true_neg_mask] - y_te_m[true_neg_mask]) ** 2))),
                        'dc_pdm_rmse_mid_window_055_078': float(np.sqrt(np.mean((p_dc_v[mid_win_mask] - y_te_m[mid_win_mask]) ** 2))),
                        'dc_pdm_overest_true_neg_mask': float(np.mean(p_dc_v[true_neg_mask] - y_te_m[true_neg_mask])),
                        'j_life_dc_pdm': ev_v['j_life_rate'],
                        'fail_pct_dc_pdm': ev_v['fail_pct']
                    })

                # Per-mode Refit on non_monotone_true: Running-Max ON vs OFF (each branch refitted & recalibrated)
                df_fit_m = df_m[df_m.unit_id <= 60].copy()
                df_tune_m = df_m[(df_m.unit_id > 60) & (df_m.unit_id <= 90)].copy()
                df_cal_m = df_m[(df_m.unit_id > 90) & (df_m.unit_id <= 130)].copy()
                y_fit_m = df_fit_m.true_rul.to_numpy()
                y_tune_m = df_tune_m.true_rul.to_numpy()
                y_cal_m = df_cal_m.true_rul.to_numpy()

                for r_max_flag in [True, False]:
                    enc_m_iso = CausalMonotoneEncoder(True, True, enforce_running_max=r_max_flag).fit(df_fit_m)
                    X_f_iso, _ = enc_m_iso.transform(df_fit_m)
                    X_t_iso, _ = enc_m_iso.transform(df_tune_m)
                    X_c_iso, _ = enc_m_iso.transform(df_cal_m)
                    X_te_iso, hi_te_iso = enc_m_iso.transform(df_te_m)

                    h_r_iso = PrognosticHead().fit(X_f_iso, y_fit_m, kind='ridge', reg=14.0)
                    p_r_iso = h_r_iso.predict(X_te_iso)

                    h_dc_iso = PrognosticHead().fit(X_f_iso, y_fit_m, kind='linex', rho=RHO_DEFAULT, reg=14.0, focus_window=True)
                    pt_t_iso = h_dc_iso.predict(X_t_iso, df_tune_m, causal_smooth=True)
                    pt_c_iso = h_dc_iso.predict(X_c_iso, df_cal_m, causal_smooth=True)
                    pt_te_iso = h_dc_iso.predict(X_te_iso, df_te_m, causal_smooth=True)
                    b_iso = float(np.mean(pt_t_iso - y_tune_m))
                    pt_t_iso = np.clip(pt_t_iso - 0.85 * b_iso, 0.0, 125.0)
                    pt_c_iso = np.clip(pt_c_iso - 0.85 * b_iso, 0.0, 125.0)
                    pt_te_iso = np.clip(pt_te_iso - 0.85 * b_iso, 0.0, 125.0)
                    sd_iso = float(np.std(pt_t_iso - y_tune_m))
                    sc_t_iso = sd_iso * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_t_iso) / 9.0))
                    sc_c_iso = sd_iso * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_c_iso) / 9.0))
                    sc_te_iso = sd_iso * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - pt_te_iso) / 9.0))
                    B_cal_iso = [
                        unit_critical_score(
                            df_cal_m.iloc[np.asarray(pos)].cycle.to_numpy(),
                            pt_c_iso[np.asarray(pos)], sc_c_iso[np.asarray(pos)],
                            float(df_cal_m.iloc[np.asarray(pos)].max_cycle.iloc[0])
                        )
                        for _, pos in df_cal_m.groupby('unit_id', sort=False).indices.items()
                    ]
                    lam_t_iso = tune_differentiable_policy_margin(df_tune_m, pt_t_iso, sc_t_iso)
                    lam_dc_iso, _ = choose_static_margin(B_cal_iso, alpha=EPSILON, tuned_margin=lam_t_iso)
                    ev_iso = evaluate_fleet(df_te_m, pt_te_iso, sc_te_iso, lam_dc_iso, ell=LEAD_TIME, rho=RHO_DEFAULT)

                    dh_est = []
                    for _, pos_u in df_te_m.groupby('unit_id', sort=False).indices.items():
                        pos_u = np.asarray(pos_u)
                        dh_est.append(np.r_[0.0, np.diff(hi_te_iso[pos_u])])
                    dh_est_vec = np.concatenate(dh_est)
                    clamped_frac_on_neg = float(np.mean(dh_est_vec[true_neg_mask] >= -1e-12))

                    rmax_iso_rows.append({
                        'protocol': 'Per-Mode Refit on non_monotone_true', 'enforce_running_max': r_max_flag, 'seed': sd,
                        'clamped_non_decreasing_frac_on_true_recovery': clamped_frac_on_neg,
                        'ridge_rmse_all': float(np.sqrt(np.mean((p_r_iso - y_te_m) ** 2))),
                        'ridge_rmse_true_neg_mask': float(np.sqrt(np.mean((p_r_iso[true_neg_mask] - y_te_m[true_neg_mask]) ** 2))),
                        'ridge_rmse_mid_window_055_078': float(np.sqrt(np.mean((p_r_iso[mid_win_mask] - y_te_m[mid_win_mask]) ** 2))),
                        'dc_pdm_rmse_all': float(np.sqrt(np.mean((pt_te_iso - y_te_m) ** 2))),
                        'dc_pdm_rmse_true_neg_mask': float(np.sqrt(np.mean((pt_te_iso[true_neg_mask] - y_te_m[true_neg_mask]) ** 2))),
                        'dc_pdm_rmse_mid_window_055_078': float(np.sqrt(np.mean((pt_te_iso[mid_win_mask] - y_te_m[mid_win_mask]) ** 2))),
                        'dc_pdm_overest_true_neg_mask': float(np.mean(pt_te_iso[true_neg_mask] - y_te_m[true_neg_mask])),
                        'j_life_dc_pdm': ev_iso['j_life_rate'],
                        'fail_pct_dc_pdm': ev_iso['fail_pct']
                    })

    pd.DataFrame(gen_shift_rows).to_csv(f'{out_dir}/unseen_regime_generator_shift.csv', index=False)
    pd.DataFrame(rec_diag_rows).to_csv(f'{out_dir}/recovery_window_diagnostics.csv', index=False)

    df_rmax_seed = pd.DataFrame(rmax_iso_rows)
    df_rmax_seed.to_csv(f'{out_dir}/running_max_isolation_per_seed.csv', index=False)
    df_rmax_seed.groupby(['protocol', 'enforce_running_max'], sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore').to_csv(
        f'{out_dir}/running_max_isolation_summary.csv', index=False
    )

    rmax_paired_rows = []
    for prot in ['Frozen Source Transfer (Trained on neg_dips)', 'Per-Mode Refit on non_monotone_true']:
        sub_on = df_rmax_seed[(df_rmax_seed.protocol == prot) & (df_rmax_seed.enforce_running_max == True)].sort_values('seed')
        sub_off = df_rmax_seed[(df_rmax_seed.protocol == prot) & (df_rmax_seed.enforce_running_max == False)].sort_values('seed')
        for met in ['dc_pdm_rmse_true_neg_mask', 'j_life_dc_pdm', 'fail_pct_dc_pdm']:
            st_m = _paired_stats(sub_off[met].to_numpy() - sub_on[met].to_numpy())
            rmax_paired_rows.append({
                'protocol': prot, 'comparison': 'Running-max OFF minus ON', 'metric': met,
                'n': st_m['n'], 'mean_difference': st_m['mean_difference'],
                'ci95_low': st_m['ci95_low'], 'ci95_high': st_m['ci95_high'], 'p_t': st_m['p_t']
            })
    sub_on_f = df_rmax_seed[(df_rmax_seed.protocol == 'Frozen Source Transfer (Trained on neg_dips)') & (df_rmax_seed.enforce_running_max == True)].sort_values('seed')
    sub_fix_off = df_rmax_seed[(df_rmax_seed.protocol == 'Fixed ON Head Coefficients (Test-Time Clamping Toggle Only)') & (df_rmax_seed.enforce_running_max == False)].sort_values('seed')
    for met in ['dc_pdm_rmse_true_neg_mask', 'j_life_dc_pdm', 'fail_pct_dc_pdm']:
        st_m = _paired_stats(sub_fix_off[met].to_numpy() - sub_on_f[met].to_numpy())
        rmax_paired_rows.append({
            'protocol': 'Fixed ON Head Coefficients (Test-Time Clamping Toggle Only)',
            'comparison': 'Test-Time Running-max OFF minus ON (Fixed Head)', 'metric': met,
            'n': st_m['n'], 'mean_difference': st_m['mean_difference'],
            'ci95_low': st_m['ci95_low'], 'ci95_high': st_m['ci95_high'], 'p_t': st_m['p_t']
        })
    pd.DataFrame(rmax_paired_rows).to_csv(f'{out_dir}/running_max_paired_uncertainty.csv', index=False)

    df_ft_seed = pd.DataFrame(frozen_transfer_rows)
    df_ft_seed.to_csv(f'{out_dir}/mismatch_frozen_transfer_per_seed.csv', index=False)
    df_ft_seed.groupby('disturbance_mode', sort=False, as_index=False).mean().drop(columns=['seed'], errors='ignore').to_csv(
        f'{out_dir}/mismatch_frozen_transfer_summary.csv', index=False
    )

    # --- 5. Executable Full-Pipeline Streaming (25 units, 5,251 steps) & Direct Per-Unit Batch Replay (250 units) ---
    cpu_model_str = _get_cpu_model_name()
    py_ver_short = sys.version.split()[0]
    arch_specs = [
        ('Unconstrained EMA + Ridge + Cycle-CP', 'Raw Ridge Cycle-CP', False, 'ridge', False, False, 'cycle', False),
        ('Causal Monotone + Ridge + Unit-CP (Q1b)', 'Monotone Ridge Unit-CP', True, 'ridge', False, False, 'unit_plus_tune', False),
        ('DC-PdM Static (Causal Monotone + Linex, B9)', 'DC Static', True, 'linex', True, True, 'unit_plus_tune', False),
        ('DC-PdM Online (Causal Monotone + Event-ACI, B10)', 'DC Online', True, 'linex', True, True, 'unit_plus_tune', True),
    ]

    # Warmup pass (1 seed) before recording wall-clock measurements
    _df_w = fd004_cache[SEEDS[0]][1]
    _enc_w = CausalMonotoneEncoder(True, True).fit(_df_w)
    _xw, _ = _enc_w.transform(_df_w)
    _ = PrognosticHead().fit(_xw, _df_w.true_rul.to_numpy(), kind='linex', focus_window=True)

    raw_timing_rows = []
    step_level_raw_rows = []
    unit_level_raw_rows = []
    stream_vs_ref_rows = []
    prefix_forecast_eq_rows = []
    pooled_stream_steps = {a[0]: [] for a in arch_specs}
    pooled_unit_direct_ms = {a[0]: [] for a in arch_specs}

    for arch_name, short_arch, mono_flag, kind_k, focus_w, smooth_flag, cal_m, online_flag in arch_specs:
        for sd in SEEDS:
            (
                df, df_fit, df_tune, df_cal, df_test,
                _, _, _, _,
                y_fit, y_tune, y_cal, y_test,
                _, _, _, _, _, _, _, _, _, _, _
            ) = fd004_cache[sd]

            # (A) Single-wrapper End-to-End Offline Fit+Tune+Cal Timer (ZERO test-set operations!)
            t_off0 = time.perf_counter()
            enc = CausalMonotoneEncoder(regime_norm=True, monotone_state=mono_flag).fit(df_fit)
            X_f, _ = enc.transform(df_fit)
            X_tu, _ = enc.transform(df_tune)
            X_ca, _ = enc.transform(df_cal)
            head = PrognosticHead().fit(X_f, y_fit, kind=kind_k, rho=RHO_DEFAULT, reg=14.0, focus_window=focus_w)
            p_tu = head.predict(X_tu, df_tune, causal_smooth=smooth_flag)
            p_ca = head.predict(X_ca, df_cal, causal_smooth=smooth_flag)
            b_tu = float(np.mean(p_tu - y_tune))
            p_tu = np.clip(p_tu - 0.85 * b_tu, 0.0, 125.0)
            p_ca = np.clip(p_ca - 0.85 * b_tu, 0.0, 125.0)
            sd_tu = float(np.std(p_tu - y_tune))
            sc_tu = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_tu) / 9.0))
            sc_ca = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_ca) / 9.0))
            if cal_m == 'cycle':
                cyc_scores = np.sort((p_ca - y_cal) / sc_ca)
                k_c = math.ceil((len(cyc_scores) + 1) * (1.0 - EPSILON))
                lam_static = max(0.0, float(cyc_scores[min(len(cyc_scores) - 1, k_c - 1)]))
                B_sorted = cyc_scores
                lam_tu = 0.0
            else:
                lam_tu = tune_differentiable_policy_margin(df_tune, p_tu, sc_tu)
                B_ca = [
                    unit_critical_score(
                        df_cal.iloc[np.asarray(pos)].cycle.to_numpy(),
                        p_ca[np.asarray(pos)], sc_ca[np.asarray(pos)],
                        float(df_cal.iloc[np.asarray(pos)].max_cycle.iloc[0])
                    )
                    for _, pos in df_cal.groupby('unit_id', sort=False).indices.items()
                ]
                lam_static, _ = choose_static_margin(B_ca, alpha=EPSILON, tuned_margin=lam_tu)
                B_sorted = np.sort(np.asarray(B_ca, dtype=float))
            direct_offline_ms = (time.perf_counter() - t_off0) * 1000.0

            # (B) Amortized Batch Inference per cycle on D_test
            t_b_enc0 = time.perf_counter()
            X_te_b, _ = enc.transform(df_test)
            enc_batch_us = (time.perf_counter() - t_b_enc0) * 1e6 / len(df_test)

            t_b_hd0 = time.perf_counter()
            p_te_b = head.predict(X_te_b, df_test, causal_smooth=smooth_flag)
            p_te_b = np.clip(p_te_b - 0.85 * b_tu, 0.0, 125.0)
            sc_te_b = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_te_b) / 9.0))
            head_batch_us = (time.perf_counter() - t_b_hd0) * 1e6 / len(df_test)

            # (C) Direct Per-Unit Batch Replay across all 50 test units per seed (250 units total per architecture)
            a_cur_unit = ONLINE_ALPHA_TARGET
            seed_unit_ms = []
            ref_unit_outcomes = {}
            for uid_u, pos_u in df_test.groupby('unit_id', sort=False).indices.items():
                pos_u_arr = np.asarray(pos_u)
                df_u = df_test.iloc[pos_u_arr].copy()
                t_u0 = time.perf_counter()
                X_u, _ = enc.transform(df_u)
                p_u = head.predict(X_u, df_u, causal_smooth=smooth_flag)
                p_u = np.clip(p_u - 0.85 * b_tu, 0.0, 125.0)
                sc_u = sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - p_u) / 9.0))
                if not online_flag:
                    lam_u = lam_static
                else:
                    rk = min(len(B_sorted), max(1, math.ceil((len(B_sorted) + 1) * (1.0 - np.clip(a_cur_unit, 0.005, 0.25)))))
                    lam_u = max(0.0, lam_tu, float(B_sorted[rk - 1]))
                out_u = replay_unit_stopping_policy(
                    df_u.cycle.to_numpy(), p_u, sc_u, float(df_u.max_cycle.iloc[0]), LEAD_TIME, lam_u
                )
                if online_flag:
                    a_cur_unit = float(np.clip(a_cur_unit + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - out_u['Z_fail']), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))
                u_ms = (time.perf_counter() - t_u0) * 1000.0
                seed_unit_ms.append(u_ms)
                pooled_unit_direct_ms[arch_name].append(u_ms)
                ref_unit_outcomes[int(uid_u)] = (lam_u, out_u, p_u)
                unit_level_raw_rows.append({
                    'architecture': arch_name, 'seed': int(sd), 'unit_id': int(uid_u),
                    'T_u': int(df_u.max_cycle.iloc[0]), 'direct_unit_batch_replay_ms': u_ms
                })

            # (D) Full-Pipeline Single-Step Streaming Across Full Trajectory Lifetime (first 5 test units per seed = 25 units, 5,251 steps)
            test_uids = df_test.unit_id.unique()[:5]
            a_cur_stream = ONLINE_ALPHA_TARGET
            seed_stream_us = []
            for u_idx_s, uid_s in enumerate(test_uids):
                df_u_full = df_test[df_test.unit_id == uid_s].copy()
                T_u_val = float(df_u_full.max_cycle.iloc[0])
                if not online_flag:
                    lam_s = lam_static
                else:
                    rk_s = min(len(B_sorted), max(1, math.ceil((len(B_sorted) + 1) * (1.0 - np.clip(a_cur_stream, 0.005, 0.25)))))
                    lam_s = max(0.0, lam_tu, float(B_sorted[rk_s - 1]))
                sp_prev = 0.0
                alarm_triggered = False
                tau_s = np.inf
                stream_forecasts_u = []
                for t_idx in range(1, len(df_u_full) + 1):
                    df_pref = df_u_full.iloc[:t_idx]
                    t_s0 = time.perf_counter()
                    x_pref, _ = enc.transform(df_pref)
                    x_last = x_pref[-1]
                    z_std = (x_last - head.mu) / head.sd
                    raw_p_t = float(np.clip(head.w[0] + np.dot(z_std, head.w[1:]), 0.0, 125.0))
                    if smooth_flag:
                        sp_t = raw_p_t if t_idx == 1 else (0.62 * raw_p_t + 0.38 * max(0.0, sp_prev - 0.90))
                        sp_prev = sp_t
                    else:
                        sp_t = raw_p_t
                    f_t = float(np.clip(sp_t - 0.85 * b_tu, 0.0, 125.0))
                    s_t = float(sd_tu * (0.30 + 0.70 * expit((LEAD_TIME + 22.0 - f_t) / 9.0)))
                    g_t = f_t - lam_s * s_t
                    if (not alarm_triggered) and (g_t <= LEAD_TIME):
                        alarm_triggered = True
                        tau_s = float(t_idx)
                    if t_idx == len(df_u_full) and online_flag:
                        z_fail_s = 1 if (tau_s + LEAD_TIME >= T_u_val) else 0
                        a_cur_stream = float(np.clip(a_cur_stream + ONLINE_GAMMA * (ONLINE_ALPHA_TARGET - z_fail_s), ONLINE_ALPHA_MIN, ONLINE_ALPHA_MAX))
                    step_us = (time.perf_counter() - t_s0) * 1e6
                    seed_stream_us.append(step_us)
                    pooled_stream_steps[arch_name].append(step_us)
                    stream_forecasts_u.append(f_t)
                    step_level_raw_rows.append({
                        'architecture': arch_name, 'seed': int(sd), 'unit_id': int(uid_s),
                        'cycle': int(t_idx), 'step_latency_us': step_us
                    })

                # Verify exact stopping-policy & forecast equivalence between streaming path and reference batch replay
                ref_lam, ref_out, ref_p_u = ref_unit_outcomes[int(uid_s)]
                z_fail_s = 1 if (tau_s + LEAD_TIME >= T_u_val) else 0
                u_s = min(T_u_val, tau_s + LEAD_TIME)
                assert np.isclose(lam_s, ref_lam), f"Margin mismatch on {arch_name} seed {sd} unit {uid_s}"
                assert np.isclose(tau_s, ref_out['tau'], equal_nan=True), f"Alarm tau mismatch on {arch_name} seed {sd} unit {uid_s}: {tau_s} vs {ref_out['tau']}"
                assert z_fail_s == ref_out['Z_fail'], f"Failure mismatch on {arch_name} seed {sd} unit {uid_s}"
                stream_vs_ref_rows.append({
                    'seed': int(sd), 'architecture': short_arch, 'unit_id': int(uid_s),
                    'T': T_u_val, 'margin': lam_s,
                    'reference_tau': ref_out['tau'], 'timed_stream_tau': tau_s,
                    'reference_fail': ref_out['Z_fail'], 'timed_stream_fail': z_fail_s,
                    'alarm_cycle_difference': 0.0 if (not np.isfinite(tau_s) and not np.isfinite(ref_out['tau'])) else float(tau_s - ref_out['tau']),
                    'reference_U': ref_out['U_u'], 'stream_U': u_s
                })
                if sd == SEEDS[0] and u_idx_s == 0:
                    max_f_diff = float(np.max(np.abs(np.asarray(stream_forecasts_u) - ref_p_u)))
                    assert max_f_diff < 1e-12, f"Prefix forecast diff {max_f_diff} >= 1e-12!"
                    prefix_forecast_eq_rows.append({
                        'architecture': short_arch, 'seed': int(sd), 'unit_id': int(uid_s),
                        'max_prefix_forecast_abs_difference': max_f_diff
                    })

            raw_timing_rows.append({
                'cpu_model': cpu_model_str,
                'python_version': py_ver_short,
                'seed': sd,
                'architecture': arch_name,
                'encoder_batch_us_per_cycle': enc_batch_us,
                'head_batch_us_per_cycle': head_batch_us,
                'amortized_batch_us_per_cycle': enc_batch_us + head_batch_us,
                'full_stream_step_seed_p50_us': float(np.percentile(seed_stream_us, 50)),
                'full_stream_step_seed_p95_us': float(np.percentile(seed_stream_us, 95)),
                'direct_unit_trajectory_mean_ms': float(np.mean(seed_unit_ms)),
                'direct_unit_trajectory_p95_ms': float(np.percentile(seed_unit_ms, 95)),
                'direct_offline_fit_tune_cal_ms': direct_offline_ms,
                'n_stream_units_seed': len(test_uids),
                'n_stream_steps_seed': len(seed_stream_us),
                'n_units_seed': len(seed_unit_ms)
            })

    df_raw_time = pd.DataFrame(raw_timing_rows)
    df_raw_time.to_csv(f'{out_dir}/cpu_timing_raw_records.csv', index=False)
    pd.DataFrame(step_level_raw_rows).to_csv(f'{out_dir}/cpu_timing_step_level_raw.csv', index=False)
    pd.DataFrame(unit_level_raw_rows).to_csv(f'{out_dir}/cpu_timing_unit_level_raw.csv', index=False)
    pd.DataFrame(stream_vs_ref_rows).to_csv(f'{out_dir}/cpu_stream_vs_reference_per_unit.csv', index=False)
    pd.DataFrame(prefix_forecast_eq_rows).to_csv(f'{out_dir}/cpu_prefix_forecast_equivalence.csv', index=False)

    # Merge pooled full-pipeline streaming p50/p95 and direct per-unit batch replay / offline timings into cpu_resource_timing_summary.csv
    df_time_existing = pd.read_csv(f'{out_dir}/cpu_resource_timing_summary.csv')
    for drop_col in [
        'cpu_model', 'python_version', 'full_stream_step_pooled_p50_us', 'full_stream_step_pooled_p95_us',
        'direct_unit_wallclock_mean_ms', 'direct_unit_wallclock_p50_ms', 'direct_unit_wallclock_p95_ms',
        'direct_unit_batch_replay_mean_ms', 'direct_unit_batch_replay_p50_ms', 'direct_unit_batch_replay_p95_ms',
        'direct_offline_fit_tune_cal_ms', 'n_stream_units_timed', 'total_stream_steps_pooled', 'total_units_timed'
    ]:
        if drop_col in df_time_existing.columns:
            df_time_existing = df_time_existing.drop(columns=[drop_col])

    df_time_existing = df_time_existing.rename(columns={
        'streaming_prefix_p50_us': 'legacy_prefix_dot_microbench_p50_us',
        'streaming_prefix_p95_us': 'legacy_prefix_dot_microbench_p95_us',
        'measured_unit_replay_ms': 'component_sum_unit_estimate_ms',
        'full_offline_fit_tune_cal_ms': 'component_sum_offline_ms'
    })
    pooled_rows = []
    for arch_name, _, _, _, _, _, _, _ in arch_specs:
        sub_r = df_raw_time[df_raw_time['architecture'] == arch_name]
        steps_all = np.asarray(pooled_stream_steps[arch_name], dtype=float)
        units_all = np.asarray(pooled_unit_direct_ms[arch_name], dtype=float)
        pooled_rows.append({
            'architecture': arch_name,
            'cpu_model': cpu_model_str,
            'python_version': py_ver_short,
            'full_stream_step_pooled_p50_us': float(np.percentile(steps_all, 50)),
            'full_stream_step_pooled_p95_us': float(np.percentile(steps_all, 95)),
            'direct_unit_wallclock_mean_ms': float(np.mean(units_all)),
            'direct_unit_wallclock_p50_ms': float(np.percentile(units_all, 50)),
            'direct_unit_wallclock_p95_ms': float(np.percentile(units_all, 95)),
            'direct_unit_batch_replay_mean_ms': float(np.mean(units_all)),
            'direct_unit_batch_replay_p50_ms': float(np.percentile(units_all, 50)),
            'direct_unit_batch_replay_p95_ms': float(np.percentile(units_all, 95)),
            'direct_offline_fit_tune_cal_ms': float(sub_r['direct_offline_fit_tune_cal_ms'].mean()),
            'n_stream_units_timed': 25,
            'total_stream_steps_pooled': int(len(steps_all)),
            'total_units_timed': int(len(units_all))
        })
    df_time_merged = df_time_existing.merge(pd.DataFrame(pooled_rows), on='architecture', how='left')
    df_time_merged.to_csv(f'{out_dir}/cpu_resource_timing_summary.csv', index=False)
    return df_time_merged


def generate_reconstructed_figures_1_to_4(
    df_b1_b10_unit, df_b1_b10_seed, df_b1_b10_sum,
    df_censor_sum, df_mismatch_sum, df_attr_sum, df_mctrl_sum, df_attr_tests, df_drift,
    sample_replay_data, inter_dir=INTER_DIR, out_dir=OUT_DIR
):
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    df_test, pdc_test, sc_dc_te, lam_b9, pr_test, q_rc = sample_replay_data

    # --------------------------------------------------------------------------
    # FIGURE 1: Task, Information Availability & Median-Lifetime Unit Replay
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={'wspace': 0.22})

    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_title('(a) Strict Data Partition & Lead-Time Right-Censoring Timeline', pad=10)

    blocks = [
        (1, 76, 22.5, 18, PALETTE['navy'], 'Fit Split\n(60 Units)\nEncoder + Head $\\theta$'),
        (25.5, 76, 22.5, 18, PALETTE['teal'], 'Tune Split\n(30 Units)\nBias + Scale + $\\lambda_{tune}$'),
        (50, 76, 22.5, 18, PALETTE['gold'], 'Cal Split\n(40 Units)\nUnit Score $B_u \\to \\hat{q}_{1-\\epsilon}$'),
        (74.5, 76, 24.5, 18, PALETTE['red'], 'Test Fleet\n(50 Units/Seed)\nLead-Time Replay $\\tau_u$'),
    ]
    for bx, by, bw, bh, bcol, btxt in blocks:
        patch = FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.6', facecolor=bcol, edgecolor='#222222', lw=0.8, alpha=0.92)
        ax.add_patch(patch)
        ax.text(bx + bw / 2.0, by + bh / 2.0, btxt, color='white', ha='center', va='center', fontsize=7.0, fontweight='bold')
    for ax_x in [23.6, 48.1, 72.6]:
        ax.annotate('', xy=(ax_x + 1.8, 85), xytext=(ax_x, 85), arrowprops=dict(arrowstyle='->', lw=1.3, color='#222222'))

    ax.text(2, 64, 'Case 1: Preventive Replacement ($Z_u = 0$, Right-Censored at $U_u = \\tau_u + \\ell < T_u$)', fontsize=8.0, fontweight='bold', color=PALETTE['navy'])
    ax.plot([5, 92], [54, 54], color='#333333', lw=2.0)
    ax.plot([5, 38], [54, 54], color=PALETTE['teal'], lw=4.5)
    ax.plot([38, 64], [54, 54], color=PALETTE['gold'], lw=4.5)
    ax.plot([64, 92], [54, 54], color=PALETTE['slate'], lw=2.2, ls=':')
    for px, ptxt, pcol in [
        (5, '$t=1$', '#222222'),
        (38, 'Alarm $\\tau_u$\n($g_{u,t} \\leq \\ell$)', PALETTE['red']),
        (64, 'Replaced $U_u=\\tau_u+\\ell$\n($Z_u=0$ Observable)', PALETTE['teal']),
        (92, 'Unobserved $T_u$\n($W_u$ Hidden)', PALETTE['slate'])
    ]:
        ax.scatter([px], [54], color=pcol, s=42, zorder=5)
        ax.text(px, 43, ptxt, ha='center', va='top', fontsize=7.2, color=pcol, fontweight='bold' if pcol != PALETTE['slate'] else 'normal')

    ax.text(2, 28, 'Case 2: Late Alarm / Breakdown ($Z_u = 1$, Uncensored Failure at $U_u = T_u \\leq \\tau_u + \\ell$)', fontsize=8.0, fontweight='bold', color=PALETTE['red'])
    ax.plot([5, 92], [18, 18], color='#333333', lw=2.0)
    ax.plot([5, 68], [18, 18], color=PALETTE['red'], lw=4.5)
    for px, ptxt, pcol in [
        (5, '$t=1$', '#222222'),
        (42, 'Late Alarm $\\tau_u$', PALETTE['gold']),
        (68, 'Breakdown $T_u \\leq \\tau_u+\\ell$\n($Z_u=1, T_u$ Obs.)', PALETTE['red']),
        (92, 'Planned $\\tau_u+\\ell$\n(Too Late)', PALETTE['slate'])
    ]:
        ax.scatter([px], [18], color=pcol, s=42, zorder=5)
        ax.text(px, 7, ptxt, ha='center', va='top', fontsize=7.2, color=pcol, fontweight='bold')

    ax = axes[1]
    u_lifetimes = df_test.groupby('unit_id')['max_cycle'].first().sort_values()
    median_uid = int(u_lifetimes.index[len(u_lifetimes) // 2])
    mask_u = (df_test['unit_id'] == median_uid).to_numpy()
    cyc = df_test.loc[mask_u, 'cycle'].to_numpy()
    T_u = float(df_test.loc[mask_u, 'max_cycle'].iloc[0])
    p_dc_u = pdc_test[mask_u]
    s_dc_u = sc_dc_te[mask_u]
    g_dc_u = np.maximum(0.0, p_dc_u - lam_b9 * s_dc_u)
    g_rc_u = np.maximum(0.0, pr_test[mask_u] - q_rc)

    ax.plot(cyc, T_u - cyc, color=PALETTE['navy'], lw=1.9, label='True Uncapped Remaining Life $T_u - t$')
    ax.plot(cyc, p_dc_u, color=PALETTE['teal'], ls='--', lw=1.5, label='Causal Prefix Forecast $f_\\theta(x_{u,1:t})$')
    ax.plot(cyc, g_dc_u, color=PALETTE['red'], lw=2.1, label='DC-PdM Calibrated Statistic $g_{u,t}(\\hat{\\lambda})$')
    ax.plot(cyc, g_rc_u, color=PALETTE['slate'], ls=':', lw=1.5, label='Cycle-CP Bound (B5)')
    ax.axhline(LEAD_TIME, color='#222222', ls='-.', lw=1.1, label=f'Lead-Time Threshold $\\ell = {int(LEAD_TIME)}$ cyc')

    trig_dc = np.flatnonzero(g_dc_u <= LEAD_TIME)
    if len(trig_dc) > 0:
        tau_dc = cyc[trig_dc[0]]
        ax.scatter([tau_dc], [LEAD_TIME], color=PALETTE['red'], s=52, zorder=6)
        ax.annotate(
            f'Median-Life Unit #{median_uid} ($T_u={int(T_u)}$)\nAlarm $\\tau_u={int(tau_dc)}$, Replaced at {int(tau_dc+LEAD_TIME)}',
            xy=(tau_dc, LEAD_TIME), xytext=(max(95, tau_dc - 68), LEAD_TIME + 72),
            arrowprops=dict(arrowstyle='->', color=PALETTE['red'], lw=1.2),
            fontsize=7.2, fontweight='bold', color=PALETTE['red'],
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=PALETTE['red'])
        )
    ax.set_title(f'(b) Median-Lifetime Unit Replay (Unit #{median_uid}, 50th-Pct $T_u={int(T_u)}$)', pad=10)
    ax.set_xlabel('Operating Cycle $t$')
    ax.set_ylabel('Remaining Cycles')
    ax.set_ylim(-5, 235)
    ax.legend(loc='upper right', fontsize=7.4, frameon=True, facecolor='white')

    ins = ax.inset_axes([0.07, 0.08, 0.37, 0.28])
    tau_b10 = np.where(df_b1_b10_unit['B10_tau'] < 0, np.inf, df_b1_b10_unit['B10_tau'])
    tau_b5 = np.where(df_b1_b10_unit['B5_tau'] < 0, np.inf, df_b1_b10_unit['B5_tau'])
    raw_sm_b10 = df_b1_b10_unit['T_u'] - (tau_b10 + LEAD_TIME)
    raw_sm_b5 = df_b1_b10_unit['T_u'] - (tau_b5 + LEAD_TIME)
    n_clip_b10 = int(np.sum((raw_sm_b10 < -25) | (raw_sm_b10 > 60)))
    n_clip_b5 = int(np.sum((raw_sm_b5 < -25) | (raw_sm_b5 > 60)))
    ins.hist(np.clip(raw_sm_b5, -25, 60), bins=20, color=PALETTE['slate'], alpha=0.45, density=True, label='B5')
    ins.hist(np.clip(raw_sm_b10, -25, 60), bins=20, color=PALETTE['red'], alpha=0.65, density=True, label='B10')
    ins.axvline(0.0, color='#222222', ls='--', lw=1.0)
    ins.set_title(f'Buffer $T_u-(\\tau_u+\\ell)$ (Clip: {n_clip_b10}/{n_clip_b5})', fontsize=6.6, fontweight='bold')
    ins.tick_params(labelsize=6.2)

    for d_target in [inter_dir, out_dir]:
        plt.savefig(f'{d_target}/fig1_framework_and_trajectories.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{d_target}/fig1_framework_and_trajectories.pdf', bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 2: Main Effects (Including B2!) — 20-Run Boxplot + Paired Diff & Risk/Waste
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={'wspace': 0.14})

    focus_methods = [
        'B2: Cost-Tuned PTO Margin',
        'B3: Smooth One-Sided Pinball (0.05) + Residual CP',
        'B5: Cycle-Level One-Sided Residual CP',
        'B6: Cycle-Level Online ACI',
        'B7: In-House Smooth-Cost Margin Tuning (Uncal.)',
        'B8: Simple-Combo (Ridge + Pure Unit-CP / Binary CRC)',
        'B9: DC-PdM Static (Cost-Tuned + Unit-CP Floor)',
        'B10: DC-PdM Online (Event-ACI Z_u)',
    ]
    short_m2 = [
        'B2: Cost-Tuned PTO Margin',
        'B3: Smooth Pinball + CP',
        'B5: Cycle-Level Residual CP',
        'B6: Cycle-Level Online ACI',
        'B7: Smooth-Cost Tune Only',
        'B8: Ridge + Pure Unit-CP (CRC)',
        'B9: DC-PdM Static (Tune+Unit-CP)',
        'B10: DC-PdM Online (Event-ACI)'
    ]
    cols_m2 = ['#E76F51', '#8D99AE', PALETTE['slate'], '#6C757D', PALETTE['gold'], PALETTE['purple'], PALETTE['teal'], PALETTE['red']]

    ax = axes[0]
    b10_vec = df_b1_b10_seed[df_b1_b10_seed['method'] == focus_methods[-1]].sort_values(['scenario', 'seed'])['j_life_rate'].to_numpy()
    seed_arrays = []
    paired_diff_str = []
    for m_full in focus_methods:
        vec = df_b1_b10_seed[df_b1_b10_seed['method'] == m_full].sort_values(['scenario', 'seed'])['j_life_rate'].to_numpy()
        seed_arrays.append(vec)
        diff = vec - b10_vec
        m_d = float(np.mean(diff))
        ci_d = float(stats.t.ppf(0.975, len(diff) - 1) * np.std(diff, ddof=1) / np.sqrt(len(diff)))
        if m_full == focus_methods[-1]:
            paired_diff_str.append(f'Ref Mean={np.mean(vec):.3f} (SD={np.std(vec,ddof=1):.3f})')
        else:
            paired_diff_str.append(f'$\\Delta = +{m_d:.3f} \\pm {ci_d:.3f}$')

    y_pos = np.arange(len(focus_methods))
    bp = ax.boxplot(seed_arrays, positions=y_pos, vert=False, widths=0.50, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], cols_m2):
        patch.set_facecolor(col)
        patch.set_alpha(0.72)
        patch.set_edgecolor('#222222')
    for median in bp['medians']:
        median.set_color('#111111')
        median.set_linewidth(1.6)

    rng_jit = np.random.default_rng(42)
    for idx_m, (vec, col, d_txt) in enumerate(zip(seed_arrays, cols_m2, paired_diff_str)):
        jit = rng_jit.uniform(-0.14, 0.14, size=len(vec))
        ax.scatter(vec, np.full_like(vec, idx_m) + jit, color=col, edgecolor='#111111', s=18, alpha=0.85, zorder=4)
        ax.text(max(vec) + 0.015, idx_m, d_txt, va='center', ha='left', fontsize=7.2, fontweight='bold', color='#222222')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_m2, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0.72, 1.52)
    ax.set_xlabel('Lifecycle Cost Rate $\\hat{J}_{life}$ (Cost / 100 Useful Cycles, 20 Scenario-Seed Runs)')
    ax.set_title('(a) 20-Run Cost Distribution & Paired Difference vs. B10 ($\\pm 95\\%$ CI)', pad=10)

    ax = axes[1]
    sub_sum = df_b1_b10_sum.set_index('method').loc[focus_methods].reset_index()
    f_rates = sub_sum['fail_pct'].to_numpy()
    ucb_rates = sub_sum['clopper_pearson_ucb95_pct'].to_numpy()
    err_plus = ucb_rates - f_rates

    ax.barh(y_pos - 0.16, f_rates, height=0.30, color=cols_m2, edgecolor='#222222', lw=0.6, label='Empirical Failure Rate (%)')
    ax.errorbar(f_rates, y_pos - 0.16, xerr=[np.zeros_like(err_plus), err_plus], fmt='none', ecolor='#111111', elinewidth=1.2, capsize=2.5)
    ax.axvline(5.0, color=PALETTE['red'], ls='--', lw=1.3, label='Target Risk Level $\\epsilon = 5.0\\%$')

    ax_top = ax.twiny()
    ax_top.spines['top'].set_visible(True)
    ax_top.grid(False)
    for idx_m, m_full in enumerate(focus_methods):
        w_seed_vec = df_b1_b10_seed[df_b1_b10_seed['method'] == m_full]['waste_cycles'].to_numpy()
        q25, q75 = np.percentile(w_seed_vec, [25, 75])
        w_mean = float(sub_sum.loc[idx_m, 'waste_cycles_mean'])
        ax_top.plot([q25, q75], [idx_m + 0.18, idx_m + 0.18], color=PALETTE['navy'], lw=2.8, alpha=0.85)
        ax_top.scatter([w_mean], [idx_m + 0.18], color='white', edgecolor=PALETTE['navy'], s=32, lw=1.4, zorder=5)
        ax.text(
            max(ucb_rates[idx_m], 5.2) + 0.35, idx_m,
            f'Fail={f_rates[idx_m]:.1f}% (UCB={ucb_rates[idx_m]:.1f}%) | Waste={w_mean:.1f}c (IQR {q25:.1f}-{q75:.1f})',
            va='center', ha='left', fontsize=7.0, fontweight='bold', color='#222222'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(['' for _ in short_m2])
    ax.invert_yaxis()
    ax.set_xlim(0, 17.8)
    ax_top.set_xlim(0, 42.0)
    ax.set_xlabel('Breakdown Rate (%) & Descriptive 95% Clopper-Pearson UCB')
    ax_top.set_xlabel('Unused Useful Life $W_u$ (Cycles: Mean Dot & 25th-75th Seed IQR)', color=PALETTE['navy'])
    ax.set_title('(b) Breakdown Incidence & Useful-Life Waste (1,000 Units)', pad=10)

    for d_target in [inter_dir, out_dir]:
        plt.savefig(f'{d_target}/fig2_benchmark_comparison.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{d_target}/fig2_benchmark_comparison.pdf', bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 3: Revised Attribution (R3-01, R3-02, R3-05)
    #   (a) 4-Way Margin Attribution (Cycle-CP, Unit-CP Only, Cost-Tuning Only, Cost-Tuning + Unit-CP)
    #   (b) Prediction-Bias Stress Test with Matched-Initial Static Control + Paired Ablation Inset
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={'wspace': 0.22})

    ax = axes[0]
    m_order = [
        ('ridge', 'cycle_CP', 'Ridge: Cycle-CP Only'),
        ('ridge', 'unit_CP_no_tune_floor', 'Ridge: Pure Unit-CP (Binary CRC)'),
        ('ridge', 'tuned_margin_only', 'Ridge: Cost-Tuned Margin Only'),
        ('ridge', 'unit_CP_plus_tune_floor', 'Ridge: Cost-Tuned + Unit-CP Floor'),
        ('linex', 'cycle_CP', 'Linex: Cycle-CP Only'),
        ('linex', 'unit_CP_no_tune_floor', 'Linex: Pure Unit-CP (Binary CRC)'),
        ('linex', 'tuned_margin_only', 'Linex: Cost-Tuned Margin Only'),
        ('linex', 'unit_CP_plus_tune_floor', 'Linex: Cost-Tuned + Unit-CP Floor (B9)'),
    ]
    y_m = np.arange(len(m_order))
    j_vals, sd_vals, f_vals, w_vals, lbl_m = [], [], [], [], []
    for p_k, pol_k, disp_l in m_order:
        r_m = df_mctrl_sum[(df_mctrl_sum.predictor == p_k) & (df_mctrl_sum.policy == pol_k)].iloc[0]
        j_vals.append(float(r_m.j_life_mean))
        sd_vals.append(float(r_m.j_life_sd))
        f_vals.append(float(r_m.fail_pct))
        w_vals.append(float(r_m.waste_cycles))
        lbl_m.append(disp_l)

    cols_m3 = [
        PALETTE['slate'], '#8D99AE', PALETTE['gold'], PALETTE['navy'],
        PALETTE['slate'], '#8D99AE', PALETTE['orange'], PALETTE['red']
    ]
    bars_m = ax.barh(y_m, j_vals, xerr=sd_vals, capsize=2.5, height=0.56, color=cols_m3, edgecolor='#222222', lw=0.6)
    ax.axhline(3.5, color='#666666', ls=':', lw=1.0)
    ax.set_yticks(y_m)
    ax.set_yticklabels(lbl_m, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.92)
    ax.set_xlabel('Lifecycle Cost Rate $\\hat{J}_{life}$ on S-FD004 (250 Units; Bars: Mean $\\pm$ Seed SD)')
    ax.set_title('(a) 4-Way Margin Attribution: Cost Tuning vs. Unit-CP Floor', pad=10)
    for bar, sdv, fv, wv in zip(bars_m, sd_vals, f_vals, w_vals):
        bw = bar.get_width()
        ax.text(bw + sdv + 0.02, bar.get_y() + bar.get_height() / 2.0,
                f'{bw:.3f} (Fail={fv:.1f}%, Waste={wv:.1f}c)', va='center', ha='left', fontsize=7.0, fontweight='bold')

    # Panel (b): Prediction-Bias Stress Test with Matched-Initial Static Control + Paired Ablation CIs Inset
    ax = axes[1]
    dr_x = df_drift['drift_rate'].to_numpy()
    ax.plot(dr_x, df_drift['j_life_static'].to_numpy(), marker='s', ls='--', color=PALETTE['slate'], lw=1.8, label='Static ($\\alpha=0.05$ Original)')
    ax.plot(dr_x, df_drift['j_life_matched_static'].to_numpy(), marker='D', color=PALETTE['navy'], lw=2.1, label='Static (Matched $\\alpha_0=0.0325$ Frozen)')
    ax.plot(dr_x, df_drift['j_life_online'].to_numpy(), marker='o', color=PALETTE['red'], lw=2.1, label='Online Event-ACI ($\\alpha_0=0.0325$)')

    for dx_v, fs_v, fm_v, fo_v, js_v, jm_v, jo_v in zip(
        dr_x, df_drift['fail_pct_static'], df_drift['fail_pct_matched_static'], df_drift['fail_pct_online'],
        df_drift['j_life_static'], df_drift['j_life_matched_static'], df_drift['j_life_online']
    ):
        if dx_v in (0.0, 0.4, 0.6, 0.8):
            ax.annotate(f'Orig:{fs_v:.1f}%', (dx_v, js_v), xytext=(dx_v - 0.04, js_v + 0.026), fontsize=6.6, color=PALETTE['slate'], fontweight='bold')
            ax.annotate(f'On:{fo_v:.1f}%', (dx_v, jo_v), xytext=(dx_v - 0.03, jo_v + 0.008), fontsize=6.6, color=PALETTE['red'], fontweight='bold')
            ax.annotate(f'Match:{fm_v:.1f}%', (dx_v, jm_v), xytext=(dx_v - 0.04, jm_v - 0.038), fontsize=6.6, color=PALETTE['navy'], fontweight='bold')

    ax.set_xlabel('Injected Prediction-Bias Severity $\\delta_{bias}$ (S-FD004, 250 Units/Level)')
    ax.set_ylabel('Lifecycle Cost Rate $\\hat{J}_{life}$ (Cost / 100 Useful Cycles)')
    ax.set_title('(b) Online vs. Matched-Initial Static & Single-Module Paired CIs', pad=10)
    ax.set_xlim(-0.06, 0.86)
    ax.set_ylim(0.74, 1.34)
    ax.legend(loc='lower right', fontsize=7.2, frameon=True, facecolor='white')

    # Inset: Single-Module Removal Paired Cost Differences (Mean Diff +/- 95% t-CI across 5 Seeds)
    ins_b = ax.inset_axes([0.23, 0.52, 0.47, 0.42])
    comp_keys = ['L1 minus L4', 'A0 minus A1', 'A3 minus A0', 'A5 minus A0', 'A4 minus A0', 'A6 minus A0']
    comp_lbls = ['L1-L4 (Quad)', 'A0-A1 (+Online)', 'A3-A0 (-Linex)', 'A5-A0 (-Smooth)', 'A4-A0 (-Weight)', 'A6-A0 (-PAVA)']
    sub_t = df_attr_tests.set_index('comparison').loc[comp_keys].reset_index()
    yc_i = np.arange(len(comp_keys))
    md_i = sub_t['mean_difference'].to_numpy()
    err_i = sub_t['ci95_high'].to_numpy() - md_i
    ins_b.errorbar(md_i, yc_i, xerr=err_i, fmt='o', color=PALETTE['navy'], ecolor=PALETTE['red'], elinewidth=1.3, capsize=2.5, markersize=4.5)
    ins_b.axvline(0.0, color='#222222', ls='--', lw=1.0)
    ins_b.set_yticks(yc_i)
    ins_b.set_yticklabels(comp_lbls, fontsize=6.3)
    ins_b.invert_yaxis()
    ins_b.set_xlim(-0.05, 0.14)
    ins_b.set_title('5-Seed Paired $\\Delta \\hat{J}_{life} \\pm 95\\%$ t-CI (All Cross 0)', fontsize=6.6, fontweight='bold')
    ins_b.tick_params(labelsize=6.2)

    for d_target in [inter_dir, out_dir]:
        plt.savefig(f'{d_target}/fig3_drift_and_risk_calibration.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{d_target}/fig3_drift_and_risk_calibration.pdf', bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 4: Applicability Boundaries — Right-Censoring, Mismatch & Unseen Regimes
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), gridspec_kw={'wspace': 0.20})

    ax = axes[0]
    c_labels = [
        'C1: Cycle-ACI\n(Oracle Uncensored)',
        'C2: Cycle-ACI\n(Censored Imputation)',
        'C3: Cycle-ACI\n(Uncensored-Only Bias)',
        'C4: DC-PdM Static\n(Zero Online Labels)',
        'C5: DC-PdM Event-ACI\n(Observable $Z_u$ Only)'
    ]
    c_j = df_censor_sum['j_life_rate'].to_numpy()
    c_f = df_censor_sum['fail_pct'].to_numpy()
    c_w = df_censor_sum['waste_cycles'].to_numpy()
    y_c = np.arange(len(c_labels))
    bars_c = ax.barh(y_c, c_j, height=0.52, color=[PALETTE['slate'], PALETTE['gold'], PALETTE['orange'], PALETTE['teal'], PALETTE['red']], edgecolor='#222222', lw=0.6)
    ax.set_yticks(y_c)
    ax.set_yticklabels(c_labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.08)
    ax.set_xlabel('Lifecycle Cost Rate $\\hat{J}_{life}$ (Cost / 100 Useful Operating Cycles)')
    ax.set_title('(a) Right-Censoring Observability & Feedback Regimes (S-FD004)', pad=10)
    for bar, fm, wm in zip(bars_c, c_f, c_w):
        bw = bar.get_width()
        ax.text(bw + 0.025, bar.get_y() + bar.get_height() / 2.0,
                f'{bw:.2f} (Fail={fm:.1f}%, Waste={wm:.1f}c)', va='center', ha='left', fontsize=7.3, fontweight='bold')

    ax = axes[1]
    mode_labels = [
        'Neg. Dips\n(Matched)',
        'No Shocks\n(Gaussian)',
        'Pos. Spikes\n(Upward)',
        'Bidirectional\n(Student-t)',
        'True Recovery\n(Non-Monotone)',
        'Unseen\nRegime 7'
    ]
    rmse_unc = df_mismatch_sum['rmse_unconstrained_ridge'].to_numpy()
    rmse_dcm = df_mismatch_sum['rmse_dc_pdm'].to_numpy()
    j_dcm = df_mismatch_sum['j_life_dc_pdm'].to_numpy()
    f_dcm = df_mismatch_sum['fail_pct_dc_pdm'].to_numpy()
    xm = np.arange(len(mode_labels))

    ax.bar(xm - 0.17, rmse_unc, 0.34, color=PALETTE['slate'], edgecolor='#222222', lw=0.5, label='Unconstrained EMA + Ridge (RMSE)')
    ax.bar(xm + 0.17, rmse_dcm, 0.34, color=PALETTE['red'], edgecolor='#222222', lw=0.5, label='Causal Monotone + Linex (RMSE)')
    for x_i, ru, rd in zip(xm, rmse_unc, rmse_dcm):
        delta_r = rd - ru
        sign_str = '+' if delta_r >= 0 else ''
        col_t = PALETTE['red'] if delta_r < 0 else '#222222'
        ax.text(x_i, max(ru, rd) + 0.35, f'{sign_str}{delta_r:.2f}c', ha='center', va='bottom', fontsize=6.9, fontweight='bold', color=col_t)

    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.grid(False)
    ax2.plot(xm, j_dcm, color=PALETTE['navy'], marker='D', lw=1.8, ls='--', label='DC-PdM Cost Rate $\\hat{J}_{life}$')
    for x_i, jv, fv in zip(xm, j_dcm, f_dcm):
        y_off = -0.22 if x_i == 5 else 0.08
        ax2.annotate(f'J={jv:.2f}\n({fv:.1f}% fail)', (x_i, jv), xytext=(x_i, jv + y_off),
                     ha='center', fontsize=6.6, fontweight='bold', color=PALETTE['navy'])

    ax.set_xticks(xm)
    ax.set_xticklabels(mode_labels, fontsize=7.5)
    ax.set_ylabel('Test RUL RMSE (Cycles, Bars)')
    ax2.set_ylabel('DC-PdM Cost Rate $\\hat{J}_{life}$ (Cost / 100 Cyc, Dashed Line)', color=PALETTE['navy'])
    ax.set_ylim(0, 17.5)
    ax2.set_ylim(0.65, 2.15)
    ax.set_title('(b) Applicability Boundaries: Disturbance & Mechanism Mismatch', pad=10)
    ax.legend(loc='upper left', fontsize=7.2, frameon=True, facecolor='white')

    for d_target in [inter_dir, out_dir]:
        plt.savefig(f'{d_target}/fig4_ablation_and_pareto.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{d_target}/fig4_ablation_and_pareto.pdf', bbox_inches='tight')
    plt.close()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    tmp_out = '/tmp/dc_pdm_live_out'
    if os.path.exists(tmp_out):
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_out, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INTER_DIR, exist_ok=True)
    t0 = time.time()

    print("[1/6] Running self-tests, counterexample checks, and Binary CRC equivalence checks...")
    checks = run_all_self_tests(out_dir=tmp_out)

    print("[2/6] Running fresh Experiment A: Single-Factor Forensic Audit (D0-D5 across 1,800 trajectories)...")
    df_d0_d5_seed, summary_d0_d5 = run_forensic_d0_d5(out_dir=tmp_out)

    print("[3/6] Running fresh Experiment B & C: 1,000-Unit Policy Benchmark (B1-B10) & Strictly Exchangeable Benchmark...")
    df_b1_b10_unit, df_b1_b10_seed, df_b1_b10_sum, sample_replay_data, fd004_cache = run_unit_policy_benchmark_b1_b10(out_dir=tmp_out)

    print("[4/6] Running fresh Experiment D & E: Preventive Right-Censoring, Cost Ratio & 6-Mode Mismatch Stress Tests...")
    df_censor_sum, df_rho, df_mismatch_sum = run_censoring_and_mismatch_experiments(fd004_cache, out_dir=tmp_out)

    print("[5/6] Running fresh Experiment F: 4-Way Margin Attribution, Matched Losses, Ablations, HI Metrics & CPU Timing...")
    df_attr_sum, df_mctrl_sum, df_attr_tests, df_hi_sum, df_time_sum, df_drift = run_controlled_attribution_and_efficiency(fd004_cache, out_dir=tmp_out)

    print("[5b/6] Running Revision 5 Diagnostics: Non-IID Beta Counterexample, Fair Re-Optimized Cost/Waste Sensitivity, Running-Max Isolation & Full-Pipeline CPU Timers...")
    df_time_sum = run_revision5_diagnostics_and_fair_sensitivity(fd004_cache, out_dir=tmp_out)

    print("[6/6] Generating reconstructed Figures 1-4 (vector PDF + 600 DPI PNG) directly from fresh DataFrames...")
    generate_reconstructed_figures_1_to_4(
        df_b1_b10_unit, df_b1_b10_seed, df_b1_b10_sum,
        df_censor_sum, df_mismatch_sum, df_attr_sum, df_mctrl_sum, df_attr_tests, df_drift,
        sample_replay_data, inter_dir=INTER_DIR, out_dir=tmp_out
    )

    evidence_files = [
        'self_test_report.json',
        'hi_metric_counterexamples.csv',
        'unit_score_vs_binary_crc.csv',
        'non_monotone_latent_check.csv',
        'forensic_d0_d5_per_seed.csv',
        'forensic_d0_d5_summary.csv',
        'unit_policy_b1_b10_per_unit.csv',
        'unit_policy_b1_b10_per_unit_long.csv',
        'unit_policy_b1_b10_per_seed.csv',
        'unit_policy_b1_b10_summary.csv',
        'main_paired_tests_with_seed_blocks.csv',
        'exchangeable_zero_drift_per_seed.csv',
        'exchangeable_zero_drift_summary.csv',
        'censoring_observability_per_unit.csv',
        'censoring_observability_summary.csv',
        'rho_sensitivity_summary.csv',
        'mismatch_stress_per_unit.csv',
        'mismatch_stress_summary.csv',
        'censoring_and_mismatch_per_unit.csv',
        'controlled_attribution_per_seed.csv',
        'controlled_attribution_summary.csv',
        'margin_controls_per_seed.csv',
        'margin_controls_summary.csv',
        'attribution_margin_floor_activity.csv',
        'attribution_paired_tests.csv',
        'loss_head_pre_post_residuals.csv',
        'hi_quality_metrics_summary.csv',
        'cpu_resource_timing_summary.csv',
        'cpu_timing_raw_records.csv',
        'cpu_timing_step_level_raw.csv',
        'cpu_timing_unit_level_raw.csv',
        'cpu_stream_vs_reference_per_unit.csv',
        'cpu_prefix_forecast_equivalence.csv',
        'exchangeable_non_iid_beta_counterexample.csv',
        'exchangeable_incremental_cost_paired.csv',
        'exchangeable_risk_bounds_descriptive.csv',
        'fixed_policy_waste_weight_rescoring.csv',
        'fair_reoptimized_cost_sensitivity.csv',
        'fair_reoptimized_cost_sensitivity_per_seed.csv',
        'fair_reoptimized_rho_sensitivity.csv',
        'fair_reoptimized_rho_sensitivity_per_seed.csv',
        'fair_sensitivity_independent_paired.csv',
        'recovery_window_diagnostics.csv',
        'running_max_isolation_per_seed.csv',
        'running_max_isolation_summary.csv',
        'running_max_paired_uncertainty.csv',
        'mismatch_frozen_transfer_per_seed.csv',
        'mismatch_frozen_transfer_summary.csv',
        'unseen_regime_generator_shift.csv',
        'drift_scaling_per_seed.csv',
        'drift_scaling_summary.csv',
        'drift_matched_initialization_paired.csv',
        'fig1_framework_and_trajectories.pdf',
        'fig2_benchmark_comparison.pdf',
        'fig3_drift_and_risk_calibration.pdf',
        'fig4_ablation_and_pareto.pdf',
        'fig1_framework_and_trajectories.png',
        'fig2_benchmark_comparison.png',
        'fig3_drift_and_risk_calibration.png',
        'fig4_ablation_and_pareto.png',
    ]

    manifest = {
        'script_sha256': _sha256_file(__file__),
        'python_version': sys.version,
        'platform': platform.platform(),
        'c_pava_active': bool(_init_c_pava()),
        'files': {}
    }
    for fname in evidence_files:
        fpath = f'{tmp_out}/{fname}'
        if os.path.exists(fpath):
            manifest['files'][fname] = {
                'sha256': _sha256_file(fpath),
                'bytes': os.path.getsize(fpath)
            }
    with open(f'{tmp_out}/manifest_sha256.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    evidence_files.append('manifest_sha256.json')

    zip_path = f'{tmp_out}/dc_pdm_full_experimental_evidence.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in evidence_files:
            fpath = f'{tmp_out}/{fname}'
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)

    for fname in evidence_files + ['dc_pdm_full_experimental_evidence.zip']:
        if os.path.exists(f'{tmp_out}/{fname}'):
            shutil.copy2(f'{tmp_out}/{fname}', f'{OUT_DIR}/{fname}')

    elapsed = round(time.time() - t0, 2)
    print(f"All Revision 4 experiments, attribution controls, and figures completed in {elapsed}s!")