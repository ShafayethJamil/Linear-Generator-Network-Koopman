"""
Gate check for the regenerated long-horizon / fine-dt NS truth (camera-ready, 2026-11-03).

Prints mean omega^2 at the first integer-time frame (t = 1) for each file, with the
standard error over trajectories, and the ratio to the Li et al. benchmark test split
(NavierStokes_V1e-5_N1200_T20.mat, trajectories 1000:1200, frame index 0).

Usage: python long_horizon/check_ic_energy.py FILE[:frame_index] [FILE[:frame_index] ...]
  frame_index = index of t = 1 in that file (0 for 1-per-unit files, 3 for the
  0.25-spaced fine-dt files whose frames start at t = 0.25).
"""
import os
import sys
import numpy as np
import h5py

# Li et al. benchmark file (see data/README.md); override with LGNKM_BENCH.
BENCH = os.environ.get('LGNKM_BENCH', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'data', 'ns', 'NavierStokes_V1e-5_N1200_T20.mat'))


def load_u(path):
    try:
        with h5py.File(path, 'r') as f:
            return np.array(f['u'])
    except OSError:                       # MATLAB v5 (the Li et al. benchmark)
        from scipy.io import loadmat
        return loadmat(path, variable_names=['u'])['u']


def frame_stats(u, idx, sl=slice(None)):
    # every file here is (N, H, W, T): HDF5 as written by the generators, and
    # the benchmark as loaded by scipy
    w = u[sl, :, :, idx].astype(np.float64)
    per = (w ** 2).mean(axis=(1, 2))
    return per.mean(), per.std(ddof=1) / np.sqrt(len(per)), len(per), u.shape


if __name__ == '__main__':
    ub = load_u(BENCH)
    b, b_se, b_n, b_shape = frame_stats(ub, 0, slice(1000, 1200))
    print(f'benchmark test split   shape {b_shape}  n={b_n}  mean w^2 = {b:.4f} +- {b_se:.4f}')
    for arg in sys.argv[1:]:
        path, _, idx = arg.partition(':')
        m, se, n, shape = frame_stats(load_u(path), int(idx or 0))
        print(f'{path}  shape {shape}  n={n}  mean w^2 = {m:.4f} +- {se:.4f}  ratio to benchmark {m / b:.3f}')
