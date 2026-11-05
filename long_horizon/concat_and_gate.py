"""Concatenate the 50-trajectory parts into one truth file and apply the energy gate.

Paper: Appendix A.9; called by long_horizon/run_pipeline.sh.

Usage:
    python long_horizon/concat_and_gate.py OUT FRAME_INDEX PART [PART ...]

Gate: mean omega^2 at t = 1 must be within [0.90, 1.10] of the benchmark test split; the
script exits with code 1 otherwise, so nothing is evaluated on a file that fails.
Benchmark path: $LGNKM_BENCH (default data/ns/NavierStokes_V1e-5_N1200_T20.mat).
"""
import json
import sys
import numpy as np
import h5py
from check_ic_energy import BENCH, load_u, frame_stats

LO, HI = 0.90, 1.10

out, idx, parts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
us, t0 = [], None
for p in parts:
    with h5py.File(p, 'r') as f:
        us.append(np.array(f['u']))
        t = np.array(f['t'])
    assert t0 is None or np.array_equal(t, t0), f'time vector differs in {p}'
    t0 = t
u = np.concatenate(us, axis=0)
with h5py.File(out, 'w') as fo:
    fo.create_dataset('u', data=u, compression='gzip')
    fo.create_dataset('t', data=t0)

b, b_se, _, _ = frame_stats(load_u(BENCH), 0, slice(1000, 1200))
m, se, n, shape = frame_stats(u, idx)
ratio = m / b
ok = LO <= ratio <= HI
res = dict(file=out, shape=list(shape), n=n, mean_w2=m, se=se, bench_mean_w2=b, bench_se=b_se,
           ratio=ratio, gate=[LO, HI], passed=bool(ok),
           per_part=[float(frame_stats(x, idx)[0] / b) for x in us])
print(json.dumps(res, indent=1))
json.dump(res, open(out.replace('.mat', '_gate.json'), 'w'), indent=1)
sys.exit(0 if ok else 1)
